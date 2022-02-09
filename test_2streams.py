import argparse
import os
import warnings
import multiprocessing
from collections import OrderedDict

import numpy as np
import torch
import pickle
from torchsummary import summary
from torch import nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.autograd import Variable
from torchnet import meter
from torchvision import models, datasets

import globalvar as gl
import torch_utils
from FixRes.transforms import get_transforms
from dataset import Hand
from get_model import get_model
from utils.datasets import *
from visualize import Visualizer
from log import logger

from models import *
from sklearn.metrics import roc_curve, auc
from scipy import interp
from itertools import cycle



def hook(model, input, output):
    global conv1024
    # conv1024 = output.detach()
    conv1024 = output


if __name__ == '__main__':
    # 忽略警告信息
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='se_viewfusion_resnet50', help='')
    parser.add_argument('--test_dir', type=str, default=r'D:\Dataset\compcars\absolute_path_test.txt', help='')
    parser.add_argument('--img_path', type=str, default=r'D:\Dataset\compcars1\compcars\compcars', help='')
    parser.add_argument('--img_size', type=int, default=224, help='')
    parser.add_argument('--batch_size', type=int, default=8, help='')
    parser.add_argument('--num_workers', type=int, default=8, help='')
    parser.add_argument('--gpu', type=tuple, default=(0,), help='')
    parser.add_argument('--model_params', type=str,
                        default=r'D:\save_model_params\se_viewfusion_resnet50_1_RA\best\best_67.pt',
                        help='')
    parser.add_argument('--pretrained', type=bool, default=False, help='')
    # yolov3
    parser.add_argument('--cfg', type=str,
                        default=r'D:\EPCNN\7.14_first_thesis\yolov3-master\cfg\yolov3-compcars-5views-tiny.cfg',
                        help='')
    parser.add_argument('--weights', type=str, default='weights/compcars5views224224/best.pt',
                        help='')
    parser.add_argument('--conf_thres', type=float, default=0.1)
    parser.add_argument('--nms_thres', type=float, default=0.5)
    opt = parser.parse_args()

    device = torch_utils.select_device()
    gl._init()
    # create dataset
    print("Loading dataset...")
    # val_data = Hand(opt.test_dir, transforms=None, train=False, img_size=opt.img_size)
    # val_dataloader = DataLoader(val_data, batch_size=opt.batch_size, shuffle=False, num_workers=opt.num_workers,drop_last=True)
    # print('val dataset len: {}'.format(len(val_dataloader.dataset)))
    transf = get_transforms(input_size=opt.img_size, test_size=opt.img_size, kind='full', crop=True,
                            need=('train', 'val'), backbone=None)
    transform_test = transf['val']
    test_set = datasets.ImageFolder(opt.img_path + '\\val', transform=transform_test)

    val_dataloader = torch.utils.data.DataLoader(
        test_set, batch_size=32, shuffle=False, num_workers=(opt.num_workers - 1),
    )

    # model
    model = get_model(opt.model, pretrained=opt.pretrained)
    model_classification = model
    model_classification.cuda()

    gl.set_value('view_info', torch.FloatTensor(2, 5).cuda())
    gl.set_value('yolo_features', torch.FloatTensor(2, 1024, 7, 7).cuda())

    # load model parameter
    if opt.model_params is not None:
        # delete "module."
        module_dict = torch.load(opt.model_params)
        del_dict = OrderedDict()
        for k, v in module_dict.items():
            # print(k)
            name = k[:]  # remove `module.`
            del_dict[name] = v
        model_classification.load_state_dict(del_dict)
        logger.info('Load ckpt from {}'.format(opt.model_params))

    # set CUDA_VISIBLE_DEVICES
    model_classification = nn.DataParallel(model_classification, device_ids=opt.gpu)
    model_classification = model_classification.cuda()

    model_classification.eval()

    # model_pos_estimate
    logger.info('yolov3')
    model_pos_estimate = Darknet(opt.cfg, opt.img_size)
    # model_info(model_pos_estimate)

    model_pos_estimate.load_state_dict(torch.load(opt.weights, map_location=device)['model'])
    model_pos_estimate.to(device).eval()
    acc = 0

    for step, (data, label, img_path) in enumerate(val_dataloader):
        # yolov3 estimate
        # bbox_info = []
        view_info = []
        yolo_features = []
        se_features = []
        model_pos_estimate.module_list._modules.get('12').register_forward_hook(hook)
        for i, i_img_path in enumerate(img_path):
            # BGR
            img0 = cv2.imread(i_img_path)
            cv2.setNumThreads(0)
            cv2.ocl.setUseOpenCL(False)
            # Padded resize
            img, _, _, _ = letterbox(img0, height=opt.img_size)
            # Normalize RGB
            img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB
            img = np.ascontiguousarray(img, dtype=np.float32)  # uint8 to float32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            img = torch.from_numpy(img).unsqueeze(0).to(device)

            pred, _ = model_pos_estimate(img)
            yolo_features.append(conv1024)
            detections_src = non_max_suppression(pred, opt.conf_thres, opt.nms_thres)[0]
            if detections_src is not None and len(detections_src) > 0:
                # bbox_info.append(torch.clamp(detections_src[-1, 0:4].view(1, 4), min=0, max=224))
                view_info.append(detections_src[-1, 7:12].view(1, 5))
            else:
                # bbox_info.append(torch.tensor([[0, 0, 224, 224]], dtype=torch.float32).cuda())
                view_info.append(torch.zeros((1, 5), dtype=torch.float32).cuda())

        view_info = torch.cat(view_info)
        yolo_features = torch.cat(yolo_features)

        # train model
        inputs = Variable(data)
        target = Variable(label)

        inputs = inputs.cuda()
        target = target.cuda()

        gl.set_value('view_info', view_info)
        gl.set_value('yolo_features', yolo_features)
        # gl.set_value('bbox_info', bbox_info)
        output = model_classification(inputs)

        for i in range(output.size()[0]):
            # print(output[i].size())
            score = F.softmax(output[i].view(1, 431), dim=1)
            _, prediction = torch.max(score.data, dim=1)
            # score, aux = model(inputs)
            # loss = criterion(score, target)
            # print(prediction)
            # print(target[i])
            a = (prediction == target[i])
            acc += torch.sum(prediction == target[i])

            # if not prediction == label:
            print(img_path[i])
            print('Prediction number: %d' % prediction[0] + ' Real number: %d' % target[i])

    acc_p = (float(acc)) / (float(len(val_dataloader.dataset)))
    print("The percentage of success is:%.6f" % acc_p + " The accurate value is:%d" % acc)




