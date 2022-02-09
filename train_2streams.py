import argparse
import os
import warnings
import multiprocessing
from collections import OrderedDict

import numpy as np
import torch
from thop import profile
from torch.backends import cudnn
from torchsummary import summary
from torch import nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts, StepLR
from torch.autograd import Variable
from torchnet import meter
from torchvision import datasets

import globalvar as gl
import torch_utils
from FixRes.samplers import RASampler
from FixRes.transforms import get_transforms
from LabelSmoothing import LSR
from dataset import Hand
from get_model import get_model
from init_weight import init_weight
from no_bias_decay import no_bias_decay
from utils.datasets import *
# from network import resnet34, resnet101

from visualize import Visualizer
from log import logger
# from network.senet import *
from models import *
from efficientnet_pytorch import EfficientNet
from torchvision.models import resnet

def hook(model, input, output):
    global conv1024
    # conv1024 = output.detach()
    conv1024 = output

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='se_viewfusion_resnet50')
    parser.add_argument('--batch_size', type=int, default=2, help='')
    parser.add_argument('--gpu', type=tuple, default=(0,), help='')
    parser.add_argument('--train_dir', type=str, default=r'D:\Dataset\compcars\train.txt', help='')
    parser.add_argument('--img_path', type=str, default=r'D:\Dataset\compcars1\compcars\compcars', help='')
    parser.add_argument('--save_freq_epoch', type=int, default=1, help='')
    parser.add_argument('--model_save_path', type=str,
                        default=r'D:\EPCNN\7.14_first_thesis\yolov3-master\saved_model_params\\',
                        help='')
    parser.add_argument('--num_class', type=int, default=431, help='')
    parser.add_argument('--img_size', type=int, default=224, help='')
    parser.add_argument('--num_workers', type=int, default=8, help='')

    # yolov3
    parser.add_argument('--cfg', type=str,
                        default=r'D:\EPCNN\7.14_first_thesis\yolov3-master\cfg\yolov3-compcars-5views-tiny.cfg',
                        help='')
    parser.add_argument('--weights', type=str, default='weights/compcars5views224224/best.pt', help='')
    parser.add_argument('--conf_thres', type=float, default=0.1)
    parser.add_argument('--nms_thres', type=float, default=0.5)
    # train
    parser.add_argument('--last_epoch', type=int, default=-1, help='')
    parser.add_argument('--max_epoch', type=int, default=120, help='')
    parser.add_argument('--init_lr', type=float, default=0.02, help='')
    parser.add_argument('--momentum', type=float, default=0.9, help='')
    parser.add_argument('--weight_decay', type=float, default=0.0005, help='')
    parser.add_argument('--nesterov', type=bool, default=True, help='')
    parser.add_argument('--T_0', type=int, default=10, help=' Number of iterations for the first restart.'
                                                            'best performing settings reported in paper are T_0 = 10')
    parser.add_argument('--T_mult', type=int, default=2,  help='A factor increases Ti after a restart.'
                                                    'best performing settings reported in paper are T_0 = 10, T_mult=2')

    # visdom
    parser.add_argument('--visdom_name', type=str, default='se-resnet50',help='')
    parser.add_argument('--tmp_loss_name', type=str, default='name')
    parser.add_argument('--loss_name', type=str, default='loss_name')

    # tricks
    parser.add_argument('--pretrained', type=bool, default=False, help='')
    parser.add_argument('--pretrained_params', type=str, default=r'D:\EPCNN\7.14_first_thesis\yolov3-master\weights\ckpt_epoch_45.pth',
                        help='')
    parser.add_argument('--no_bias_decay', type=bool, default=True,
                        help='')
    parser.add_argument('--label_smoothing', type=bool, default=True, help='')

    opt = parser.parse_args()

    device = torch_utils.select_device()

    # create dataset
    print("Loading dataset...")

    transf = get_transforms(input_size=opt.img_size, test_size=opt.img_size, kind='full', crop=True,
                            need=('train', 'val'), backbone=None)
    transform_train = transf['train']
    transform_test = transf['val']
    train_set = datasets.ImageFolder(opt.img_path + '\\train', transform=transform_train)
    train_sampler = RASampler(
        train_set, len(opt.gpu), 0, len(train_set), opt.batch_size, repetitions=3, len_factor=2.0, shuffle=True, drop_last=False)
    train_dataloader = torch.utils.data.DataLoader(
        train_set,
        batch_size=opt.batch_size,
        num_workers=(opt.num_workers - 1),
        sampler=train_sampler,
    )

    gl._init()
    gl.set_value('view_info', torch.FloatTensor(2, 5).cuda())
    gl.set_value('yolo_features', torch.FloatTensor(2, 1024, 7, 7).cuda())
    gl.set_value('bbox_info', torch.FloatTensor(2, 4).cuda())

    # model_classification
    print("Loading Model...")
    model = get_model(opt.model, pretrained=opt.pretrained)
    model_classification = model
    model_classification = model_classification.cuda()
    # summary(model_classification.cuda(), (3, opt.img_size, opt.img_size))

    if opt.pretrained_params is not None:
        pretrained_dict = torch.load(opt.pretrained_params)
        # delete module
        del_dict = OrderedDict()
        for k, v in pretrained_dict.items():
            # print(k)
            name = k[14:]  # remove `module.`
            del_dict[name] = v
        pretrained_dict = del_dict

        model_dict = model_classification.state_dict()
        # pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}  # filter out unnecessary keys
        pretrained_dict = {k: v for k, v in pretrained_dict.items()
                           if (k in model_dict) and (pretrained_dict[k].size() == model_dict[k].size())
                           }

        print('\nloaded dicts are:')
        for k, v in model_dict.items():
            if k in pretrained_dict:
                print(k + '\t\t\t\t\t\t' + str(v.size()))
        print('\nunloaded dicts are:')
        for k, v in model_dict.items():
            if not k in pretrained_dict:
                print(k + '\t\t\t\t\t\t' + str(v.size()))
        model_dict.update(pretrained_dict)
        model_classification.load_state_dict(model_dict)

    # optimizer
    if opt.no_bias_decay:
        trainable_vars = no_bias_decay(model_classification)
    else:
        trainable_vars = [param for param in model_classification.parameters() if param.requires_grad]
    print("Training with sgd")
    linear_scaled_lr = 8.0 * opt.init_lr * opt.batch_size / 512.0
    optimizer = torch.optim.SGD(trainable_vars, lr=linear_scaled_lr, momentum=opt.momentum,
                                weight_decay=opt.weight_decay, nesterov=opt.nesterov)

    # learning rate
    lr_scheduler = StepLR(optimizer, step_size=30)
    # lr_scheduler = CosineAnnealingWarmRestarts(optimizer, opt.T_0, T_mult=opt.T_mult)

    # Train
    vis = Visualizer(env=opt.visdom_name)
    model_classification = model_classification.cuda()
    model_classification.train()

    # model_pos_estimate
    logger.info('yolov3')
    model_pos_estimate = Darknet(opt.cfg, opt.img_size)
    inputs = torch.randn(1, 3, 224, 224)

    flops, params = profile(model_pos_estimate, inputs=(inputs,))
    print('flops:', flops)
    print('params:', params)
    model_pos_estimate.load_state_dict(torch.load(opt.weights, map_location=device)['model'], strict=False)
    model_pos_estimate.to(device).eval()

    best_loss = np.inf
    # meters
    if opt.label_smoothing:
        criterion = LSR()
    else:
        criterion = nn.CrossEntropyLoss()
    loss_meter = meter.AverageValueMeter()
    last_epoch = opt.last_epoch
    iters = len(train_dataloader)

    for epoch in range(0, opt.max_epoch):

        loss_meter.reset()
        last_epoch += 1
        logger.info('Start training epoch {}'.format(last_epoch))
        lr_scheduler.step(epoch)
        for step, (data, label, img_path) in enumerate(train_dataloader):
            # yolov3 estimate
            bbox_info = []
            view_info = []
            yolo_features = []
            asda = model_pos_estimate.module_list._modules
            model_pos_estimate.module_list._modules.get('12').register_forward_hook(hook)
            for i, i_img_path in enumerate(img_path):
                # BGR
                img0 = cv2.imread(i_img_path)
                # Padded resize
                img, _, _, _ = letterbox(img0, height=opt.img_size)
                # Normalize RGB
                img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB
                img = np.ascontiguousarray(img, dtype=np.float32)  # uint8 to float32
                img /= 255.0  # 0 - 255 to 0.0 - 1.0 归一化
                img = torch.from_numpy(img).unsqueeze(0).to(device)

                pred, _ = model_pos_estimate(img)
                yolo_features.append(conv1024)
                detections_src = non_max_suppression(pred, opt.conf_thres, opt.nms_thres)[0]
                # print('\nimage %s: ' % (i_img_path), end='')
                if detections_src is not None and len(detections_src) > 0:
                    bbox_info.append(torch.clamp(detections_src[-1, 0:4].view(1, 4), min=0, max=224))
                    view_info.append(detections_src[-1, 7:12].view(1, 5))
                    # print(detections_src[-1, :7])
                else:
                    bbox_info.append(torch.tensor([[0, 0, 224, 224]], dtype=torch.float32).cuda())
                    view_info.append(torch.zeros((1, 5), dtype=torch.float32).cuda())

            bbox_info = torch.cat(bbox_info)
            bbox_info = torch.round(bbox_info / 224 * 6).int()
            view_info = torch.cat(view_info)
            yolo_features = torch.cat(yolo_features)

            # train model
            inputs = Variable(data)
            target = Variable(label)
            inputs = inputs.cuda()
            target = target.cuda()

            gl.set_value('view_info', view_info)
            gl.set_value('yolo_features', yolo_features)
            gl.set_value('bbox_info', bbox_info)

            optimizer.zero_grad()
            score = model_classification(inputs)
            loss = criterion(score, target)
            loss.backward()
            optimizer.step()
            # lr_scheduler.step(int(epoch + step / iters))

            # meters update
            loss_meter.add(loss.item())  # 1.0  # loss_meter.add(loss.data[0])  # 0.4

            if step % 10 == 0:
                print('[epoch:%3d - %4d] loss of train: %.6f' % (last_epoch, step + 1, loss_meter.value()[0]))
                vis.plot(opt.tmp_loss_name, loss_meter.value()[0])
                pass

        model_save_path = opt.model_save_path + opt.model
        best_model_save_path = model_save_path + '\\best'
        if not os.path.exists(model_save_path):
            os.makedirs(model_save_path)
        if not os.path.exists(best_model_save_path):
            os.makedirs(best_model_save_path)

        if (last_epoch % opt.save_freq_epoch == 0) or (last_epoch == opt.max_epoch - 1):
            save_name = model_save_path + '\\ckpt_epoch_{}.pth'.format(last_epoch)
            torch.save(model_classification.state_dict(), save_name)

        if loss_meter.value()[0] < best_loss:
            logger.info('Found a better ckpt ({:.3f} -> {:.3f}) from epoch{} '.format(best_loss, loss_meter.value()[0], last_epoch))
            best_loss = loss_meter.value()[0]
            torch.save(model_classification.state_dict(), best_model_save_path + '\\best_{}.pt'.format(last_epoch))

        vis.plot(opt.loss_name, loss_meter.value()[0])