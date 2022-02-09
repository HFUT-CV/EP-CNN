import json
import scipy.io as sio
import numpy as np
from collections import defaultdict

# coco dataset
name_box_id = defaultdict(list)
id_name = dict()
f = open("/home/tuan/Dataset/coco/annotations/instances_train2014.json", encoding='utf-8')
data = json.load(f)

annotations = data['annotations']
for ant in annotations:
    id = ant['image_id']  # image_id
    name = 'coco/images/train2014/%012d.jpg' % id  # path
    cat = ant['category_id']  # class

    if cat == 3:
        # cat = cat - 1
        name_box_id[name].append([ant['bbox'], cat])

    # if cat >= 1 and cat <= 11:
    #     cat = cat - 1
    # elif cat >= 13 and cat <= 25:
    #     cat = cat - 2
    # elif cat >= 27 and cat <= 28:
    #     cat = cat - 3
    # elif cat >= 31 and cat <= 44:
    #     cat = cat - 5
    # elif cat >= 46 and cat <= 65:
    #     cat = cat - 6
    # elif cat == 67:
    #     cat = cat - 7
    # elif cat == 70:
    #     cat = cat - 9
    # elif cat >= 72 and cat <= 82:
    #     cat = cat - 10
    # elif cat >= 84 and cat <= 90:
    #     cat = cat - 11
    #
    # name_box_id[name].append([ant['bbox'], cat])  # image_path.append(bbox(左上角x,y,w,h), class)

# # compcars dataset
# name_box_id = defaultdict(list)
# image_root = "/home/tuan/Dataset/"
#
# with open(image_root+'compcars/train.txt', 'r') as f:
#     for i in f.readlines():
#         image_path = image_root + i.split(' ')[0]
#         image_class = i.split(' ')[1][:-1]
#
#         bbox = []
#         with open(image_path.replace('images', 'label').replace('jpg', 'txt'), 'r') as f2:
#             src_bbox = f2.readlines()[2]
#             # print(src_bbox[:-2].split(' ')[0])
#             bbox.append(int(src_bbox[:-1].split(' ')[0]))
#             bbox.append(int(src_bbox[:-1].split(' ')[1]))
#             bbox.append(int(src_bbox[:-1].split(' ')[2]) - int(src_bbox.split()[0]))
#             bbox.append(int(src_bbox[:-1].split(' ')[3]) - int(src_bbox.split()[1]))
#
#         name_box_id[image_path].append([bbox, image_class])  # image_path.append(bbox(左上角x,y,w,h), class)


# ###下面是讲解python怎么读取.mat文件以及怎么处理得到的结果###
# load_data = sio.loadmat('/home/tuan/Dataset/stanfordcars_label&bbox/cars_train_annos.mat')
# # 假设文件中存有字符变量是matrix，例如matlab中save(load_fn, 'matrix');当然可以保存多个save(load_fn, 'matrix_x', 'matrix_y', ...);
# load_matrix = load_data['annotations']
# for i in load_matrix:
#     print(i)
# load_matrix_row = load_matrix[0]  # 取了当时matlab中matrix的第一行，python中数组行排列
#
# ###下面是讲解python怎么保存.mat文件供matlab程序使用###
# save_fn = 'xxx.mat'
# save_array = np.array([1, 2, 3, 4])
# sio.savemat(save_fn, {'array': save_array})  # 和上面的一样，存在了array变量的第一行
#
# save_array_x = np.array([1, 2, 3, 4])
# save_array_y = np.array([5, 6, 7, 8])
# sio.savemat(save_fn, {'array_x': save_array_x, 'array_x': save_array_x})  # 同理，只是存入了两个不同的变量供使用
#
# # stanfordcars dataset
# name_box_id = defaultdict(list)
# image_root = "/home/tuan/Dataset/"
#
# with open(image_root+'compcars/train.txt', 'r') as f:
#     for i in f.readlines():
#         image_path = image_root + i.split(' ')[0]
#         image_class = i.split(' ')[1][:-1]
#
#         bbox = []
#         with open(image_path.replace('images', 'label').replace('jpg', 'txt'), 'r') as f2:
#             src_bbox = f2.readlines()[2]
#             # print(src_bbox[:-2].split(' ')[0])
#             bbox.append(int(src_bbox[:-1].split(' ')[0]))
#             bbox.append(int(src_bbox[:-1].split(' ')[1]))
#             bbox.append(int(src_bbox[:-1].split(' ')[2]) - int(src_bbox.split()[0]))
#             bbox.append(int(src_bbox[:-1].split(' ')[3]) - int(src_bbox.split()[1]))
#
#         name_box_id[image_path].append([bbox, image_class])  # image_path.append(bbox(左上角x,y,w,h), class)


f = open('train.txt', 'w')
for key in name_box_id.keys():
    f.write(key)
    box_infos = name_box_id[key]
    for info in box_infos:
        x_min = int(info[0][0])
        y_min = int(info[0][1])
        x_max = x_min + int(info[0][2])
        y_max = y_min + int(info[0][3])

        box_info = " %d,%d,%d,%d,%d" % (
            x_min, y_min, x_max, y_max, int(info[1]))
        f.write(box_info)
    f.write('\n')
f.close()