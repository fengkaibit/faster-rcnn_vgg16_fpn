import os
import numpy as np
import xml.etree.ElementTree as ET
import sys
sys.path.append('/home/fengkai/PycharmProjects/faster-rcnn-vgg16-fpn5_7')
from data.util import read_image
import cv2

class VOCBboxDataset:
    def __init__(self, data_dir, split='trainval',
                 use_difficult=False, return_difficult=False):
        id_list_file = os.path.join(
            data_dir, 'ImageSets/Main/{0}.txt'.format(split))

        self.ids = [id_.strip() for id_ in open(id_list_file)]
        self.data_dir = data_dir
        self.use_difficult = use_difficult
        self.return_difficult = return_difficult
        self.label_names = VOC_BBOX_LABEL_NAMES

    def __len__(self):
        return len(self.ids)

    def get_example(self,i):
        id_ = self.ids[i]
        anno = ET.parse(
            os.path.join(self.data_dir,'Annotations',id_ + '.xml'))   #打开xml文档
        bbox = list()
        label = list()
        difficult = list()
        for obj in anno.findall('object'):
            if not self.use_difficult and int(obj.find('difficult').text) == 1:
                continue

            difficult.append(int(obj.find('difficult').text))
            bndbox_anno = obj.find('bndbox')
            bbox.append([
                int(bndbox_anno.find(tag).text) - 1
                for tag in ('ymin', 'xmin', 'ymax', 'xmax')])
            name = obj.find('name').text.lower().strip()   #转换大写字母为小写并删除两端空格
            label.append(VOC_BBOX_LABEL_NAMES.index(name))
        bbox = np.stack(bbox).astype(np.float32)
        label = np.stack(label).astype(np.int32)
        difficult = np.array(difficult,dtype=np.bool).astype(np.uint8)

        img_file = os.path.join(self.data_dir, 'images/data', id_ + '.jpg')
        img = cv2.imread(img_file)
        #img = read_image(img_file,color=True)
        return img, bbox, label, difficult

    __getitem__ = get_example


VOC_BBOX_LABEL_NAMES = (
    'aeroplane',
    'bicycle',
    'bird',
    'boat',
    'bottle',
    'bus',
    'car',
    'cat',
    'chair',
    'cow',
    'diningtable',
    'dog',
    'horse',
    'motorbike',
    'person',
    'pottedplant',
    'sheep',
    'sofa',
    'train',
    'tvmonitor')

if __name__ == '__main__':
    import tqdm
    obj = VOCBboxDataset('/media/fengkai/Seagate Backup Plus Drive/Dataset/voc/VOCdevkit/VOC2007/')
    for num in tqdm(range(obj.__len__())):

        src_img, bbox, label, difficult, id = obj.get_example(num)
        img = np.ones((src_img.shape[0], src_img.shape[1]), dtype=np.int16)
        H, W, C = src_img.shape
        ymin = bbox[:, 0]
        xmin = bbox[:, 1]
        ymax = bbox[:, 2]
        xmax = bbox[:, 3]

        for j in range(H):
            for k in range(W):
                for i in range(bbox.shape[0]):
                    if (j >= ymin[i] and (j <= ymax[i])
                            and (k >= xmin[i]) and (k <= xmax[i])):
                        img[j,k] = 0

        for j in range(H):
            for k in range(W):
                if img[j,k] == 0:
                    continue
                else:
                    src_img[j,k,:] = 0
        cv2.imwrite('/home/fengkai/data/{}.jpg'.format(id),src_img)







