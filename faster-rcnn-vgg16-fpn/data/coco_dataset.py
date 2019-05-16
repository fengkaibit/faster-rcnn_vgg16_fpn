import os
import json
import numpy as np
from data.util import read_image

COCO_LABEL_NAMES = ('person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
                 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
                 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
                 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite',
                 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle',
                 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
                 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
                 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
                 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book',
                 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush')

coco_id_name_map={1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle', 5: 'airplane',
                   6: 'bus', 7: 'train', 8: 'truck', 9: 'boat', 10: 'traffic light',
                   11: 'fire hydrant', 13: 'stop sign', 14: 'parking meter', 15: 'bench',
                   16: 'bird', 17: 'cat', 18: 'dog', 19: 'horse', 20: 'sheep', 21: 'cow',
                   22: 'elephant', 23: 'bear', 24: 'zebra', 25: 'giraffe', 27: 'backpack',
                   28: 'umbrella', 31: 'handbag', 32: 'tie', 33: 'suitcase', 34: 'frisbee',
                   35: 'skis', 36: 'snowboard', 37: 'sports ball', 38: 'kite', 39: 'baseball bat',
                   40: 'baseball glove', 41: 'skateboard', 42: 'surfboard', 43: 'tennis racket',
                   44: 'bottle', 46: 'wine glass', 47: 'cup', 48: 'fork', 49: 'knife', 50: 'spoon',
                   51: 'bowl', 52: 'banana', 53: 'apple', 54: 'sandwich', 55: 'orange',
                   56: 'broccoli', 57: 'carrot', 58: 'hot dog', 59: 'pizza', 60: 'donut',
                   61: 'cake', 62: 'chair', 63: 'couch', 64: 'potted plant', 65: 'bed', 67: 'dining table',
                   70: 'toilet', 72: 'tv', 73: 'laptop', 74: 'mouse', 75: 'remote', 76: 'keyboard',
                   77: 'cell phone', 78: 'microwave', 79: 'oven', 80: 'toaster', 81: 'sink',
                   82: 'refrigerator', 84: 'book', 85: 'clock', 86: 'vase', 87: 'scissors',
                   88: 'teddy bear', 89: 'hair drier', 90: 'toothbrush'}

class cocoDataset:
    def __init__(self, data_dir, split='val',
                 use_difficult=False, return_difficult=False):
        self.json_path = os.path.join(data_dir, 'coco_{0}_2017.json'.format(split))
        self.image_dir = os.path.join(data_dir, '{0}2017'.format(split))
        self.use_difficult = use_difficult
        self.return_difficult = return_difficult
        self.coco_label_name = COCO_LABEL_NAMES

    def get_example(self, i):
        with open(self.json_path, 'r') as f:
            dataset = json.load(f)
        annotations = dataset['annotations']
        gt_bbox = list()
        gt_label = list()
        difficult = list()
        example = annotations[i]
        self.image_id = example[0]['image_id']

        for entry in example:
            cat_id = entry['category_id']
            bbox = np.asarray(entry['bbox'], dtype=np.float32)
            xmin, ymin, xmax, ymax = bbox[0], bbox[1], bbox[2] + bbox[0], bbox[3] + bbox[1]
            gt_bbox.append([ymin, xmin, ymax, xmax])
            difficult.append(0)  #coco数据集没有difficult参数，此处为了和voc数据集保持一致，所有difficult值设为0
            gt_label.append(COCO_LABEL_NAMES.index(coco_id_name_map[cat_id]))

        gt_bbox = np.stack(gt_bbox).astype(np.float32)
        difficult = np.array(difficult, dtype=np.bool).astype(np.uint8)
        gt_label = np.stack(gt_label).astype(np.int32)
        img_file = os.path.join(self.image_dir, '%012d.jpg'%self.image_id)
        img = read_image(img_file, color=True)
        return img, gt_bbox, gt_label, difficult

if __name__ == '__main__':
    a = cocoDataset('/media/fengkai/Seagate Backup Plus Drive/Dataset/COCO2017')
    img, gt_bbox, gt_label, difficult = a.get_example(1)
    print(gt_label)
