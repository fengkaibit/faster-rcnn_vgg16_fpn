import os
import json
import numpy as np
import cv2

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

def convert_dataset():
    json_path = '/media/fengkai/Seagate Backup Plus Drive/Dataset/COCO2017/annotations/instances_val2017.json'
    imgs_dir = '/media/fengkai/Seagate Backup Plus Drive/Dataset/COCO2017/val2017'

    with open(json_path, 'r') as f:
        dataset = json.load(f)
    annotations = dataset['annotations']

    display = 10
    invalid_files = []
    dataset_dict = {}
    for i, entry in enumerate(annotations):
        obj_id = entry['id']
        cat_id = entry['category_id']
        area = entry['area']
        img_id = entry['image_id']
        bbox = entry['bbox']
        image_file = os.path.join(imgs_dir, '%012d.jpg' %img_id)
        if not os.path.exists(image_file):
            invalid_files.append(image_file)
            continue
        if img_id not in dataset_dict.keys():
            dataset_dict[img_id] = []
        dataset_dict[img_id].append(dict(id=obj_id, category_id=cat_id, area=area, bbox=bbox, image_id=img_id))
        if (i+1)%display == 0:
            print('processed {}/{}'.format(i+1, len(annotations)))
    print('{} files is not exist'.format(len(invalid_files)))
    print(invalid_files)

    with open('coco_train_2017.json', 'w+') as f:
        dataset_list = []
        for k, v in dataset_dict.items():
            dataset_list.append(v)
        coco_image_json = dict(annotations=dataset_list)
        f.write(json.dumps(coco_image_json))
    print('Done')

def test_dataset():
    json_path = './coco_train_2017.json'
    imgs_dir = '/media/fengkai/Seagate Backup Plus Drive/Dataset/COCO2017/val2017'

    with open(json_path, 'r') as f:
        dataset = json.load(f)
    annotations = dataset['annotations']

    samples_index = np.random.choice(np.arange(len(annotations)), size=100, replace=False)
    for index in samples_index:
        sample = annotations[index]
        image_id = sample[0]['image_id']
        image_file = os.path.join(imgs_dir, '%012d.jpg' % image_id)
        image = cv2.imread(image_file)

        for entry in sample:
            cat_id = entry['category_id']
            bbox = np.asarray(entry['bbox'], dtype=np.float32)
            x, y, w, h = bbox[0], bbox[1], bbox[2], bbox[3]
            cv2.rectangle(image, (x, y), (x + w, y + h), (0,255,0), 2)
            cv2.putText(image, coco_id_name_map[cat_id], (int(x), int(y-15)), cv2.FONT_HERSHEY_PLAIN, 0.9, (0,0,255))
        cv2.imwrite('./test/%012d.jpg' %image_id, image)

if __name__ == '__main__':
    #convert_dataset()
    test_dataset()


