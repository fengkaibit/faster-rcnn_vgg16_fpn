import numpy as np
import random
from PIL import Image

def read_image(img_path, dtype=np.float32, color=True):
    img = Image.open(img_path)
    try:
        if color:
            img = img.convert('RGB')
        else:
            img = img.convert('P')  #模式“P”为8位彩色图像，它的每个像素用8个bit表示，其对应的彩色值是按照调色板查询出来的.
        img = np.asarray(img, dtype=dtype)    #将图片转为array形式
    finally:
        if hasattr(img, 'close'):  #判断img是否有'close'
            img.close()

    if img.ndim == 2:
        return img[np.newaxis]
    if img.ndim == 3:
        return img.transpose(2, 0, 1)  #转换为CHW格式


def resize_bbox(bbox, in_size, out_size):
    bbox = bbox.copy()
    y_scale = out_size[0] / in_size[0]
    x_scale = out_size[1] / in_size[1]
    bbox[:, 0] = y_scale * bbox[:, 0]
    bbox[:, 1] = x_scale * bbox[:, 1]
    bbox[:, 2] = y_scale * bbox[:, 2]
    bbox[:, 3] = x_scale * bbox[:, 3]
    return bbox

def random_flip_image(img, y_random = False, x_random = True,
                      return_param = False, copy = False):
    y_flip, x_flip = False, False
    if y_random:
        y_flip = random.choice([True, False])
    if x_random:
        x_flip = random.choice([True, False])

    if y_flip:
        img = img[:, ::-1, :]
    if x_flip:
        img = img[:, :, ::-1]
    if copy:
        img = img.copy()
    if return_param:
        return img, {'y_flip': y_flip, 'x_flip': x_flip}
    else:
        return img

def flip_bbox(bbox, img_size, y_flip=False, x_flip=False):
    H, W = img_size
    bbox = bbox.copy()
    if y_flip:
        y_max = H - bbox[:, 0]
        y_min = H - bbox[:, 2]
        bbox[:, 0] = y_min
        bbox[:, 2] = y_max
    if x_flip:
        x_max = W - bbox[:, 1]
        x_min = W - bbox[:, 3]
        bbox[:, 1] = x_min
        bbox[:, 3] = x_max
    return bbox