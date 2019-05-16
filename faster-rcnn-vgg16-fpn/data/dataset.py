from __future__ import absolute_import
from __future__ import division
import torch
from torchvision import transforms
from skimage import transform as skt
import numpy as np
from data import util
from utils.config import opt
from data.voc_dataset import VOCBboxDataset

#去正则化
def inverse_normalize(img):
    if opt.caffe_pretrain:
        img = img + (np.array([122.7717, 115.9465, 102.9801]).reshape(3, 1, 1))
        return img[::-1, :, :]
    # approximate un-normalize for visualize
    return (img * 0.225 + 0.45).clip(min=0, max=1) * 255

def pytorch_normalize(img):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    img = normalize(torch.from_numpy(img))
    return img.numpy()

#caffe的正则化，[0-255]RGB转换为[-125,125]BGR
def caffe_normalize(img):
    """
    return appr -125-125 BGR
    """
    img = img[[2, 1, 0], :, :]  # RGB-BGR
    img = img * 255
    mean = np.array([122.7717, 115.9465, 102.9801]).reshape(3, 1, 1)
    img = (img - mean).astype(np.float32, copy=True)
    return img

#图像预处理函数，首先将读入的图片（CHW格式）转换为[0-1]，然后用长边不超过1000，短边不超过600进行尺度缩放，最后用caffe
#正则化或者pytorch正则化处理图片
def preprocess(img, min_size=600, max_size=1000):
    C, H, W = img.shape   #读入图片CHW格式
    scale1 = min_size / min(H, W)
    scale2 = max_size / max(H, W)
    scale = min(scale1, scale2)
    img = img / 255.
    img = skt.resize(img, (C, H * scale, W * scale), mode='reflect', anti_aliasing=False)
    img = pytorch_normalize(img)
    return img

class Transform(object):
    def __init__(self, min_size=600, max_size=1000):
        self.min_size = min_size
        self.max_size = max_size

    def __call__(self, in_data):
        img, bbox, label = in_data
        C, H, W = img.shape
        img = preprocess(img, self.min_size, self.max_size)
        _C, _H, _W = img.shape
        scale = _H / H
        bbox = util.resize_bbox(bbox, (H,W), (_H, _W))

        img, params = util.random_flip_image(img, x_random=True, return_param=True)
        bbox = util.flip_bbox(bbox,(_H, _W),x_flip=params['x_flip'])
        return img, bbox, label, scale

#从数据集存储路径中将例子一个个的获取出来，然后调用前面的Transform函数将图片
#label进行最小值最大值放缩归一化，重新调整bboxes的大小，然后随机反转，最后将数据集返回。
class Dataset:
    def __init__(self, opt):
        self.opt = opt
        self.dataset = VOCBboxDataset(opt.voc_data_dir)
        self.transform = Transform(opt.min_size, opt.max_size)

    def __getitem__(self, item):
        ori_img, bbox, label, difficult = self.dataset.get_example(item)
        img, bbox, label, scale = self.transform((ori_img,bbox,label))
        return img.copy(), bbox.copy(), label.copy(), scale #返回处理后的图片(CHW,[-1,1]),bbox,label,原图和调整后图的比例因子

    def __len__(self):
        return len(self.dataset)

class TestDataset:
    def __init__(self, opt, split='test', use_difficult=True):
        self.opt = opt
        self.dataset = VOCBboxDataset(opt.voc_data_dir, split=split, use_difficult=use_difficult)

    def __getitem__(self, item):
        ori_img, bbox, label, difficult = self.dataset.get_example(item)
        img = preprocess(ori_img)
        return img, ori_img.shape[1:], bbox, label, difficult

    def __len__(self):
        return len(self.dataset)


if __name__ == '__main__':
    obj = Dataset(opt)
    obj.__getitem__(337)


