from __future__ import absolute_import
from __future__ import division
import torch
import numpy as np
import cupy as cp
from utils import array_tool
from model.utils.bbox_tools import loc2bbox
from model.utils.nms import non_maximum_suppression

from data.dataset import preprocess
from torch import nn
from torch.nn import functional
from utils.config import opt

def nogard(f):
    def new_f(*args, **kwargs):
        with torch.no_grad():
            return f(*args, **kwargs)
    return new_f

class FasterRCNN(nn.Module):

    def __init__(self, C2, C3, C4,
                 fpn,
                 rpn,
                 head,
                 loc_normalize_mean = (0., 0., 0., 0.),
                 loc_normalize_std = (0.1, 0.1, 0.2, 0.2)):
        super(FasterRCNN, self).__init__()
        self.C2 = C2
        self.C3 = C3
        self.C4 = C4
        self.fpn = fpn
        self.rpn = rpn
        self.head = head

        self.loc_normalize_mean = loc_normalize_mean
        self.loc_normalize_std = loc_normalize_std
        self.use_preset('evaluate')

    @property
    def n_class(self):
        return self.head.n_class

    def forward(self, x, scale=1.):
        img_size = x.shape[2:]   #处理后图片的h和w

        c2_out = self.C2(x)
        c3_out = self.C3(c2_out)
        c4_out = self.C4(c3_out)

        p2, p3, p4, p5 = self.fpn(c2_out, c3_out, c4_out)
        features_maps = [p2, p3, p4, p5]
        rcnn_maps = [p2, p3, p4]
        rpn_locs, rpn_scores, rois, roi_indices, anchor = self.rpn(features_maps, img_size, scale)   #rpn网络
        roi_cls_locs, roi_scores = self.head(rcnn_maps, rois, roi_indices)
        return roi_cls_locs, roi_scores, rois, roi_indices


    def use_preset(self, preset):
        if preset == 'visualize':
            self.nms_thresh = 0.3
            self.score_thresh = 0.7
        elif preset == 'evaluate':
            self.nms_thresh = 0.3
            self.score_thresh = 0.05
        else:
            raise ValueError('preset must be visualize or evaluate')

    def _suppress(self, raw_cls_bbox, raw_prob):  #将roihead的预测结果利用score_thresh和nms_thresh进行过滤
        bbox = list()
        label = list()
        score = list()
        for l in range(1, self.n_class):  #0为背景
            cls_bbox_l = raw_cls_bbox.reshape((-1, self.n_class, 4))[:, l, :]
            prob_l = raw_prob[:, l]
            mask = prob_l > self.score_thresh

            cls_bbox_l = cls_bbox_l[mask]
            prob_l = prob_l[mask]
            keep = non_maximum_suppression(
                cp.array(cls_bbox_l), self.nms_thresh, prob_l)
            keep = cp.asnumpy(keep)
            bbox.append(cls_bbox_l[keep])
            label.append((l-1) * np.ones((len(keep),)))
            score.append(prob_l[keep])
        bbox = np.concatenate(bbox, axis=0).astype(np.float32)
        label = np.concatenate(label, axis=0).astype(np.int32)
        score = np.concatenate(score, axis=0).astype(np.float32)
        return bbox, label, score


    @nogard
    def predict(self, imgs, sizes = None, visualize=False):
        self.eval()
        if visualize:
            self.use_preset('visualize')
            prepared_imgs = list()
            sizes = list()
            for img in imgs:
                size = img.shape[1:]
                img = preprocess(array_tool.tonumpy(img))
                prepared_imgs.append(img)
                sizes.append(size)
        else:
            prepared_imgs = imgs
        bboxes = list()
        labels = list()
        scores = list()
        for img, size in zip(prepared_imgs, sizes):
            img = array_tool.totensor(img[None]).float()   #img增加一维(_, C, H, W)
            scale = img.shape[3] / size[1]   # W' / W, 处理后图像和原图比例
            roi_cls_locs, roi_scores, rois, roi_indices = self(img, scale=scale)

            #batch size为1
            roi_score = roi_scores.data
            roi_cls_loc = roi_cls_locs.data
            roi = array_tool.totensor(rois) / scale

            mean = torch.Tensor(self.loc_normalize_mean).cuda().repeat(self.n_class)[None]  #(1,84)
            std = torch.Tensor(self.loc_normalize_std).cuda().repeat(self.n_class)[None]

            roi_cls_loc = (roi_cls_loc * std + mean)
            roi_cls_loc = roi_cls_loc.view(-1, self.n_class, 4)  #(R, 21 ,4)

            roi = roi.view(-1, 1, 4).expand_as(roi_cls_loc)  #扩充维度  #(R, 21, 4)
            cls_bbox = loc2bbox(array_tool.tonumpy(roi).reshape(-1,4),
                                array_tool.tonumpy(roi_cls_loc).reshape(-1,4))
            cls_bbox = array_tool.totensor(cls_bbox)
            cls_bbox = cls_bbox.view(-1, self.n_class * 4)  #(R, 84)
            cls_bbox[:, 0::2] = (cls_bbox[:, 0::2]).clamp(min=0, max=size[0])
            cls_bbox[:, 1::2] = (cls_bbox[:, 1::2]).clamp(min=0, max=size[1]) #裁剪预测bbox不超出原图尺寸

            prob = array_tool.tonumpy(
                functional.softmax(array_tool.totensor(roi_score), dim=1))

            raw_cls_bbox = array_tool.tonumpy(cls_bbox)
            raw_prob = array_tool.tonumpy(prob)

            bbox, label, score = self._suppress(raw_cls_bbox, raw_prob)
            bboxes.append(bbox)
            labels.append(label)
            scores.append(score)   #将每个batch_size的压在一起

        self.use_preset('evaluate')
        self.train()
        return bboxes, labels, scores

    def get_optimizer(self):
        lr =opt.lr
        params = []
        for key, value in dict(self.named_parameters()).items():
            if value.requires_grad:
                if 'bias' in key:
                    params += [{'params': [value], 'lr': lr * 2, 'weight_decay': 0}]
                else:
                    params += [{'params': [value], 'lr': lr, 'weight_decay': opt.weight_decay}]

        if opt.use_adam:
            self.optimizer = torch.optim.Adam(params)
        else:
            self.optimizer = torch.optim.SGD(params, momentum=0.9)
        return self.optimizer

    def scale_lr(self, decay=0.1):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] *= decay
        return self.optimizer


if __name__ == '__main__':
    img = np.ones((3,5,5),dtype=np.float32)
    b = array_tool.totensor(img[None]).float()
    loc_normalize_mean = (0., 0., 0., 0.)
    roi_cls_loc = np.ones((1,84),dtype=np.float32)
    mean = torch.Tensor(loc_normalize_mean).cuda()
    mean = mean.repeat(21)[None]
    loc_normalize_std = (0.1, 0.1, 0.2, 0.2)
    std = torch.Tensor(loc_normalize_std).cuda()
    std = std.repeat(21)[None]
    roi_cls_loc = array_tool.totensor(roi_cls_loc)
    roi_cls_loc = roi_cls_loc.data
    roi_cls_loc = (roi_cls_loc * std + mean)
    print(roi_cls_loc.size())
    roi_cls_loc = roi_cls_loc.view(-1, 21, 4)
    print(roi_cls_loc.size())
    roi = np.zeros((1,4),dtype=np.float32)
    roi = array_tool.totensor(roi)
    scale = array_tool.totensor(np.array([1.])).float()
    roi = roi / scale
    roi = roi.view(-1,1,4).expand_as(roi_cls_loc)
    print(roi.size())
