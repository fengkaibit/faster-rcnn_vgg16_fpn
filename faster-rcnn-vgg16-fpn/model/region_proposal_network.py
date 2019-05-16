import sys
sys.path.append('/home/fengkai/PycharmProjects/my-faster-rcnn')

import numpy as np
from torch.nn import functional
import torch

from model.utils.bbox_tools import generate_anchor_base
from model.utils.creator_tool import ProposalCreator
from utils import array_tool

class RegionProposalNetwork(torch.nn.Module):

    def __init__(
            self, in_channels=512, mid_channels=512, ratios=[0.5,1,2],
            anchor_scales=[2,4,8,16,32], feat_stride=[4, 8, 16, 32],
            proposal_creator_params=dict()):
        super(RegionProposalNetwork, self).__init__()

        self.ratios = ratios
        self.anchor_scales = anchor_scales
        self.feat_stride = feat_stride

        self.proposal_layer = ProposalCreator(self, **proposal_creator_params)

        num_anchor_base = 3
        self.conv1 = torch.nn.Conv2d(in_channels, mid_channels, 3, 1, 1)
        self.score = torch.nn.Conv2d(mid_channels, num_anchor_base*2, 1, 1, 0) #二分类，obj or nobj
        self.loc = torch.nn.Conv2d(mid_channels, num_anchor_base*4, 1, 1, 0) #坐标回归
        normal_init(self.conv1, 0, 0.01)
        normal_init(self.score, 0, 0.01)
        normal_init(self.loc, 0, 0.01)

    def forward(self, feature_maps, img_size, scale=1.):
        feature_maps_num = len(feature_maps)
        all_anchors = list()
        all_rois = list()
        all_roi_indices = list()
        all_rpn_locs = []
        all_rpn_fg_scores = []
        all_rpn_scores = []


        for i in range(feature_maps_num):
            batch_size, _, hh, ww = feature_maps[i].shape  # x为feature map, n为batch_size,此版本代码为1. _为512, hh, ww即为特征图宽高
            if i == 0:
                anchor_base = generate_anchor_base(anchor_scales=[4], ratios=self.ratios)
            if i == 1:
                anchor_base = generate_anchor_base(anchor_scales=[8], ratios=self.ratios)
            if i == 2:
                anchor_base = generate_anchor_base(anchor_scales=[16], ratios=self.ratios)
            if i == 3:
                anchor_base = generate_anchor_base(anchor_scales=[32], ratios=self.ratios)
            anchor = _enumerate_shifted_anchor(
                np.array(anchor_base), self.feat_stride[i], hh, ww)
            all_anchors.append(anchor)
            num_anchor = anchor.shape[0] // (hh * ww)  #
            h = functional.relu(self.conv1(feature_maps[i]), inplace=True)  #(batch_size, 512, hh, ww)

            rpn_locs = self.loc(h)   #(batch_size, 9*4, hh, ww)
            rpn_locs = rpn_locs.permute(0, 2, 3, 1).contiguous().view(batch_size,-1, 4)  #转换为(batch_size,hh, ww, 9*4)在转换为(batch_size, hh*ww*9, 4)
            rpn_scores = self.score(h)
            rpn_scores = rpn_scores.permute(0, 2, 3, 1).contiguous()  #转换为(batch_size,hh, ww, 9*2)
            rpn_softmax_scores = functional.softmax(rpn_scores.view(batch_size, hh, ww, num_anchor, 2), dim=4)  #TODO 维度问题
            rpn_fg_scores = rpn_softmax_scores[:, :, :, :, 1].contiguous()  #得到前景的分类概率

            rpn_fg_scores = rpn_fg_scores.view(batch_size, -1) #得到所有anchor的前景分类概率
            rpn_scores = rpn_scores.view(batch_size, -1, 2)

            all_rpn_locs.append(rpn_locs)
            all_rpn_fg_scores.append(rpn_fg_scores)
            all_rpn_scores.append(rpn_scores)

        all_rpn_locs = torch.cat(all_rpn_locs, 1)
        all_rpn_fg_scores = torch.cat(all_rpn_fg_scores, 1)
        all_rpn_scores = torch.cat(all_rpn_scores, 1)
        all_anchors = np.concatenate(all_anchors, axis=0)

        for i in range(batch_size):
            roi = self.proposal_layer(
                all_rpn_locs[i].cpu().data.numpy(),
                all_rpn_fg_scores[i].cpu().data.numpy(),
                all_anchors, img_size, scale=scale)
            #rpn_locs维度（hh * ww * 9，4），rpn_fg_scores维度为（hh * ww * 9），
            #anchor的维度为（hh * ww * 9，4）， img_size的维度为（3，H，W），H和W是经过数据预处理后的。
            #计算（H / 16）x( W / 16)x9(大概20000)
            #个anchor属于前景的概率，取前12000个并经过NMS得到2000个近似目标框G ^ 的坐标。roi的维度为(2000, 4)

            batch_index = i * np.ones((len(roi),),dtype=np.int32)   #(len(roi), )
            all_rois.append(roi)
            all_roi_indices.append(batch_index)  #记录roi的batch批次

        all_rois = np.concatenate(all_rois,axis=0)  #按列排所有的roi， rois格式（R， 4），R为所有batch的roi数量
        all_roi_indices = np.concatenate(all_roi_indices, axis=0) #按列排所有roi的批次编号，格式同rois

        # rpn_locs的维度（hh*ww*9，4），rpn_scores维度为（hh*ww*9，2），
        # rois的维度为（2000,4），roi_indices用不到(因为此代码训练时batch为1)，anchor的维度为（hh*ww*9，4）
        return all_rpn_locs, all_rpn_scores, all_rois, all_roi_indices, all_anchors


def normal_init(m, mean, stddev, truncated=False):
    if truncated:
        m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean)   #截断产生正态分布
    else:
        m.weight.data.normal_(mean, stddev)   #普通产生正态分布
        m.bias.data.zero_()

def _enumerate_shifted_anchor(anchor_base, feat_stride, height, width):
    shift_y = np.arange(0, height * feat_stride, feat_stride)
    shift_x = np.arange(0, width * feat_stride, feat_stride)
    shift_x, shift_y = np.meshgrid(shift_x, shift_y)  #产生x,y坐标网格
    shift = np.stack((shift_y.ravel(), shift_x.ravel(),
                     shift_y.ravel(), shift_x.ravel()), axis=1)  #产生坐标偏移矩阵（w*h, 4）

    A = anchor_base.shape[0]  #特征图上每一个点产生anchor数目,9
    K = shift.shape[0]  #坐标偏移矩阵行数(即特征图的像素点个数, w*h)
    #(1, A ,4) + (K, 1, 4) = (K, A, 4)
    anchor = anchor_base.reshape(1, A, 4) + shift.reshape((1, K, 4)).transpose((1,0,2))
    anchor = anchor.reshape((K * A, 4)).astype(np.float32)  #修改尺寸为(K * A, 4)
    return anchor

def _enumerate_shifted_anchor_torch(anchor_base, feat_stride, height, width):
    shift_y = torch.arange(0, height * feat_stride, feat_stride)
    shift_x = torch.arange(0, width * feat_stride, feat_stride)
    shift_x, shift_y = np.meshgrid(shift_x, shift_y)  #产生x,y坐标网格
    shift = np.stack((shift_y.ravel(), shift_x.ravel(),
                     shift_y.ravel(), shift_x.ravel()), axis=1)  #产生坐标偏移矩阵（w*h, 4）

    A = anchor_base.shape[0]  #特征图上每一个点产生anchor数目,9
    K = shift.shape[0]  #坐标偏移矩阵行数(即特征图的像素点个数, w*h)
    #(1, A ,4) + (K, 1, 4) = (K, A, 4)
    anchor = anchor_base.reshape(1, A, 4) + shift.reshape((1, K, 4)).transpose((1,0,2))
    anchor = anchor.reshape((K * A, 4)).astype(np.float32)  #修改尺寸为(K * A, 4)
    return anchor


if __name__ == '__main__':
    anchor_base = generate_anchor_base(anchor_scales=[8],ratios=[0.5, 1, 2])
    print(anchor_base)
    feat_stride = [4, 8, 16, 32, 64]
    height = [200, 100, 50, 25, 12]
    width = [200, 100, 50, 25, 12]
    all_anchor = list()

    for i in range(5):
        anchor = _enumerate_shifted_anchor(anchor_base, feat_stride[i], height[i], width[i])
        all_anchor.append(anchor)
    all_anchor = np.concatenate(all_anchor,axis=0)
    print(all_anchor)
