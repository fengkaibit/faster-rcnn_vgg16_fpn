import sys
sys.path.append('/home/fengkai/PycharmProjects/my-faster-rcnn')

import numpy as np
import cupy as cp

from model.utils.bbox_tools import bbox2loc, loc2bbox, bbox_iou
from model.utils.nms.non_maximum_suppression import non_maximum_suppression

#由于RoIs给出的2000个候选框，分别对应feature map不同大小的区域。
# 首先利用ProposalTargetCreator 挑选出128个sample_rois,
# 然后使用了RoIPooling 将这些不同尺寸的区域全部pooling到同一个尺度（7×7）上。
#RoIs和gt_bboxes 的IoU大于0.5的，选择一些（比如32个）
#选择 RoIs和gt_bboxes的IoU小于等于0（或者0.1）的选择一些（比如 128-32=96个）作为负样本
#ProposalCreator产生2000个ROIS，但是这些ROIS并不都用于训练，经过本ProposalTargetCreator的筛选产生128个用于自身的训练
#输入：2000个rois、一个batch（一张图）中所有的bbox ground truth（R，4）、对应bbox所包含的label（R，1）（VOC2007来说20类0-19）
#输出：128个sample roi（128，4）、128个gt_roi_loc（128，4）、128个gt_roi_label（128，1）

class ProposalTargetCreator(object):
    def __init__(self,
                 n_sample = 128,
                 pos_ratio=0.25, pos_iou_thresh=0.5,
                 neg_iou_thresh_high = 0.5, neg_iou_thresh_low = 0.0):
        self.n_sample = n_sample
        self.pos_ratio = pos_ratio
        self.pos_iou_thresh = pos_iou_thresh
        self.neg_iou_thresh_high = neg_iou_thresh_high
        self.neg_iou_thresh_low = neg_iou_thresh_low

    def __call__(self, roi, bbox, label,
                 loc_normalize_mean=(0., 0., 0., 0.),
                 loc_normalize_std=(0.1, 0.1, 0.2, 0.2)):

        n_bbox, _ = bbox.shape  #(R,4),gt的边界框的坐标。

        roi = np.concatenate((roi,bbox),axis=0) #首先将2000个roi和R个bbox给连接一下成为新的roi (2000+R, 4),即将gt也放入roi中便于训练
        pos_roi_per_image = np.round(self.n_sample * self.pos_ratio)  #每张图片产生32个正样本
        iou = bbox_iou(roi, bbox)  #计算每一个roi与每一个gt bbox的iou, roi返回格式(2000 + R, R)

        # 按行找到最大值索引，返回最大值对应的序号以及其真正的IOU。返回的是每个roi与哪个bbox的最大
        gt_assignment = iou.argmax(axis=1)
        # 按行找到iou最大值
        max_iou = iou.max(axis=1)
        gt_roi_label = label[gt_assignment] + 1 #从1开始的类别序号，给每个类得到真正的label(将0-19变为1-20), 0为背景

        pos_index = np.where(max_iou >= self.pos_iou_thresh)[0] #根据iou的最大值将正负样本找出来，pos_iou_thresh=0.5
        # 需要保留的roi个数（满足大于pos_iou_thresh条件的roi与32之间较小的一个,即正样本数目小于等于32）
        pos_roi_per_this_image = int(min(pos_roi_per_image, pos_index.size))

        if pos_index.size > 0:
            pos_index = np.random.choice(
                pos_index, size=pos_roi_per_this_image, replace=False)

        neg_index = np.where((max_iou < self.neg_iou_thresh_high)
                             & (max_iou >= self.neg_iou_thresh_low))[0]
        neg_roi_per_this_image = self.n_sample - pos_roi_per_this_image
        neg_roi_per_this_image = int(min(neg_roi_per_this_image, neg_index.size))
        if neg_index.size > 0:
            neg_index = np.random.choice(
                neg_index, size=neg_roi_per_this_image, replace=False)

        keep_index = np.append(pos_index, neg_index)
        gt_roi_label = gt_roi_label[keep_index]
        gt_roi_label[pos_roi_per_this_image:] = 0  #负样本标签为0
        sample_roi = roi[keep_index]
        # 那么此时输出的128*4的sample_roi就可以去扔到 RoIHead网络里去进行分类与回归了。
        # 同样， RoIHead网络利用这sample_roi+featue为输入，输出是分类（21类）和回归（进一步微调bbox）的预测值，
        gt_roi_loc = bbox2loc(sample_roi, bbox[gt_assignment[keep_index]])
        gt_roi_loc = ((gt_roi_loc - np.array(loc_normalize_mean, np.float32)) / np.array(loc_normalize_std, np.float32))

        return sample_roi, gt_roi_loc, gt_roi_label

#负责在训练RPN的时候，从上万个anchor中选择一些(比如256)进行训练，
# 以使得正负样本比例大概是1:1. 同时给出训练的位置参数目标。
# 即返回gt_rpn_loc和gt_rpn_label。
#作用是生成训练要用的anchor(与对应框iou值最大或者最小的各128个框的坐标和256个label（0或者1）)
#AnchorTargetCreator产生的带标签的样本就是给RPN网络进行训练学习用哒
class AnchorTargetCreator(object):
    def __init__(self,
                 n_sample=256,
                 pos_iou_thresh=0.7, neg_iou_thresh=0.3,
                 pos_ratio=0.5):
        self.n_sample = n_sample
        self.pos_iou_thresh = pos_iou_thresh
        self.neg_iou_thresh = neg_iou_thresh
        self.pos_ratio = pos_ratio

    def __call__(self, bbox, anchor, img_size):
        img_H, img_W = img_size

        n_anchor = len(anchor)     #anchor:(S,4),S为anchor数
        inside_index = _get_inside_index(anchor, img_H, img_W)  #将那些超出图片范围的anchor全部去掉,只保留位于图片内部的序号
        anchor = anchor[inside_index]
        argmax_ious, label = self._create_label(inside_index, anchor, bbox)

        # anchor格式(S,4), bbox[argmax_ious]格式(S,4)，计算每个anchor与这个anchor和gtiou最大的边框回归，loc格式也为(S,4)
        loc = bbox2loc(anchor, bbox[argmax_ious])
        # 将位于图片内部的框的label对应到所有生成的20000个框中（label原本为所有在图片中的框的）
        label = _unmap(label, n_anchor, inside_index, fill=-1)
        # 将回归的框对应到所有生成的20000个框中（label原本为所有在图片中的框的）
        loc = _unmap(loc, n_anchor, inside_index, fill=0)

        return loc, label

    def _create_label(self, inside_index, anchor, bbox):
        # label: 1 is positive, 0 is negative, -1 is dont care
        label = np.empty((len(inside_index),),dtype=np.int32)
        label.fill(-1)   #lavel初始化全部填充为-1

        argmax_ious, max_iou, gt_argmax_iou = self._calc_ious(anchor, bbox, inside_index)

        label[max_iou < self.neg_iou_thresh] = 0  #每个anchor的最大iou与负样本阈值比较，低于阈值设为0
        label[gt_argmax_iou] = 1  #与每个gt iou最大的anchor设为正样本，设为1

        label[max_iou > self.pos_iou_thresh] = 1 #每个anchor的最大iou与正样本比较，高于阈值设为1
        n_pos = int(self.n_sample * self.pos_ratio)

        pos_index = np.where(label == 1)[0]

        if len(pos_index) > n_pos:
            disable_index = np.random.choice(
                pos_index, size=(len(pos_index) - n_pos), replace=False)
            label[disable_index] = -1  # 如果正样本数量大于128，就随机挑选多余的正样本设为-1

        n_neg = int(self.n_sample  - np.sum(label == 1))   #负样本数量
        neg_index = np.where(label == 0)[0]
        if len(neg_index) > n_neg:
            disable_index = np.random.choice(
                neg_index, size=(len(neg_index) - n_neg), replace=False)
            label[disable_index] = -1  #如果负样本数量大于256 - n_pos,就随机挑选多余的负样本设为-1

        return argmax_ious, label

    def _calc_ious(self, anchor, bbox, inside_index):
        ious = bbox_iou(anchor, bbox)  #ious格式(S, N), S为anchor数目， N为gt bbox数目

        argmax_ious = ious.argmax(axis=1)   #按行找到iou最大值索引
        max_iou = ious[np.arange(len(inside_index)), argmax_ious]   #求出每个anchor与哪个bbox的iou最大，以及最大值，max_ious:[1,S]
        gt_argmax_iou = ious.argmax(axis=0)  #按列找到iou最大值索引
        gt_max_iou = ious[gt_argmax_iou, np.arange(ious.shape[1])]  #求出每个bbox与哪个anchor的iou最大，以及最大值,gt_max_ious:[1,N]
        gt_argmax_iou = np.where(ious == gt_max_iou)[0]  #每个bbox与anchor最大iou的索引，每个bbox与哪个anchor的iou最大
        return argmax_ious, max_iou, gt_argmax_iou  #每个anchor与gt的最大iou索引号，每个anchor的最大iou值，gt与哪个anchoriou最大索引号

def _get_inside_index(anchor, H, W):
    index_inside = np.where(
        (anchor[:, 0] >=0) &
        (anchor[:, 1] >=0) &
        (anchor[:, 2] <=H) &
        (anchor[:, 3] <=W))[0]
    return index_inside

def _unmap(data, count, index, fill=0):
    if len(data.shape) == 1:   #映射label
        ret = np.empty((count,), dtype=data.dtype)
        ret.fill(fill)
        ret[index] = data   #将图片内的anchor的label复制到对应的ret中
    else:    #映射loc
        ret = np.empty((count,) + data.shape[1:], dtype=data.dtype)  #（n_anchor,4）
        ret.fill(fill)
        ret[index, :] = data  #将图片内的anchor的loc复制到对应的ret中
    return ret

#在RPN中，从上万个anchor中，选择一定数目（2000或者300），调整大小和位置，生成RoIs，用以Fast R-CNN训练或者测试。
# 这部分的操作不需要进行反向传播，因此可以利用numpy/tensor实现
#对于每张图片，利用RPN计算大概20000个anchor属于前景的概率，
# 然后训练时从中选取概率较大的12000个，利用位置回归参数，修正这12000个anchor的位置，
# 利用非极大值抑制，选出2000个ROIS以及对应的位置参数。
#测试时选取概率较大的6000个，用同样的方法选取300ROIS.
class ProposalCreator:
    def __init__(self,
                 parent_model,
                 nms_thresh=0.7,
                 n_train_pre_nms=12000,
                 n_train_post_nms=2000,
                 n_test_pre_nms=6000,
                 n_test_post_nms=300,
                 min_size=16):
        self.parent_model = parent_model
        self.nms_thresh = nms_thresh
        self.n_train_pre_nms = n_train_pre_nms
        self.n_train_post_nms = n_train_post_nms
        self.n_test_pre_nms = n_test_pre_nms
        self.n_test_post_nms = n_test_post_nms
        self.min_size = min_size

    def __call__(self, loc, score, anchor, img_size, scale=1.):  #这里的loc和score是经过region_proposal_network中经过1x1卷积分类和回归得到的

        if self.parent_model.training:
            n_pre_nms = self.n_train_pre_nms
            n_post_nms = self.n_train_post_nms
        else:
            n_pre_nms = self.n_test_pre_nms
            n_post_nms = self.n_test_post_nms

        roi = loc2bbox(anchor, loc)
        #裁剪将rois的ymin,ymax限定在[0,H]
        #slice() 函数实现切片对象, clip函数实现截断对象
        roi[:, slice(0,4,2)] = np.clip(
            roi[:, slice(0,4,2)], 0, img_size[0])
        roi[:, slice(1,4,2)] = np.clip(
            roi[:, slice(1,4,2)], 0, img_size[1])

        min_size = self.min_size * scale  #设定roi的最小尺寸
        hs = roi[:, 2] - roi[:, 0]  #roi的高度
        ws = roi[:, 3] - roi[:, 1]  #roi的宽度
        keep = np.where((hs >= min_size) &(ws >= min_size))[0]   #挑出大于16*16的roi
        roi = roi[keep, :]
        score = score[keep]

        order = score.ravel().argsort()[::-1]  #分数从大到小排列
        if n_pre_nms > 0:
            order = order[:n_pre_nms]  #train时从20000中取前12000个rois，test取前6000个
        roi = roi[order, :]

        #使用nms过一遍排序后的roi
        keep = non_maximum_suppression(
            cp.ascontiguousarray(cp.asarray(roi)),
            thresh=self.nms_thresh)
        if n_post_nms > 0:
            keep = keep[:n_post_nms]
        roi = roi[keep]
        return roi


if __name__ == '__main__':
    from model.region_proposal_network import _enumerate_shifted_anchor
    from model.utils.bbox_tools import generate_anchor_base
    anchor_base = generate_anchor_base()

    anchor = _enumerate_shifted_anchor(anchor_base, 16, 60, 35)
    bbox = np.array([[100,100,200,200],[400,400,650,500]])
    anchor_target = AnchorTargetCreator()
    loc, label = anchor_target.__call__(bbox,anchor,(960,560))

    roi = np.zeros((2000,4),dtype=float)
    label = np.zeros((2000,),dtype=int)


    proposal = ProposalTargetCreator()
    sample_roi, gt_roi_loc, gt_roi_label = proposal.__call__(roi,bbox,label)



