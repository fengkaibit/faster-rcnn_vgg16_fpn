from __future__ import absolute_import
import os
from collections import namedtuple
import time
from torch.nn import functional
from model.utils.creator_tool import AnchorTargetCreator, ProposalTargetCreator

from torch import nn
import torch
from utils import array_tool
from utils.config import opt
from torchnet.meter import ConfusionMeter, AverageValueMeter
from utils.vis_tool import Visualizer

LossTuple = namedtuple('LossTuple',
                       ['rpn_loc_loss',
                        'rpn_cls_loss',
                        'roi_loc_loss',
                        'roi_cls_loss',
                        'total_loss'])

class FasterRCNNTrainer(nn.Module):

    def __init__(self, faster_rcnn):
        super(FasterRCNNTrainer, self).__init__()

        self.faster_rcnn = faster_rcnn
        self.rpn_sigma = opt.rpn_sigma     #是在_faster_rcnn_loc_loss调用用来计算位置损失函数用到的超参数，
        self.roi_sigma = opt.roi_sigma

        self.anchor_target_creator = AnchorTargetCreator()   #从上万个anchor中挑选256个来训练rpn，其中正样本不超过128
        self.proposal_target_creator = ProposalTargetCreator()  #从rpn给的2000个框中挑出128个来训练roihead，其中正样本不超过32个

        self.loc_normalize_mean = faster_rcnn.loc_normalize_mean
        self.loc_normalize_std = faster_rcnn.loc_normalize_std

        self.optimizer = self.faster_rcnn.get_optimizer()
        #可视化
        self.vis = Visualizer(env=opt.env)

        #验证预测值和真实值的精度
        self.rpn_cm = ConfusionMeter(2) #混淆矩阵，就是验证预测值与真实值精确度的矩阵ConfusionMeter(2)括号里的参数指的是类别数
        self.roi_cm = ConfusionMeter(21)
        self.meters = {k: AverageValueMeter() for k in LossTuple._fields}  #验证平均loss

    def forward(self, imgs, bboxes, labels, scale):
        '''
        :param imgs:  (~torch.autograd.Variable)  一个批次的图片
        :param bboxes: (~torch.autograd.Variable)  (N, R, 4)
        :param labels:  (~torch.autograd..Variable)  (N, R)  [0 - L-1] L为类别数
        :param scale:   (float)  原图经过preprocessing处理后的缩放比
        :return:  namedtuple of 5 losses
        '''

        n = bboxes.shape[0]  #batch_size 数量
        if n != 1:
            raise ValueError('Currently only batch size 1 is supported')

        _, _, H, W = imgs.shape
        img_size = (H, W)

        c2_out = self.faster_rcnn.C2(imgs)
        c3_out = self.faster_rcnn.C3(c2_out)
        c4_out = self.faster_rcnn.C4(c3_out)

        p2, p3, p4, p5 = self.faster_rcnn.fpn(c2_out, c3_out, c4_out)
        feature_maps = [p2, p3, p4, p5]
        rcnn_maps = [p2, p3, p4]

        # rpn_locs的维度（hh*ww*9，4），rpn_scores维度为（hh*ww*9，2）， rois的维度为（2000,4），
        # roi_indices用不到，anchor的维度为（hh*ww*9，4），H和W是经过数据预处理后的。
        # 计算（H/16）x(W/16)x9(大概20000)个anchor属于前景的概率，取前12000个并经过NMS得到2000个近似目标框G^的坐标。
        # roi的维度为(2000,4)
        rpn_locs, rpn_scores, rois, roi_indices, anchor = self.faster_rcnn.rpn(
            feature_maps, img_size, scale)

        bbox = bboxes[0]
        label = labels[0]
        rpn_score = rpn_scores[0]  #（hh*ww*9，2）
        rpn_loc = rpn_locs[0]   #(hh*ww*9,4)
        roi = rois   #(2000,4)

        # 调用proposal_target_creator函数生成sample roi（128，4）、gt_roi_loc（128，4）、
        # gt_roi_label（128，1），RoIHead网络利用这sample_roi+featue为输入，
        # 输出是分类（21类）和回归（进一步微调bbox）的预测值，
        # 那么分类回归的groud truth就是ProposalTargetCreator输出的gt_roi_label和gt_roi_loc。
        sample_roi, gt_roi_loc, gt_roi_label = self.proposal_target_creator(
            roi,
            array_tool.tonumpy(bbox),
            array_tool.tonumpy(label),
            self.loc_normalize_mean,
            self.loc_normalize_std)

        sample_roi_index = torch.zeros(len(sample_roi))

        roi_cls_loc, roi_score = self.faster_rcnn.head(
            rcnn_maps,
            sample_roi,
            sample_roi_index)


        #------------------RPN loss------------------#
        gt_rpn_loc, gt_rpn_label = self.anchor_target_creator(
            array_tool.tonumpy(bbox),
            anchor,
            img_size)
        gt_rpn_label = array_tool.totensor(gt_rpn_label).long()
        gt_rpn_loc = array_tool.totensor(gt_rpn_loc)
        #rpn的回归l1smooth损失
        rpn_loc_loss = _fast_rcnn_loc_loss(
            rpn_loc,
            gt_rpn_loc,
            gt_rpn_label.data,
            self.rpn_sigma)
        #rpn的分类交叉熵损失
        rpn_cls_loss = functional.cross_entropy(rpn_score, gt_rpn_label.cuda(), ignore_index=-1)
        _gt_rpn_label = gt_rpn_label[gt_rpn_label > -1]
        _gt_rpn_score = rpn_score[gt_rpn_label > -1]
        _rpn_score = array_tool.tonumpy(rpn_score)[array_tool.tonumpy(gt_rpn_label) > -1]
        self.rpn_cm.add(array_tool.totensor(_rpn_score, False), _gt_rpn_label.data.long())

        #------------------------ROI loss------------------------#
        n_sample = roi_cls_loc.shape[0]   #n_sample为128 , roi_cls_loc为VGG16RoIHead的输出（128*84）
        roi_cls_loc = roi_cls_loc.view(n_sample, -1, 4) # roi_cls_loc=（128,21,4）
        roi_loc = roi_cls_loc[torch.arange(0, n_sample).long().cuda(), \
                                array_tool.totensor(gt_roi_label).long()]  # (128,4),按照label编号从21类中挑出当前标签的loc，从（128,21,4）降为（128,4）
        gt_roi_label = array_tool.totensor(gt_roi_label).long()
        gt_roi_loc = array_tool.totensor(gt_roi_loc)

        #roi的回归l1smooth损失
        roi_loc_loss = _fast_rcnn_loc_loss(
            roi_loc.contiguous(),
            gt_roi_loc,
            gt_roi_label.data,
            self.roi_sigma)
        roi_cls_loss = nn.CrossEntropyLoss()(roi_score, gt_roi_label.cuda())  #roi的交叉熵损失
        self.roi_cm.add(array_tool.totensor(roi_score, False), gt_roi_label.data.long())

        losses = [rpn_loc_loss, rpn_cls_loss, roi_loc_loss, roi_cls_loss]
        losses = losses + [sum(losses)]  #总loss，增加losses列表长度到5

        return LossTuple(*losses)

    def train_step(self, imgs, bboxes, labels, scale):
        self.optimizer.zero_grad()
        losses = self.forward(imgs, bboxes, labels, scale)
        losses.total_loss.backward()
        self.optimizer.step()
        self.update_meters(losses)
        return losses

    def save(self, save_optimizer=False, save_path=None, **kwargs):
        save_dict= dict()

        save_dict['model'] = self.faster_rcnn.state_dict()
        save_dict['config'] = opt._state_dict()
        save_dict['other_info'] = kwargs
        save_dict['vis_info'] = self.vis.state_dict()

        if save_optimizer:
            save_dict['optimizer'] = self.optimizer.state_dict()
        if save_path is None:
            timestr = time.strftime('%m%d%H%M')
            save_path = 'checkpoints/fasterrcnn_%s' % timestr
            for k_, v_ in kwargs.items():
                save_path += '_%s' % v_

        save_dir = os.path.dirname(save_path)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        torch.save(save_dict, save_path)
        self.vis.save([self.vis.env])
        return save_path

    def load(self, path, load_optimizer=True, parse_opt=False, ):
        state_dict = torch.load(path)
        if 'model' in state_dict:
            self.faster_rcnn.load_state_dict(state_dict['model'])
        else:  # legacy way, for backward compatibility
            self.faster_rcnn.load_state_dict(state_dict)
            return self
        if parse_opt:
            opt._parse(state_dict['config'])
        if 'optimizer' in state_dict and load_optimizer:
            self.optimizer.load_state_dict(state_dict['optimizer'])
        return self


    def update_meters(self, losses):
        loss_d = {k: array_tool.scalar(v) for k, v in losses._asdict().items()}
        for key, meter in self.meters.items():
            meter.add(loss_d[key])

    def reset_meters(self):
        for key, meter in self.meters.items():
            meter.reset()
        self.roi_cm.reset()
        self.rpn_cm.reset()

    def get_meter_data(self):
        return {k: v.value()[0] for k, v in self.meters.items()}

def _smooth_l1_loss(x, t, in_weight, sigma):
    sigma2 = sigma ** 2
    diff = in_weight * (x - t)
    abs_diff = diff.abs()
    flag = (abs_diff < (1. / sigma2)).float()
    y = (flag * (sigma2 / 2.) * (diff ** 2) +
         (1 - flag) * (abs_diff - 0.5 / sigma2))
    return y.sum()

def _fast_rcnn_loc_loss(pred_loc, gt_loc, gt_label, sigma):
    in_weight = torch.zeros(gt_loc.shape).cuda()
    in_weight[(gt_label > 0).view(-1, 1).expand_as(in_weight).cuda()] = 1
    loc_loss = _smooth_l1_loss(pred_loc, gt_loc, in_weight, sigma)
    #计算平均loss，加上负样本
    loc_loss /= ((gt_label >=0).sum().float())
    return loc_loss




