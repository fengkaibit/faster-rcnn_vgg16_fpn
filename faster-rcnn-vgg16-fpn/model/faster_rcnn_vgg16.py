from __future__ import absolute_import
import sys
sys.path.append('/home/fengkai/PycharmProjects/faster-rcnn-resnet50-fpn')
import torch
import numpy as np
from torchvision.models import vgg16, resnet50
from model.region_proposal_network import RegionProposalNetwork
from model.faster_rcnn import FasterRCNN
from model.roi_module import RoIPooling2D
from roi_align.functions.roi_align import RoIAlignFunction
from utils.config import opt
from utils import array_tool
from torch.nn import functional
from model.fpn import FPN

def decom_vgg16():
    model = vgg16(not opt.load_path)

    C2 = list(model.features)[:16]
    C3 = list(model.features)[16:23]
    C4 = list(model.features)[23:30]

    classifier = model.classifier  # vgg16分类层
    classifier = list(classifier)
    del classifier[6]  # 删除最后一层分类层，Linear（4096 -> 1000）
    if not opt.use_dropout:
        del classifier[5]
        del classifier[2]
    classifier = torch.nn.Sequential(*classifier)

    return torch.nn.Sequential(*C2), torch.nn.Sequential(*C3), torch.nn.Sequential(*C4), classifier

class FasterRCNNVGG16RPN(FasterRCNN):

    def __init__(self,
                 n_fg_class= opt.class_num,
                 ratios=[0.5, 1, 2],
                 anchor_scales=[8, 16],
                 feat_stride=[4, 8, 16, 32]):
        C2, C3, C4, classifier = decom_vgg16()

        fpn = FPN(
            out_channels=512
        )

        rpn = RegionProposalNetwork(
            512, 512,
            ratios=ratios,
            anchor_scales=anchor_scales,
            feat_stride=feat_stride,
        )

        head = VGG16RoIHead(
            n_class=n_fg_class + 1,
            roi_size=7,
            feat_stride=[4, 8, 16, 32],
            classifier=classifier,
        )

        super(FasterRCNNVGG16RPN, self).__init__(
            C2, C3, C4,
            fpn,
            rpn,
            head,
        )

class VGG16RoIHead(torch.nn.Module):
    def __init__(self, n_class, roi_size, feat_stride, classifier):
        super(VGG16RoIHead, self).__init__()

        self.classifier = classifier
        self.cls_loc = torch.nn.Linear(4096, n_class * 4)
        self.score = torch.nn.Linear(4096, n_class)
        normal_init(self.cls_loc, 0, 0.01)
        normal_init(self.score, 0, 0.01)

        self.n_class = n_class
        self.roi_size = roi_size
        self.feat_stride = feat_stride
        self.spatial_scale = [1. / i for i in feat_stride]

    def forward(self, features_maps, rois, roi_indices):
        roi_indices = array_tool.totensor(roi_indices).float()
        rois = array_tool.totensor(rois).float()
        roi_level = self._PyramidRoI_Feat(rois)
        indices_and_rois = torch.cat([roi_indices[:, None], rois], dim=1)
        xy_indices_and_rois = indices_and_rois[:, [0, 2, 1, 4, 3]]  # yx->xy
        indices_and_rois = xy_indices_and_rois.contiguous()  # 把tensor变成在内存中连续分布的形式

        roi_pool_feats = []
        roi_to_levels = []

        for i, l in enumerate(range(2, 5)):
            if (roi_level == l).sum() == 0:
                continue
            idx_l = (roi_level == l).nonzero()
            roi_to_levels.append(idx_l)

            keep_indices_and_rois = indices_and_rois[idx_l]
            keep_indices_and_rois = keep_indices_and_rois.view(-1, 5)
            #roi_pooling = RoIPooling2D(self.roi_size, self.roi_size, self.spatial_scale[i])
            roi_align = RoIAlignFunction(self.roi_size, self.roi_size, self.spatial_scale[i])
            #pool = roi_pooling(features_maps[i], keep_indices_and_rois)   #通过roi_pooling
            pool = roi_align(features_maps[i], keep_indices_and_rois)
            roi_pool_feats.append(pool)
        roi_pool_feats = torch.cat(roi_pool_feats, 0)
        roi_to_levels = torch.cat(roi_to_levels, 0)
        roi_to_levels = roi_to_levels.squeeze()
        idx_sorted, order = torch.sort(roi_to_levels)
        roi_pool_feats = roi_pool_feats[order]

        pool = roi_pool_feats.view(roi_pool_feats.size(0), -1)  # batch_size, CHW拉直

        fc7 = self.classifier(pool)
        roi_cls_locs = self.cls_loc(fc7)  # （1000->84）每一类坐标回归
        roi_scores = self.score(fc7)  # （1000->21） 每一类类别预测

        return roi_cls_locs, roi_scores

    def _PyramidRoI_Feat(self, rois):
        roi_h = rois[:, 2] - rois[:, 0] + 1
        roi_w = rois[:, 3] - rois[:, 1] + 1
        roi_level = torch.log(torch.sqrt(roi_h*roi_w)/224.0) /np.log(2)
        roi_level = torch.round(roi_level + 4)
        roi_level[roi_level < 2] = 2
        roi_level[roi_level > 4] = 4
        return roi_level


def normal_init(m, mean, stddev, truncated=False):
    if truncated:
        m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean)   #截断产生正态分布
    else:
        m.weight.data.normal_(mean, stddev)   #普通产生正态分布
        m.bias.data.zero_()

def weights_init(m, mean, stddev, truncated=False):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

if __name__ == '__main__':
    from data.util import read_image
    import cv2
    cv_img = cv2.imread('/home/fengkai/dog.jpg')
    src_img = read_image('/home/fengkai/dog.jpg')

    from data.dataset import preprocess
    img = preprocess(array_tool.tonumpy(src_img))
    img = torch.from_numpy(img)[None]
    C2, C3, C4, classifier = decom_vgg16()
    c2_out = C2(img)
    print(c2_out.shape)
    c3_out = C3(c2_out)
    print(c3_out.shape)
    c4_out = C4(c3_out)
    print(c4_out.shape)

    '''
    import numpy as np
    from model.fpn import FPN
    fpn = FPN(256)
    p2, p3, p4, p5, p6 = fpn.forward(c2_out, c3_out, c4_out, c5_out)

    rcnn_maps = [p2, p3, p4, p5]
    feat_stride = [4, 8, 16, 32, 64]
    spatial_scale = [1. / i for i in feat_stride]
    for i, l in enumerate(range(2, 6)):
        print(rcnn_maps[i].shape)
        print(spatial_scale[i])

    feature_maps = [p2, p3, p4, p5, p6]
    from model.region_proposal_network import RegionProposalNetwork
    rpn = RegionProposalNetwork()
    img_size = [src_img.shape[1], src_img.shape[2]]
    all_rpn_locs, all_rpn_scores, all_rois, all_roi_indices, all_anchors = rpn.forward(feature_maps, img_size)
    '''

    '''
    features_base, features_top, classifier = decom_resnet50()

    roihead = Resnet50RoIHead(21,14, 1. / 16, features_top, classifier)
    roihead = roihead.cuda()
    ## fake data###
    B, N, C, H, W, PH, PW = 1, 8, 1024, 32, 32, 40, 40

    bottom_data = torch.randn(B, C, H, W).cuda()
    spatial_scale = 1. / 16
    outh, outw = PH, PW
    x = bottom_data.requires_grad_()
    import numpy as np
    rois = np.ones((128, 4),dtype=np.float32)
    roi_indices = np.zeros((128,),dtype=np.int32)
    roi_cls_locs, roi_scores = roihead.forward(x, rois, roi_indices)
    roi_cls_locs = roi_cls_locs.view(128, -1, 4)
    label = np.ones((128, ),dtype=np.uint8)
    label = array_tool.totensor(label).long()
    a = torch.arange(0, 128).long().cuda()
    b = roi_cls_locs[a, label]
    '''
