from __future__ import absolute_import
import cupy as cp
import os
import matplotlib
import ipdb
from tqdm import tqdm  #进度条库

from utils.config import opt
from utils import array_tool
from data.dataset import Dataset, TestDataset, inverse_normalize
from model.faster_rcnn_vgg16 import FasterRCNNVGG16RPN
from torch.utils import data as data_
from trainer import FasterRCNNTrainer
from utils.vis_tool import visdom_bbox
from utils.eval_tool import eval_detection_voc
from model.faster_rcnn_resnet50 import FasterRCNNResNet50

import resource  #用于查询或修改当前系统资源限制设置

rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (20480, rlimit[1]))

matplotlib.use('agg')


def eval(dataloader, faster_rcnn, test_num=10000):
    pred_bboxes, pred_labels, pred_scores = list(), list(), list()
    gt_bboxes, gt_labels, gt_difficults = list(), list(), list()
    for ii, (imgs, sizes, gt_bboxes_, gt_labels_, gt_difficults_) in tqdm(enumerate(dataloader)):
        sizes = [sizes[0][0].item(), sizes[1][0].item()]
        pred_bboxes_ , pred_labels_, pred_scores_ = faster_rcnn.predict(imgs, [sizes])
        gt_bboxes += list(gt_bboxes_.numpy())
        gt_labels += list(gt_labels_.numpy())
        gt_difficults += list(gt_difficults_.numpy())
        pred_bboxes += pred_bboxes_
        pred_labels += pred_labels_
        pred_scores += pred_scores_
        if ii == test_num: break

    result = eval_detection_voc(
        pred_bboxes, pred_labels, pred_scores,
        gt_bboxes, gt_labels, gt_difficults,
        use_07_metric=True)
    return result

def train(**kwargs):
    opt._parse(kwargs)  #将调用函数时候附加的参数用，config.py文件里面的opt._parse()进行解释，然后获取其数据存储的路径，之后放到Dataset里面！

    dataset = Dataset(opt)
    print('load data')
    dataloader = data_.DataLoader(dataset,
                                  batch_size=1,
                                  shuffle=True,
                                  num_workers=opt.num_workers)

    testset = TestDataset(opt)
    test_dataloader = data_.DataLoader(testset,
                                       batch_size=1,
                                       num_workers=opt.test_num_workers,
                                       shuffle=False,
                                       #pin_memory=True
                                       )   #pin_memory锁页内存,开启时使用显卡的内存，速度更快

    faster_rcnn = FasterRCNNVGG16RPN()
    print('model construct completed')
    trainer = FasterRCNNTrainer(faster_rcnn).cuda()
    #判断opt.load_path是否存在，如果存在，直接从opt.load_path读取预训练模型，然后将训练数据的label进行可视化操作
    if opt.load_path:
        trainer.load(opt.load_path)
        print('load pretrained model from %s' %opt.load_path)
    trainer.vis.text(dataset.dataset.label_names, win='labels')
    best_map = 0
    lr_ = opt.lr
    # 之后用一个for循环开始训练过程，而训练迭代的次数opt.epoch=14也在config.py文件中都预先定义好，属于超参数
    for epoch in range(opt.epoch):
        print('epoch {}/{}'.format(epoch, opt.epoch))
        trainer.reset_meters()  #首先在可视化界面重设所有数据
        for ii, (img, bbox_, label_, scale) in tqdm(enumerate(dataloader)):
            scale = array_tool.scalar(scale)
            img, bbox, label = img.cuda().float(), bbox_.cuda(), label_.cuda()
            trainer.train_step(img, bbox, label, scale)

            if (ii+1) % opt.plot_every == 0:
                if os.path.exists(opt.debug_file):
                    ipdb.set_trace()

                #可视化画出loss
                trainer.vis.plot_many(trainer.get_meter_data())
                #可视化画出groudtruth bboxes
                ori_img_ = inverse_normalize(array_tool.tonumpy(img[0]))
                gt_img = visdom_bbox(ori_img_,
                                     array_tool.tonumpy(bbox_[0]),
                                     array_tool.tonumpy(label_[0]))
                trainer.vis.img('gt_img', gt_img)

                #可视化画出预测bboxes
                # 调用faster_rcnn的predict函数进行预测，预测的结果保留在以_下划线开头的对象里面
                _bboxes, _labels, _scores = trainer.faster_rcnn.predict([ori_img_], visualize=True)
                pred_img = visdom_bbox(ori_img_,
                                       array_tool.tonumpy(_bboxes[0]),
                                       array_tool.tonumpy(_labels[0]).reshape(-1),
                                       array_tool.tonumpy(_scores[0]))
                trainer.vis.img('pred_img', pred_img)
                # 调用 trainer.vis.text将rpn_cm也就是RPN网络的混淆矩阵在可视化工具中显示出来
                trainer.vis.text(str(trainer.rpn_cm.value().tolist()), win='rpn_cm')
                #将roi_cm也就是roihead网络的混淆矩阵在可视化工具中显示出来
                trainer.vis.img('roi_cm', array_tool.totensor(trainer.roi_cm.conf, False).float())
        eval_result = eval(test_dataloader, faster_rcnn, test_num=opt.test_num)
        trainer.vis.plot('test_map', eval_result['map'])
        lr_ = trainer.faster_rcnn.optimizer.param_groups[0]['lr']
        log_info = 'lr:{}, map:{}, loss:{}'.format(str(lr_),
                                          str(eval_result['map']),
                                          str(trainer.get_meter_data()))
        trainer.vis.log(log_info)  #将学习率以及map等信息及时显示更新

        if eval_result['map'] > best_map:
            best_map = eval_result['map']
            best_path = trainer.save(best_map=best_map)
        if epoch == 9:   #if判断语句如果学习的epoch达到了9就将学习率*0.1变成原来的十分之一
            trainer.load(best_path)
            trainer.faster_rcnn.scale_lr(opt.lr_decay)
            lr_ = lr_ * opt.lr_decay

        if epoch == 13:
            break

if __name__ == '__main__':
    import fire

    fire.Fire()
    train()