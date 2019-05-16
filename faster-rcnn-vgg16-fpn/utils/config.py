from pprint import pprint #提供了打印出任何python数据结构类和方法。

class config:
    voc_data_dir = '/media/fengkai/Seagate Backup Plus Drive/Dataset/voc/VOCdevkit/VOC2007/'
    coco_data_dir = ''
    class_num = 20  #voc 20, coco 80
    min_size = 600
    max_size = 1000
    num_workers = 8   # #每次提取数据多进进程
    test_num_workers = 2

    # sigma for l1_smooth_loss
    rpn_sigma = 3.
    roi_sigma = 1.

    weight_decay = 0.0005
    lr = 0.001
    lr_decay = 0.1

    # visualization
    env = 'faster-rcnn'  # visdom env
    port = 8097
    plot_every = 10  # vis every N iter

    # training
    epoch = 14

    use_dropout = False  #是否在分类时使用use_dropout。（vgg16的classifier部分）
    use_adam = False

    # debug
    debug_file = '/tmp/debugf'

    test_num = 10000

    # load model
    load_path = None

    caffe_pretrain = False
    caffe_pretrain_path = 'checkpoints/vgg16_caffe.pth'

    def _parse(self, kwargs):
        state_dict = self._state_dict()
        for k, v in kwargs.items():
            if k not in state_dict:
                raise ValueError('UnKnown Option: "--%s"' % k)
            setattr(self, k, v)

        print('======user config========')
        pprint(self._state_dict())
        print('==========end============')

    def _state_dict(self):
        return {k: getattr(self, k) for k, _ in config.__dict__.items() \
                if not k.startswith('_')}

opt = config()