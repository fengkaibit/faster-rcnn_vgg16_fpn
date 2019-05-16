import torch
from torch import nn
from torchvision import transforms, utils
from torch.autograd import Variable, gradcheck
import sys
sys.path.append('/home/fengkai/PycharmProjects/my-faster-rcnn')
from roi_align.functions.roi_align import RoIAlignFunction
import cv2
from skimage.io import imread
import matplotlib.pyplot as plt

def crop_and_resize(pool_size, feature_map, boxes, box_ind):
    if boxes.shape[1]==5:
        x1, y1, x2, y2, _= boxes.chunk(5, dim=1)
    else:
        x1, y1, x2, y2= boxes.chunk(4, dim=1)
    
    box_ind=box_ind.view(-1,1).float()

    boxes = torch.cat((box_ind, x1, y1, x2, y2), 1)
    return RoIAlignFunction(pool_size[0],pool_size[1], 1)(feature_map, boxes)


def to_varabile(tensor, requires_grad=False, is_cuda=True):
    if is_cuda:
        tensor = tensor.cuda()
    var = Variable(tensor, requires_grad=requires_grad)
    return var

if __name__=='__main__':
    pool_width=7
    pool_height=7

    is_cuda = torch.cuda.is_available()

    img_path1 = '/home/fengkai/darknet/data/eagle.jpg'
    img_path2 = '/home/fengkai/darknet/data/horses.jpg'

    boxes_data = torch.FloatTensor([[0, 0, 200, 200], [0, 0, 200, 200]])
    box_index_data = torch.IntTensor([0, 1])

    image_data1 = transforms.ToTensor()(imread(img_path1)).unsqueeze(0)
    image_data2 = transforms.ToTensor()(imread(img_path2)).unsqueeze(0)

    image_data = torch.cat((image_data1, image_data2), 0)
    print(image_data.shape)

    image_torch = to_varabile(image_data, is_cuda=is_cuda)
    boxes = to_varabile(boxes_data, is_cuda=is_cuda)
    box_index = to_varabile(box_index_data, is_cuda=is_cuda)

    crops_torch = crop_and_resize((pool_width, pool_height), image_torch, boxes, box_index)

    print(crops_torch.data.size())
    crops_torch_data = crops_torch.data.cpu().numpy().transpose(0, 2, 3, 1)
    fig = plt.figure()
    plt.subplot(121)
    plt.imshow(crops_torch_data[0])
    plt.subplot(122)
    plt.imshow(crops_torch_data[1])
    plt.show()
