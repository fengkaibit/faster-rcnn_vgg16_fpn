import numpy as np
import six    #兼容python2和python3的差异
from six import __init__

#已知源框和位置偏差求目标框,loc格式(R,4),d_y, d_x, d_h, d_w
def loc2bbox(src_bbox, loc):
    if src_bbox.shape[0] == 0:
        return np.zeros((0,4), dtype=loc.dtype)

    src_bbox = src_bbox.astype(src_bbox.dtype, copy=False)

    src_height = src_bbox[:, 2] - src_bbox[:, 0]
    src_width = src_bbox[:, 3] - src_bbox[:, 1]
    src_center_x = src_bbox[:, 1] + 0.5 * src_width
    src_center_y = src_bbox[:, 0] + 0.5 * src_height

    dy = loc[:, 0::4]
    dx = loc[:, 1::4]
    dh = loc[:, 2::4]
    dw = loc[:, 3::4]

    dst_center_y = dy * src_height[:,np.newaxis] + src_center_y[:, np.newaxis]  #(R,1) = (R,1)*(R,0)+(R,0) 因此要新增一个轴
    dst_center_x = dx * src_width[:, np.newaxis] + src_center_x[:, np.newaxis]
    dst_height = np.exp(dh) * src_height[:, np.newaxis]
    dst_width = np.exp(dw) * src_width[:, np.newaxis]

    dst_bbox = np.zeros(loc.shape, dtype=loc.dtype)
    dst_bbox[:, 0::4] = dst_center_y - 0.5 * dst_height
    dst_bbox[:, 1::4] = dst_center_x - 0.5 * dst_width
    dst_bbox[:, 2::4] = dst_center_y + 0.5 * dst_height
    dst_bbox[:, 3::4] = dst_center_x + 0.5 * dst_width   #由中心点转换为左上角点和右下角点格式[ymin,xmin,ymax,xmax]

    return dst_bbox

#已知源框和目标框求出其位置偏差
def bbox2loc(src_bbox,dst_bbox):
    src_height = src_bbox[:, 2] - src_bbox[:, 0]
    src_width = src_bbox[:, 3] - src_bbox[:, 1]
    src_center_x = src_bbox[:, 1] + 0.5 * src_width
    src_center_y = src_bbox[:, 0] + 0.5 * src_height

    dst_height = dst_bbox[:, 2] - dst_bbox[:, 0]
    dst_width = dst_bbox[:, 3] - dst_bbox[:, 1]
    dst_center_x = dst_bbox[:, 1] + 0.5 * dst_width
    dst_center_y = dst_bbox[:, 0] + 0.5 * dst_height

    eps = np.finfo(src_height.dtype).eps     #得到最小正数eps
    src_height = np.maximum(src_height, eps)
    src_width = np.maximum(src_width, eps)   #将src_height和src_width与eps比较保证两者都不为负

    dy = (dst_center_y - src_center_y) / src_height
    dx = (dst_center_x - src_center_x) / src_width
    dw = np.log(dst_width / src_width)
    dh = np.log(dst_height / src_height)

    loc = np.vstack((dy,dx,dh,dw)).transpose()  #np.vstack按照行的顺序把数组给堆叠起来
    return loc

def bbox_iou(bbox_a, bbox_b):    #bbox_a:(N,4), bbox_b:(K,4)
    if bbox_a.shape[1] != 4 or bbox_b.shape[1] != 4:
        raise IndexError  #raise抛出异常

    tl = np.maximum(bbox_a[:,None,:2], bbox_b[:, :2]) #tl为交叉部分框左上角坐标, tl:(N,K,2)

    br = np.minimum(bbox_a[:,None,2:], bbox_b[:, 2:]) #br为交叉部分框右下角坐标, br:(N,K,2)
    area_i = np.prod(br - tl, axis=2) * (tl < br).all(axis=2) #在轴2上计算br-tl的乘积，
                                                            # 然后判断br是否都小于tl(两个框不相交的情况), (N,K)
    area_a = np.prod(bbox_a[:,2:] - bbox_a[:,:2],axis=1) #在轴1(行)上计算(ymax,xmax)-(ymin,xmin)的乘积, (N,)
    area_b = np.prod(bbox_b[:, 2:] - bbox_b[:, :2], axis=1)  #(K,)
    iou = area_i / (area_a[:, None] + area_b - area_i)  #(N,K)
    return iou

def generate_anchor_base(base_size=16, ratios=[0.5, 1, 2],anchor_scales=[8,16,32]):
    center_x = base_size / 2
    center_y = base_size / 2
    anchor_base = np.zeros((len(ratios) * len(anchor_scales),4),dtype=np.float32)

    for i in six.moves.range(len(ratios)):
        for j in six.moves.range(len(anchor_scales)):
            anchor_height = base_size * anchor_scales[j] * np.sqrt(ratios[i])
            anchor_width = base_size * anchor_scales[j] * np.sqrt(1. / ratios[i])

            index = i * len(anchor_scales) + j
            anchor_base[index, 0] = center_y - anchor_height / 2.
            anchor_base[index, 1] = center_x - anchor_width / 2.
            anchor_base[index, 2] = center_y + anchor_height /2.
            anchor_base[index, 3] = center_x + anchor_width / 2.
    return anchor_base   #(9,4)

if __name__ == '__main__':
    anchor_base = generate_anchor_base()
    print(anchor_base)


