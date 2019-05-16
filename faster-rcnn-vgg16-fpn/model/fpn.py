from __future__ import absolute_import
import torch
from torch.nn import functional

class FPN(torch.nn.Module):
    def __init__(self, out_channels):
        super(FPN, self).__init__()
        self.out_channels = out_channels

        self.P5 = torch.nn.MaxPool2d(kernel_size=1, stride=2, padding=0)

        self.P4_conv1 = torch.nn.Conv2d(512, self.out_channels, kernel_size=1, stride=1, padding=0)
        self.P4_conv2 = torch.nn.Conv2d(self.out_channels, self.out_channels, 3, 1, 1)

        self.P3_conv1 = torch.nn.Conv2d(512, self.out_channels, kernel_size=1, stride=1, padding=0)
        self.P3_conv2 = torch.nn.Conv2d(self.out_channels, self.out_channels, 3, 1, 1)

        self.P2_conv1 = torch.nn.Conv2d(256, self.out_channels, kernel_size=1, stride=1, padding=0)
        self.P2_conv2 = torch.nn.Conv2d(self.out_channels, self.out_channels, 3, 1, 1)

        normal_init(self.P4_conv1, 0, 0.01)
        normal_init(self.P4_conv2, 0, 0.01)
        normal_init(self.P3_conv1, 0, 0.01)
        normal_init(self.P3_conv2, 0, 0.01)
        normal_init(self.P2_conv1, 0, 0.01)
        normal_init(self.P2_conv2, 0, 0.01)


    def forward(self, C2, C3, C4):

        p4_out = self.P4_conv1(C4)

        p5_out = self.P5(p4_out)

        p3_out = self._upsample_add(p4_out, self.P3_conv1(C3))
        p2_out = self._upsample_add(p3_out, self.P2_conv1(C2))

        p4_out = self.P4_conv2(p4_out)
        p3_out = self.P3_conv2(p3_out)
        p2_out = self.P2_conv2(p2_out)

        return p2_out, p3_out, p4_out, p5_out

    def _upsample_add(self, x, y):
        '''Upsample and add two feature maps.
        Args:
          x: (Variable) top feature map to be upsampled.
          y: (Variable) lateral feature map.
        Returns:
          (Variable) added feature map.
        Note in PyTorch, when input size is odd, the upsampled feature map
        with `F.upsample(..., scale_factor=2, mode='nearest')`
        maybe not equal to the lateral feature map size.
        e.g.
        original input size: [N,_,15,15] ->
        conv2d feature map size: [N,_,8,8] ->
        upsampled feature map size: [N,_,16,16]
        So we choose bilinear upsample which supports arbitrary output sizes.
        '''
        _,_,H,W = y.size()
        return functional.interpolate(x, size=(H,W), mode='bilinear') + y

def normal_init(m, mean, stddev, truncated=False):
    """
    weight initalizer: truncated normal and random normal.
    """
    # x is a parameter
    if truncated:
        m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean)  # not a perfect approximation
    else:
        m.weight.data.normal_(mean, stddev)
        m.bias.data.zero_()