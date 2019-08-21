import math
import numbers

import torch
import numpy as np
from torch import nn
from torch.nn import functional as F

# Taken from: https://discuss.pytorch.org/t/is-there-anyway-to-do-gaussian-filtering-for-an-image-2d-3d-in-pytorch/12351/10
class GaussianSmoothing(nn.Module):
    """
    Apply gaussian smoothing on a
    1d, 2d or 3d tensor. Filtering is performed seperately for each channel
    in the input using a depthwise convolution.
    Arguments:
        channels (int, sequence): Number of channels of the input tensors. Output will
            have this number of channels as well.
        kernel_size (int, sequence): Size of the gaussian kernel.
        sigma (float, sequence): Standard deviation of the gaussian kernel.
        dim (int, optional): The number of dimensions of the data.
            Default value is 2 (spatial).
    """
    def __init__(self, channels, kernel_size, sigma, dim=2):
        super(GaussianSmoothing, self).__init__()
        if isinstance(kernel_size, numbers.Number):
            kernel_size = [kernel_size] * dim
        if isinstance(sigma, numbers.Number):
            sigma = [sigma] * dim

        # The gaussian kernel is the product of the
        # gaussian function of each dimension.
        kernel = 1
        meshgrids = torch.meshgrid(
            [
                torch.arange(size, dtype=torch.float32)
                for size in kernel_size
            ]
        )
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= 1 / (std * math.sqrt(2 * math.pi)) * \
                      torch.exp(-((mgrid - mean) / std) ** 2 / 2)

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(kernel)

        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

        self.register_buffer('weight', kernel)
        self.groups = channels

        if dim == 1:
            self.conv = F.conv1d
        elif dim == 2:
            self.conv = F.conv2d
        elif dim == 3:
            self.conv = F.conv3d
        else:
            raise RuntimeError(
                'Only 1, 2 and 3 dimensions are supported. Received {}.'.format(dim)
            )

    def forward(self, inp):
        """
        Apply gaussian filter to input.
        Arguments:
            input (torch.Tensor): Input to apply gaussian filter on.
        Returns:
            filtered (torch.Tensor): Filtered output.
        """
        return self.conv(inp, weight=self.weight, groups=self.groups)

class ElasticDeformation(nn.Module):
    def __init__(self, im_size, filter_size, std):
        super().__init__()
        self.im_size = im_size
        self.filter_size = filter_size
        self.std = std
        self.kernel = GaussianSmoothing(2, self.filter_size, self.std).cuda()

        self._get_base_flow()

    def _get_base_flow(self):
        xflow, yflow = np.meshgrid(
                np.linspace(-1, 1, self.im_size, dtype='float32'),
                np.linspace(-1, 1, self.im_size, dtype='float32'))
        flow = np.stack((xflow, yflow), axis=-1)
        flow = np.expand_dims(flow, axis=0)
        self.base_flow = nn.Parameter(torch.from_numpy(flow)).cuda().detach()

    def warp(self, im, flow):
        return F.grid_sample(im, flow, mode='bilinear')

    def forward(self, im, params):
        flow = F.pad(params, ((self.filter_size - 1) // 2, ) * 4 , mode='reflect')
        local_flow = self.kernel(flow).transpose(1, 3).transpose(1, 2)
        return self.warp(im, local_flow + self.base_flow)
