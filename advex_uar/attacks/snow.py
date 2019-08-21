import math
import numbers
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn

# modification of code from pytorch forums
class GaussianSmoothing(nn.Module):
    """
    Apply gaussian smoothing on a tensor. Filtering is performed seperately for each channel
    in the input using a depthwise convolution.
    Arguments:
        channels (int, sequence): Number of channels of the input tensors. Output will
            have this number of channels as well.
        kernel_size (int, sequence): Size of the gaussian kernel.
        sigma (float, sequence): Standard deviation of the gaussian kernel.
    """
    def __init__(self, kernel_size, sigma, channels=1):
        super(GaussianSmoothing, self).__init__()
        if isinstance(kernel_size, numbers.Number):
            kernel_size = [kernel_size] * 2
        if isinstance(sigma, numbers.Number):
            sigma = [sigma] * 2

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
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1)).cuda()

        self.register_buffer('weight', kernel)
        self.groups = channels

        self.conv = F.conv2d

    def forward(self, input, padding=0):
        """
        Apply gaussian filter to input.
        Arguments:
            input (torch.Tensor): Input to apply gaussian filter on.
        Returns:
            filtered (torch.Tensor): Filtered output.
        """
        return self.conv(input, weight=self.weight, groups=self.groups, padding=padding)


# from stackoverflow
def trapez(y,y0,w):
    return np.clip(np.minimum(y+1+w/2-y0, -y+1+w/2+y0),0,1)


def weighted_line(r0, c0, r1, c1, w, rmin=0, rmax=np.inf):
    # The algorithm below works fine if c1 >= c0 and c1-c0 >= abs(r1-r0).
    # If either of these cases are violated, do some switches.
    if abs(c1-c0) < abs(r1-r0):
        # Switch x and y, and switch again when returning.
        xx, yy, val = weighted_line(c0, r0, c1, r1, w, rmin=rmin, rmax=rmax)
        return (yy, xx, val)

    # At this point we know that the distance in columns (x) is greater
    # than that in rows (y). Possibly one more switch if c0 > c1.
    if c0 > c1:
        return weighted_line(r1, c1, r0, c0, w, rmin=rmin, rmax=rmax)

    # The following is now always < 1 in abs
    slope = (r1-r0) / (c1-c0)

    # Adjust weight by the slope
    w *= np.sqrt(1+np.abs(slope)) / 2

    # We write y as a function of x, because the slope is always <= 1
    # (in absolute value)
    x = np.arange(c0, c1+1, dtype=float)
    y = x * slope + (c1*r0-c0*r1) / (c1-c0)

    # Now instead of 2 values for y, we have 2*np.ceil(w/2).
    # All values are 1 except the upmost and bottommost.
    thickness = np.ceil(w/2)
    yy = (np.floor(y).reshape(-1,1) + np.arange(-thickness-1,thickness+2).reshape(1,-1))
    xx = np.repeat(x, yy.shape[1])
    vals = trapez(yy, y.reshape(-1,1), w).flatten()

    yy = yy.flatten()

    # Exclude useless parts and those outside of the interval
    # to avoid parts outside of the picture
    mask = np.logical_and.reduce((yy >= rmin, yy < rmax, vals > 0))

    return (yy[mask].astype(int), xx[mask].astype(int), vals[mask])


def make_kernels(snow_length_bound=13, blur=True):
    kernels = []

    flip = np.random.uniform() < 0.5

    for i in range(7):
        k_size = snow_length_bound
        mid = k_size//2
        k_npy = np.zeros((k_size, k_size))
        rr, cc, val = weighted_line(
            mid, mid, np.random.randint(mid+2,k_size), np.random.randint(mid+2,k_size),
            np.random.choice([1,3,5], p=[0.6, 0.3, 0.1]), mid, k_size)

        k_npy[rr, cc] = val
        k_npy[:mid+1, :mid+1] = k_npy[::-1,::-1][:mid+1,:mid+1]

        if flip:
            k_npy = k_npy[:, ::-1]

        kernel = torch.FloatTensor(k_npy.copy()).view(1,1,k_size,k_size).cuda()

        if blur:
            blurriness = np.random.uniform(0.41, 0.6)
            gaussian_blur = GaussianSmoothing(int(np.ceil(5 * blurriness)), blurriness)
            kernel = gaussian_blur(kernel, padding=1)
        kernels.append(kernel)

    return kernels


def snow_creator(intensities, k, resol):
    flake_grids = []
    k = torch.cat(k, 1)

    intensities_pow = torch.pow(intensities, 4)
    flake_grids = torch.zeros((intensities.size(0), k.size(1), resol, resol)).cuda()

    for i in range(4):
        flake_grids[:, i, ::4,i::4] = intensities_pow[:,i]
    for i in range(3):
        flake_grids[:, i+4, i+1::4,::4] = intensities_pow[:,4+i]

    snow = F.conv2d(flake_grids, k, padding=k.size(-1)//2)

    return snow
