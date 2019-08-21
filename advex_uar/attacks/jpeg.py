import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import itertools

# From https://gist.github.com/deanmark/9aec75b7dc9fa71c93c4bc85c5438777
def tensordot_pytorch(a, b, dims=2):
    axes=dims
    # code adapted from numpy
    try:
        iter(axes)
    except Exception:
        axes_a = list(range(-axes, 0))
        axes_b = list(range(0, axes))
    else:
        axes_a, axes_b = axes
    try:
        na = len(axes_a)
        axes_a = list(axes_a)
    except TypeError:
        axes_a = [axes_a]
        na = 1
    try:
        nb = len(axes_b)
        axes_b = list(axes_b)
    except TypeError:
        axes_b = [axes_b]
        nb = 1
    
    # uncomment in pytorch >= 0.5
    # a, b = torch.as_tensor(a), torch.as_tensor(b)
    as_ = a.shape
    nda = a.dim()
    bs = b.shape
    ndb = b.dim()
    equal = True
    if na != nb:
        equal = False
    else:
        for k in range(na):
            if as_[axes_a[k]] != bs[axes_b[k]]:
                equal = False
                break
            if axes_a[k] < 0:
                axes_a[k] += nda
            if axes_b[k] < 0:
                axes_b[k] += ndb
    if not equal:
        raise ValueError("shape-mismatch for sum")

    # Move the axes to sum over to the end of "a"
    # and to the front of "b"
    notin = [k for k in range(nda) if k not in axes_a]
    newaxes_a = notin + axes_a
    N2 = 1
    for axis in axes_a:
        N2 *= as_[axis]
    newshape_a = (int(np.multiply.reduce([as_[ax] for ax in notin])), N2)
    olda = [as_[axis] for axis in notin]

    notin = [k for k in range(ndb) if k not in axes_b]
    newaxes_b = axes_b + notin
    N2 = 1
    for axis in axes_b:
        N2 *= bs[axis]
    newshape_b = (N2, int(np.multiply.reduce([bs[ax] for ax in notin])))
    oldb = [bs[axis] for axis in notin]

    at = a.permute(newaxes_a).reshape(newshape_a)
    bt = b.permute(newaxes_b).reshape(newshape_b)

    res = at.matmul(bt)
    return res.reshape(olda + oldb)

# based on https://github.com/rshin/differentiable-jpeg/blob/master/jpeg.py
# Thanks to Dan Hendrycks for adapting the code to pytorch

def rgb_to_ycbcr(image):
  matrix = np.array(
      [[65.481, 128.553, 24.966],
       [-37.797, -74.203, 112.],
       [112., -93.786, -18.214]],
      dtype=np.float32).T / 255
  shift = torch.as_tensor([16., 128., 128.], device="cuda")

  # result = torch.tensordot(image, torch.as_tensor(matrix, device="cuda"), dims=1) + shift
  result = tensordot_pytorch(image, matrix, dims=1) + shift
  result.view(image.size())
  return result


def rgb_to_ycbcr_jpeg(image):
  matrix = np.array(
      [[0.299, 0.587, 0.114],
       [-0.168736, -0.331264, 0.5],
       [0.5, -0.418688, -0.081312]],
      dtype=np.float32).T
  shift = torch.as_tensor([0., 128., 128.], device="cuda")

  # result = torch.tensordot(image, torch.as_tensor(matrix, device="cuda"), dims=1) + shift
  result = tensordot_pytorch(image, torch.as_tensor(matrix, device='cuda'), dims=1) + shift
  result.view(image.size())
  return result


# 2. Chroma subsampling
def downsampling_420(image):
  # input: batch x height x width x 3
  # output: tuple of length 3
  #   y:  batch x height x width
  #   cb: batch x height/2 x width/2
  #   cr: batch x height/2 x width/2
  y, cb, cr = image[...,0], image[...,1], image[...,2]
  # requires that height and width are divisible by 2 to avoid
  # padding issues
  cb = F.avg_pool2d(cb, kernel_size=2)
  cr = F.avg_pool2d(cr, kernel_size=2)
  return y, cb, cr
  # return (y.squeeze(), cb.squeeze(), cr.squeeze())


# 3. Block splitting
# From https://stackoverflow.com/questions/41564321/split-image-tensor-into-small-patches
def image_to_patches(image):
  # input: batch x h x w
  # output: batch x h*w/64 x h x w
  k = 8
  batch_size, height, width = image.size()
  image_reshaped = image.view(batch_size, height // k, k, -1, k)
  image_transposed = torch.transpose(image_reshaped, 2, 3)
  return image_transposed.contiguous().view(batch_size, -1, k, k)


# 4. DCT
def dct_8x8_ref(image):
  image = image - 128
  result = np.zeros((8, 8), dtype=np.float32)
  for u, v in itertools.product(range(8), range(8)):
    value = 0
    for x, y in itertools.product(range(8), range(8)):
      value += image[x, y] * np.cos((2 * x + 1) * u * np.pi / 16) * np.cos(
          (2 * y + 1) * v * np.pi / 16)
    result[u, v] = value
  alpha = np.array([1. / np.sqrt(2)] + [1] * 7)
  scale = np.outer(alpha, alpha) * 0.25
  return result * scale


def dct_8x8(image):
  image = image - 128
  tensor = np.zeros((8, 8, 8, 8), dtype=np.float32)
  for x, y, u, v in itertools.product(range(8), repeat=4):
    tensor[x, y, u, v] = np.cos((2 * x + 1) * u * np.pi / 16) * np.cos(
        (2 * y + 1) * v * np.pi / 16)
  alpha = np.array([1. / np.sqrt(2)] + [1] * 7)
  scale = torch.FloatTensor(np.outer(alpha, alpha) * 0.25).cuda()
  #result = scale * torch.tensordot(image, torch.as_tensor(tensor, device="cuda"), dims=2)
  result = scale * tensordot_pytorch(image, torch.as_tensor(tensor, device="cuda"), dims=2)
  result.view(image.size())
  return result


def make_quantization_tables(self):
    # 5. Quantizaztion
    self.y_table = torch.as_tensor(np.array(
        [[16, 11, 10, 16, 24, 40, 51, 61],
         [12, 12, 14, 19, 26, 58, 60, 55],
         [14, 13, 16, 24, 40, 57, 69, 56],
         [14, 17, 22, 29, 51, 87, 80, 62],
         [18, 22, 37, 56, 68, 109, 103, 77],
         [24, 35, 55, 64, 81, 104, 113, 92],
         [49, 64, 78, 87, 103, 121, 120, 101],
         [72, 92, 95, 98, 112, 100, 103, 99]],
        dtype=np.float32).T, device="cuda")
    c_table = np.empty((8, 8), dtype=np.float32)
    c_table.fill(99)
    c_table[:4, :4] = np.array([[17, 18, 24, 47], [18, 21, 26, 66],
                                [24, 26, 56, 99], [47, 66, 99, 99]]).T
    self.c_table = torch.as_tensor(c_table, device="cuda")


def y_quantize(self, image, rounding, rounding_var, factor=1):
  image = image / (self.y_table * factor)
  image = rounding(image, rounding_var)
  return image


def c_quantize(self, image, rounding, rounding_var, factor=1):
  image = image / (self.c_table * factor)
  image = rounding(image, rounding_var)
  return image


# -5. Dequantization
def y_dequantize(self, image, factor=1):
  return image * (self.y_table * factor)


def c_dequantize(self, image, factor=1):
  return image * (self.c_table * factor)


# -4. Inverse DCT
def idct_8x8_ref(image):
  alpha = np.array([1. / np.sqrt(2)] + [1] * 7)
  alpha = np.outer(alpha, alpha)
  image = image * alpha

  result = np.zeros((8, 8), dtype=np.float32)
  for u, v in itertools.product(range(8), range(8)):
    value = 0
    for x, y in itertools.product(range(8), range(8)):
      value += image[x, y] * np.cos((2 * u + 1) * x * np.pi / 16) * np.cos(
          (2 * v + 1) * y * np.pi / 16)
    result[u, v] = value
  return result * 0.25 + 128


def idct_8x8(image):
  alpha = np.array([1. / np.sqrt(2)] + [1] * 7)
  alpha = torch.FloatTensor(np.outer(alpha, alpha)).cuda()
  image = image * alpha

  tensor = np.zeros((8, 8, 8, 8), dtype=np.float32)
  for x, y, u, v in itertools.product(range(8), repeat=4):
    tensor[x, y, u, v] = np.cos((2 * u + 1) * x * np.pi / 16) * np.cos(
        (2 * v + 1) * y * np.pi / 16)
  # result = 0.25 * torch.tensordot(image, torch.as_tensor(tensor, device="cuda"), dims=2) + 128
  result = 0.25 * tensordot_pytorch(image, torch.as_tensor(tensor, device="cuda"), dims=2) + 128
  result.view(image.size())
  return result


# -3. Block joining
def patches_to_image(patches, height, width):
  # input: batch x h*w/64 x h x w
  # output: batch x h x w
  height = int(height)
  width = int(width)
  k = 8
  batch_size = patches.size(0)
  image_reshaped = patches.view(batch_size, height // k, width // k, k, k)
  image_transposed = torch.transpose(image_reshaped, 2, 3)
  return image_transposed.contiguous().view(batch_size, height, width)

# -2. Chroma upsampling
def upsampling_420(y, cb, cr):
  # input:
  #   y:  batch x height x width
  #   cb: batch x height/2 x width/2
  #   cr: batch x height/2 x width/2
  # output:
  #   image: batch x height x width x 3
  def repeat(x, k=2):
    height, width = x.size()[1:3]
    x = x.unsqueeze(-1)
    x = x.repeat((1, 1, k, k))
    x = x.view(-1, height * k, width * k)
    return x

  cb = repeat(cb)
  cr = repeat(cr)
  return torch.stack((y, cb, cr), dim=-1)


# -1. YCbCr -> RGB
def ycbcr_to_rgb(image):
  matrix = np.array(
      [[298.082, 0, 408.583],
       [298.082, -100.291, -208.120],
       [298.082, 516.412, 0]],
      dtype=np.float32).T / 256
  shift = torch.as_tensor([-222.921, 135.576, -276.836], device="cuda")

  # result = torch.tensordot(image, torch.tensor(matrix, device="cuda"), dims=1) + shift
  result = tensordot_pytorch(image, torch.tensor(matrix, device="cuda"), dims=1) + shift
  result.view(image.size())
  return result


def ycbcr_to_rgb_jpeg(image):
  matrix = np.array(
      [[1., 0., 1.402],
       [1, -0.344136, -0.714136],
       [1, 1.772, 0]],
      dtype=np.float32).T
  shift = torch.FloatTensor([0, -128, -128]).cuda()

  # result = torch.tensordot(image + shift, torch.tensor(matrix, device="cuda"), dims=1)
  result = tensordot_pytorch(image + shift, torch.tensor(matrix, device="cuda"), dims=1)
  result.view(image.size())
  return result


def jpeg_compress_decode(self, image_channels_first, rounding_vars, lambder, downsample_c=True,
                         factor=1):
  def noisy_round(x, noise):
    return x + lambder[:, None, None, None] * (noise - 0.5)
  
  #image = torch.as_tensor(image)
  image = torch.transpose(image_channels_first, 1, 3)
  height, width = image.size()[1:3]

  orig_height, orig_width = height, width
  if height % 16 != 0 or width % 16 != 0:
    # Round up to next multiple of 16
    height = ((height - 1) // 16 + 1) * 16
    width = ((width - 1) // 16 + 1) * 16

    vpad = height - orig_height
    wpad = width - orig_width
    top = vpad // 2
    bottom = vpad - top
    left = wpad // 2
    right = wpad - left

    # image = tf.pad(image, [[0, 0], [top, bottom], [left, right], [0, 0]], 'SYMMETRIC')
    # image = tf.pad(image, [[0, 0], [0, vpad], [0, wpad], [0, 0]], 'SYMMETRIC')
    # image = F.pad(image, (vpad, wpad), 'replicate')
    image = F.pad(image, (left, right, top, bottom), 'replicate')

  # "Compression"
  image = rgb_to_ycbcr_jpeg(image)
  if downsample_c:
    y, cb, cr = downsampling_420(image)
  else:
    y, cb, cr = torch.split(image, 3, dim=3)
  components = {'y': y, 'cb': cb, 'cr': cr}
  for k in components.keys():
    comp = components[k]
    comp = image_to_patches(comp)
    comp = dct_8x8(comp)
    if k == 'y':
        comp = y_quantize(self, comp, noisy_round, 0.5 + 0.5 * rounding_vars[0], factor)
    elif k  == 'cb':
        comp = c_quantize(self, comp, noisy_round, 0.5 + 0.5 * rounding_vars[1], factor)
    else:
        comp = c_quantize(self, comp, noisy_round, 0.5 + 0.5 * rounding_vars[2], factor)
    components[k] = comp

  # Decoding
  for k in components.keys():
    comp = components[k]
    comp = c_dequantize(self, comp, factor) if k in ('cb', 'cr') else y_dequantize(
        self, comp, factor)
    comp = idct_8x8(comp)
    if k in ('cb', 'cr'):
      if downsample_c:
        comp = patches_to_image(comp, height / 2, width / 2)
      else:
        comp = patches_to_image(comp, height, width)
    else:
      comp = patches_to_image(comp, height, width)
    components[k] = comp

  y, cb, cr = components['y'], components['cb'], components['cr']
  if downsample_c:
    image = upsampling_420(y, cb, cr)
  else:
    image = torch.stack((y, cb, cr), dim=-1)
  image = ycbcr_to_rgb_jpeg(image)

  # Crop to original size
  if orig_height != height or orig_width != width:
    #image = image[:, top:-bottom, left:-right]
    image = image[:, :-vpad, :-wpad]

  # Hack: RGB -> YUV -> RGB sometimes results in incorrect values
  #    min_value = tf.minimum(tf.reduce_min(image), 0.)
  #    max_value = tf.maximum(tf.reduce_max(image), 255.)
  #    value_range = max_value - min_value
  #    image = 255 * (image - min_value) / value_range
  image = torch.clamp(image, 0, 255)

  return torch.transpose(image, 1, 3)

def quality_to_factor(quality):
    if quality < 50:
        return 50./quality
    else:
        return (200. - quality * 2)/100.


class JPEG(nn.Module):
    def __init__(self):
        super(JPEG, self).__init__()
        make_quantization_tables(self)

    def forward(self, pixel_inp, rounding_vars, epsilon):
        return jpeg_compress_decode(self, pixel_inp, rounding_vars, epsilon)
        

