import torch
import torch.nn as nn
import torchvision

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

def get_eps_params(base_eps, resol):
    eps_list = []
    max_list = []
    min_list = []
    for i in range(3):
        eps_list.append(torch.full((resol, resol), base_eps, device='cuda'))
        min_list.append(torch.full((resol, resol), 0., device='cuda'))
        max_list.append(torch.full((resol, resol), 255., device='cuda'))

    eps_t = torch.unsqueeze(torch.stack(eps_list), 0)
    max_t = torch.unsqueeze(torch.stack(max_list), 0)
    min_t = torch.unsqueeze(torch.stack(min_list), 0)
    return eps_t, max_t, min_t

def get_imagenet_params(resol):
    mean_list = []
    std_list = []
    for i in range(3):
        mean_list.append(torch.full((resol, resol), IMAGENET_MEAN[i], device='cuda'))
        std_list.append(torch.full((resol, resol), IMAGENET_STD[i], device='cuda'))
    return torch.unsqueeze(torch.stack(mean_list), 0), torch.unsqueeze(torch.stack(std_list), 0)

class ImagenetTransform(nn.Module):
    def __init__(self, resol):
        super().__init__()
        self.mean, self.std = get_imagenet_params(resol)
        
    def forward(self, x):
        '''
        Parameters:
            x: input image with pixels normalized to [0, 255]
        '''
        x = x / 255.
        x = x.sub(self.mean)
        x = x.div(self.std)
        return x

class InverseImagenetTransform(nn.Module):
    def __init__(self, resol):
        super().__init__()
        self.mean, self.std = get_imagenet_params(resol)

    def forward(self, x):
        '''
        Parameters:
            x: input image with pixels normalized to ([0, 1] - IMAGENET_MEAN) / IMAGENET_STD
        '''
        x = x.mul(self.std)
        x = x.add(self.mean)
        x = x * 255.
        return x

class PixelModel(nn.Module):
    def __init__(self, model, resol):
        super().__init__()
        self.model = model
        self.transform = ImagenetTransform(resol)

    def forward(self, x):
        '''
        Parameters:
            x: input image with pixels normalized to [0, 255]
        '''
        x = self.transform(x)
        # x is now normalized as the model expects
        x = self.model(x)
        return x

class AttackWrapper(nn.Module):
    def __init__(self, resol):
        super().__init__()
        self.resol = resol
        self.transform = ImagenetTransform(resol)
        self.inverse_transform = InverseImagenetTransform(resol)
        self.epoch = 0
        
    def forward(self, model, img, *args, **kwargs):
        was_training = model.training
        pixel_model = PixelModel(model, self.resol)
        pixel_model.eval()
        pixel_img = self.inverse_transform(img.detach())
        pixel_ret = self._forward(pixel_model, pixel_img, *args, **kwargs)
        if was_training:
            pixel_model.train()
        ret = self.transform(pixel_ret)
        return ret

    def set_epoch(self, epoch):
        self.epoch = epoch
        self._update_params(epoch)

    def _update_params(self, epoch):
        pass
