import torch
import torch.nn as nn

""" Adapted from https://github.com/eladrich/pixel2style2pixel/tree/master/criteria/lpips"""

from criteria.perceptual_loss.lpips.networks import LinLayers
from criteria.perceptual_loss.lpips.utils import get_state_dict
from criteria.perceptual_loss.perceptual_loss import PerceptualLoss

class LPIPS(nn.Module):
    r"""Creates a criterion that measures
    Learned Perceptual Image Patch Similarity (LPIPS).
    Arguments:
        config (dict)

        device (str)

    """
    def __init__(self, config, device, multi_scale):
        super(LPIPS, self).__init__()

        self.config = config
        self.perceptual_loss = PerceptualLoss(config, device, multi_scale)
        if self.config.network_type == 'alexnet':
            net_type='alex'
            n_channels_list = [64, 192, 384, 256, 256]
        elif self.config.network_type == 'squeezenet1_1':
            net_type = 'squeeze'
            n_channels_list = [64, 128, 256, 384, 384, 512, 512]
        elif self.config.network_type == 'vgg16':
            net_type = 'vgg'
            n_channels_list = [64, 128, 256, 512, 512]
        else:
            raise NotImplementedError('choose net_type from [alex, squeeze, vgg].')
        
        # linear layers
        self.lin = LinLayers(n_channels_list).to("cuda")
        self.lin.load_state_dict(get_state_dict(dir=self.config.network_dir,net_type=net_type))

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        feat_x = self.perceptual_loss.compute_perceptual_features(x, return_features=True)
        feat_y = self.perceptual_loss.compute_perceptual_features(y, return_features=True)
        
        feat_x_list = []
        for xs in feat_x[0]:
            for x in xs:
                feat_x_list.append(x)
        
        feat_y_list = []
        for xs in feat_y[0]:
            for x in xs:
                feat_y_list.append(x)

        diff = [(fx - fy) ** 2 for fx, fy in zip(feat_x_list, feat_y_list)]
        res = [l(d).mean((2, 3), True) for d, l in zip(diff, self.lin)]

        return torch.sum(torch.cat(res, 0)) / x.shape[0]