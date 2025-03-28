# -*- coding: utf-8 -*-
import torch
from typing import Sequence
from collections import OrderedDict
import torch
from itertools import chain
import torch.nn as nn
from torchvision import models

class PerceptualLoss(torch.nn.Module):
    def __init__(self,
                 device: str = 'cuda',
                 multi_scale: bool = False,
                 channel_iterative_mode: bool = True,
                 in_channels: int = 3,
                 pre_trained: bool = False,
                 resize_input: bool = False,
                 style_layer_ids: list = [],
                 style_block_ids: list = [],
                 feature_layer_ids: list = [4,9,16,23,30],
                 feature_block_ids: list = [0,1,2,3,4],
                 alpha_style: float = 0,
                 alpha_feature: float = 1,
        ):
        r''' Class that computes the Perceptual Loss based on selected Network.
            For more details : See Johnson et al. - Perceptual losses for real-time style transfer and super-resolution.
            (https://arxiv.org/pdf/1603.08155)

        Arguments :
                - device: (str) - Device where to store the Neural Network (default = 'cuda')
                - multi_scale: (bool) - Multi Scale mode to compute Perceptual Loss at different scales (default = False)
                - channel_iterative_mode: (bool) - To compute the Perceptual Loss over channels of the input (default = False)
                - in_channels: (int) - Number of input channels for perceptual Loss - [Used only if channel_iterative_mode=False] (default = 1)
                - pre_trained: (bool) - To use a pre-trained or a random version of the VGG16 (default = False)
                - resize_input: (bool) - To adapt input size to ImageNet Dataset size (224x224) (default = False)
                - style_layer_ids: (list) - Ids of Style Layers used for Perceptual Loss (default = [])
                - feature_layer_ids: (list) - Ids of Feature Layers used for Perceptual Loss (default = [4,9,16,23,30])
                - alpha_style: (float) - Weight of Style Loss (default = 0)
                - alpha_feature (float) - Weight of Feature Loss (default = 1)
         '''
        super(PerceptualLoss, self).__init__()

        self.device=device
        if 'cuda' in device and not torch.cuda.is_available() : 
            self.device = 'cpu'
        
        # Iteration over channels for perceptual loss
        self.channel_iterative_mode = channel_iterative_mode # whether to iterates over the channel for perceptual loss
        self.in_channels = in_channels # Number of input channel for VGG16
        self.pre_trained = pre_trained
        self.resize_input = resize_input

        # Memory for features
        self.features_memory=None
        self.styles_memory=None 

        # Style layer
        self.style_layer_ids = style_layer_ids
        self.style_block_ids=style_block_ids
        self.feature_block_ids=feature_block_ids
        self.alpha_style = alpha_style

        # Features layers
        self.feature_layer_ids = feature_layer_ids
        self.alpha_feature = alpha_feature

        # Multi-scale mode
        self.multi_scale = multi_scale
        self.scaling_factor = [0]
        if multi_scale :
            for i in range(3):
                self.scaling_factor.append(2**i)
        
        self._set_network()

    def _set_blocks(self):
        r''' Set the blocks of layers from the neural network 
        
        Return : 
            blocks: (list)
        '''
        blocks = []
        
        if not self.channel_iterative_mode and self.in_channels != 3:
            # The VGG16 is adapted so that the first layer can receive the required number of channels.
            layers = [nn.Conv2d(in_channels=self.in_channels, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)).eval()]

            for id_layer in range(1, self.feature_layer_ids[0]+1):
                layers.append(self.network.features[id_layer].eval())

            blocks.append(nn.Sequential(*layers))

            for id in range(1, len(self.feature_layer_ids)-1):
                blocks.append(self.network.features[self.feature_layer_ids[id]:self.feature_layer_ids[id+1]].eval())

        else :
            blocks.append(self.network.features[:self.feature_layer_ids[0]].eval())
            for id in range(len(self.feature_layer_ids)-1):
                blocks.append(self.network.features[self.feature_layer_ids[id]:self.feature_layer_ids[id+1]].eval())

        return blocks
    
    def _downscale(self,
                  x : torch.Tensor,
                  scale_times=1,
                  mode='bilinear'
        ):
        
        for _ in range(scale_times):
            x = torch.nn.functional.interpolate(x, scale_factor=0.5, mode=mode)

        return x
    
    def _set_network(self):
        r''' Set the VGG16 from torchvision.
        
        Trained version obtained : "https://download.pytorch.org/models/vgg16-397923af.pth"
        '''
        self.size_resize=[224,224]
        self.network = models.vgg16(pretrained=self.pre_trained).to(self.device)
        
        blocks =  self._set_blocks()

        for bl in blocks:
            for p in bl.parameters():
                p.requires_grad = False

        self.blocks = torch.nn.ModuleList(blocks)
    
    def _forward_net_single_img(self,
                               x: torch.Tensor
        ):
        r''' Forward the Network features and styles for a single image. 

        Arguments :
            x: (torch.Tensor)
            
        Return : 
            features : (list)
            styles : (list)
        '''

        # if shape is (B,H,W)
        if len(x.shape)==3 : 
            x = x.unsqueeze(1)

        if self.channel_iterative_mode:
            if x.shape[1] == 1 :
                x = x.repeat(1, 3, 1, 1)
            elif x.shape[1] != 3 :
                raise ValueError(f'Excpecting input to have 3 channels but it has {x.shape[1]}')
        else :
            if x.shape[1] != self.in_channels :
                raise ValueError(f'Excpecting input to have {self.in_channels} channels but it has {x.shape[1]}')
        
        if self.resize_input:
            x = torch.nn.functional.interpolate(x, mode='bilinear', size=self.size_resize, align_corners=False)

        features = []
        styles= []

        for i, block in enumerate(self.blocks):
            x = block(x)
            if i in self.feature_block_ids:
                features.append(x)
            if i in self.style_block_ids: 
                act_x = x.reshape(x.shape[0], x.shape[1], -1)
                gram_x = act_x @ act_x.permute(0, 2, 1)
                styles.append(gram_x)
        
        return features, styles
    
    def compute_perceptual_features(self,
                                    x: torch.Tensor,
                                    return_features_and_styles:bool = False
        ):
        r''' Compute the features of a single image.

        Arguments :
            x: (torch.Tensor)
            return_features_and_styles: (bool)

        Note : Useful in case you need to compute the perceptual loss between the same original image and other images.
        
        '''
        features = []
        styles = []
        for scaling_factor in self.scaling_factor:
            if self.multi_scale:
                x = self._downscale(x, scaling_factor)

            if self.channel_iterative_mode:
                for channel_id in range(x.shape[1]):
                    feature, style = self._forward_net_single_img(x[:, channel_id, :, :])
                    features.append(feature)
                    styles.append(style)
            else :
                feature, style = self._forward_net_single_img(x)
                features.append(feature)
                styles.append(style)

        self.features_memory=features
        self.styles_memory=styles 

        if return_features_and_styles:
            return features, styles

    
    def _perceptual_loss_given_features_and_target(self,
                                                  x: torch.Tensor,
                                                  features_y: list,
                                                  styles_y: list
        ):
        r''' Computes the Perceptual Loss given features and a target image.

        Arguments :
            x: (torch.Tensor)
            features_y : (list)
            styles_y : (list)
        
        Return :
            loss : (troch.Tensor)
        '''

        features_x, styles_x = self._forward_net_single_img(x)

        loss = 0.0
        for i, _ in enumerate(self.blocks):
            if i in self.feature_block_ids:
                x = features_x[i]
                y = features_y[i]
                loss_features = torch.nn.functional.l1_loss(x, y)
                loss += self.alpha_feature*loss_features
            if i in self.style_block_ids: 
                gram_x = styles_x[i]
                gram_y = styles_y[i]
                loss_style = torch.nn.functional.l1_loss(gram_x, gram_y)
                loss += self.alpha_style*loss_style
        return loss

    def _perceptual_loss_given_input_and_target(self,
                                               x : torch.Tensor,
                                               y : torch.Tensor
        ):
        r''' Computes the Perceptual Loss between two images 
        
        Arguments :
            x: (torch.Tensor)
            y: (torch.Tensor)

        Return :
            loss : (troch.Tensor)'''

        features_x, styles_x = self._forward_net_single_img(x)

        return self._perceptual_loss_given_features_and_target(
            x=y,
            features_y=features_x,
            styles_y=styles_x
        )

    def forward(self,
                x: torch.Tensor,
                y: torch.Tensor = None
        ):
        r''' Computes the Perceptual loss between two images 
        Arguments :
            x: (torch.Tensor)
            y: (torch.Tensor) (default=None)

        Note : 
            If y is None, the features of y needs to be computed before by calling the function : compute_perceptual_features
            
        '''

        perceptual_loss = torch.tensor(0.).to(self.device)

        if y is not None:
            for scaling_factor in self.scaling_factor:
                if self.multi_scale:
                    x = self._downscale(x, scaling_factor)
                    y = self._downscale(y, scaling_factor)
                    
                if self.channel_iterative_mode:
                    for channel_id in range(x.shape[1]):
                        perceptual_loss += self._perceptual_loss_given_input_and_target( 
                            x = x[:, channel_id, :, :],
                            y = y[:, channel_id, :, :]
                        )
                    perceptual_loss /= x.shape[1]
                else :
                    perceptual_loss += self._perceptual_loss_given_input_and_target( 
                            x = x,
                            y = y
                        )
        else :
            if self.features_memory is None and self.styles_memory is None :
                print('Warning: The features needs to be computed before. To do so call the function : compute_perceptual_features ')
                raise ValueError
            for id_scaling_factor, scaling_factor in enumerate(self.scaling_factor):
                if self.multi_scale:
                    x = self._downscale(x, scaling_factor)
                else :
                    id_scaling_factor = 0
                if self.channel_iterative_mode:
                    for channel_id in range(x.shape[1]):
                        features_y=self.features_memory[id_scaling_factor*x.shape[1]+channel_id]
                        if len(self.style_layer_ids):
                            styles_y=self.styles_memory[id_scaling_factor*x.shape[1]+channel_id]
                        else :
                            styles_y=None
                        perceptual_loss += self._perceptual_loss_given_features_and_target(
                            x=x[:, channel_id, :, :],
                            features_y=features_y, 
                            styles_y=styles_y
                        )
                    perceptual_loss /= x.shape[1]
                else :

                    features_y=self.features_memory[id_scaling_factor]
                    if len(self.style_layer_ids):
                        styles_y=self.styles_memory[id_scaling_factor]
                    else :
                        styles_y=None

                    perceptual_loss += self._perceptual_loss_given_features_and_target(
                        x=x,
                        features_y=features_y, 
                        styles_y=styles_y,
                    )

            
        return perceptual_loss



class LPIPS(nn.Module):
    r''' Creates a criterion that measures Learned Perceptual Image Patch Similarity (LPIPS).
        For more info see : Zhang et al. The Unreasonable Effectiveness of Deep Features as a Perceptual Metric 
        (https://arxiv.org/pdf/1801.03924)

        This code is inspired from : https://github.com/richzhang/PerceptualSimilarity/

        Arguments :
                device: (str) - Device where to store the Neural Network (default = 'cuda')
                multi_scale: (bool) - Multi Scale mode to compute Perceptual Loss at different scales (default = False)
                channel_iterative_mode: (bool) - To compute the Perceptual Loss over channels of the input (default = False)
                in_channels: (int) - Number of input channels for perceptual Loss - [Used only if channel_iterative_mode=False] (default = 1)
                pre_trained: (bool) - To use a pre-trained or a random version of the VGG16 (default = False)
                resize_input: (bool) - To adapt input size to ImageNet Dataset size (224x224) (default = False)
                style_layer_ids: (list) - Ids of Style Layers used for Perceptual Loss (default = [])
                feature_layer_ids: (list) - Ids of Feature Layers used for Perceptual Loss (default = [4,9,16,23,30])
                alpha_style: (float) - Weight of Style Loss (default = 0)
                alpha_feature (float) - Weight of Feature Loss (default = 1)
    '''
    def __init__(self, 
                 device: str = 'cuda',
                 multi_scale: bool = False,
                 channel_iterative_mode: bool = False,
                 in_channels: int = 1,
                 pre_trained: bool = False,
                 resize_input: bool = False,
                 style_layer_ids: list = [],
                 style_block_ids: list = [],
                 feature_layer_ids: list = [4,9,16,23,30],
                 feature_block_ids: list = [0,1,2,3,4],
                 alpha_style: float = 0,
                 alpha_feature: float = 1
        ):
        super(LPIPS, self).__init__()

        
        self.perceptual_loss = PerceptualLoss(
            device=device,
            multi_scale=multi_scale,
            channel_iterative_mode=channel_iterative_mode,
            in_channels=in_channels,
            pre_trained=pre_trained,
            resize_input=resize_input,
            style_layer_ids=style_layer_ids,
            style_block_ids=style_block_ids,
            feature_layer_ids=feature_layer_ids,
            feature_block_ids=feature_block_ids,
            alpha_style=alpha_style,
            alpha_feature=alpha_feature
        )

        n_channels_list = [64, 128, 256, 512, 512]
        
        # linear layers
        self.lin = LinLayers(n_channels_list).to("cuda")
        self.lin.load_state_dict(self._get_state_dict())

    def _get_state_dict(self, net_type: str = 'vgg16', version: str = '0.1'):
        # build url
        url = 'https://raw.githubusercontent.com/richzhang/PerceptualSimilarity/' \
            + f'master/lpips/weights/v{version}/{net_type}.pth'

        # download
        old_state_dict = torch.hub.load_state_dict_from_url(
            url, progress=True,
            map_location=None if torch.cuda.is_available() else torch.device('cpu')
        )

        # rename keys
        new_state_dict = OrderedDict()
        for key, val in old_state_dict.items():
            new_key = key
            new_key = new_key.replace('lin', '')
            new_key = new_key.replace('model.', '')
            new_state_dict[new_key] = val
        
        return new_state_dict

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        feat_x, _ = self.perceptual_loss.compute_perceptual_features(x, return_features_and_styles=True)
        feat_y, _ = self.perceptual_loss.compute_perceptual_features(y, return_features_and_styles=True)
        
        feat_x_list = []
        for xs in feat_x:
            for x in xs:
                feat_x_list.append(x)
        
        feat_y_list = []
        for xs in feat_y:
            for x in xs:
                feat_y_list.append(x)

        diff = [(fx - fy) ** 2 for fx, fy in zip(feat_x_list, feat_y_list)]
        res = [l(d).mean((2, 3), True) for d, l in zip(diff, self.lin)]

        return torch.sum(torch.cat(res, 0)) / x.shape[0]




def normalize_activation(x, eps=1e-10):
    norm_factor = torch.sqrt(torch.sum(x ** 2, dim=1, keepdim=True))
    return x / (norm_factor + eps)


def get_state_dict(dir, net_type: str = 'alex'):

    old_state_dict = torch.load(
        dir+f'lpips/{net_type}.pth',
        map_location=None if torch.cuda.is_available() else torch.device('cpu')
    )

    # rename keys
    new_state_dict = OrderedDict()
    for key, val in old_state_dict.items():
        new_key = key
        new_key = new_key.replace('lin', '')
        new_key = new_key.replace('model.', '')
        new_state_dict[new_key] = val

    return new_state_dict

def get_network(net_type: str):
    if net_type == 'vgg':
        return VGG16()
    else:
        raise NotImplementedError('choose net_type from [alex, squeeze, vgg].')


class LinLayers(nn.ModuleList):
    def __init__(self, n_channels_list: Sequence[int]):
        super(LinLayers, self).__init__([
            nn.Sequential(
                nn.Identity(),
                nn.Conv2d(nc, 1, 1, 1, 0, bias=False)
            ) for nc in n_channels_list
        ])

        for param in self.parameters():
            param.requires_grad = False


class BaseNet(nn.Module):
    def __init__(self):
        super(BaseNet, self).__init__()

        # register buffer
        self.register_buffer(
            'mean', torch.Tensor([-.030, -.088, -.188])[None, :, None, None])
        self.register_buffer(
            'std', torch.Tensor([.458, .448, .450])[None, :, None, None])

    def set_requires_grad(self, state: bool):
        for param in chain(self.parameters(), self.buffers()):
            param.requires_grad = state

    def z_score(self, x: torch.Tensor):
        return (x - self.mean) / self.std

    def forward(self, x: torch.Tensor):
        x = self.z_score(x)

        output = []
        for i, (_, layer) in enumerate(self.layers._modules.items(), 1):
            x = layer(x)
            if i in self.target_layers:
                output.append(normalize_activation(x))
            if len(output) == len(self.target_layers):
                break
        return output

class VGG16(BaseNet):
    def __init__(self):
        super(VGG16, self).__init__()

        self.layers = models.vgg16(True).features
        self.target_layers = [4, 9, 16, 23, 30]
        self.n_channels_list = [64, 128, 256, 512, 512]

        self.set_requires_grad(False)