# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import numpy as np
from copy import deepcopy
from criteria.perceptual_loss.networks import set_vgg16, set_vgg11, set_vgg13, set_vgg19, set_squeezenet1_1, set_vit_b_16
from criteria.perceptual_loss.networks import set_alexnet, set_resnet101, set_resnet152, set_resnet18, set_resnet34, set_resnet50
import argparse

class MultiPerceptualLoss(torch.nn.Module):
    def __init__(self,
                 config: argparse.Namespace = None,
                 device: str = 'cuda'
        ):
        super(MultiPerceptualLoss, self).__init__()
        assert isinstance(config.network_type, list)

        self.perceptual_losses = []
        network_type_list = deepcopy(config.network_type)
        for net_type in network_type_list :
            config.network_type = net_type
            self.perceptual_losses.append(deepcopy(PerceptualLoss(config=config, device=device)))

    def compute_perceptual_features(self,
                                    img: torch.Tensor,
                                    normalize: bool =True
                                    ):
        for id, perceptual_loss in enumerate(self.perceptual_losses) :
                perceptual_loss.compute_perceptual_features(img, normalize=normalize)

    def forward(self, img_gen, input_img=None, normalize=True):
        perceptual_loss_total = 0
        for id, perceptual_loss in enumerate(self.perceptual_losses) :
            loss =  perceptual_loss(img_gen, input_img, normalize)
            perceptual_loss_total += loss

        return perceptual_loss_total


class PerceptualLoss(torch.nn.Module):
    def __init__(self,
                 config: argparse.Namespace = None,
                 device: str = 'cuda',
                 multi_scale: bool = False
        ):
        r''' Class that computes the Perceptual Loss based on selected Network.
        Arguments:
            config (argparse.Namespace): 
            device (str): 
            multi_scale (bool): 
         '''
        super(PerceptualLoss, self).__init__()

        self.config=config
        self.device=device
        
        self.set_network()

        self.transform = torch.nn.functional.interpolate
        self.resize = self.config.resize_input

        # Features memory in case they need to be memorized 
        # ex : For optimization procedure we have to compare the same input
        # features all along the optimization process. So there is no need
        # to compute the input features at each steps of optimization
        
        self.features_input_img=None
        self.styles_input_img=None 

        self.multi_scale = multi_scale
        self.scaling_factor = [0]
        if multi_scale :
            for i in range(3):
                self.scaling_factor.append(2**i)

        
    
    def downscale(self,
                  x : torch.Tensor,
                  scale_times=1,
                  mode='bilinear'
        ):
        
        for _ in range(scale_times):
            x = torch.nn.functional.interpolate(x, scale_factor=0.5, mode=mode)

        return x

    def set_network(self):
        try :
            print(f'Init Network {self.config.network_type}')
            blocks = eval('set_'+self.config.network_type)
        except :
            print(f'NotImplementedError: Network type : {self.config.network_type} is not implemented')
            raise NotImplementedError

        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(self.device))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(self.device))
        
        if self.config.channel_computation=='sol4':
            # Solution 4
            self.grayscale_to_rgb = nn.Sequential(
                                    nn.Conv2d(in_channels=1, out_channels=3, kernel_size=1),
                                    nn.ReLU()
            )

        for bl in blocks:
            for p in bl.parameters():
                p.requires_grad = False

        self.blocks = torch.nn.ModuleList(blocks)
    
    def forward_net_single_img(self,
                               input_img: torch.Tensor,
                               feature_layers: list = [0,1,2,3,4],
                               style_layers: list = [],
                               compute_all_features: bool = False,
                               normalize: bool = True
        ):
        r''' Forward the Network features and styles for a single image '''

        if normalize:
            # Normalize input from [-1,1] range to [0,1] range
            input_img = (input_img+1)/2

        if len(input_img.shape)==3 :
            input_img = input_img.unsqueeze(1).repeat(1, 3, 1, 1)
        
        if self.resize:
            input_img = self.transform(input_img, mode='bilinear', size=self.size_resize, align_corners=False)
            
        features = []
        styles= []

        # the samples have to be in range [0, 1] and normalized using
        # mean = [0.485, 0.456, 0.406] and std = [0.229, 0.224, 0.225]
        if self.config.channel_computation != 'sol5' :
            x = (input_img-self.mean) / self.std
        else :
            # grayscale imagenet's train dataset mean and standard deviation 
            
            grayscale_mean = 0.44531356896770125
            grayscale_std = 0.2692461874154524
            x = (input_img-grayscale_mean) / grayscale_std

        for i, block in enumerate(self.blocks):
            x = block(x)
            if i in feature_layers or compute_all_features:
                if self.config.network_type == 'set_vit_b_16' and i == 1:
                    b, c, h, w = x.shape
                    x = x.reshape(b,h*w,c)
                features.append(x)
            if i in style_layers: 
                act_x = x.reshape(x.shape[0], x.shape[1], -1)
                gram_x = act_x @ act_x.permute(0, 2, 1)
                styles.append(gram_x)
        
        return features, styles
    
    def compute_perceptual_features(self,
                                    img: torch.Tensor,
                                    compute_all_features: bool = True,
                                    normalize: bool = True,
                                    return_features:bool = False):
        r''' Compute the features of a single image with respect to the chosen solution and save them in the memory '''
        features = []
        styles = []
        for scaling_factor in self.scaling_factor:
            if self.multi_scale:
                x = self.downscale(img, scaling_factor)
            else :
                x = img

            for channel_id in range(x.shape[1]):
                feature, style = self.forward_net_single_img(
                                                    input_img = x[:, channel_id, :, :],
                                                    feature_layers = self.config.feature_layers,
                                                    style_layers = self.config.style_layers,
                                                    compute_all_features=compute_all_features,
                                                    normalize=normalize
                        )
                features.append(feature)
                styles.append(style)


        if not return_features:
            self.features_input_img=features
            # print('self.features_input_img length :', len(self.features_input_img))
            self.styles_input_img=styles 
        else :
            return features, styles

    
    def perceptual_loss_given_features_and_target(self,
                                                  target_img: torch.Tensor,
                                                  feature_layers: list = [0,1,2,3,4],
                                                  features_input_img: list = None,
                                                  style_layers:list = [],
                                                  styles_input_img:list = None,
                                                  alpha_feature:float = 1.0,
                                                  alpha_style:float = 0.01,
                                                  normalize: bool = True
        ):
        r''' Computes the Perceptual Loss given features of an image and a target image '''

        features_target_img, styles_target_img = self.forward_net_single_img(target_img, normalize=normalize)

        loss = 0.0
        for i, _ in enumerate(self.blocks):
            if i in feature_layers:
                x = features_input_img[i]
                y = features_target_img[i]
                loss_features = torch.nn.functional.l1_loss(x, y)
                loss += alpha_feature*loss_features
            if i in style_layers: 
                gram_x = styles_input_img[i]
                gram_y = styles_target_img[i]
                loss_style = torch.nn.functional.l1_loss(gram_x, gram_y)
                loss += alpha_style*loss_style
        return loss

    def perceptual_loss_given_input_and_target(self,
                                               input_img : torch.Tensor,
                                               target_img : torch.Tensor,
                                               feature_layers: list = [0,1,2,3],
                                               style_layers: list = [],
                                               alpha_feature: float = 1.0,
                                               alpha_style: float = 0.01,
                                               normalize: bool = True
        ):
        r''' Computes the VGG Loss given an input image and a target image '''

        features_input_img, styles_input_img = self.forward_net_single_img(input_img, normalize=normalize)

        return self.perceptual_loss_given_features_and_target(
            target_img=target_img,
            feature_layers=feature_layers,
            features_input_img=features_input_img,
            style_layers=style_layers,
            styles_input_img=styles_input_img,
            alpha_feature=alpha_feature,
            alpha_style=alpha_style,
            normalize=normalize
        )

    def forward(self,
                img_gen: torch.Tensor,
                input_img : torch.Tensor = None,
                normalize : bool = True
        ):
        r''' Computes the VGG loss between two images with respect to the chosen solution 
        '''

        perceptual_loss = torch.tensor(0.).to(self.device)

        if input_img is not None:
            for scaling_factor in self.scaling_factor:
                if self.multi_scale:
                    x = self.downscale(img_gen, scaling_factor)
                    y = self.downscale(input_img, scaling_factor)
                else :
                    x = img_gen
                    y = input_img

                for channel_id in range(x.shape[1]):
                    perceptual_loss += self.perceptual_loss_given_input_and_target( 
                        input_img = x[:, channel_id, :, :],
                        target_img = y[:, channel_id, :, :],
                        feature_layers = self.config.feature_layers,
                        style_layers = self.config.style_layers,
                        alpha_feature = self.config.alpha_feature,
                        alpha_style = self.config.alpha_style,
                        normalize = normalize
                    )
                perceptual_loss /= x.shape[1]

        else :
            if self.features_input_img is None and self.styles_input_img is None :
                print('Warning: The features needs to be computed beforehand')
                raise ValueError
            for id_scaling_factor, scaling_factor in enumerate(self.scaling_factor):
                if self.multi_scale:
                    x = self.downscale(img_gen, scaling_factor)
                else :
                    x = img_gen
                    id_scaling_factor = 0

                for channel_id in range(x.shape[1]):
                    features_input_img=self.features_input_img[id_scaling_factor*x.shape[1]+channel_id]
                    if self.config.style_layers:
                        styles_input_img=self.styles_input_img[id_scaling_factor*x.shape[1]+channel_id]
                    else :
                        styles_input_img=None
                    perceptual_loss += self.perceptual_loss_given_features_and_target(
                        target_img=x[:, channel_id, :, :],
                        features_input_img=features_input_img, 
                        styles_input_img=styles_input_img,
                        feature_layers = self.config.feature_layers,
                        style_layers = self.config.style_layers,
                        alpha_feature = self.config.alpha_feature,
                        alpha_style = self.config.alpha_style,
                        normalize = normalize
                    )
                perceptual_loss /= x.shape[1]
            
        return perceptual_loss