# -*- coding: utf-8 -*-
import torch
import torchvision
import torch
import torch.nn as nn 

class PerceptualLoss(torch.nn.Module):
    def __init__(self,
                 device: str = 'cuda',
                 multi_scale: bool = False,
                 channel_iterative_mode: bool = False,
                 in_channels: int = 1,
                 pre_trained: bool = True,
                 resize_input: bool = False,
                 style_layer_ids: list = [],
                 feature_layer_ids: list = [4,9,16,23,30],
                 alpha_style: float = 0,
                 alpha_feature: float = 1,
        ):
        r''' Class that computes the Perceptual Loss based on selected Network.
                device: (str) - (default = 'cuda')
                multi_scale: (bool) - (default = False)
                channel_iterative_mode: (bool) - (default = False)
                in_channels: (int) - (default = 1)
                pre_trained: (bool) - (default = True)
                resize_input: (bool) - (default = False)
                style_layer_ids: (list) - (default = [])
                feature_layer_ids: (list) - (default = [4,9,16,23,30])
                alpha_style: (float) - (default = 0)
                alpha_feature (float) - (default = 1)
         '''
        super(PerceptualLoss, self).__init__()

        self.set_network()

        self.device=device
        
        # Iteration over channels for perceptual loss
        self.channel_iterative_mode = channel_iterative_mode # whether to iterates over the channel for perceptual loss
        self.in_channels = in_channels # Number of input channel for VGG16
        self.pre_trained = pre_trained
        self.resize_input = resize_input

        # Memory for features
        self.features_input_img=None
        self.styles_input_img=None 

        # Style layer
        self.style_layer_ids = style_layer_ids
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

        
    def set_blocks(self):
        r''' '''
        blocks = []
        
        if self.channel_iterative_mode :
            layers = [nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)).eval()]
        else :
            layers = [nn.Conv2d(in_channels=self.in_channels, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)).eval()]

        for id_layer in range(1, self.feature_layer_ids[0]+1):
            layers.append(self.network.features[id_layer].eval())
        blocks.append(nn.Sequential(*layers))

        for id in range(len(self.feature_layer_ids)-1):
            blocks.append(self.network.features[self.feature_layer_ids[id]:self.feature_layer_ids[id+1]].eval())

        return blocks
    
    def downscale(self,
                  x : torch.Tensor,
                  scale_times=1,
                  mode='bilinear'
        ):
        
        for _ in range(scale_times):
            x = torch.nn.functional.interpolate(x, scale_factor=0.5, mode=mode)

        return x

    def set_network(self):
        # Trained version obtained : "https://download.pytorch.org/models/vgg16-397923af.pth"
        self.size_resize=[224,224]
        self.network = torchvision.models.vgg16(pretrained=self.pre_trained).to(self.device)
        
        blocks =  self.set_blocks()

        for bl in blocks:
            for p in bl.parameters():
                p.requires_grad = False

        self.blocks = torch.nn.ModuleList(blocks)
    
    def forward_net_single_img(self,
                               input_img: torch.Tensor,
                               feature_layers: list = [0,1,2,3,4],
                               style_layers: list = [],
                               compute_all_features: bool = False
        ):
        r''' Forward the Network features and styles for a single image '''

        if len(input_img.shape)==3 :
            input_img = input_img.unsqueeze(1).repeat(1, 3, 1, 1)
        
        if self.resize_input:
            input_img = torch.nn.functional.interpolate(input_img, mode='bilinear', size=self.size_resize, align_corners=False)
            
        features = []
        styles= []

        for i, block in enumerate(self.blocks):
            x = block(x)
            if i in feature_layers or compute_all_features:
                features.append(x)
            if i in style_layers: 
                act_x = x.reshape(x.shape[0], x.shape[1], -1)
                gram_x = act_x @ act_x.permute(0, 2, 1)
                styles.append(gram_x)
        
        return features, styles
    
    def compute_perceptual_features(self,
                                    img: torch.Tensor,
                                    compute_all_features: bool = True,
                                    return_features:bool = False):
        r''' Compute the features of a single image with respect to the chosen solution and save them in the memory '''
        features = []
        styles = []
        for scaling_factor in self.scaling_factor:
            if self.multi_scale:
                x = self.downscale(img, scaling_factor)
            else :
                x = img

            if self.channel_iterative_mode:
                for channel_id in range(x.shape[1]):
                    feature, style = self.forward_net_single_img(
                                                        input_img = x[:, channel_id, :, :],
                                                        compute_all_features=compute_all_features
                            )
                    features.append(feature)
                    styles.append(style)
            else :
                feature, style = self.forward_net_single_img(
                                                        input_img = x[:, channel_id, :, :],
                                                        compute_all_features=compute_all_features
                )
                features.append(feature)
                styles.append(style)

        if not return_features:
            self.features_input_img=features
            self.styles_input_img=styles 
        else :
            return features, styles

    
    def perceptual_loss_given_features_and_target(self,
                                                  target_img: torch.Tensor,
                                                  features_input_img: list = None,
                                                  styles_input_img: list = None
        ):
        r''' Computes the Perceptual Loss given features of an image and a target image '''

        features_target_img, styles_target_img = self.forward_net_single_img(target_img)

        loss = 0.0
        for i, _ in enumerate(self.blocks):
            if i in self.feature_layer_ids:
                x = features_input_img[i]
                y = features_target_img[i]
                loss_features = torch.nn.functional.l1_loss(x, y)
                loss += self.alpha_feature*loss_features
            if i in self.style_layer_ids: 
                gram_x = styles_input_img[i]
                gram_y = styles_target_img[i]
                loss_style = torch.nn.functional.l1_loss(gram_x, gram_y)
                loss += self.alpha_style*loss_style
        return loss

    def perceptual_loss_given_input_and_target(self,
                                               input_img : torch.Tensor,
                                               target_img : torch.Tensor
        ):
        r''' Computes the VGG Loss given an input image and a target image '''

        features_input_img, styles_input_img = self.forward_net_single_img(input_img)

        return self.perceptual_loss_given_features_and_target(
            target_img=target_img,
            features_input_img=features_input_img,
            styles_input_img=styles_input_img
        )

    def forward(self,
                img_gen: torch.Tensor,
                input_img : torch.Tensor = None
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
                if self.channel_iterative_mode:
                    for channel_id in range(x.shape[1]):
                        perceptual_loss += self.perceptual_loss_given_input_and_target( 
                            input_img = x[:, channel_id, :, :],
                            target_img = y[:, channel_id, :, :]
                        )
                    perceptual_loss /= x.shape[1]
                else :
                    perceptual_loss += self.perceptual_loss_given_input_and_target( 
                            input_img = x,
                            target_img = y
                        )
        else :
            if self.features_input_img is None and self.styles_input_img is None :
                print('Warning: The features needs to be computed before. To do so call the function : compute_perceptual_features ')
                raise ValueError
            for id_scaling_factor, scaling_factor in enumerate(self.scaling_factor):
                if self.multi_scale:
                    x = self.downscale(img_gen, scaling_factor)
                else :
                    x = img_gen
                    id_scaling_factor = 0
                if self.channel_iterative_mode:
                    for channel_id in range(x.shape[1]):
                        features_input_img=self.features_input_img[id_scaling_factor*x.shape[1]+channel_id]
                        if len(self.style_layer_ids):
                            styles_input_img=self.styles_input_img[id_scaling_factor*x.shape[1]+channel_id]
                        else :
                            styles_input_img=None
                        perceptual_loss += self.perceptual_loss_given_features_and_target(
                            target_img=x[:, channel_id, :, :],
                            features_input_img=features_input_img, 
                            styles_input_img=styles_input_img
                        )
                    perceptual_loss /= x.shape[1]
                else :

                    features_input_img=self.features_input_img[id_scaling_factor]
                    if len(self.style_layer_ids):
                        styles_input_img=self.styles_input_img[id_scaling_factor]
                    else :
                        styles_input_img=None

                    perceptual_loss += self.perceptual_loss_given_features_and_target(
                        target_img=x,
                        features_input_img=features_input_img, 
                        styles_input_img=styles_input_img,
                    )

            
        return perceptual_loss