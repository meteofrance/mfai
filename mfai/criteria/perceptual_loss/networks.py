# -*- coding: utf-8 -*-
import torchvision
import torch
import torch.nn as nn 
import os

def set_blocks_from_direct_features(config, network, feature_layers):
    blocks = []
    if config.features_after_relu:
        feature_layers = [i+1 for i in feature_layers]

    if config.channel_computation!='sol5' :
        # All solutions except 5
        blocks.append(network.features[:feature_layers[0]].eval())
    else :        
        # Solution 5
        if 'vgg' in config.network_type :
            layers = [nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)).eval()]
        elif config.network_type == 'squeezenet1_1':
            layers = [nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(3, 3), stride=(2, 2)).eval()]
        elif config.network_type == 'alexnet':
            layers = [nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(11, 11), stride=(4,4), padding=(2,2)).eval()]
        elif 'resnet' in config.network_type :
            layers = [nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(7, 7), stride=(2,2), padding=(3,3)).eval()]
        elif config.network_type == 'set_vit_b_16':
            raise NotImplementedError
        
        for id_layer in range(1, feature_layers[0]+1):
            layers.append(network.features[id_layer].eval())
        blocks.append(nn.Sequential(*layers))

    for id in range(len(feature_layers)-1):
        blocks.append(network.features[feature_layers[id]:feature_layers[id+1]].eval())

    return blocks

def set_features_from_squeezenet1_1(config, network, feature_layers):
    blocks = []

    feature_extractor = nn.Sequential(*list(network.children())[:-1])
    features = nn.Sequential(*list(feature_extractor[0].children())[:-1])

    if config.channel_computation!='sol5' :
        # All solutions except 5
        blocks.append(features[:feature_layers[0]])
    else :        
        # Solution 5
        blocks.append(nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(3, 3), stride=(2, 2)).eval())
        for id_layer in range(1, feature_layers[0]+1):
            blocks.append(features[id_layer].eval())
        
    for id in range(len(feature_layers)-1):
        blocks.append(features[feature_layers[id]:feature_layers[id+1]])
    
    return nn.Sequential(*blocks)

def set_features_from_resnet(config, network, feature_layers):
    blocks = []
    features = nn.Sequential(*list(network.children())[:-1])

    if config.channel_computation!='sol5' :
        # All solutions except 5
        blocks.append(features[:feature_layers[0]])
    else :        
        # Solution 5
        blocks.append(nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(7, 7), stride=(2, 2), padding=(3,3)).eval())
        for id_layer in range(1, feature_layers[0]+1):
            blocks.append(features[id_layer].eval())
    
    for id in range(len(feature_layers)-1):
        blocks.append(features[feature_layers[id]:feature_layers[id+1]])
    
    return nn.Sequential(*blocks)



def set_features_from_vit(config, network):
    blocks = []
    feature_extractor = nn.Sequential(*list(network.children())[:-1])
    layers = nn.Sequential(*list(feature_extractor[1].children())[:-1])
    encoders = nn.Sequential(*list(layers[1].children())[:-1])

    if config.channel_computation!='sol5' :
        # All solutions except 5
        blocks.append(feature_extractor[0].eval())
    else :   
        # Solution 5
        blocks.append(nn.Conv2d(in_channels=1, out_channels=768, kernel_size=(16, 16), stride=(16, 16)).eval())
      
    blocks.append(nn.Sequential(*[layers[0], encoders[0]]).eval())

    for id in range(1, len(encoders)):
        blocks.append(encoders[id].eval())

    return blocks

def load_or_save_weight(network, dir, type, pre_trained=False):
    r''' Load network weights if they exists, otherwise creates them '''
    if pre_trained :
        network_path = dir + f"{type}_trained.pth"
    else :
        network_path = dir + f"{type}_random.pth"
    if os.path.isfile(network_path):
        network.load_state_dict(torch.load(network_path))
    else :
        os.makedirs(dir, exist_ok=True)
        torch.save(network.state_dict(), network_path)

def set_alexnet(self):
    # Trained version obtained : https://download.pytorch.org/models/alexnet-owt-7be5be79.pth
    self.size_resize=[224,224]
    network = torchvision.models.alexnet(weights=None).to(self.device)
    load_or_save_weight(network, dir=self.config.network_dir, type='alexnet', pre_trained=self.config.pre_trained)
    feature_layers = [1, 4, 7, 9, 11]
    return set_blocks_from_direct_features(config=self.config, network=network, feature_layers=feature_layers)


def set_vgg16(self):
    # Trained version obtained : "https://download.pytorch.org/models/vgg16-397923af.pth"
    self.size_resize=[224,224]
    network = torchvision.models.vgg16(weights=None).to(self.device)
    load_or_save_weight(network, dir=self.config.network_dir, type='vgg16', pre_trained=self.config.pre_trained)
    feature_layers = [3,8,15,22,29] 
    return set_blocks_from_direct_features(config=self.config, network=network, feature_layers=feature_layers)

def set_vgg11(self):
    # Trained version obtained : https://download.pytorch.org/models/vgg11-8a719046.pth
    self.size_resize=[224,224]
    network = torchvision.models.vgg11(weights=None).to(self.device)
    load_or_save_weight(network, dir=self.config.network_dir, type='vgg11', pre_trained=self.config.pre_trained)
    feature_layers = [1,4,9,14,19]
    return set_blocks_from_direct_features(config=self.config, network=network, feature_layers=feature_layers)

def set_vgg19(self):
    # Trained version obtained : https://download.pytorch.org/models/vgg19-dcbb9e9d.pth
    self.size_resize=[224,224]
    network = torchvision.models.vgg19(weights=None).to(self.device)
    load_or_save_weight(network, dir=self.config.network_dir, type='vgg19', pre_trained=self.config.pre_trained)
    feature_layers = [3,8,17,26,35]
    return set_blocks_from_direct_features(config=self.config, network=network, feature_layers=feature_layers)

def set_vgg13(self):
    # Trained version obtained : https://download.pytorch.org/models/vgg13-19584684.pth
    self.size_resize=[224,224]
    network = torchvision.models.vgg13(weights=None).to(self.device)
    load_or_save_weight(network, dir=self.config.network_dir, type='vgg13', pre_trained=self.config.pre_trained)
    feature_layers = [3,8,13,18,23]
    return set_blocks_from_direct_features(config=self.config, network=network, feature_layers=feature_layers)

def set_squeezenet1_1(self):
    # Trained version obtained : https://download.pytorch.org/models/squeezenet1_1-b8a52dc0.pth
    self.size_resize=[224,224]
    network = torchvision.models.squeezenet1_1(weights=None).to(self.device)
    load_or_save_weight(network, dir=self.config.network_dir, type='squeezenet1_1', pre_trained=self.config.pre_trained)
    feature_layers = [2,5,8,10,11,12,13]
    return set_features_from_squeezenet1_1(config=self.config, network=network, feature_layers=feature_layers)

def set_resnet18(self):
    # Trained version obtained : https://download.pytorch.org/models/resnet18-f37072fd.pth
    self.size_resize=[224,224]
    network = torchvision.models.resnet18(weights=None).to(self.device)
    load_or_save_weight(network, dir=self.config.network_dir, type='resnet18', pre_trained=self.config.pre_trained)
    feature_layers = [2,5,6,7,8]
    return set_features_from_resnet(config=self.config, network=network, feature_layers=feature_layers)

def set_resnet34(self):
    # Trained version obtained : https://download.pytorch.org/models/resnet34-b627a593.pth
    self.size_resize=[224,224]
    network = torchvision.models.resnet34(weights=None).to(self.device)
    load_or_save_weight(network, dir=self.config.network_dir, type='resnet34', pre_trained=self.config.pre_trained)
    feature_layers = [2,5,6,7,8]
    return set_features_from_resnet(config=self.config, network=network, feature_layers=feature_layers)

def set_resnet50(self):
    # Trained version obtained : https://download.pytorch.org/models/resnet50-0676ba61.pth 
    self.size_resize=[224,224]
    network = torchvision.models.resnet50(weights=None).to(self.device)
    load_or_save_weight(network, dir=self.config.network_dir, type='resnet50', pre_trained=self.config.pre_trained)
    feature_layers = [2,5,6,7,8]
    return set_features_from_resnet(config=self.config, network=network, feature_layers=feature_layers)

def set_resnet101(self):
    # Trained version obtained : https://download.pytorch.org/models/resnet101-63fe2227.pth
    self.size_resize=[224,224]
    network = torchvision.models.resnet101(weights=None).to(self.device)
    load_or_save_weight(network, dir=self.config.network_dir, type='resnet101', pre_trained=self.config.pre_trained)
    feature_layers = [2,5,6,7,8]
    return set_features_from_resnet(config=self.config, network=network, feature_layers=feature_layers)

def set_resnet152(self):
    # Trained version obtained : https://download.pytorch.org/models/resnet152-394f9c45.pth
    self.size_resize=[224,224]
    network = torchvision.models.resnet152(weights=None).to(self.device)
    load_or_save_weight(network, dir=self.config.network_dir, type='resnet152', pre_trained=self.config.pre_trained)
    feature_layers = [2,5,6,7,8]
    return set_features_from_resnet(config=self.config, network=network, feature_layers=feature_layers)

def set_vit_b_16(self):
    # Trained version obtained : https://download.pytorch.org/models/vit_b_16-c867db91.pth
    self.size_resize=None
    network = torchvision.models.vit_b_16(weights=None).to(self.device)
    load_or_save_weight(network, dir=self.config.network_dir, type='vit_b_16', pre_trained=self.config.pre_trained)
    return set_features_from_vit(config=self.config, network=network)