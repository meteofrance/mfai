import csv
import cartopy
import numpy as np
from typing import Any
from py4cast.datasets import get_datasets
from py4cast.plots import plot_prediction, DomainInfo
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os

OUTPUT_DIR = "../internship-cerfacs-archive/output_data/"
GRAPH_STEP = 10 #pas d'enregistrement des graphiques de comparaison.
EPOCH_COUNT = 2 #nombre d'époch
TIMESTEPS = 10 #nombre de timesteps

#loading the dataset - code from oscar

dataset_configuration: dict[str, Any] = {
    "periods": {
      "train": {
        "start": 20230101,
        "end": 20230101,
        "obs_step": 3600,
      },
      "valid": {
        "start": 20230102,
        "end": 20230102,
        "obs_step": 3600,
        "obs_step_btw_t0": 10800,
      },
      "test": {
        "start": 20230103,
        "end": 20230103,
        "obs_step": 3600,
        "obs_step_btw_t0": 10800,
      },
    },
    "grid": {
      "name": "PAAROME_1S40",
      "border_size": 0,
      "subdomain": [100, 612, 240, 880],
      "proj_name": "PlateCarree",
      "projection_kwargs": {},
    },
    "settings": {
      "standardize": True,
      "file_format": "npy",
    },
    "params": {
      "aro_t2m": {
        "levels": [2],
        "kind": "input_output",
        },
      "aro_r2": {
        "levels": [2],
        "kind": "input_output",
        },
      "aro_tp": {
        "levels": [0],
        "kind": "input_output",
        },
      "aro_u10": {
        "levels": [10],
        "kind": "input_output",
        },
      "aro_v10": {
        "levels": [10],
        "kind": "input_output",
        },
      "aro_t": {
        "levels": [250, 500, 700, 850],
        "kind": "input_output",
        },
      "aro_u": {
        "levels": [250, 500, 700, 850],
        "kind": "input_output",
        },
      "aro_v": {
        "levels": [250, 500, 700, 850],
        "kind": "input_output",
        },
      "aro_z": {
        "levels": [250, 500, 700, 850],
        "kind": "input_output",
        },
      "arp_t": {
        "levels": [250, 500, 700, 850],
        "kind": "input",
      },
      "arp_u":{
        "levels": [250, 500, 700, 850],
        "kind": "input",
      },
      "arp_v":{
        "levels": [250, 500, 700, 850],
        "kind": "input",
      },
      "arp_z":{
        "levels": [250, 500, 700, 850],
        "kind": "input",
      },
    },
}

#extracting data from the dataset

train, test, val = get_datasets(
    name="titan_aro_arp",
    num_input_steps=1,
    num_pred_steps_train=1,
    num_pred_steps_val_test=1,
    dataset_conf=dataset_configuration,
)

print(train)
print(type(train))


#loading the model

from mfai.pytorch.models.gaussian_diffusion import GaussianDiffusionSettings, GaussianDiffusion

settings = GaussianDiffusionSettings(
    timesteps = TIMESTEPS,
    sampling_timesteps = None,
    objective = "pred_v",
    beta_schedule = "sigmoid",
    schedule_fn_kwargs = {},
    ddim_sampling_eta = 0.0,
    auto_normalize = True,
    offset_noise_strength = (0.0,),
    min_snr_loss_weight = False,
    min_snr_gamma = 5,
    immiscible = False
)

model = GaussianDiffusion(
  in_channels = 21, #correspond bien au nombre de features
  out_channels = 21,
  input_shape = (train.grid.x,train.grid.y), #on part du principe que les datasets auront tous la même grille
  settings = settings
)

loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

loss_scores = []

def save_loss_scores_entry():
  global loss_scores
  loss_to_save = loss_scores.pop(0)
  with open(OUTPUT_DIR+"loss_scores.csv","a") as lossfile:
    wr = csv.writer(lossfile)
    wr.writerow(loss_to_save)

domain_info = train.domain_info
interior_mask = torch.zeros_like(train[0].forcing.tensor.permute(0,3,1,2).detach()[:,0,:,:].squeeze(0))

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR+"figures/", exist_ok=True)
os.makedirs(OUTPUT_DIR+"models/", exist_ok=True)
with open(OUTPUT_DIR+"loss_scores.csv","w") as lossfile:
  wr = csv.writer(lossfile)
  wr.writerow(["epoch","train_i","loss"])

for epoch in range(EPOCH_COUNT):
  for i,item in enumerate(train):
    input, target = item.forcing.tensor, item.outputs.tensor
    input_tensor = input.permute(0,3,1,2) #reshape to fit format
    target_tensor = target.permute(0,3,1,2)

    # Maintenant tu peux appeler ton modèle avec input pour générer et target pour avoir une loss

    # Zero your gradients for every batch!
    optimizer.zero_grad()
    output = model(input_tensor)
    #plotting output against target
    if i%GRAPH_STEP == 0:
      for j,feature_name in enumerate(item.outputs.feature_names):
        pred = output.detach()[:,j,:,:].squeeze(0)
        plot_target = target_tensor.detach()[:,j,:,:].squeeze(0)
        fig = plot_prediction(pred=pred,target=plot_target,interior_mask=interior_mask,domain_info=domain_info,title=feature_name+" "+str(i),vrange=None)
        fig.savefig(OUTPUT_DIR+f"figures/target_output_{feature_name}_{epoch}-{i}.png")
        plt.close(fig)

    # Compute the loss and its gradients
    loss = loss_fn(output, target_tensor)
    loss.backward()
    loss_scores.append([epoch,i,loss.item()])
    print(f"new loss score: {loss_scores[-1][-1]}")
    if len(loss_scores)>1 and loss_scores[-1][-1] < loss_scores[-2][-1]:
      print(f"latest loss is better ({loss_scores[-1][-1]} < {loss_scores[-2][-1]}), saving model number {epoch}-{i} as best_loss")
      torch.save({"epoch":epoch,"global_step":i,"state_dict":model.state_dict()}, OUTPUT_DIR+f"models/model_weights_best_loss.pth")
    if len(loss_scores)>2:
      save_loss_scores_entry()
    # Adjust learning weights
    optimizer.step()
    print(f"{len(train)-i} left in training, epoch {epoch}/{EPOCH_COUNT-1}")

#save remaining loss
while len(loss_scores)>0:
  save_loss_scores_entry()

torch.save({"epoch":epoch,"global_step":i,"state_dict":model.state_dict()}, OUTPUT_DIR+f"models/model_weights_final.pth")