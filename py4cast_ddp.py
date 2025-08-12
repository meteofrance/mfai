from typing import Any
from py4cast.datasets import get_datasets
import torch
import os

#loading the dataset - code from oscar

SAVE_EVERY_EPOCH = True

dataset_configuration: dict[str, Any] = {
    "periods": {
      "train": {
        "start": 20200101,
        "end": 20221231,
        "obs_step": 3600,
      },
      "valid": {
        "start": 20230101,
        "end": 20231231,
        "obs_step": 3600,
        "obs_step_btw_t0": 10800,
      },
      "test": {
        "start": 20240101,
        "end": 20240831,
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
    timesteps = 100,
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
  in_channels = len(train.params), #correspond bien au nombre de features
  out_channels = len(train.params),
  input_shape = (train.grid.x,train.grid.y), #on part du principe que les datasets auront tous la même grille
  settings = settings
)

loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
epoch_count = 1

print("cuda available? ",torch.cuda.is_available())

for epoch in range(epoch_count):
    for item in train:
        input_tensor = item.inputs.tensor.permute(0,3,1,2) #reshape to fit format
        target_tensor = item.outputs.tensor.permute(0,3,1,2)
        # Maintenant tu peux appeler ton modèle avec input pour générer et target pour avoir une loss
    
        # Zero your gradients for every batch!
        optimizer.zero_grad()
        output = model(input_tensor)
    
        # Compute the loss and its gradients
        loss = loss_fn(output, target_tensor)
        loss.backward()
    
        # Adjust learning weights
        optimizer.step()
    
    if SAVE_EVERY_EPOCH:
        torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss.item(),
        }, f"checkpoint_epoch_{epoch}.pth")
torch.save({
    "epoch": epoch,
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "loss": loss.item(),
}, "checkpoint_last.pth")
