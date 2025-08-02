from mfai.pytorch.models.gaussian_diffusion import GaussianDiffusion
from typing import Any
from py4cast.datasets import get_datasets

#loading the dataset - code from oscar

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
      "arp_u": {
        "levels": [250, 500, 700, 850],
        "kind": "input",
        },
      "arp_v": {
        "levels": [250, 500, 700, 850],
        "kind": "input"
        },
      "arp_z": {
        "levels": [250, 500, 700, 850],
        "kind": "input",
        },
    },
}

#loading the model

#TODO

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

#upscaling the data (w/ plots)

#TODO

#applying the model