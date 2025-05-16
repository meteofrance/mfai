"""
Tests for our pytorch metrics
"""

import numpy as np
import pytest
import torch
from torch import Tensor

from mfai.torch.metrics import FAR, FNR, PR_AUC, CSINeighborood


@pytest.mark.parametrize("num_neighbors,expected_value", [(0, 0.36), (1, 0.91)])
def test_csi_binary(num_neighbors: int, expected_value: float):
    """
    Build tensors of size (2, 1, 5, 5), compute the CSI for binary task and check if the output is
    the result expected.
    """
    y_true = torch.tensor(
        np.array(
            [
                [
                    [
                        [0, 0, 0, 1, 0],
                        [0, 0, 1, 0, 0],
                        [0, 0, 1, 0, 0],
                        [0, 1, 0, 0, 0],
                        [0, 1, 0, 0, 0],
                    ]
                ],
                [
                    [
                        [0, 0, 0, 1, 0],
                        [0, 0, 1, 0, 0],
                        [0, 0, 1, 0, 0],
                        [0, 1, 0, 0, 0],
                        [0, 1, 0, 0, 0],
                    ]
                ],
            ]
        )
    )
    y_hat = torch.tensor(
        np.array(
            [
                [
                    [
                        [0, 0, 0, 1, 0],
                        [0, 0, 1, 1, 0],
                        [0, 1, 0, 1, 0],
                        [0, 1, 1, 0, 0],
                        [1, 1, 1, 0, 0],
                    ]
                ],
                [
                    [
                        [0, 0, 0, 1, 0],
                        [0, 0, 1, 1, 0],
                        [0, 1, 0, 1, 0],
                        [0, 1, 1, 0, 0],
                        [1, 1, 1, 0, 0],
                    ]
                ],
            ]
        )
    )

    csi = CSINeighborood(num_neighbors=num_neighbors, task="binary")
    csi.update(preds=y_hat, targets=y_true)
    csi_score = torch.round(csi.compute(), decimals=2).float()

    assert pytest.approx(csi_score.cpu(), 0.001) == expected_value


@pytest.mark.parametrize("num_neighbors,expected_value", [(0, 0.43), (1, 0.79)])
def test_csi_multiclass(num_neighbors: int, expected_value: Tensor):
    """
    Build tensors of size (1, 1, 5, 5), compute the CSI for multiclass taskand check if the output is
    the result expected.
    """
    y_true = torch.tensor(
        np.array(
            [
                [
                    [
                        [0, 0, 0, 1, 0],
                        [0, 0, 1, 2, 0],
                        [0, 0, 1, 0, 2],
                        [0, 1, 0, 0, 0],
                        [0, 3, 0, 0, 0],
                    ]
                ]
            ]
        )
    )
    y_hat = torch.tensor(
        np.array(
            [
                [
                    [
                        [0, 0, 0, 1, 0],
                        [0, 0, 1, 2, 0],
                        [0, 1, 0, 2, 0],
                        [0, 3, 1, 0, 0],
                        [1, 3, 1, 0, 0],
                    ]
                ]
            ]
        )
    )

    csi = CSINeighborood(num_neighbors, "multiclass", 4)
    csi.update(preds=y_hat, targets=y_true)
    csi_score = torch.round(csi.compute(), decimals=2).float()

    assert pytest.approx(csi_score.cpu(), 0.001) == expected_value


@pytest.mark.parametrize("num_neighbors,expected_value", [(0, 0.36), (1, 0.81)])
def test_csi_multilabel(num_neighbors: int, expected_value: Tensor):
    """
    Build tensors of size (1, 3, 5, 5), compute the CSI for multilabel task and check if the output is
    the result expected.
    """
    y_true = torch.tensor(
        np.array(
            [
                [
                    [
                        [0, 0, 0, 1, 0],
                        [0, 0, 1, 0, 0],
                        [0, 0, 1, 0, 0],
                        [0, 1, 0, 0, 0],
                        [0, 0, 0, 0, 0],
                    ],
                    [
                        [0, 0, 0, 0, 0],
                        [0, 0, 0, 1, 0],
                        [0, 0, 0, 0, 1],
                        [0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0],
                    ],
                    [
                        [0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0],
                        [0, 1, 0, 0, 0],
                    ],
                ]
            ]
        )
    )
    y_hat = torch.tensor(
        np.array(
            [
                [
                    [
                        [0, 0, 0, 1, 0],
                        [0, 0, 1, 0, 0],
                        [0, 1, 0, 0, 0],
                        [0, 0, 1, 0, 0],
                        [1, 0, 1, 0, 0],
                    ],
                    [
                        [0, 0, 0, 0, 0],
                        [0, 0, 0, 1, 0],
                        [0, 0, 0, 1, 0],
                        [0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0],
                    ],
                    [
                        [0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0],
                        [0, 1, 0, 0, 0],
                        [0, 1, 0, 0, 0],
                    ],
                ]
            ]
        )
    )

    csi = CSINeighborood(num_neighbors, "multilabel", 3)
    csi.update(preds=y_hat, targets=y_true)
    csi_score = torch.round(csi.compute(), decimals=2).float()

    assert pytest.approx(csi_score.cpu(), 0.001) == expected_value


def test_pr_auc():
    """
    Test of the compute of the Precision-Recall Area Under the Curve.
    """
    preds = torch.tensor([0.0, 1.0, 0.0, 1.0])
    targets = torch.tensor([0, 0, 1, 1])
    far = PR_AUC()
    far.update(preds, targets)
    auc_value = far.compute()
    expected_value = 0.125
    assert pytest.approx(auc_value.cpu(), 0.001) == expected_value


def test_far():
    """
    Test of the compute of the False Alarm Rate.
    """
    preds = torch.tensor([0.0, 1.0, 0.0, 1.0])
    targets = torch.tensor([0, 0, 1, 1])
    far = FAR("binary")
    far.update(preds, targets)
    auc_value = far.compute()
    expected_value = 0.5
    assert pytest.approx(auc_value.cpu(), 0.001) == expected_value


def test_fnr():
    """
    Test of the compute of the False Alarm Rate.
    """
    preds = torch.tensor([0.0, 1.0, 0.0, 1.0])
    targets = torch.tensor([0, 0, 1, 1])
    fnr = FNR()
    fnr.update(preds, targets)
    auc_value = fnr.compute()
    expected_value = 0.5
    assert pytest.approx(auc_value.cpu(), 0.001) == expected_value
