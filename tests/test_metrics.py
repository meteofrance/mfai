"""
Tests for our pytorch metrics
"""

import numpy as np
import pytest
import torch
from mfai.torch.metrics import CSINeighborood


@pytest.mark.parametrize("num_neighbors,expected", [(0, 0.36), (1, 0.91)])
def test_csi_binary(num_neighbors: int, expected: float):
    """
    Build two 5x5 tensors and compute the CSI for binary task.
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

    assert (
        csi_score - expected < 0.01
    ), f"Failed to compute the CSI, return {csi_score} instead of {expected}."


@pytest.mark.parametrize("num_neighbors,expected", [(0, 0.43), (1, 0.79)])
def test_csi_multiclass(num_neighbors: int, expected: torch.Tensor):
    """
    Build two 5x5 tensors and compute the CSI for binary task.
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

    print(y_hat.shape)

    csi = CSINeighborood(num_neighbors, "multiclass", 4)
    csi.update(preds=y_hat, targets=y_true)
    csi_score = torch.round(csi.compute(), decimals=2).float()

    print(csi.true_positives)
    print(csi.false_positives)
    print(csi.false_negatives)
    print(csi_score)

    assert (
        torch.sum(csi_score - expected) < 0.01
    ), f"Failed to compute the CSI, return {csi_score} instead of {expected}."


@pytest.mark.parametrize("num_neighbors,expected", [(0, 0.36), (1, 0.81)])
def test_csi_multilabel(num_neighbors: int, expected: torch.Tensor):
    """
    Build two 5x5 tensors and compute the CSI for binary task.
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

    assert (
        torch.sum(csi_score - expected) < 0.001
    ), f"Failed to compute the CSI, return {csi_score} instead of {expected}."
