"""
Tests for our pytorch metrics
"""

import numpy as np
import pytest
import torch
from torch import Tensor

from mfai.pytorch.metrics import FAR, FNR, FSS, PR_AUC, CSINeighborhood


@pytest.mark.parametrize("num_neighbors,expected_value", [(0, 0.36), (1, 0.91)])
def test_csi_binary(num_neighbors: int, expected_value: float) -> None:
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

    csi = CSINeighborhood(num_neighbors=num_neighbors, task="binary")
    csi.update(preds=y_hat, targets=y_true)
    csi_score = torch.round(csi.compute(), decimals=2).float()

    assert pytest.approx(csi_score.cpu(), 0.001) == expected_value


@pytest.mark.parametrize("num_neighbors,expected_value", [(0, 0.43), (1, 0.79)])
def test_csi_multiclass(num_neighbors: int, expected_value: Tensor) -> None:
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

    csi = CSINeighborhood(num_neighbors, "multiclass", 4)
    csi.update(preds=y_hat, targets=y_true)
    csi_score = torch.round(csi.compute(), decimals=2).float()

    assert pytest.approx(csi_score.cpu(), 0.001) == expected_value


@pytest.mark.parametrize("num_neighbors,expected_value", [(0, 0.36), (1, 0.81)])
def test_csi_multilabel(num_neighbors: int, expected_value: Tensor) -> None:
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

    csi = CSINeighborhood(num_neighbors, "multilabel", 3)
    csi.update(preds=y_hat, targets=y_true)
    csi_score = torch.round(csi.compute(), decimals=2).float()

    assert pytest.approx(csi_score.cpu(), 0.001) == expected_value


def test_pr_auc() -> None:
    """
    Test of the compute of the Precision-Recall Area Under the Curve.
    """
    preds = torch.tensor([0.0, 1.0, 0.0, 1.0])
    targets = torch.tensor([0, 0, 1, 1])
    pr_auc = PR_AUC()
    pr_auc.update(preds, targets)
    pr_auc_value = pr_auc.compute()
    expected_value = 0.125
    assert pytest.approx(pr_auc_value.cpu(), 0.001) == expected_value


def test_far() -> None:
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


def test_fnr() -> None:
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


def test_fss() -> None:
    preds: Tensor = torch.tensor(
        [
            [
                [
                    [0.0, 0.0, 1.0, 1.0],
                    [0.0, 0.0, 1.0, 1.0],
                    [0.0, 0.0, 0.0, 2.0],
                    [0.0, 0.0, 0.0, 2.0],
                ]
            ]
        ]
    )
    targets: Tensor = torch.tensor(
        [
            [
                [
                    [0.0, 0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0, 0.0],
                    [1.0, 1.0, 2.0, 0.0],
                    [1.0, 1.0, 2.0, 0.0],
                ]
            ]
        ]
    )
    mask: Tensor = torch.tensor(
        [
            [0, 0, 1, 1],
            [0, 0, 1, 1],
            [0, 0, 1, 1],
            [0, 0, 1, 1],
        ]
    )
    fss_1 = FSS(neighborood=1, thresholds=0.5)
    fss_1_masked = FSS(neighborood=1, thresholds=0.5, mask=mask)
    fss_2 = FSS(neighborood=2, thresholds=0.5)
    fss_2_masked = FSS(neighborood=2, thresholds=0.5, mask=mask)
    fss_4 = FSS(neighborood=4, thresholds=0.5)

    # Check perfect prediction lead to perfect score (1)
    fss_1.update(targets, targets)
    fss_1_masked.update(targets, targets)
    fss_2.update(targets, targets)
    fss_2_masked.update(targets, targets)
    fss_4.update(targets, targets)

    torch.testing.assert_close(fss_1.compute(), torch.tensor(1.0))
    torch.testing.assert_close(fss_1_masked.compute(), torch.tensor(1.0))
    torch.testing.assert_close(fss_2.compute(), torch.tensor(1.0))
    torch.testing.assert_close(fss_2_masked.compute(), torch.tensor(1.0))
    torch.testing.assert_close(fss_4.compute(), torch.tensor(1.0))

    # Check for a first batch
    fss_1.reset()
    fss_1_masked.reset()
    fss_2.reset()
    fss_2_masked.reset()
    fss_4.reset()

    fss_1.update(preds, targets)
    fss_1_masked.update(preds, targets)
    fss_2.update(preds, targets)
    fss_2_masked.update(preds, targets)
    fss_4.update(preds, targets)

    torch.testing.assert_close(
        fss_1.compute(), torch.tensor(0.0)
    )  # predictions are disjoint from observations
    torch.testing.assert_close(fss_1.compute(), torch.tensor(0.0))
    torch.testing.assert_close(fss_2.compute(), torch.tensor(8 / 41))
    torch.testing.assert_close(fss_2_masked.compute(), torch.tensor(1 / 3))
    torch.testing.assert_close(fss_4.compute(), torch.tensor(84 / 85))

    # Check for a second batch
    fss_2.update(targets, targets)

    torch.testing.assert_close(fss_2.compute(), torch.tensor(50 / 83))

    # Check for multiple thresholds
    fss_1 = FSS(neighborood=1, thresholds=[0.5, 1.5])
    fss_1_masked = FSS(neighborood=1, thresholds=[0.5, 1.5], mask=mask)
    fss_2 = FSS(neighborood=2, thresholds=[0.5, 1.5])
    fss_2_masked = FSS(neighborood=2, thresholds=[0.5, 1.5], mask=mask)
    fss_4 = FSS(neighborood=4, thresholds=[0.5, 1.5])

    fss_1.update(targets, targets)
    fss_1_masked.update(targets, targets)
    fss_2.update(targets, targets)
    fss_2_masked.update(targets, targets)
    fss_4.update(targets, targets)

    torch.testing.assert_close(fss_1.compute(), torch.tensor([1.0, 1.0]))
    torch.testing.assert_close(fss_1_masked.compute(), torch.tensor([1.0, 1.0]))
    torch.testing.assert_close(fss_2.compute(), torch.tensor([1.0, 1.0]))
    torch.testing.assert_close(fss_2_masked.compute(), torch.tensor([1.0, 1.0]))
    torch.testing.assert_close(fss_4.compute(), torch.tensor([1.0, 1.0]))

    # Check for multiple classes
    fss_1 = FSS(neighborood=1, num_classes=1)
    fss_1_masked = FSS(neighborood=1, num_classes=1, mask=mask)
    fss_2 = FSS(neighborood=2, num_classes=1)
    fss_2_masked = FSS(neighborood=2, num_classes=1, mask=mask)
    fss_4 = FSS(neighborood=4, num_classes=1)

    fss_1.update(preds, targets)
    fss_1_masked.update(preds, targets)
    fss_2.update(preds, targets)
    fss_2_masked.update(preds, targets)
    fss_4.update(preds, targets)

    torch.testing.assert_close(
        fss_1.compute(), torch.tensor(0.0)
    )  # predictions are disjoint from observations
    torch.testing.assert_close(fss_1.compute(), torch.tensor(0.0))
    torch.testing.assert_close(fss_2.compute(), torch.tensor(8 / 41))
    torch.testing.assert_close(fss_2_masked.compute(), torch.tensor(1 / 3))
    torch.testing.assert_close(fss_4.compute(), torch.tensor(84 / 85))

    # Check raise of exceptions
    with pytest.raises(ValueError):
        FSS(neighborood=1)
        FSS(neighborood=1, thresholds=0.5, num_classes=2)
