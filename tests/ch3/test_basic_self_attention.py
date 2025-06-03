from llm_dev.ch3.basic_self_attention import (
    _compute_attention_score,
    _compute_attention_weight,
    compute_context_vector_v1,
)
import torch
import numpy as np


def test__compute_attention_score():
    inputs = torch.tensor(
        [
            [0.43, 0.15, 0.89],
            [0.55, 0.87, 0.66],
            [0.57, 0.85, 0.64],
            [0.22, 0.58, 0.33],
            [0.77, 0.25, 0.10],
            [0.05, 0.80, 0.55],
        ]
    )
    attention_scores = _compute_attention_score(inputs)
    assert attention_scores.shape[0] == 6
    assert attention_scores.shape[1] == 6
    assert np.allclose(
        attention_scores[1], [0.9544, 1.4950, 1.4754, 0.8434, 0.7070, 1.0865]
    )


def test__compute_attention_weight(inputs):

    scores = _compute_attention_score(inputs)
    weights = _compute_attention_weight(scores)
    assert weights.shape[0] == 6
    assert weights.shape[1] == 6
    assert np.allclose(
        weights[1], [0.1385, 0.2379, 0.2333, 0.1240, 0.1082, 0.1581], rtol=1e-3
    )
