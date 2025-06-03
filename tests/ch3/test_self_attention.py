import torch
import logging
from llm_dev.ch3.self_attention import SelfAttention_v1
import numpy as np


def test_self_attention_v1(inputs):
    torch.manual_seed(123)
    d_in = inputs.shape[1]
    d_out = 2
    sa_v1 = SelfAttention_v1(d_in=d_in, d_out=d_out)
    context_vector = sa_v1(inputs)
    assert context_vector.shape[0] == 6
    assert context_vector.shape[1] == 2
