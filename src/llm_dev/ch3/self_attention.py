import torch.nn as nn
import torch


class SelfAttention_v1(
    nn.Module
):  # nn.Module is a fundamental building block of PyTorch models.
    def __init__(self, d_in, d_out) -> None:
        super().__init__()
        # notice that we are initialising the Parameter layer
        # with random values here manually.
        self.W_query = nn.Parameter(torch.rand(d_in, d_out))
        self.W_key = nn.Parameter(torch.rand(d_in, d_out))
        self.W_value = nn.Parameter(torch.rand(d_in, d_out))

    def forward(self, x):
        keys = x @ self.W_key
        queries = x @ self.W_query
        values = x @ self.W_value

        attn_scores = queries @ keys.T
        attn_weights = torch.softmax(attn_scores / keys.shape[1] ** 0.5, dim=-1)
        context_vec = attn_weights @ values
        return context_vec


class SelfAttention_v2(
    nn.Module
):  # nn.Module is a fundamental building block of PyTorch models.
    def __init__(self, d_in, d_out, qkv_bias=False) -> None:
        super().__init__()
        # Linear layer has a significant advantage over Parameter,
        # which is a sophisticated values initialisation method.
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

    def forward(self, x):
        keys = x @ self.W_key
        queries = x @ self.W_query
        values = x @ self.W_value

        attn_scores = queries @ keys.T
        attn_weights = torch.softmax(attn_scores / keys.shape[1] ** 0.5, dim=-1)
        context_vec = attn_weights @ values
        return context_vec
