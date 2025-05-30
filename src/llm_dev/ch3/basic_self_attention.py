import torch
from torch import Tensor


def _compute_attention_score(input_token_embeds: Tensor):
    return input_token_embeds @ input_token_embeds.T


def _compute_attention_weight(input_scores: Tensor):
    return torch.softmax(input_scores, dim=-1)


def compute_context_vector_v1(inputs_token_embeds: Tensor):
    scores = _compute_attention_score(input_token_embeds=inputs_token_embeds)
    weights = _compute_attention_weight(input_scores=scores)
    return weights @ inputs_token_embeds
