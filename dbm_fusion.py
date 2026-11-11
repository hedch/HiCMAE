"""Two-hidden-layer Gaussian-visible DBM and gradient reversal.

The undirected energy defines tied bottom-up/top-down mean-field updates.
Parameters are trained discriminatively through unrolled inference; this module
has no generative likelihood/contrastive-divergence training phase.
"""
import torch
from torch import nn
import torch.nn.functional as F


class GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, features, coefficient):
        ctx.coefficient = float(coefficient)
        return features.view_as(features)

    @staticmethod
    def backward(ctx, gradient):
        return -ctx.coefficient * gradient, None


def reverse_gradient(features, coefficient):
    return GradientReversal.apply(features, coefficient)


class MultimodalDBM(nn.Module):
    """Two continuous visible groups coupled to h1, with undirected h1--h2.

    E(v,a,h1,h2) = .5||v||² + .5||a||² - v Wv h1 - a Wa h1
                   - h1 W12 h2 - b1 h1 - b2 h2.
    Visible groups have fixed unit variance and are clamped to encoder embeddings.
    Hidden units are Bernoulli; inference uses their mean probabilities.
    """
    def __init__(self, visible_dim=128, hidden_dims=(256, 128), steps=5):
        super().__init__()
        if visible_dim < 1 or len(hidden_dims) != 2 or min(hidden_dims) < 1 or steps < 1:
            raise ValueError('DBM requires positive visible dimension, two positive hidden sizes and steps >= 1')
        self.steps = int(steps)
        h1, h2 = hidden_dims
        self.w_video = nn.Parameter(torch.empty(visible_dim, h1))
        self.w_audio = nn.Parameter(torch.empty(visible_dim, h1))
        self.w_hidden = nn.Parameter(torch.empty(h1, h2))
        self.b1 = nn.Parameter(torch.zeros(h1))
        self.b2 = nn.Parameter(torch.zeros(h2))
        for weight in (self.w_video, self.w_audio, self.w_hidden):
            nn.init.normal_(weight, std=.02)

    def energy(self, video, audio, h1, h2):
        return (.5 * video.square().sum(-1) + .5 * audio.square().sum(-1)
                - ((video @ self.w_video + audio @ self.w_audio) * h1).sum(-1)
                - (h1 @ self.w_hidden * h2).sum(-1)
                - (h1 * self.b1).sum(-1) - (h2 * self.b2).sum(-1))

    def forward(self, video, audio, return_states=False):
        visible_drive = video @ self.w_video + audio @ self.w_audio + self.b1
        h1 = torch.sigmoid(visible_drive)
        h2 = torch.sigmoid(h1 @ self.w_hidden + self.b2)
        for _ in range(self.steps):
            # The SAME hidden weight participates in both directions.
            h1 = torch.sigmoid(visible_drive + F.linear(h2, self.w_hidden))
            h2 = torch.sigmoid(h1 @ self.w_hidden + self.b2)
        return (h1, h2) if return_states else h2
