"""The cube-finding CNN from lesson 05, in its own file so lesson 06 can reuse it."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class CubeFinder(nn.Module):
    """Fully convolutional net -> one heatmap -> soft-argmax -> (u, v) in pixels.

    The heatmap says "how likely is the cube centre here" for every cell; softmax turns it into
    probabilities; the predicted position is the probability-weighted average of cell centres.
    """

    def __init__(self, stride=4):
        super().__init__()
        self.stride = stride

        def block(cin, cout, s):
            return nn.Sequential(nn.Conv2d(cin, cout, 3, s, 1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))

        self.features = nn.Sequential(
            block(3, 32, 1), block(32, 32, 2),  # H/2
            block(32, 64, 1), block(64, 64, 2),  # H/4
            block(64, 64, 1), block(64, 64, 1),
        )
        self.heatmap = nn.Conv2d(64, 1, 1)

    def forward(self, x):
        logits = self.heatmap(self.features(x))  # (B, 1, h, w)
        b, _, h, w = logits.shape
        prob = F.softmax(logits.view(b, -1), dim=1).view(b, h, w)
        # pixel coordinates of each heatmap cell centre
        xs = (torch.arange(w, device=x.device, dtype=x.dtype) + 0.5) * self.stride - 0.5
        ys = (torch.arange(h, device=x.device, dtype=x.dtype) + 0.5) * self.stride - 0.5
        u = (prob.sum(dim=1) * xs).sum(dim=1)
        v = (prob.sum(dim=2) * ys).sum(dim=1)
        return torch.stack([u, v], dim=1), prob
