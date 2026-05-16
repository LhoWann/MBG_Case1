import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLossWithSmoothing(nn.Module):
    """
    Focal Loss + Label Smoothing combined.

    Why both:
    - Focal loss handles class imbalance (down-weights easy majority-class examples).
    - Label smoothing helps ambiguous classes (Lainnya, Tata Kelola) by preventing
      overconfident predictions on noisy labels.
    """

    def __init__(
        self,
        alpha: torch.Tensor,
        gamma: float = 2.0,
        smoothing: float = 0.1,
        n_classes: int = 8,
    ):
        super().__init__()
        self.alpha     = alpha
        self.gamma     = gamma
        self.smoothing = smoothing
        self.n_classes = n_classes

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            smooth = torch.full_like(logits, self.smoothing / (self.n_classes - 1))
            smooth.scatter_(1, targets.view(-1, 1), 1.0 - self.smoothing)

        log_p = F.log_softmax(logits, dim=-1)
        ce    = -(smooth * log_p).sum(dim=-1)

        pt      = log_p.exp().gather(1, targets.view(-1, 1)).squeeze(1)
        focal_w = (1.0 - pt) ** self.gamma
        alpha_t = self.alpha[targets]

        return (alpha_t * focal_w * ce).mean()


def rdrop_kl_loss(logits1: torch.Tensor, logits2: torch.Tensor) -> torch.Tensor:
    """Symmetric KL divergence between two stochastic forward passes (R-Drop)."""
    p1  = F.softmax(logits1, dim=-1)
    p2  = F.softmax(logits2, dim=-1)
    kl1 = F.kl_div(F.log_softmax(logits1, dim=-1), p2, reduction="batchmean")
    kl2 = F.kl_div(F.log_softmax(logits2, dim=-1), p1, reduction="batchmean")
    return 0.5 * (kl1 + kl2)
