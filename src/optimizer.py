import torch
from torch.optim import AdamW


def get_llrd_optimizer(
    model,
    lr: float,
    llrd_factor: float,
    weight_decay: float,
) -> AdamW:
    """
    Layer-wise LR decay optimizer.

    Embeddings get the lowest LR (lr * factor^(n_layers+1)).
    Encoder layers get decaying LR moving bottom-up.
    Classifier head gets the full LR.

    This prevents catastrophic forgetting of pre-trained representations
    in deeper (earlier) layers while allowing the head to adapt quickly.
    """
    no_decay = {"bias", "LayerNorm.weight", "LayerNorm.bias"}
    params   = []

    for n, p in model.named_parameters():
        if "classifier" in n or "pooler" in n:
            wd = 0.0 if any(nd in n for nd in no_decay) else weight_decay
            params.append({"params": [p], "lr": lr, "weight_decay": wd})

    try:
        n_layers = model.encoder.config.num_hidden_layers
    except AttributeError:
        n_layers = 12

    for layer_idx in range(n_layers - 1, -1, -1):
        layer_lr = lr * (llrd_factor ** (n_layers - layer_idx))
        for n, p in model.named_parameters():
            if f".{layer_idx}." in n and "classifier" not in n:
                wd = 0.0 if any(nd in n for nd in no_decay) else weight_decay
                params.append({"params": [p], "lr": layer_lr, "weight_decay": wd})

    emb_lr = lr * (llrd_factor ** (n_layers + 1))
    for n, p in model.named_parameters():
        if "embeddings" in n:
            wd = 0.0 if any(nd in n for nd in no_decay) else weight_decay
            params.append({"params": [p], "lr": emb_lr, "weight_decay": wd})

    seen = set()
    unique = []
    for g in params:
        filtered = [p for p in g["params"] if id(p) not in seen]
        seen.update(id(p) for p in filtered)
        if filtered:
            unique.append({**g, "params": filtered})

    return AdamW(unique)


class AWP:
    """
    Adversarial Weight Perturbation.
    Perturbs embedding weights → forward again → restore.
    Adds robustness at the cost of ~30% extra training time.
    Enable via cfg.use_awp=True, starts at cfg.awp_start_epoch.
    """

    def __init__(self, model, optimizer, lr: float = 1e-4, eps: float = 1e-2):
        self.model     = model
        self.optimizer = optimizer
        self.lr        = lr
        self.eps       = eps
        self._backup   = {}
        self._eps_map  = {}

    def perturb(self, emb_name: str = "embeddings"):
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name and param.grad is not None:
                self._backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    r = self.lr * param.grad / norm
                    param.data.add_(r)
                    eps_t = self.eps * param.abs().detach()
                    self._eps_map[name] = (
                        self._backup[name] - eps_t,
                        self._backup[name] + eps_t,
                    )
                    lo, hi = self._eps_map[name]
                    param.data = torch.max(torch.min(param.data, hi), lo)

    def restore(self):
        for name, param in self.model.named_parameters():
            if name in self._backup:
                param.data = self._backup[name]
        self._backup.clear()
        self._eps_map.clear()
