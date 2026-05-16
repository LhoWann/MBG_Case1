import torch
import torch.nn as nn
from transformers import AutoModel


class MBGClassifier(nn.Module):
    """
    Encoder + custom pooling + linear head.

    pooling options:
      'cls'      — [CLS] token only (default AutoModelForSequenceClassification)
      'mean'     — mean of all token embeddings (masked)
      'cls_mean' — concat([CLS], mean) → 2×hidden; best in practice
    """

    def __init__(
        self,
        model_name: str,
        n_classes: int,
        pooling: str = "cls_mean",
        hidden_dropout: float = 0.1,
        use_gradient_checkpointing: bool = False,
    ):
        super().__init__()
        self.pooling = pooling
        self.encoder = AutoModel.from_pretrained(model_name, ignore_mismatched_sizes=True)

        if use_gradient_checkpointing:
            self.encoder.gradient_checkpointing_enable()

        hidden = self.encoder.config.hidden_size
        in_features = hidden * 2 if pooling == "cls_mean" else hidden

        self.dropout    = nn.Dropout(hidden_dropout)
        self.classifier = nn.Linear(in_features, n_classes)
        nn.init.normal_(self.classifier.weight, std=0.02)
        nn.init.zeros_(self.classifier.bias)

    def _mean_pool(self, hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask_exp = mask.unsqueeze(-1).expand(hidden.size()).float()
        return torch.sum(hidden * mask_exp, 1) / torch.clamp(mask_exp.sum(1), min=1e-9)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor = None,
    ) -> torch.Tensor:
        out = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        h = out.last_hidden_state  # (B, T, H)

        if self.pooling == "cls":
            pooled = h[:, 0, :]
        elif self.pooling == "mean":
            pooled = self._mean_pool(h, attention_mask)
        else:  # cls_mean
            pooled = torch.cat([h[:, 0, :], self._mean_pool(h, attention_mask)], dim=-1)

        return self.classifier(self.dropout(pooled))
