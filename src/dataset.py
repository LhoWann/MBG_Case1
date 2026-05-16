import numpy as np
import torch
from torch.utils.data import Dataset, WeightedRandomSampler


class TweetDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len: int):
        self.texts     = texts
        self.labels    = labels
        self.tokenizer = tokenizer
        self.max_len   = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


def make_weighted_sampler(labels: np.ndarray) -> WeightedRandomSampler:
    counts = np.bincount(labels)
    weights = (1.0 / counts)[labels]
    return WeightedRandomSampler(
        weights=weights,
        num_samples=len(weights),
        replacement=True,
    )


def compute_class_weights(labels: np.ndarray, n_classes: int, device) -> torch.Tensor:
    counts = np.bincount(labels, minlength=n_classes).astype(np.float32)
    w = len(labels) / (n_classes * counts)
    return torch.tensor(w, dtype=torch.float32, device=device)
