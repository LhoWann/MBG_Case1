import random
import numpy as np
import torch
from sklearn.preprocessing import LabelEncoder
from .config import LABEL_ORDER


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def build_label_encoder() -> LabelEncoder:
    le = LabelEncoder()
    le.fit(LABEL_ORDER)
    return le
