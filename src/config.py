from dataclasses import dataclass

LABEL_ORDER = [
    "Anggaran", "Distribusi", "Ekonomi", "Kualitas Pangan",
    "Lainnya", "Politik", "Sasaran Penerima", "Tata Kelola",
]


@dataclass
class Config:
    # ── paths ─────────────────────────────────────────────────────
    data_dir      : str = "./data"
    output_dir    : str = "./data/outputs"
    labeled_file  : str = "labeled/hasil_label_baru.xlsx"
    test_file     : str = "raw/case_1_text_to_predict.xlsx"
    template_file : str = "raw/case_1_template_sheet.xlsx"

    # ── model ─────────────────────────────────────────────────────
    model_name    : str   = "xlm-roberta-base"
    max_len       : int   = 256
    pooling       : str   = "cls_mean"   # cls | mean | cls_mean

    # ── training ──────────────────────────────────────────────────
    seed          : int   = 42
    n_folds       : int   = 5
    batch_size    : int   = 16
    grad_accum    : int   = 2            # effective batch = batch_size * grad_accum
    epochs        : int   = 12           # was 8; XLM-R still improved at ep 8
    patience      : int   = 4            # was 3; extra patience for slower models
    lr            : float = 2e-5
    weight_decay  : float = 0.01
    warmup_ratio  : float = 0.1
    max_grad_norm : float = 1.0
    scheduler     : str   = "cosine"     # cosine | linear

    # ── advanced ──────────────────────────────────────────────────
    llrd_factor        : float = 0.9
    label_smoothing    : float = 0.1
    focal_gamma        : float = 2.0
    use_rdrop          : bool  = True
    rdrop_alpha        : float = 0.3
    use_awp            : bool  = False
    awp_lr             : float = 1e-4
    awp_eps            : float = 1e-2
    awp_start_epoch    : int   = 2

    # ── Optuna ────────────────────────────────────────────────────
    optuna_folds       : int   = 3
    optuna_trials      : int   = 40
    optuna_epochs      : int   = 6


def get_indobert_cfg(**overrides) -> Config:
    defaults = dict(
        model_name="indobenchmark/indobert-base-p2",
        epochs=12,
        patience=4,
    )
    return Config(**{**defaults, **overrides})


def get_xlmr_cfg(**overrides) -> Config:
    defaults = dict(
        model_name="xlm-roberta-base",
        epochs=12,
        patience=4,
    )
    return Config(**{**defaults, **overrides})
