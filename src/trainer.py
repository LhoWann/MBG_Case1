import gc
from pathlib import Path

import numpy as np
import torch
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from sklearn.metrics import balanced_accuracy_score, classification_report
from sklearn.model_selection import StratifiedKFold
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup

from .config import Config, LABEL_ORDER
from .dataset import TweetDataset, compute_class_weights, make_weighted_sampler
from .losses import FocalLossWithSmoothing, rdrop_kl_loss
from .model import MBGClassifier
from .optimizer import AWP, get_llrd_optimizer
from .utils import set_seed

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_epoch(model, loader, optimizer, scheduler, scaler, criterion, cfg, epoch, awp=None):
    model.train()
    total_loss = 0.0
    optimizer.zero_grad()
    use_amp = DEVICE.type == "cuda"

    for step, batch in enumerate(loader):
        ids   = batch["input_ids"].to(DEVICE)
        mask  = batch["attention_mask"].to(DEVICE)
        ttype = batch.get("token_type_ids")
        if ttype is not None:
            ttype = ttype.to(DEVICE)
        labs = batch["labels"].to(DEVICE)

        if cfg.use_rdrop:
            if use_amp:
                with autocast():
                    l1      = model(ids, mask, ttype)
                    l2      = model(ids, mask, ttype)
                    loss_ce = 0.5 * (criterion(l1, labs) + criterion(l2, labs))
                    loss    = (loss_ce + cfg.rdrop_alpha * rdrop_kl_loss(l1, l2)) / cfg.grad_accum
            else:
                l1      = model(ids, mask, ttype)
                l2      = model(ids, mask, ttype)
                loss_ce = 0.5 * (criterion(l1, labs) + criterion(l2, labs))
                loss    = (loss_ce + cfg.rdrop_alpha * rdrop_kl_loss(l1, l2)) / cfg.grad_accum
        else:
            if use_amp:
                with autocast():
                    logits = model(ids, mask, ttype)
                    loss   = criterion(logits, labs) / cfg.grad_accum
            else:
                logits = model(ids, mask, ttype)
                loss   = criterion(logits, labs) / cfg.grad_accum

        if use_amp:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        total_loss += loss.item() * cfg.grad_accum

        if (step + 1) % cfg.grad_accum == 0:
            if awp is not None and epoch >= cfg.awp_start_epoch:
                awp.perturb()
                if use_amp:
                    with autocast():
                        l_awp = model(ids, mask, ttype)
                        loss_awp = criterion(l_awp, labs) / cfg.grad_accum
                    scaler.scale(loss_awp).backward()
                else:
                    (criterion(model(ids, mask, ttype), labs) / cfg.grad_accum).backward()
                awp.restore()

            if use_amp:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                optimizer.step()

            scheduler.step()
            optimizer.zero_grad()

    return total_loss / len(loader)


@torch.no_grad()
def eval_epoch(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    all_preds, all_labs, all_logits = [], [], []

    for batch in loader:
        ids   = batch["input_ids"].to(DEVICE)
        mask  = batch["attention_mask"].to(DEVICE)
        ttype = batch.get("token_type_ids")
        if ttype is not None:
            ttype = ttype.to(DEVICE)
        labs   = batch["labels"].to(DEVICE)
        logits = model(ids, mask, ttype)

        total_loss += criterion(logits, labs).item()
        all_preds.extend(logits.argmax(-1).cpu().numpy())
        all_labs.extend(labs.cpu().numpy())
        all_logits.append(logits.cpu().float().numpy())

    bal = balanced_accuracy_score(all_labs, all_preds)
    return total_loss / len(loader), bal, np.array(all_preds), np.array(all_labs), np.concatenate(all_logits)


@torch.no_grad()
def predict_logits(model, texts_list, tokenizer, cfg) -> np.ndarray:
    model.eval()
    ds  = TweetDataset(texts_list, None, tokenizer, cfg.max_len)
    ldr = DataLoader(ds, batch_size=cfg.batch_size * 2, shuffle=False, num_workers=0)
    out = []
    for batch in ldr:
        ids   = batch["input_ids"].to(DEVICE)
        mask  = batch["attention_mask"].to(DEVICE)
        ttype = batch.get("token_type_ids")
        if ttype is not None:
            ttype = ttype.to(DEVICE)
        out.append(model(ids, mask, ttype).cpu().float().numpy())
    return np.concatenate(out)


def run_fold(fold, tr_texts, tr_labels, val_texts, val_labels, tokenizer, cfg, ckpt_path):
    n_classes = len(LABEL_ORDER)
    cw = compute_class_weights(tr_labels, n_classes, DEVICE)
    criterion = FocalLossWithSmoothing(
        alpha=cw, gamma=cfg.focal_gamma,
        smoothing=cfg.label_smoothing, n_classes=n_classes,
    )

    pin = DEVICE.type == "cuda"
    tr_ds  = TweetDataset(tr_texts, tr_labels, tokenizer, cfg.max_len)
    val_ds = TweetDataset(val_texts, val_labels, tokenizer, cfg.max_len)
    tr_ldr = DataLoader(tr_ds, batch_size=cfg.batch_size,
                         sampler=make_weighted_sampler(tr_labels),
                         num_workers=0, pin_memory=pin)
    val_ldr = DataLoader(val_ds, batch_size=cfg.batch_size * 2,
                          shuffle=False, num_workers=0, pin_memory=pin)

    model = MBGClassifier(cfg.model_name, n_classes, cfg.pooling).to(DEVICE)
    total_steps  = (len(tr_ldr) // cfg.grad_accum) * cfg.epochs
    warmup_steps = int(total_steps * cfg.warmup_ratio)
    optimizer    = get_llrd_optimizer(model, cfg.lr, cfg.llrd_factor, cfg.weight_decay)

    if cfg.scheduler == "cosine":
        scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    else:
        scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    scaler = GradScaler() if DEVICE.type == "cuda" else None
    awp    = AWP(model, optimizer, cfg.awp_lr, cfg.awp_eps) if cfg.use_awp else None

    # ── checkpoint resume ─────────────────────────────────────────
    # If checkpoint already exists, skip training this fold entirely
    if Path(ckpt_path).exists():
        print(f"  fold {fold}: checkpoint found, loading and skipping training")
        model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
        _, best_acc, val_preds, val_true, val_logits = eval_epoch(model, val_ldr, criterion)
        print(f"  fold {fold} (resumed) acc={best_acc:.4f}")
        return model, best_acc, val_preds, val_true, val_logits

    best_acc = 0.0
    patience_left = cfg.patience
    print(f"  fold {fold} | tr={len(tr_ds)} val={len(val_ds)} steps={total_steps}")

    for epoch in range(1, cfg.epochs + 1):
        tr_loss = train_epoch(model, tr_ldr, optimizer, scheduler, scaler, criterion, cfg, epoch, awp)
        _, val_acc, _, _, _ = eval_epoch(model, val_ldr, criterion)
        marker = ""
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), ckpt_path)
            patience_left = cfg.patience
            marker = " ★"
        else:
            patience_left -= 1
        print(f"    ep {epoch}/{cfg.epochs}  loss={tr_loss:.4f}  acc={val_acc:.4f}{marker}")
        if patience_left == 0:
            print("    early stop")
            break

    model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
    _, best_acc, val_preds, val_true, val_logits = eval_epoch(model, val_ldr, criterion)
    print(f"  fold {fold} best={best_acc:.4f}")
    return model, best_acc, val_preds, val_true, val_logits


def run_cv(cfg: Config, texts, labels, predict_texts, label="exp"):
    """Full 5-fold CV with incremental checkpoint save + OOF accumulation."""
    set_seed(cfg.seed)
    n_classes = len(LABEL_ORDER)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)

    ckpt_dir = Path(cfg.output_dir) / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Load previously computed OOF if available (resume after crash)
    oof_path  = ckpt_dir / f"{label}_oof_logits.npy"
    test_path = ckpt_dir / f"{label}_test_logits.npy"
    done_path = ckpt_dir / f"{label}_done_folds.txt"

    oof_logits  = np.zeros((len(texts), n_classes), dtype=np.float32)
    test_logits = np.zeros((len(predict_texts), n_classes), dtype=np.float32)
    done_folds  = set()

    if oof_path.exists():
        oof_logits = np.load(str(oof_path))
        print(f"  Loaded existing OOF logits from {oof_path}")
    if test_path.exists():
        test_logits = np.load(str(test_path))
        print(f"  Loaded existing test logits from {test_path}")
    if done_path.exists():
        done_folds = {int(x) for x in done_path.read_text().split() if x.strip()}
        print(f"  Already completed folds: {done_folds}")

    skf = StratifiedKFold(n_splits=cfg.n_folds, shuffle=True, random_state=cfg.seed)
    fold_scores = []

    for fold, (tr_idx, val_idx) in enumerate(skf.split(texts, labels), 1):
        print(f"\n=== FOLD {fold}/{cfg.n_folds} [{label}] ===")
        ckpt = str(ckpt_dir / f"{label}_fold{fold}.pt")
        tr_t = [texts[i] for i in tr_idx]
        val_t = [texts[i] for i in val_idx]
        tr_l = labels[tr_idx]
        val_l = labels[val_idx]

        model, score, _, _, val_logits = run_fold(
            fold, tr_t, tr_l, val_t, val_l, tokenizer, cfg, ckpt
        )
        oof_logits[val_idx] = val_logits
        test_logits += predict_logits(model, predict_texts, tokenizer, cfg) / cfg.n_folds
        fold_scores.append(score)

        # save incremental state
        np.save(str(oof_path), oof_logits)
        np.save(str(test_path), test_logits)
        done_folds.add(fold)
        done_path.write_text(" ".join(str(f) for f in sorted(done_folds)))

        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    oof_preds   = oof_logits.argmax(axis=1)
    oof_bal_acc = balanced_accuracy_score(labels, oof_preds)

    print(f"\n{'='*52}")
    print(f"[{label}] folds: {[round(s, 4) for s in fold_scores]}")
    print(f"[{label}] mean ± std: {np.mean(fold_scores):.4f} ± {np.std(fold_scores):.4f}")
    print(f"[{label}] OOF balanced accuracy: {oof_bal_acc:.4f}")
    print("="*52)
    print(classification_report(
        labels, oof_preds, target_names=LABEL_ORDER, digits=3, zero_division=0
    ))

    return {
        "label"       : label,
        "cfg"         : cfg,
        "oof_logits"  : oof_logits,
        "test_logits" : test_logits,
        "fold_scores" : fold_scores,
        "oof_bal_acc" : oof_bal_acc,
    }
