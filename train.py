import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(".")
OUT = ROOT / "prepared"
PLOTS = ROOT / "plots"
PLOTS.mkdir(exist_ok=True)
SEED = 1337

CFG = dict(
    d_model=128,
    nhead=4,
    num_layers=3,
    dim_feedforward=512,
    dropout=0.1,
    epochs=60,
    batch_size=256,
    lr=3e-4,
    weight_decay=0.01,
    warmup_epochs=1,
    label_smoothing=0.05,
    n_mels=80,
    n_frames=101,
    num_classes=12,
)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class MelDataset(Dataset):
    def __init__(self, split, stats, train=False):
        self.X = np.load(OUT / f"X_{split}.npy", mmap_mode="r")
        self.y = np.load(OUT / f"y_{split}.npy")
        self.mean = np.float32(stats["mean"])
        self.std = np.float32(stats["std"])
        self.train = train

    def __len__(self):
        return len(self.y)

    def augment(self, x):
        n_frames = x.shape[-1]
        shift = random.randint(-8, 8)
        if shift != 0:
            x = torch.roll(x, shifts=shift, dims=-1)
            if shift > 0:
                x[..., :shift] = 0
            else:
                x[..., shift:] = 0
        for _ in range(2):
            f = random.randint(0, 8)
            f0 = random.randint(0, x.shape[-2] - f)
            x[f0 : f0 + f, :] = 0
            t = random.randint(0, 12)
            t0 = random.randint(0, n_frames - t)
            x[:, t0 : t0 + t] = 0
        return x

    def __getitem__(self, i):
        x = torch.from_numpy(np.array(self.X[i], dtype=np.float32))
        x = (x - self.mean) / self.std
        if self.train:
            x = self.augment(x)
        return x.unsqueeze(0), int(self.y[i])


class AudioTransformer(nn.Module):
    def __init__(self, cfg, n_frames):
        super().__init__()
        d = cfg["d_model"]
        self.proj = nn.Linear(cfg["n_mels"], d)
        self.cls = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        self.pos = nn.Parameter(torch.randn(1, n_frames + 1, d) * 0.02)
        self.drop = nn.Dropout(cfg["dropout"])
        layer = nn.TransformerEncoderLayer(
            d_model=d,
            nhead=cfg["nhead"],
            dim_feedforward=cfg["dim_feedforward"],
            dropout=cfg["dropout"],
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, cfg["num_layers"])
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, cfg["num_classes"])

    def forward(self, x):
        x = x.squeeze(1).transpose(1, 2)
        x = self.proj(x)
        b = x.size(0)
        x = torch.cat([self.cls.expand(b, -1, -1), x], dim=1) + self.pos
        x = self.encoder(self.drop(x))
        return self.head(self.norm(x[:, 0]))


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, n = 0.0, 0, 0
    preds, targets = [], []
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            logits = model(x)
            loss = criterion(logits, y)
        total_loss += loss.item() * y.size(0)
        p = logits.argmax(1)
        correct += (p == y).sum().item()
        n += y.size(0)
        preds.append(p.cpu())
        targets.append(y.cpu())
    return total_loss / n, correct / n, torch.cat(preds).numpy(), torch.cat(targets).numpy()


def plot_curves(history):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    ep = np.arange(1, len(history["train_loss"]) + 1)
    axes[0].plot(ep, history["train_loss"], label="train", lw=2)
    axes[0].plot(ep, history["val_loss"], label="val", lw=2)
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("cross-entropy")
    axes[0].set_title("Loss")
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    axes[1].plot(ep, history["train_acc"], label="train", lw=2)
    axes[1].plot(ep, history["val_acc"], label="val", lw=2)
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("accuracy")
    axes[1].set_title("Accuracy")
    axes[1].set_ylim(0, 1.02)
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    fig.suptitle("Audio Transformer on Speech Commands v2 (12 classes)")
    fig.tight_layout()
    fig.savefig(PLOTS / "training_curves.png", dpi=140)
    plt.close(fig)


def plot_confusion(cm, classes):
    fig, ax = plt.subplots(figsize=(8.5, 7.5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(classes)), classes, rotation=45, ha="right")
    ax.set_yticks(range(len(classes)), classes)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    norm = cm.sum(axis=1, keepdims=True).clip(min=1)
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(
                j, i, str(cm[i, j]),
                ha="center", va="center", fontsize=7,
                color="white" if cm[i, j] > cm.max() * 0.5 else "black",
            )
    ax.set_title("Test confusion matrix")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(PLOTS / "confusion_matrix.png", dpi=140)
    plt.close(fig)


def plot_per_class(metrics, classes):
    fig, ax = plt.subplots(figsize=(10, 4.5))
    x = np.arange(len(classes))
    w = 0.27
    ax.bar(x - w, metrics["precision"], w, label="precision")
    ax.bar(x, metrics["recall"], w, label="recall")
    ax.bar(x + w, metrics["f1"], w, label="F1")
    ax.set_xticks(x, classes, rotation=30, ha="right")
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("score")
    ax.set_title("Per-class test metrics")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOTS / "per_class_metrics.png", dpi=140)
    plt.close(fig)


def plot_predictions(model, dataset, preds, device, classes, mean, std):
    correct = np.where(preds == dataset.y)[0]
    wrong = np.where(preds != dataset.y)[0]
    chosen = list(wrong[:4]) + list(correct[:: max(1, len(correct) // 12)][:12])
    chosen = chosen[:16]
    fig, axes = plt.subplots(4, 4, figsize=(13, 9))
    model.eval()
    for ax, idx in zip(axes.ravel(), chosen):
        mel = np.array(dataset.X[idx], dtype=np.float32)
        ax.imshow(mel, aspect="auto", origin="lower", cmap="magma")
        ok = preds[idx] == dataset.y[idx]
        ax.set_title(
            f"true: {classes[dataset.y[idx]]}  |  pred: {classes[preds[idx]]}",
            fontsize=9,
            color="green" if ok else "crimson",
        )
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("Test predictions (green = correct, red = wrong)")
    fig.tight_layout()
    fig.savefig(PLOTS / "test_predictions.png", dpi=140)
    plt.close(fig)


def main():
    set_seed(SEED)
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("CUDA not available - refusing to train on CPU")
    print(f"device: {device} ({torch.cuda.get_device_name(0)})")
    print(f"torch {torch.__version__} | cuda {torch.version.cuda}")

    stats = json.loads((OUT / "stats.json").read_text())
    classes = stats["classes"]
    cfg = CFG

    train_ds = MelDataset("train", stats, train=True)
    val_ds = MelDataset("val", stats)
    test_ds = MelDataset("test", stats)
    train_loader = DataLoader(
        train_ds, batch_size=cfg["batch_size"], shuffle=True,
        num_workers=8, pin_memory=True, drop_last=True, persistent_workers=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=512, shuffle=False, num_workers=4, pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=512, shuffle=False, num_workers=4, pin_memory=True,
    )

    model = AudioTransformer(cfg, stats["n_frames"]).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model parameters: {n_params:,}")
    print(model)

    criterion = nn.CrossEntropyLoss(label_smoothing=cfg["label_smoothing"])
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    steps_per_epoch = len(train_loader)
    total_steps = cfg["epochs"] * steps_per_epoch
    warmup_steps = cfg["warmup_epochs"] * steps_per_epoch

    def lr_at(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        t = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * t))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "lr": []}
    best_acc, best_epoch = 0.0, -1
    ckpt = ROOT / "checkpoints"
    ckpt.mkdir(exist_ok=True)
    t_start = time.time()

    for epoch in range(cfg["epochs"]):
        model.train()
        run_loss, run_correct, run_n = 0.0, 0, 0
        pbar = tqdm(train_loader, desc=f"epoch {epoch + 1}/{cfg['epochs']}", leave=False)
        for x, y in pbar:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(x)
                loss = criterion(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            run_loss += loss.item() * y.size(0)
            run_correct += (logits.argmax(1) == y).sum().item()
            run_n += y.size(0)
            pbar.set_postfix(
                loss=f"{run_loss / run_n:.3f}",
                acc=f"{run_correct / run_n:.3f}",
                mem=f"{torch.cuda.max_memory_allocated() / 2**30:.1f}GB",
            )
        tr_loss, tr_acc = run_loss / run_n, run_correct / run_n
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, device)
        lr_now = sched.get_last_lr()[0]

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["lr"].append(lr_now)
        print(
            f"epoch {epoch + 1:02d} | train loss {tr_loss:.3f} acc {tr_acc:.3f} "
            f"| val loss {val_loss:.3f} acc {val_acc:.3f} | lr {lr_now:.1e}"
        )

        if val_acc > best_acc:
            best_acc, best_epoch = val_acc, epoch + 1
            torch.save({"model": model.state_dict(), "cfg": cfg, "epoch": epoch + 1,
                        "val_acc": val_acc, "n_frames": stats["n_frames"]},
                       ckpt / "best_model.pt")

    elapsed = time.time() - t_start
    print(f"training done in {elapsed / 60:.1f} min | best val acc {best_acc:.4f} @ epoch {best_epoch}")

    model.load_state_dict(torch.load(ckpt / "best_model.pt", map_location=device)["model"])
    test_loss, test_acc, preds, targets = evaluate(model, test_loader, criterion, device)
    print(f"TEST loss {test_loss:.3f} | accuracy {test_acc:.4f}")

    cm = np.zeros((len(classes), len(classes)), dtype=np.int64)
    for t, p in zip(targets, preds):
        cm[t, p] += 1
    tp = np.diag(cm).astype(np.float64)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    precision = np.where(tp + fp > 0, tp / (tp + fp), 0)
    recall = np.where(tp + fn > 0, tp / (tp + fn), 0)
    f1 = np.where(precision + recall > 0, 2 * precision * recall / (precision + recall), 0)

    metrics = {
        "test_loss": test_loss,
        "test_accuracy": test_acc,
        "best_val_accuracy": best_acc,
        "best_epoch": best_epoch,
        "train_time_min": elapsed / 60,
        "num_parameters": n_params,
        "confusion_matrix": cm.tolist(),
        "classes": classes,
        "precision": precision.tolist(),
        "recall": recall.tolist(),
        "f1": f1.tolist(),
        "history": history,
    }
    (PLOTS / "metrics.json").write_text(json.dumps(metrics, indent=2))

    plot_curves(history)
    plot_confusion(cm, classes)
    plot_per_class(metrics, classes)
    plot_predictions(model, test_ds, preds, device, classes, stats["mean"], stats["std"])
    torch.save(model.state_dict(), ROOT / "final_model.pt")
    print("saved checkpoints/best_model.pt, final_model.pt and plots/ figures")


if __name__ == "__main__":
    main()
