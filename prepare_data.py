import json
import os
import random
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
import torchaudio
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("data/speech_commands")
OUT = Path("prepared")
PLOTS = Path("plots")
OUT.mkdir(exist_ok=True)
PLOTS.mkdir(exist_ok=True)

TARGET_WORDS = ["yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go"]
CLASSES = TARGET_WORDS + ["silence", "unknown"]
SR = 16000
CLIP_LEN = SR
N_MELS = 80
N_FFT = 400
HOP = 160

UNKNOWN_CAP = {"train": 3000, "val": 800, "test": 800}
SILENCE_COUNT = {"train": 1500, "val": 400, "test": 400}
SEED = 1337


def read_list(name):
    return set((ROOT / name).read_text().splitlines())


def build_manifest():
    val_list = read_list("validation_list.txt")
    test_list = read_list("testing_list.txt")

    all_words = sorted(
        p.name for p in ROOT.iterdir() if p.is_dir() and p.name != "_background_noise_"
    )
    unknown_words = [w for w in all_words if w not in TARGET_WORDS]

    rng = random.Random(SEED)
    manifest = {}
    for split in ("train", "val", "test"):
        split_map = {"val": val_list, "test": test_list}.get(split)

        def in_split(rel):
            if split == "train":
                return rel not in val_list and rel not in test_list
            return rel in split_map

        target_files = [
            (str(ROOT / w / f), TARGET_WORDS.index(w))
            for w in TARGET_WORDS
            for f in sorted(os.listdir(ROOT / w))
            if in_split(f"{w}/{f}")
        ]

        unknown_pool = [
            (str(ROOT / w / f), unknown_words.index(w) + 10)
            for w in unknown_words
            for f in sorted(os.listdir(ROOT / w))
            if in_split(f"{w}/{f}")
        ]

        n_unknown = min(UNKNOWN_CAP[split], len(unknown_pool))
        unknown_files = rng.sample(unknown_pool, n_unknown)
        silence_files = [("__silence__", -1)] * SILENCE_COUNT[split]

        files = target_files + unknown_files + silence_files
        rng.shuffle(files)
        manifest[split] = {
            "files": files,
            "n_target": len(target_files),
            "n_unknown": len(unknown_files),
            "n_silence": len(silence_files),
        }
        print(
            f"{split}: {len(files)} clips "
            f"(target={len(target_files)}, unknown={len(unknown_files)}, "
            f"silence={len(silence_files)})"
        )
    return manifest


def load_waveform(path, rng=None):
    if path == "__silence__":
        if rng is None:
            rng = random.Random(SEED)
        noise_files = sorted((ROOT / "_background_noise_").glob("*.wav"))
        noise = noise_files[rng.randrange(len(noise_files))]
        info = sf.info(str(noise))
        start = rng.randrange(max(1, info.frames - CLIP_LEN))
        wav, _ = sf.read(str(noise), start=start, frames=CLIP_LEN, dtype="float32")
    else:
        wav, _ = sf.read(path, dtype="float32")
        if len(wav) > CLIP_LEN:
            if rng is not None:
                start = rng.randrange(len(wav) - CLIP_LEN)
            else:
                start = (len(wav) - CLIP_LEN) // 2
            wav = wav[start : start + CLIP_LEN]
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if len(wav) < CLIP_LEN:
        wav = np.pad(wav, (0, CLIP_LEN - len(wav)))
    return wav.astype(np.float32)


def compute_logmel(wavs, device):
    x = torch.from_numpy(np.stack(wavs)).to(device)
    mel = torchaudio.transforms.MelSpectrogram(
        sample_rate=SR,
        n_fft=N_FFT,
        win_length=N_FFT,
        hop_length=HOP,
        n_mels=N_MELS,
        f_min=20,
        f_max=7600,
        power=2.0,
    ).to(device)
    with torch.no_grad():
        m = mel(x)
        m = 10.0 * torch.log10(m.clamp_min(1e-10))
    return m.cpu().numpy().astype(np.float16)


def main():
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    random.seed(SEED)
    np.random.seed(SEED)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"computing log-mel on {device}")
    manifest = build_manifest()

    stats = {"classes": CLASSES, "sr": SR, "n_mels": N_MELS, "n_frames": None}
    previews = {}

    for split in ("train", "val", "test"):
        files = manifest[split]["files"]
        rng = random.Random(SEED + hash(split) % 1000)
        labels = np.array(
            [10 if lab < 0 else (11 if lab >= 10 else lab) for _, lab in files],
            dtype=np.int64,
        )

        mels = []
        batch = []
        for i, (path, _) in enumerate(tqdm(files, desc=split)):
            batch.append(load_waveform(path, rng if split == "train" else None))
            if len(batch) == 1024:
                mels.append(compute_logmel(batch, device))
                batch = []
        if batch:
            mels.append(compute_logmel(batch, device))

        X = np.concatenate(mels, axis=0)
        np.save(OUT / f"X_{split}.npy", X)
        np.save(OUT / f"y_{split}.npy", labels)
        print(f"{split}: X={X.shape} y={labels.shape}")

        if split == "test":
            stats["n_frames"] = int(X.shape[2])
            for cls_idx in range(len(CLASSES)):
                idx = int(np.where(labels == cls_idx)[0][0])
                previews[CLASSES[cls_idx]] = {
                    "wav": load_waveform(files[idx][0], None),
                    "mel": X[idx].astype(np.float32),
                }
            stats["preview_files"] = {}
            for cls_idx in range(len(CLASSES)):
                idx = int(np.where(labels == cls_idx)[0][0])
                stats["preview_files"][CLASSES[cls_idx]] = (
                    files[idx][0] if files[idx][0] != "__silence__" else "background noise"
                )

    train_mean = float(np.mean(np.load(OUT / "X_train.npy", mmap_mode="r").astype(np.float32)))
    train_std = float(np.std(np.load(OUT / "X_train.npy", mmap_mode="r").astype(np.float32)))
    stats["mean"] = train_mean
    stats["std"] = train_std
    stats["counts"] = {
        s: {
            "target": manifest[s]["n_target"],
            "unknown": manifest[s]["n_unknown"],
            "silence": manifest[s]["n_silence"],
        }
        for s in ("train", "val", "test")
    }
    (OUT / "stats.json").write_text(json.dumps(stats, indent=2))
    print(f"train mean={train_mean:.2f} std={train_std:.2f}")

    plot_examples(previews, stats)
    plot_counts(manifest)
    print("done")


def plot_examples(previews, stats):
    fig, axes = plt.subplots(len(CLASSES), 2, figsize=(10, 2.0 * len(CLASSES)))
    for r, cls in enumerate(CLASSES):
        wav = previews[cls]["wav"]
        mel = previews[cls]["mel"]
        ax = axes[r, 0]
        ax.plot(np.arange(len(wav)) / SR, wav, lw=0.5)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(cls, loc="left", fontsize=9, fontweight="bold")
        ax = axes[r, 1]
        ax.imshow(mel, aspect="auto", origin="lower", cmap="magma")
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("Google Speech Commands v2 - example clips (left: waveform, right: 80-bin log-mel dB)")
    fig.tight_layout()
    fig.savefig(PLOTS / "dataset_samples.png", dpi=130)
    plt.close(fig)


def plot_counts(manifest):
    fig, ax = plt.subplots(figsize=(9, 4))
    x = np.arange(3)
    bottom = np.zeros(3)
    colors = ["#4C72B0", "#DD8452", "#55A868"]
    for key, color, label in zip(
        ("target", "unknown", "silence"),
        colors,
        ("10 target words", "unknown words", "silence/noise"),
    ):
        vals = np.array([manifest[s][f"n_{key}"] for s in ("train", "val", "test")])
        ax.bar(x, vals, 0.55, bottom=bottom, color=color, label=label)
        bottom += vals
    for i, s in enumerate(("train", "val", "test")):
        ax.text(x[i], bottom[i] + 400, str(int(bottom[i])), ha="center", fontsize=10)
    ax.set_xticks(x, ["train", "val", "test"])
    ax.set_ylabel("clips")
    ax.set_title("Split composition (12-class setup)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOTS / "class_distribution.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
