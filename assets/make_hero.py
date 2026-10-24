import json
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
import torchaudio
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))
from train import AudioTransformer, CFG

BG_TOP, BG_BOT = "#070C18", "#0C1730"
PANEL = "#101A30"
TEXT, MUTED = "#E6EDF3", "#8FA0B8"
CYAN, VIOLET, GREEN, AMBER = "#38BDF8", "#A78BFA", "#34D399", "#FBBF24"
DIM = "#22314E"

W, H = 18.0, 8.6
fig = plt.figure(figsize=(W, H), dpi=170, facecolor=BG_BOT)

MONO = "JetBrains Mono" if any(
    "JetBrains Mono" in f.name for f in fm.fontManager.ttflist
) else "DejaVu Sans Mono"
plt.rcParams["font.family"] = "DejaVu Sans"

bg = fig.add_axes([0, 0, 1, 1], zorder=-10)
bg.axis("off")
bg.imshow(np.linspace(0, 1, 300).reshape(-1, 1), aspect="auto",
          cmap=LinearSegmentedColormap.from_list("bg", [BG_TOP, BG_BOT]),
          extent=[0, 1, 0, 1], zorder=-10)

stats = json.loads((PROJECT / "prepared" / "stats.json").read_text())
classes = stats["classes"]
mean, std = stats["mean"], stats["std"]
device = "cuda" if torch.cuda.is_available() else "cpu"

model = AudioTransformer(CFG, stats["n_frames"]).to(device)
model.load_state_dict(torch.load(PROJECT / "final_model.pt", map_location=device))
model.eval()

mel_tf = torchaudio.transforms.MelSpectrogram(
    sample_rate=16000, n_fft=400, win_length=400, hop_length=160,
    n_mels=80, f_min=20, f_max=7600, power=2.0,
).to(device)


def logmel(wavs):
    with torch.no_grad():
        m = mel_tf(torch.from_numpy(np.stack(wavs)).to(device))
        m = 10.0 * torch.log10(m.clamp_min(1e-10))
    return m


files = sorted((PROJECT / "data" / "speech_commands" / "stop").glob("*.wav"))[:48]
wavs = []
for f in files:
    w, _ = sf.read(str(f), dtype="float32")
    if len(w) > 16000:
        w = w[:16000]
    wavs.append(np.pad(w, (0, 16000 - len(w))))
wavs = np.stack(wavs)

with torch.no_grad():
    mels = logmel(wavs).cpu().numpy()
    x = torch.from_numpy((mels - mean) / std).unsqueeze(1).to(device)
    probs_all = F.softmax(model(x).float(), dim=1).cpu().numpy()

stop_idx = classes.index("stop")
best = int(np.argmax(probs_all[:, stop_idx]))
wav, mel, probs = wavs[best], mels[best], probs_all[best]
print(f"sample: {files[best].name} | stop={probs[stop_idx]:.3f} "
      f"pred={classes[int(probs.argmax())]}")

with torch.no_grad():
    xb = torch.from_numpy((mel - mean) / std)[None, None].to(device)
    tokens = model.proj(xb.squeeze(1).transpose(1, 2))
    tokens = torch.cat([model.cls.expand(1, -1, -1), tokens], dim=1) + model.pos
    layer = model.encoder.layers[0]
    h = layer.norm1(tokens)
    qkv = F.linear(h, layer.self_attn.in_proj_weight, layer.self_attn.in_proj_bias)
    q, k, _ = qkv.chunk(3, dim=-1)
    hd = CFG["d_model"] // CFG["nhead"]
    q = q.view(1, -1, CFG["nhead"], hd).transpose(1, 2)
    k = k.view(1, -1, CFG["nhead"], hd).transpose(1, 2)
    attn = (q @ k.transpose(-2, -1) / math.sqrt(hd)).softmax(-1).mean(1)[0].cpu().numpy()
    emb = tokens[0, 1:].T.cpu().numpy()
    attn = attn[1:, 1:]


def panel(x, y, w, h, color=PANEL, r=0.012, ec="none", lw=0, z=1):
    fig.patches.append(
        FancyBboxPatch(
            (x, y), w, h,
            boxstyle=f"round,pad=0.001,rounding_size={r}",
            facecolor=color, edgecolor=ec, linewidth=lw, zorder=z,
        )
    )


def arrow(x0, x1, y):
    fig.patches.append(
        FancyArrowPatch(
            (x0, y), (x1, y), arrowstyle="-|>", mutation_scale=19,
            lw=2.1, color=CYAN, alpha=0.95, zorder=6, shrinkA=0, shrinkB=0,
        )
    )


def stage(x, w, label, desc):
    fig.text(x + w / 2, 0.798, label, ha="center", va="bottom",
             fontsize=9.5, color=CYAN, fontfamily=MONO, fontweight="bold")
    fig.text(x + w / 2, 0.324, desc, ha="center", va="top",
             fontsize=8.5, color=MUTED, fontfamily=MONO)


fig.text(0.033, 0.962, "SPEECH COMMANDS", fontsize=31, fontweight="bold",
         color=TEXT, va="top")
fig.text(0.033, 0.902, "TRANSFORMER", fontsize=31, fontweight="bold",
         color=CYAN, va="top")
fig.text(0.033, 0.838, "A minimal 3-layer audio transformer for keyword spotting",
         fontsize=12.5, color=MUTED, va="top")
fig.text(0.967, 0.905, "Google Speech Commands v2", fontsize=12.5,
         fontweight="bold", color=TEXT, ha="right", va="top")
fig.text(0.967, 0.858, "105,829 clips  ·  35 words  ·  16 kHz mono",
         fontsize=10, color=MUTED, ha="right", va="top", fontfamily=MONO)

Y, HH = 0.345, 0.430
wave = (0.033, Y, 0.135, HH)
mel_r = (0.192, Y, 0.146, HH)
tok = (0.362, Y, 0.100, HH)
enc = (0.486, Y, 0.158, HH)
prd = (0.668, Y, 0.299, HH)
for r in (wave, mel_r, tok, enc, prd):
    panel(*r, ec=DIM, lw=0.8)

for x0, x1 in ((0.169, 0.190), (0.339, 0.360), (0.463, 0.484), (0.645, 0.666)):
    arrow(x0, x1, 0.560)

stage(wave[0], wave[2], "01  RAW AUDIO", "16 kHz · 1.0 s")
stage(mel_r[0], mel_r[2], "02  LOG-MEL", "80 mels × 101")
stage(tok[0], tok[2], "03  TOKENS", "101 + CLS · d=128")
stage(enc[0], enc[2], "04  ENCODER", "3 layers · 4 heads")
stage(prd[0], prd[2], "05  PREDICTION", "12-class softmax")

ax = fig.add_axes([0.044, 0.405, 0.113, 0.310], facecolor="none")
t = np.arange(16000) / 16000
ax.plot(t, wav, color=CYAN, lw=0.55, alpha=0.95)
ax.fill_between(t, wav, 0, color=CYAN, alpha=0.10)
ax.set_xlim(0, 1)
ax.axis("off")

ax = fig.add_axes([0.200, 0.373, 0.130, 0.374], facecolor="none")
ax.imshow(mel, aspect="auto", origin="lower", cmap="magma")
ax.set_xticks([]); ax.set_yticks([])
for s in ax.spines.values():
    s.set_color(DIM); s.set_linewidth(1.0)

ax = fig.add_axes([0.367, 0.373, 0.090, 0.374], facecolor="none")
ax.imshow(emb, aspect="auto", origin="lower", cmap="viridis", interpolation="nearest")
ax.set_xticks([]); ax.set_yticks([])
for s in ax.spines.values():
    s.set_color(DIM); s.set_linewidth(1.0)

# encoder: three stacked cards + real layer-1 attention
for i, (dx, col) in enumerate(((0.010, "#141D36"), (0.005, "#182242"), (0.0, "#1B2547"))):
    panel(enc[0] + dx, enc[1] + dx, enc[2] - dx - 0.004, enc[3] - dx - 0.004,
          color=col, ec=VIOLET, lw=0.9, z=2 + i)

ax = fig.add_axes([0.494, 0.396, 0.140, 0.296], facecolor="none", zorder=5)
ax.imshow(attn, aspect="auto", cmap="magma", origin="lower")
ax.set_xticks([]); ax.set_yticks([])
for s in ax.spines.values():
    s.set_color("#2A3A66"); s.set_linewidth(0.8)
fig.text(enc[0] + 0.011, 0.726, "layer-1 attention", fontsize=7.6,
         color="#C4B5FD", fontfamily=MONO, zorder=6)
fig.text(enc[0] + enc[2] - 0.011, 0.726, "×3", fontsize=12.5, fontweight="bold",
         color=VIOLET, ha="right", zorder=6)

pred_idx = int(probs.argmax())
order = np.argsort(probs)[::-1]
fig.text(prd[0] + 0.018, 0.740, classes[pred_idx], fontsize=19, fontweight="bold",
         color=GREEN, va="top")
fig.text(prd[0] + 0.018, 0.706, f"confidence {probs[pred_idx] * 100:.1f}%",
         fontsize=8.5, color=MUTED, fontfamily=MONO, va="top")

ax = fig.add_axes([0.695, 0.368, 0.261, 0.300], facecolor="none")
ypos = np.arange(len(order))[::-1]
ax.barh(ypos, probs[order] * 100, color=[GREEN if i == pred_idx else "#2F4A7A" for i in order],
        height=0.62, alpha=0.95)
ax.set_yticks(ypos, [classes[i] for i in order], fontsize=8.1, color="#B6C2D4",
              fontfamily=MONO)
for lbl, i in zip(ax.get_yticklabels(), order):
    if i == pred_idx:
        lbl.set_color(GREEN)
        lbl.set_fontweight("bold")
ax.set_xlim(0, 104)
ax.set_xticks([])
for s in ax.spines.values():
    s.set_visible(False)
ax.tick_params(axis="y", length=0, pad=4)
for y, i in zip(ypos, order):
    if probs[i] * 100 >= 1:
        ax.text(probs[i] * 100 + 2.2, y, f"{probs[i] * 100:.0f}", va="center",
                fontsize=6.8, color=MUTED, fontfamily=MONO)

cards = [
    ("TEST ACCURACY", "93.34%", "12-class held-out set", GREEN),
    ("PARAMETERS", "620,172", "3 layers · d=128 · 4 heads", CYAN),
    ("TRAIN TIME", "1.9 min", "60 epochs · batch 256", AMBER),
    ("GPU", "RTX 3080 Ti", "CUDA 13.0 · bf16 autocast", VIOLET),
]
cw, gap, cy, ch = 0.214, 0.0155, 0.088, 0.160
for i, (label, value, sub, color) in enumerate(cards):
    x = 0.033 + i * (cw + gap)
    panel(x, cy, cw, ch, r=0.010)
    fig.patches.append(Rectangle((x + 0.008, cy + 0.020), 0.0028, ch - 0.040,
                                 facecolor=color, edgecolor="none", zorder=3))
    fig.text(x + 0.020, cy + ch - 0.030, label, fontsize=8.2, color=color,
             fontfamily=MONO, va="top")
    fig.text(x + 0.020, cy + 0.062, value,
             fontsize=20 if len(value) < 10 else 17, fontweight="bold",
             color=TEXT, va="bottom")
    fig.text(x + 0.020, cy + 0.024, sub, fontsize=7.8, color=MUTED,
             fontfamily=MONO, va="bottom")

fig.text(0.033, 0.038,
         "log-mel features cached to disk  ·  SpecAugment + time shift  ·  AdamW + cosine schedule",
         fontsize=8, color="#5C6B82", fontfamily=MONO)
fig.text(0.967, 0.038, "torch 2.14  ·  torchaudio 2.11  ·  trained on-device",
         fontsize=8, color="#5C6B82", fontfamily=MONO, ha="right")

out = Path(__file__).parent / "hero.png"
fig.savefig(out, dpi=170, facecolor=BG_BOT)
print(f"saved {out}")
