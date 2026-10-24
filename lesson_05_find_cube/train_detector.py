"""Lesson 05, step 2: train a small CNN to find the red cube in the camera image. See lesson.md.

Pure PyTorch - no Isaac Sim needed. Reads data\\cube_dataset.npz from generate_data.py.

Run from Learning\\Robotics:
    .\\run.bat lesson_05_find_cube\\train_detector.py
    .\\run.bat lesson_05_find_cube\\train_detector.py --epochs 10   # quicker
Outputs: models\\cube_detector.pt, output\\predictions.png, output\\training_curve.png
"""
import argparse
import os
import sys
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from cube_finder import CubeFinder  # the network (separate file so lesson 06 can reuse it)
parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=40)
parser.add_argument("--batch", type=int, default=64)
parser.add_argument("--lr", type=float, default=1e-3)
args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)

# ---------------- data ----------------
data = np.load(os.path.join(HERE, "data", "cube_dataset.npz"))
rgb, depth, pixel, position = data["rgb"], data["depth"].astype(np.float32), data["pixel"], data["position"]
K, V, cube_size = data["K"], data["V"], float(data["cube_size"])
N, H, W, _ = rgb.shape
print(f"{N} images of {W}x{H}, training on {device}")

idx = np.random.default_rng(0).permutation(N)
n_val = N // 5
val_idx, train_idx = idx[:n_val], idx[n_val:]

images = torch.from_numpy(rgb).permute(0, 3, 1, 2).float().div(255)  # (N, 3, H, W) in [0, 1]
targets = torch.from_numpy(pixel)  # (N, 2) true (u, v) in pixels


# ---------------- model (see cube_finder.py) ----------------
model = CubeFinder().to(device)
opt = torch.optim.Adam(model.parameters(), lr=args.lr)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
print(f"model parameters: {sum(p.numel() for p in model.parameters()):,}")


def augment(x):
    # brightness/contrast jitter: a cheap form of domain randomisation (lighting changes)
    gain = torch.empty(x.shape[0], 1, 1, 1, device=x.device).uniform_(0.7, 1.3)
    bias = torch.empty(x.shape[0], 1, 1, 1, device=x.device).uniform_(-0.1, 0.1)
    return (x * gain + bias).clamp(0, 1)


def evaluate(ids):
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(ids), 256):
            p, _ = model(images[ids[i : i + 256]].to(device))
            preds.append(p.cpu())
    return torch.cat(preds).numpy()


# ---------------- train ----------------
history = []
start = time.time()
for epoch in range(1, args.epochs + 1):
    model.train()
    perm = np.random.permutation(train_idx)
    losses = []
    for i in range(0, len(perm), args.batch):
        b = perm[i : i + args.batch]
        x, y = augment(images[b].to(device)), targets[b].to(device)
        pred, _ = model(x)
        loss = F.smooth_l1_loss(pred, y)  # error in pixels
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    sched.step()
    val_err = np.linalg.norm(evaluate(val_idx) - pixel[val_idx], axis=1)
    history.append((np.mean(losses), val_err.mean()))
    if epoch == 1 or epoch % 5 == 0 or epoch == args.epochs:
        print(f"epoch {epoch:3d}  train loss {np.mean(losses):6.3f}  val pixel error: mean {val_err.mean():.2f} px, "
              f"median {np.median(val_err):.2f} px  ({time.time() - start:.0f} s)")


# ---------------- pixel + depth -> 3D (lesson 04, done by hand) ----------------
K_inv, V_inv = np.linalg.inv(K), np.linalg.inv(V)


def pixel_to_world(u, v, depth_img):
    """Back-project a pixel using the depth image, the lens (K) and the camera pose (V)."""
    r, c = int(round(v)), int(round(u))
    win = depth_img[max(r - 1, 0) : r + 2, max(c - 1, 0) : c + 2]  # 3x3 median: robust to one bad pixel
    d = float(np.median(win))
    p_cam = d * (K_inv @ np.array([u, v, 1.0]))  # point in camera frame (ROS axes: z forward)
    return (V_inv @ np.append(p_cam, 1.0))[:3]  # camera frame -> world frame


pred_val = evaluate(val_idx)
rows = []
for k, i in enumerate(val_idx):
    true_world = position[i]
    from_true_px = pixel_to_world(*pixel[i], depth[i])  # best case: perfect detector
    from_pred_px = pixel_to_world(*pred_val[k], depth[i])  # our detector
    corrected = from_pred_px.copy()
    corrected[2] = cube_size / 2  # prior knowledge: the cube rests on the floor
    rows.append([np.linalg.norm(from_true_px - true_world), np.linalg.norm(from_pred_px - true_world),
                 np.linalg.norm(corrected - true_world)])
rows = np.array(rows) * 100  # cm
pix_err = np.linalg.norm(pred_val - pixel[val_idx], axis=1)
print("\n=== validation results ===")
print(f"pixel error (detector):                 mean {pix_err.mean():.2f} px   95% < {np.percentile(pix_err, 95):.2f} px")
print(f"3D error, true pixel + depth:           mean {rows[:, 0].mean():.2f} cm  (limit: depth sees the top face)")
print(f"3D error, predicted pixel + depth:      mean {rows[:, 1].mean():.2f} cm")
print(f"3D error, predicted + 'on the floor':   mean {rows[:, 2].mean():.2f} cm   95% < {np.percentile(rows[:, 2], 95):.2f} cm")

# ---------------- save ----------------
os.makedirs(os.path.join(HERE, "models"), exist_ok=True)
os.makedirs(os.path.join(HERE, "output"), exist_ok=True)
torch.save({"state_dict": model.state_dict(), "K": K, "V": V, "resolution": (W, H)},
           os.path.join(HERE, "models", "cube_detector.pt"))

hist = np.array(history)
fig, ax = plt.subplots(figsize=(6, 3.5))
ax.plot(range(1, len(hist) + 1), hist[:, 1], label="val pixel error (mean)")
ax.set_xlabel("epoch"), ax.set_ylabel("pixels"), ax.set_yscale("log"), ax.grid(True, alpha=0.3), ax.legend()
fig.tight_layout(), fig.savefig(os.path.join(HERE, "output", "training_curve.png"), dpi=100), plt.close(fig)

show = val_idx[:8]
model.eval()
with torch.no_grad():
    pred, prob = model(images[show].to(device))
fig, axes = plt.subplots(2, 8, figsize=(16, 4))
for j, i in enumerate(show):
    axes[0, j].imshow(rgb[i])
    axes[0, j].plot(*pixel[i], "o", mfc="none", mec="lime", ms=10)
    axes[0, j].plot(*pred[j].cpu().numpy(), "x", color="cyan", ms=8, mew=2)
    axes[0, j].set_title(f"err {np.linalg.norm(pred[j].cpu().numpy() - pixel[i]):.1f}px", fontsize=8)
    axes[1, j].imshow(prob[j].cpu().numpy(), cmap="magma")
    for a in axes[:, j]:
        a.axis("off")
axes[0, 0].set_ylabel("image"), axes[1, 0].set_ylabel("heatmap")
fig.suptitle("green o = true cube centre, cyan x = predicted; bottom = network heatmap")
fig.tight_layout(), fig.savefig(os.path.join(HERE, "output", "predictions.png"), dpi=90), plt.close(fig)
print("\nsaved models\\cube_detector.pt, output\\predictions.png, output\\training_curve.png")
