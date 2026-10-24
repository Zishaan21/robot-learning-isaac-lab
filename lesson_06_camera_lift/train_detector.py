"""Lesson 06, step 2: retrain lesson 05's CubeFinder on images from the real lift scene. See lesson.md.

Pure PyTorch. Reads data\\lift_camera_dataset.npz from collect_data.py.

Run from Learning\\Robotics:
    .\\run.bat lesson_06_camera_lift\\train_detector.py
Outputs: models\\lift_cube_detector.pt, output\\predictions.png
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
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "lesson_05_find_cube"))
from cube_finder import CubeFinder  # same network as lesson 05

parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=40)
parser.add_argument("--batch", type=int, default=64)
parser.add_argument("--lr", type=float, default=1e-3)
args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)

data = np.load(os.path.join(HERE, "data", "lift_camera_dataset.npz"))
rgb, depth = data["rgb"], data["depth"].astype(np.float32)
pixel, zc, zs, K = data["pixel"], data["centre_depth"], data["surface_depth"], data["K"]
N, H, W, _ = rgb.shape
print(f"{N} images of {W}x{H}, training on {device}")

idx = np.random.default_rng(0).permutation(N)
n_val = N // 5
val_idx, train_idx = idx[:n_val], idx[n_val:]

# Depth calibration: the depth camera sees the cube's SURFACE; the policy needs its CENTRE.
# Measure the average gap on the training set and add it back later (a learned constant).
depth_offset = float(np.median(zc[train_idx] - zs[train_idx]))
print(f"depth offset (centre - surface), calibrated on training images: {depth_offset * 100:.2f} cm")

images = torch.from_numpy(rgb).permute(0, 3, 1, 2).float().div(255)
targets = torch.from_numpy(pixel)

model = CubeFinder().to(device)
opt = torch.optim.Adam(model.parameters(), lr=args.lr)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)


def augment(x):
    gain = torch.empty(x.shape[0], 1, 1, 1, device=x.device).uniform_(0.7, 1.3)
    bias = torch.empty(x.shape[0], 1, 1, 1, device=x.device).uniform_(-0.1, 0.1)
    return (x * gain + bias).clamp(0, 1)


def evaluate(ids):
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(ids), 256):
            out.append(model(images[ids[i : i + 256]].to(device))[0].cpu())
    return torch.cat(out).numpy()


start = time.time()
for epoch in range(1, args.epochs + 1):
    model.train()
    perm = np.random.permutation(train_idx)
    losses = []
    for i in range(0, len(perm), args.batch):
        b = perm[i : i + args.batch]
        pred, _ = model(augment(images[b].to(device)))
        loss = F.smooth_l1_loss(pred, targets[b].to(device))
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    sched.step()
    if epoch == 1 or epoch % 5 == 0 or epoch == args.epochs:
        err = np.linalg.norm(evaluate(val_idx) - pixel[val_idx], axis=1)
        print(f"epoch {epoch:3d}  train loss {np.mean(losses):6.3f}  val pixel error: mean {err.mean():.2f} px, "
              f"median {np.median(err):.2f} px  ({time.time() - start:.0f} s)")

# ---- 3D error in the camera frame (frame-independent, so we don't need each env's pose) ----
K_inv = np.linalg.inv(K)


def depth_at(img, u, v):
    r, c = int(round(v)), int(round(u))
    win = img[max(r - 1, 0) : r + 2, max(c - 1, 0) : c + 2]
    return float(np.median(np.nan_to_num(win, nan=100.0, posinf=100.0)))


pred = evaluate(val_idx)
errs = []
for k, i in enumerate(val_idx):
    true_p = zc[i] * (K_inv @ np.array([*pixel[i], 1.0]))
    z_hat = depth_at(depth[i], *pred[k]) + depth_offset
    est_p = z_hat * (K_inv @ np.array([*pred[k], 1.0]))
    errs.append(np.linalg.norm(est_p - true_p))
errs = np.array(errs) * 100
pix_err = np.linalg.norm(pred - pixel[val_idx], axis=1)
print("\n=== validation results ===")
print(f"pixel error:       mean {pix_err.mean():.2f} px   95% < {np.percentile(pix_err, 95):.2f} px")
print(f"3D position error: mean {errs.mean():.2f} cm   median {np.median(errs):.2f} cm   95% < {np.percentile(errs, 95):.2f} cm")
print(f"                   worst 1%: > {np.percentile(errs, 99):.1f} cm  (see the WORST images in output\\predictions.png)")

os.makedirs(os.path.join(HERE, "models"), exist_ok=True)
os.makedirs(os.path.join(HERE, "output"), exist_ok=True)
torch.save({"state_dict": model.state_dict(), "depth_offset": depth_offset, "resolution": (W, H)},
           os.path.join(HERE, "models", "lift_cube_detector.pt"))

worst = val_idx[np.argsort(errs)[-4:]]
show = np.concatenate([val_idx[:8], worst])
model.eval()
with torch.no_grad():
    p, prob = model(images[show].to(device))
fig, axes = plt.subplots(2, 12, figsize=(22, 4))
for j, i in enumerate(show):
    axes[0, j].imshow(rgb[i])
    axes[0, j].plot(*pixel[i], "o", mfc="none", mec="lime", ms=10)
    axes[0, j].plot(*p[j].cpu().numpy(), "x", color="red", ms=8, mew=2)
    axes[0, j].set_title(("WORST " if j >= 8 else "") + f"{np.linalg.norm(p[j].cpu().numpy() - pixel[i]):.1f}px",
                         fontsize=8)
    axes[1, j].imshow(prob[j].cpu().numpy(), cmap="magma")
    for a in axes[:, j]:
        a.axis("off")
fig.suptitle("green o = true cube centre, red x = predicted; bottom = heatmap. Last 4 = worst validation images")
fig.tight_layout()
fig.savefig(os.path.join(HERE, "output", "predictions.png"), dpi=80)
plt.close(fig)
print("\nsaved models\\lift_cube_detector.pt and output\\predictions.png")
