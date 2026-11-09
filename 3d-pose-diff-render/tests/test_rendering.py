"""
Gradient verification for src/rendering.py. No SMPL needed: uses a sphere.

Run in Colab (after installing PyTorch3D) from the repo root:
    !python tests/test_rendering.py

Checks:
  1. Gradients exist, are finite, and are non-zero for both losses.
  2. Finite-difference gradcheck on the reprojection loss (double precision).
  3. "Recovery test": render a sphere at a known position, start from a wrong
     position, and confirm optimization recovers the true position using
     silhouette-only, keypoint-only, and both.
"""

import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from pytorch3d.utils import ico_sphere  # noqa: E402

from src.rendering import (  # noqa: E402
    make_cameras,
    make_silhouette_renderer,
    project_points,
    render_silhouette,
    reprojection_loss,
    silhouette_loss,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
H = W = 256

# Simple pinhole camera at the origin looking down +Z (OpenCV convention).
K = torch.tensor([[[300.0, 0.0, W / 2], [0.0, 300.0, H / 2], [0.0, 0.0, 1.0]]], device=device)
R = torch.eye(3, device=device)[None]
t = torch.zeros(1, 3, device=device)
size_hw = torch.tensor([[H, W]], device=device)

renderer = make_silhouette_renderer(make_cameras(R, t, K, size_hw), (H, W))

sphere = ico_sphere(3, device)
base_verts = sphere.verts_padded() * 0.5  # radius 0.5 m
faces = sphere.faces_padded()
joint_idx = torch.arange(0, base_verts.shape[1], 40, device=device)  # fake "joints"

TRUE_OFFSET = torch.tensor([[0.3, -0.2, 4.0]], device=device)
with torch.no_grad():
    target_verts = base_verts + TRUE_OFFSET[:, None]
    target_mask = render_silhouette(renderer, target_verts, faces)
    target_kps = project_points(target_verts[:, joint_idx], R, t, K)
conf = torch.ones(target_kps.shape[:2], device=device)


def losses(offset):
    verts = base_verts + offset[:, None]
    l_sil = silhouette_loss(render_silhouette(renderer, verts, faces), target_mask)
    l_kp = reprojection_loss(project_points(verts[:, joint_idx], R, t, K), target_kps, conf, norm=W)
    return l_sil, l_kp


def check_gradients():
    offset = torch.tensor([[0.0, 0.0, 4.5]], device=device, requires_grad=True)
    for name, loss in zip(["silhouette", "reprojection"], losses(offset)):
        (g,) = torch.autograd.grad(loss, offset)
        assert torch.isfinite(g).all(), f"{name}: non-finite gradient"
        assert g.abs().sum() > 0, f"{name}: zero gradient"
        print(f"[ok] {name:12s} loss={loss.item():.5f} grad={g.squeeze().tolist()}")


def check_gradcheck():
    torch.manual_seed(0)
    pts = torch.randn(1, 5, 3, dtype=torch.double) + torch.tensor([0, 0, 4.0], dtype=torch.double)
    pts.requires_grad_(True)
    Kd, Rd, td = K.double().cpu(), R.double().cpu(), t.double().cpu()
    kps = torch.rand(1, 5, 2, dtype=torch.double) * W
    c = torch.ones(1, 5, dtype=torch.double)
    f = lambda p: reprojection_loss(project_points(p, Rd, td, Kd), kps, c, norm=W)  # noqa: E731
    assert torch.autograd.gradcheck(f, (pts,)), "gradcheck failed"
    print("[ok] reprojection gradcheck (finite differences match autograd)")


def recovery(use_sil, use_kp, steps=300, lr=0.02):
    offset = torch.tensor([[0.0, 0.0, 4.5]], device=device, requires_grad=True)
    opt = torch.optim.Adam([offset], lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        l_sil, l_kp = losses(offset)
        loss = use_sil * l_sil + use_kp * l_kp
        loss.backward()
        opt.step()
    err = (offset.detach() - TRUE_OFFSET).norm().item()
    label = "+".join(n for n, u in [("silhouette", use_sil), ("keypoints", use_kp)] if u)
    print(f"[{'ok' if err < 0.05 else 'FAIL'}] recovery with {label:22s} final error = {err:.4f} m")
    return offset.detach()


if __name__ == "__main__":
    print(f"device: {device}")
    check_gradients()
    check_gradcheck()
    recovery(1, 0)
    recovery(0, 1)
    final = recovery(1, 1)

    try:  # visual sanity check
        import matplotlib.pyplot as plt

        with torch.no_grad():
            sil = render_silhouette(renderer, base_verts + final[:, None], faces)
        fig, ax = plt.subplots(1, 2, figsize=(8, 4))
        ax[0].imshow(target_mask[0].cpu(), cmap="gray"); ax[0].set_title("target")
        ax[0].scatter(*target_kps[0].cpu().T, s=8, c="r")
        ax[1].imshow(sil[0].cpu(), cmap="gray"); ax[1].set_title("recovered")
        for a in ax:
            a.axis("off")
        plt.savefig("rendering_check.png", bbox_inches="tight")
        print("saved rendering_check.png (red keypoints should sit on the white sphere)")
    except ImportError:
        pass
