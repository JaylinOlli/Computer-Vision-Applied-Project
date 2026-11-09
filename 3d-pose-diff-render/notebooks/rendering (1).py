"""
Differentiable rendering engine for SMPL pose refinement.

Provides:
  - make_cameras:             3DPW / OpenCV camera params -> PyTorch3D cameras
  - make_silhouette_renderer: soft silhouette renderer (differentiable)
  - render_silhouette:        verts + faces -> (B, H, W) soft mask in [0, 1]
  - project_points:           3D joints -> 2D pixels (OpenCV convention, pure torch)
  - silhouette_loss:          rendered mask vs. target person mask
  - reprojection_loss:        projected joints vs. 2D keypoints (confidence-weighted)

Conventions (agree on these with the team):
  verts   : (B, V, 3) float, world coordinates, meters
  faces   : (B, F, 3) long  (or (F, 3), broadcast to the batch)
  joints  : (B, J, 3) float, world coordinates, meters
  R, t, K : (B, 3, 3), (B, 3), (B, 3, 3) in OpenCV convention (x_cam = R @ x_world + t)
  masks   : (B, H, W) float in [0, 1]
  kps_2d  : (B, J, 2) pixel coords (x right, y down), same resolution as the render
  conf    : (B, J) per-keypoint confidence / visibility weights
"""

import math

import torch
from pytorch3d.renderer import (
    BlendParams,
    MeshRasterizer,
    MeshRenderer,
    RasterizationSettings,
    SoftSilhouetteShader,
)
from pytorch3d.structures import Meshes
from pytorch3d.utils import cameras_from_opencv_projection


# --------------------------------------------------------------------------- #
# Cameras
# --------------------------------------------------------------------------- #
def scale_intrinsics(K: torch.Tensor, scale: float) -> torch.Tensor:
    """Rescale intrinsics when rendering at lower resolution (e.g. 1920x1080 -> 480x270)."""
    K = K.clone()
    K[:, :2, :] *= scale
    return K


def make_cameras(R, t, K, image_size_hw):
    """
    Build PyTorch3D cameras from OpenCV-style params (what 3DPW provides).

    image_size_hw: (B, 2) tensor of (height, width) in pixels.
    Handles PyTorch3D's flipped axis convention (+X left, +Y up) for you.
    """
    return cameras_from_opencv_projection(R=R, tvec=t, camera_matrix=K, image_size=image_size_hw)


# --------------------------------------------------------------------------- #
# Renderer
# --------------------------------------------------------------------------- #
def make_silhouette_renderer(cameras, image_size_hw, sigma=1e-4, faces_per_pixel=50):
    """
    Soft silhouette renderer.

    sigma controls edge softness: larger -> blurrier edges -> gradients reach
    further from the boundary (good for bad initializations), but less precise.
    """
    blend = BlendParams(sigma=sigma, gamma=1e-4)
    raster = RasterizationSettings(
        image_size=tuple(int(x) for x in image_size_hw),  # (H, W)
        blur_radius=math.log(1.0 / 1e-4 - 1.0) * sigma,
        faces_per_pixel=faces_per_pixel,
    )
    return MeshRenderer(
        rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster),
        shader=SoftSilhouetteShader(blend_params=blend),
    )


def render_silhouette(renderer, verts, faces):
    """Render a soft mask. Returns (B, H, W) in [0, 1]; differentiable w.r.t. verts."""
    if faces.dim() == 2:
        faces = faces[None].expand(verts.shape[0], -1, -1)
    meshes = Meshes(verts=verts, faces=faces)
    images = renderer(meshes)  # (B, H, W, 4); alpha channel holds the silhouette
    return images[..., 3]


# --------------------------------------------------------------------------- #
# Projection (pure torch, OpenCV convention -> no axis-flip confusion)
# --------------------------------------------------------------------------- #
def project_points(points_3d, R, t, K, eps=1e-6):
    """Project world points (B, J, 3) to pixels (B, J, 2) with x_cam = R x + t, uv = K x_cam / z."""
    cam = points_3d @ R.transpose(1, 2) + t[:, None, :]
    z = cam[..., 2:].clamp(min=eps)
    uv1 = torch.cat([cam[..., :2] / z, torch.ones_like(z)], dim=-1)
    return (uv1 @ K.transpose(1, 2))[..., :2]


# --------------------------------------------------------------------------- #
# Losses
# --------------------------------------------------------------------------- #
def silhouette_loss(pred_sil, target_mask, mode="l2"):
    """
    pred_sil, target_mask: (B, H, W) in [0, 1].
    mode="l2"  : mean squared pixel difference (smooth, standard in PyTorch3D tutorials)
    mode="iou" : 1 - soft IoU (insensitive to how big the person is in the frame)
    """
    if mode == "l2":
        return ((pred_sil - target_mask) ** 2).mean()
    if mode == "iou":
        inter = (pred_sil * target_mask).sum(dim=(1, 2))
        union = (pred_sil + target_mask - pred_sil * target_mask).sum(dim=(1, 2))
        return (1.0 - inter / (union + 1e-6)).mean()
    raise ValueError(f"unknown mode: {mode}")


def gmof(sq_residual, rho):
    """Geman-McClure robustifier (as in SMPLify): caps the influence of outlier keypoints."""
    return (rho ** 2) * sq_residual / (sq_residual + rho ** 2)


def reprojection_loss(proj_2d, kps_2d, conf, norm=1.0, rho=None):
    """
    Confidence-weighted 2D keypoint loss.

    proj_2d, kps_2d: (B, J, 2) pixels; conf: (B, J).
    norm: divide residuals by this (e.g. image width) so the loss scale is
          comparable to the silhouette loss regardless of resolution.
    rho:  if set, apply Geman-McClure with this threshold (in normalized units).
    """
    sq = (((proj_2d - kps_2d) / norm) ** 2).sum(dim=-1)  # (B, J)
    if rho is not None:
        sq = gmof(sq, rho)
    return (conf * sq).sum() / conf.sum().clamp(min=1e-6)
