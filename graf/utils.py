import numpy as np
import torch
import imageio
import os


def get_nsamples(data_loader, N):
  x = []
  x_pose = []
  n = 0
  while n < N:
    x_next, pose_next = next(iter(data_loader))
    x.append(x_next)
    x_pose.append(pose_next)
    n += x_next.size(0)
  x = torch.cat(x, dim=0)[:N]
  x_pose = torch.cat(x_pose, dim=0)[:N]
  return x, x_pose


def count_trainable_parameters(model):
  model_parameters = filter(lambda p: p.requires_grad, model.parameters())
  return sum([np.prod(p.size()) for p in model_parameters])


def save_video(imgs, fname, as_gif=False, fps=24, quality=8):
    # convert to np.uint8
    imgs = (255 * np.clip(imgs.permute(0, 2, 3, 1).detach().cpu().numpy() / 2 + 0.5, 0, 1)).astype(np.uint8)
    imageio.mimwrite(fname, imgs, fps=fps, quality=quality)
    
    if as_gif:  # save as gif, too
        os.system(f'ffmpeg -i {fname} -r 15 '
                  f'-vf "scale=512:-1,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse" {os.path.splitext(fname)[0] + ".gif"}')


def color_depth_map(depths, scale=None):
    """
    Color an input depth map.

    Arguments:
        depths -- HxW numpy array of depths
        [scale=None] -- scaling the values (defaults to the maximum depth)

    Returns:
        colored_depths -- HxWx3 numpy array visualizing the depths
    """

    _color_map_depths = np.array([
      [0, 0, 0],  # 0.000
      [0, 0, 255],  # 0.114
      [255, 0, 0],  # 0.299
      [255, 0, 255],  # 0.413
      [0, 255, 0],  # 0.587
      [0, 255, 255],  # 0.701
      [255, 255, 0],  # 0.886
      [255, 255, 255],  # 1.000
      [255, 255, 255],  # 1.000
    ]).astype(float)
    _color_map_bincenters = np.array([
      0.0,
      0.114,
      0.299,
      0.413,
      0.587,
      0.701,
      0.886,
      1.000,
      2.000,  # doesn't make a difference, just strictly higher than 1
    ])
  
    if scale is None:
      scale = depths.max()
  
    values = np.clip(depths.flatten() / scale, 0, 1)
    # for each value, figure out where they fit in in the bincenters: what is the last bincenter smaller than this value?
    lower_bin = ((values.reshape(-1, 1) >= _color_map_bincenters.reshape(1, -1)) * np.arange(0, 9)).max(axis=1)
    lower_bin_value = _color_map_bincenters[lower_bin]
    higher_bin_value = _color_map_bincenters[lower_bin + 1]
    alphas = (values - lower_bin_value) / (higher_bin_value - lower_bin_value)
    colors = _color_map_depths[lower_bin] * (1 - alphas).reshape(-1, 1) + _color_map_depths[
      lower_bin + 1] * alphas.reshape(-1, 1)
    return colors.reshape(depths.shape[0], depths.shape[1], 3).astype(np.uint8)


# Virtual camera utils


def to_sphere(u, v):
    theta = 2 * np.pi * u
    phi = np.arccos(1 - 2 * v)
    cx = np.sin(phi) * np.cos(theta)
    cy = np.sin(phi) * np.sin(theta)
    cz = np.cos(phi)
    s = np.stack([cx, cy, cz])
    return s


def polar_to_cartesian(r, theta, phi, deg=True):
    if deg:
        phi = phi * np.pi / 180
        theta = theta * np.pi / 180
    cx = np.sin(phi) * np.cos(theta)
    cy = np.sin(phi) * np.sin(theta)
    cz = np.cos(phi)
    return r * np.stack([cx, cy, cz])


def to_uv(loc):
    # normalize to unit sphere
    loc = loc / loc.norm(dim=1, keepdim=True)

    cx, cy, cz = loc.t()
    v = (1 - cz) / 2

    phi = torch.acos(cz)
    sin_phi = torch.sin(phi)

    # ensure we do not divide by zero
    eps = 1e-8
    sin_phi[sin_phi.abs() < eps] = eps

    theta = torch.acos(cx / sin_phi)

    # check for sign of phi
    cx_rec = sin_phi * torch.cos(theta)
    if not np.isclose(cx.numpy(), cx_rec.numpy(), atol=1e-5).all():
        sin_phi = -sin_phi

    # check for sign of theta
    cy_rec = sin_phi * torch.sin(theta)
    if not np.isclose(cy.numpy(), cy_rec.numpy(), atol=1e-5).all():
        theta = -theta

    u = theta / (2 * np.pi)
    assert np.isclose(to_sphere(u, v).detach().cpu().numpy(), loc.t().detach().cpu().numpy(), atol=1e-5).all()

    return u, v


def to_phi(u):
    return 360 * u  # 2*pi*u*180/pi


def to_theta(v):
    return np.arccos(1 - 2 * v) * 180. / np.pi


def sample_on_sphere(range_u=(0, 1), range_v=(0, 1)):
    u = np.random.uniform(*range_u)
    v = np.random.uniform(*range_v)
    return to_sphere(u, v)


def look_at(eye, at=np.array([0, 0, 0]), up=np.array([0, 0, 1]), eps=1e-5):
    at = at.astype(float).reshape(1, 3)
    up = up.astype(float).reshape(1, 3)

    eye = eye.reshape(-1, 3)
    up = up.repeat(eye.shape[0] // up.shape[0], axis=0)
    eps = np.array([eps]).reshape(1, 1).repeat(up.shape[0], axis=0)

    z_axis = eye - at   # (1, 3)
    z_axis /= np.max(np.stack([np.linalg.norm(z_axis, axis=1, keepdims=True), eps]))  # (2, 1, 1)

    x_axis = np.cross(up, z_axis) # (1, 3)
    x_axis /= np.max(np.stack([np.linalg.norm(x_axis, axis=1, keepdims=True), eps]))

    y_axis = np.cross(z_axis, x_axis)
    y_axis /= np.max(np.stack([np.linalg.norm(y_axis, axis=1, keepdims=True), eps]))

    r_mat = np.concatenate((x_axis.reshape(-1, 3, 1), y_axis.reshape(-1, 3, 1), z_axis.reshape(-1, 3, 1)), axis=2)

    return r_mat


def look_at_torch_single(eye, at=np.array([0, 0, 0]), up=np.array([0, 0, 1]), eps=1e-5):
    at = torch.from_numpy(at.astype(float).reshape(1, 3)).to(eye.device)
    up = torch.from_numpy(up.astype(float).reshape(1, 3)).to(eye.device)

    eye = eye.reshape(-1, 3)
    up = up.repeat(eye.shape[0] // up.shape[0], 1)
    eps = torch.from_numpy(np.array([eps]).reshape(1, 1).repeat(up.shape[0], 1)).to(eye.device)

    z_axis = eye - at

    new_z_axis = z_axis / torch.max(torch.stack([torch.norm(z_axis, dim=1).unsqueeze(1), eps]))

    x_axis = torch.cross(up, z_axis)
    new_x_axis = x_axis / torch.max(torch.stack([torch.norm(x_axis, dim=1).unsqueeze(1), eps]))

    y_axis = torch.cross(z_axis, x_axis)
    new_y_axis = y_axis / torch.max(torch.stack([torch.norm(y_axis, dim=1).unsqueeze(1), eps]))

    r_mat = torch.cat((new_x_axis.reshape(-1, 3, 1), new_y_axis.reshape(-1, 3, 1), new_z_axis.reshape(-1, 3, 1)), dim=2)

    return r_mat

def look_at_torch(eye, at=np.array([0, 0, 0]), up=np.array([0, 0, 1]), eps=1e-5):
    bs = len(eye)       # eye: (B, 3)
    at = torch.from_numpy(at.astype(float).reshape(1, 3)).to(eye.device).repeat(bs, 1)
    up = torch.from_numpy(up.astype(float).reshape(1, 3)).to(eye.device)

    eye = eye.reshape(bs, -1)
    up = up.repeat(eye.shape[0] // up.shape[0], 1)
    eps = torch.from_numpy(np.array([eps]).reshape(1, 1).repeat(up.shape[0])).to(eye.device).unsqueeze(1)

    z_axis = eye - at

    new_z_axis = z_axis / torch.max(torch.stack([torch.norm(z_axis, dim=1).unsqueeze(1), eps]))

    x_axis = torch.cross(up, z_axis)
    new_x_axis = x_axis / torch.max(torch.stack([torch.norm(x_axis, dim=1).unsqueeze(1), eps]))

    y_axis = torch.cross(z_axis, x_axis)
    new_y_axis = y_axis / torch.max(torch.stack([torch.norm(y_axis, dim=1).unsqueeze(1), eps]))

    r_mat = torch.cat((new_x_axis.reshape(-1, 3, 1), new_y_axis.reshape(-1, 3, 1), new_z_axis.reshape(-1, 3, 1)), dim=2)

    return r_mat