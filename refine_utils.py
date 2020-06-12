import numpy as np
import torch

voxsize = 32

#camera parameters
F_MM = 35.  # Focal length
PIXEL_ASPECT_RATIO = 1.
CAM_MAX_DIST = 1.75
SENSOR_SIZE_MM = 32.
RESOLUTION_PCT = 100.

img_w = 137
img_h = 137

scale = RESOLUTION_PCT / 100
f_u = F_MM * img_w * scale / SENSOR_SIZE_MM
f_v = F_MM * img_h * scale * PIXEL_ASPECT_RATIO / SENSOR_SIZE_MM
u_0 = img_w * scale / 2
v_0 = img_h * scale / 2

switch_axis = np.array([[1, 0, 0, 0],
                        [0, 0, 1, 0],
                        [0, -1, 0, 0],
                        [0, 0, 0, 1]])

K = np.array([[ f_u    ,  0.    , u_0    ,  0.    ],
              [ 0.     ,  f_v   , v_0    ,  0.    ],
              [ 0.     ,  0.    , 1      ,  0     ],
              [ 0.     ,  0.    , 0.     ,  1.    ]])

K_inv = np.linalg.inv(K)


# https://github.com/darylclimb/cvml_project/blob/master/projections/inverse_projection/geometry_utils.py

def pixel_coord_np(width, height):
    """
    Pixel in homogenous coordinate
    Returns:
        Pixel coordinate:       [3, width * height]
    """
    x = np.linspace(0, width - 1, width).astype(np.int)
    y = np.linspace(0, height - 1, height).astype(np.int)
    [x, y] = np.meshgrid(x, y)
    return np.vstack((x.flatten(), y.flatten(), np.ones_like(x.flatten())))
    
pixel_coords = pixel_coord_np(img_w, img_h)

# compute radius of minimum sphere
def get_radius(depth_img_raw, distance, vox_size = 32):
    
    cam_coords = K_inv[:3, :3] @ pixel_coords * depth_img_raw.flatten().numpy()
    cam_coords = cam_coords[:, np.where(np.logical_and(cam_coords[2] > 0.15/1.15, cam_coords[2] < 1))[0]]
    cam_coords[2] -= distance/1.15
    cam_coords = switch_axis[:3, :3] @ cam_coords
    
    radius = np.max(np.linalg.norm(cam_coords, axis = 0))
    
    return radius
    
# compute depth close index
def get_depth_close_idx(depth_img_raw, distance, radius, vox_size = 32):
    
    pixel_coords = pixel_coord_np(img_w, img_h)
    
    cam_coords = K_inv[:3, :3] @ pixel_coords * depth_img_raw.flatten().numpy()
    cam_coords = cam_coords[:, np.where(np.logical_and(cam_coords[2] > 0.15/1.15, cam_coords[2] < 1))[0]]
    cam_coords[2] -= distance/1.15
    cam_coords = switch_axis[:3, :3] @ cam_coords
    
    
    new_coords = ((cam_coords/radius/2 + 0.5)*vox_size).astype(int)
    new_coords = np.clip(new_coords, 0, vox_size-1)
    
    projected_vox = torch.zeros(vox_size, vox_size, vox_size)
    projected_vox[new_coords[0], new_coords[1], new_coords[2]] = 1
    
    projected_depth = torch.argmax(projected_vox*torch.arange(voxsize-1, -1, -1).unsqueeze(0).unsqueeze(2).repeat(32, 1, 32), 1)
    projected_depth[projected_depth == 31] = 0
    
    close_index = torch.zeros((vox_size, vox_size, vox_size)).byte()
    tmp = torch.t(projected_depth[torch.sum(projected_vox, 1).bool()].expand(vox_size, -1)) > torch.arange(0, vox_size)
    close_index[torch.sum(projected_vox, 1).bool()] = tmp.byte()
    
    close_index = torch.transpose(close_index, 1, 2)
    
    return projected_vox, close_index
    
