import numpy as np
import torch
import torch.nn.functional as F
import skimage.measure as sk

import time
import pyrender
import pymesh
import trimesh

from pyemd import emd_samples
import chamfer_python

import binvox_rw
from glob import glob

D2R = np.pi/180.0
voxsize = 32
sample_size = 2048

def RotatePhi(phi):
    
    return np.array([[1, 0, 0, 0], 
                     [0, np.cos(D2R*phi), np.sin(D2R*phi), 0], 
                     [0, -np.sin(D2R*phi), np.cos(D2R*phi), 0], 
                     [0, 0, 0, 1]])
                     
                     
def RotateAzimuth(phi):
    
    return np.array([[np.cos(D2R*phi), np.sin(D2R*phi), 0, 0], 
                     [-np.sin(D2R*phi), np.cos(D2R*phi), 0, 0], 
                     [0, 0, 1, 0], 
                     [0, 0, 0, 1]])
                     
                     
def RotateAlongAxis(theta, a, b, c):
    
    return np.array([[a**2*(1-np.cos(D2R*theta)) + np.cos(D2R*theta), a*b*(1-np.cos(D2R*theta)) - c*np.sin(D2R*theta), a*c*(1-np.cos(D2R*theta)) + b*np.sin(D2R*theta), 0], 
                     [a*b*(1-np.cos(D2R*theta)) + c*np.sin(D2R*theta), b**2*(1-np.cos(D2R*theta)) + np.cos(D2R*theta), b*c*(1-np.cos(D2R*theta)) - a*np.sin(D2R*theta), 0], 
                     [a*c*(1-np.cos(D2R*theta)) - b*np.sin(D2R*theta), b*c*(1-np.cos(D2R*theta)) + a*np.sin(D2R*theta), c**2*(1-np.cos(D2R*theta)) + np.cos(D2R*theta), 0], 
                     [0, 0, 0, 1]])
                     
               
# generate meshgrid           
# [depth, height, width]
def get_meshgrid(depth = voxsize, height = voxsize, width = voxsize, ratio = 1.0):
    x_mesh = np.repeat(np.repeat(np.linspace(-ratio, ratio, width)[np.newaxis, :], height, axis=0)[np.newaxis, :, :], depth, axis=0)
    y_mesh = np.repeat(np.repeat(np.linspace(-ratio, ratio, height)[:, np.newaxis], width, axis=-1)[np.newaxis, :, :], depth, axis=0)
    z_mesh = np.repeat(np.repeat(np.linspace(-ratio, ratio, depth)[:, np.newaxis], height, axis= -1)[:,:, np.newaxis], width, axis=-1)
    
    x_expand = np.expand_dims(x_mesh, axis = -1)
    y_expand = np.expand_dims(y_mesh, axis = -1)
    z_expand = np.expand_dims(z_mesh, axis = -1)
    
    meshgrid = np.concatenate((x_expand, np.concatenate((y_expand, z_expand), axis = -1)), axis = -1)

    return meshgrid
    
# transform meshgrid given transformation matrix
def get_transformed_meshgrid(meshgrid, transform_matrix, depth = voxsize, height = voxsize, width = voxsize):
    meshgrid_flat = meshgrid.transpose(3, 0, 1, 2).reshape(3,-1)
    one = np.ones((1, meshgrid_flat.shape[1]))
    meshgrid_expand = np.vstack((meshgrid_flat, one))
    transformed_meshgrid = (transform_matrix @ meshgrid_expand)
    
    transformed_meshgrid = (transformed_meshgrid[0:3, :]/transformed_meshgrid[3, :]).reshape(3, depth, height, width).transpose(1, 2, 3, 0)
    
    return torch.tensor(transformed_meshgrid, dtype=torch.float)


######################
#  single transform  #
######################

# compute transformation matrix
def get_transform_matrix(azimuth, elevation, scale = np.sqrt(3)):
    
    rot_base = RotateAlongAxis(90, 0, 0, 1) @ RotateAlongAxis(-90, 1, 0, 0)
    rot_m = RotateAlongAxis(azimuth, 0, 1, 0) @ RotateAlongAxis(-elevation, 1, 0, 0) @ rot_base
    
    
    sca_m = np.array([[scale, 0, 0, 0],
                      [0, scale, 0, 0],
                      [0, 0, scale, 0],
                      [0, 0, 0, 1]])
    
    return rot_m @ sca_m
    

# group function for transform voxel in pytorch tensor
def get_transformed_vox(vox_torch, azimuth, elevation, scale = np.sqrt(3)):
    meshgird = get_transformed_meshgrid(get_meshgrid(voxsize, voxsize, voxsize), get_transform_matrix(azimuth, elevation, scale), voxsize, voxsize, voxsize)
    transformedVox = F.grid_sample(vox_torch, meshgird.unsqueeze(0), mode='bilinear', padding_mode='zeros', align_corners=False)
    
    return transformedVox[0]


########################
#  Relative transform  #
########################


def get_relative_transform_matrix(azimuth_1, elevation_1, azimuth_2, elevation_2):
    
    rot_m = RotateAlongAxis(elevation_1, 0, 0, 1) @ RotateAlongAxis(azimuth_1, 1, 0, 0)  @ RotateAlongAxis(azimuth_2, 1, 0, 0) @ RotateAlongAxis(elevation_2, 0, 0, 1)
    
    scale = 1
    #scale = 1/np.sqrt(3)
    sca_m = np.array([[scale, 0, 0, 0],
                      [0, scale, 0, 0],
                      [0, 0, scale, 0],
                      [0, 0, 0, 1]])
    
    return rot_m @ sca_m


def get_relative_transformed_vox(vox_torch, azimuth_1, elevation_1, azimuth_2, elevation_2, device, voxsize = 32, align_mode = 'zeros'):
    meshgird = get_transformed_meshgrid(get_meshgrid(voxsize, voxsize, voxsize), 
                                        get_relative_transform_matrix(azimuth_1, elevation_1, azimuth_2, elevation_2), 
                                        voxsize, voxsize, voxsize).to(device)
    transformedVox = F.grid_sample(vox_torch, meshgird.unsqueeze(0), mode='bilinear', padding_mode=align_mode, align_corners=False)
    
    return transformedVox

#########
#  SDF  #
#########

# transformation function to rotate sdf indice
def get_transform_matrix_sdf(azimuth, elevation, scale = 1.0):
    
    rot_base = RotateAlongAxis(90, 0, 1, 0) @ RotateAlongAxis(-90, 1, 0, 0)
    rot_m = RotateAlongAxis(azimuth, 0, 1, 0) @ RotateAlongAxis(-elevation, 0, 0, 1) @ rot_base
    
    
    sca_m = np.array([[scale, 0, 0, 0],
                      [0, scale, 0, 0],
                      [0, 0, scale, 0],
                      [0, 0, 0, 1]])
    
    return rot_m @ sca_m


# group function to get transformed sdf indice
def get_transformed_indices(indices, azimuth, elevation, scale = 1/np.sqrt(3)):
    
    transform_matrix = get_transform_matrix_sdf(-azimuth, -elevation, scale)[0:3, 0:3]
    transformed_indices = indices @ transform_matrix
    
    return transformed_indices
    


# convert sdf to voxel
def sdf2Voxel(sample_pt, sample_sdf_val, fill = 0):
    
    sample_pt = ((sample_pt + np.array([0.5, 0.5, 0.5]))* voxsize).astype(int)
    sample_pt = np.clip(sample_pt, 0, voxsize-1)
    
    v = fill * np.ones((voxsize, voxsize, voxsize))
    v[sample_pt[:,0], sample_pt[:,1], sample_pt[:,2]] = sample_sdf_val
    
    return v
    
    
# advanced indexing 2x2x2 context from voxel
def getContext(sample_pt_query, vox):
    
    # sample_pt bxcxdimxdimxdim
    # vox bxmx3
    
    channel_size = vox.shape[1]
    batch_size, sample_size, _ = sample_pt_query.shape
    meshgrid_base = torch.Tensor(np.meshgrid(np.arange(0, batch_size), np.arange(0, channel_size), np.arange(0, 2), np.arange(0, 2), np.arange(0, 2))).int()
    context = torch.empty((batch_size, sample_size, channel_size, 2, 2, 2))

    for j in range(context.shape[1]):
        context[:, j, :, :, :, :] = vox[
                    meshgrid_base[0].long(),
                    meshgrid_base[1].long(),
                    (meshgrid_base[2] + sample_pt_query[:, j, 0].reshape(1, -1, 1, 1, 1)).long(), 
                    (meshgrid_base[3] + sample_pt_query[:, j, 1].reshape(1, -1, 1, 1, 1)).long(), 
                    (meshgrid_base[4] + sample_pt_query[:, j, 2].reshape(1, -1, 1, 1, 1)).long()
                ].transpose(0, 1)
    
    # b x c x m x 2 x 2 x 2
    return context.transpose(1, 2)


def trilinearInterpolation(context, dx, dy, dz):
    
    v0 = context[:, :, :, 0, 0, 0]*(1-dx)*(1-dy)*(1-dz)
    v1 = context[:, :, :, 1, 0, 0]*dx*(1-dy)*(1-dz)
    v2 = context[:, :, :, 0, 1, 0]*(1-dx)*dy*(1-dz)
    v3 = context[:, :, :, 1, 1, 0]*dx*dy*(1-dz)
    v4 = context[:, :, :, 0, 0, 1]*(1-dx)*(1-dy)*dz
    v5 = context[:, :, :, 1, 0, 1]*dx*(1-dy)*dz
    v6 = context[:, :, :, 0, 1, 1]*(1-dx)*dy*dz
    v7 = context[:, :, :, 1, 1, 1]*dx*dy*dz
    
    # b x c x m 1
    return v0 + v1 + v2 + v3 + v4 + v5 + v6 + v7
    



# generate mesh from continuous model
def generate_mesh(continuous, unet, out_vox, z, device, vox_res = 32, grid_res = 64, batch_size = 32, azimuth = 0, elevation = 0, isosurface = 0.0, conditional=True):
    
    start_time = time.time()
    
    vox = np.zeros((grid_res, grid_res, grid_res))
    idx = np.array(np.where(vox == 0))
    
    # normalize
    sample_pt = (torch.t(torch.tensor(idx/grid_res, dtype=torch.float)) - 0.5)
    sample_pt = sample_pt.reshape(-1, sample_size, 3)
    
    sample_pt_normalized = sample_pt + torch.tensor([0.5, 0.5, 0.5])
    # (0, 63)
    sample_pt_scale = torch.clamp(sample_pt_normalized* (vox_res-1), 0, (vox_res-1)-1e-5)
    # (0, 62]
    sample_pt_query = torch.clamp((sample_pt_scale).int(), 0, (vox_res-2))
    sample_pt_distance = sample_pt_scale - sample_pt_query
    
    
    vox_feature = unet(out_vox, z).repeat(batch_size, 1, 1, 1, 1).detach().cpu() if conditional else unet(out_vox).repeat(batch_size, 1, 1, 1, 1).detach().cpu()
    
    #print("--- %s seconds ---" % (time.time() - start_time))
    
    #print("Data generation")
    
    pre_sdf_list = []
    
    for i in range(int(sample_pt.shape[0]/batch_size)):
        
        start = i*batch_size
        end = (i + 1)*batch_size
        
        context = getContext(sample_pt_query[start:end, :, :], vox_feature)
        
        dx = sample_pt_distance[start:end, :, 0].unsqueeze(1)
        dy = sample_pt_distance[start:end, :, 1].unsqueeze(1)
        dz = sample_pt_distance[start:end, :, 2].unsqueeze(1)
        # local feature
        con = trilinearInterpolation(context, dx, dy, dz).to(device)
        # global feature
        latent = z.squeeze(-1).squeeze(-1).repeat(batch_size, 1, sample_size)       
        # point
        sample_pt_batch = sample_pt[start:end, :, :].transpose(-1, -2).to(device)
        
        
        sample_pt_batch = sample_pt_batch.transpose(-1, -2).reshape(-1, 3)
        con_batch = con.transpose(-1, -2).reshape(-1, 32)
        z_batch = latent.transpose(-1, -2).reshape(-1, 256)
        
        # avoid occupying gpu memory
        pred_sdf_batch = continuous(sample_pt_batch, 
                                    con_batch,
                                    z_batch,
                                   ).squeeze(1).detach().cpu()
        
        pre_sdf_list.append(pred_sdf_batch)
        
    pred_sdf = torch.cat(pre_sdf_list).reshape(-1,)
    
    vox[tuple([idx[0], idx[1], idx[2]])] = pred_sdf[:].numpy()
    
    #print(vox.shape)
    
    #print("--- %s seconds ---" % (time.time() - start_time))
    
    #print("Success generation")
    try:
        verts, faces, _, _ =  sk.marching_cubes_lewiner(vox, level=isosurface)
        #mesh = pymesh.form_mesh(verts, faces)
        #transform_matrix = get_relative_transform_matrix(azimuth, elevation, 0, 0)[0:3, 0:3]
        #transformed_vertices = mesh.vertices @ transform_matrix
        mesh = trimesh.Trimesh(verts, faces)
        #trimesh.repair.fix_inversion(mesh)
        trimesh.repair.fill_holes(mesh)
        mesh_py = pymesh.form_mesh(mesh.vertices, mesh.faces)
        return mesh_py
    except:
        print("Failed generation")
        return None
    
    
    
    
# generate mesh from voxel
def mesh_from_voxel(vox_torch):
    
    verts, faces, _, _ =  sk.marching_cubes_lewiner(vox_torch.detach().cpu().numpy(), level=0.5)
    mesh_py = pymesh.form_mesh(2*verts, faces)
    mesh = trimesh.Trimesh(mesh_py.vertices,mesh_py.faces)
    trimesh.repair.fix_inversion(mesh)
    trimesh.repair.fill_holes(mesh)
    
    return mesh
    
# render a mesh with pyrender render
def render(mesh):
    model = trimesh.Trimesh(mesh.vertices,mesh.faces)
    mesh_py = pyrender.Mesh.from_trimesh(model)
    scene = pyrender.Scene()
    scene.add(mesh_py)
    viewer = pyrender.Viewer(scene, use_raymond_lighting=True, point_size=2)
    
    
def mesh_test(mesh_py, dim = 64, count = 16384):
    
    mesh = trimesh.Trimesh(mesh_py.vertices, mesh_py.faces)
    samples, _ = trimesh.sample.sample_surface(mesh, count)
    
    samples_batch = torch.tensor(samples.reshape(64, -1, 3), dtype = torch.float)
    
    grid = pymesh.VoxelGrid(2./dim)
    grid.insert_mesh(mesh_py)
    grid.create_grid()
    
    idx = ((grid.mesh.vertices + 1.1) / 2.4 * dim).astype(np.int)
    v = np.zeros([dim, dim, dim])
    v[idx[:,0], idx[:,1], idx[:,2]] = 1
    
    return samples_batch, samples, v


# compute chamfer distance, earth movers' distacne and intersection over union between two meshes
def get_test_results(mesh_py_1, mesh_py_2):
    
    samples_batch_1, samples_1, v1 = mesh_test(mesh_py_1)
    samples_batch_2, samples_2, v2 = mesh_test(mesh_py_2)
    
    dist1, dist2, _, _ = chamfer_python.distChamfer(samples_batch_1, samples_batch_2)
    chamfer_dist = torch.mean(dist1) + torch.mean(dist2)
    
    emd = emd_samples(samples_1, samples_2)
    
    intersection = np.sum(np.logical_and(v1, v2))
    union = np.sum(np.logical_or(v1, v2))
    
    iou = intersection/union
    
    return chamfer_dist, emd, iou
    
