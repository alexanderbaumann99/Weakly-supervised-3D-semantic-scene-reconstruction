import trimesh
import logging
logging.getLogger("trimesh").setLevel(9000)
import numpy as np
from sklearn.neighbors import KDTree
import math
import torch
from dgl.geometry import farthest_point_sampler as fps

def sample_uniform_points(amount):
    '''changed to box instead of sphere'''
    uniform_points = np.random.uniform(-0.5, 0.5, size=(amount, 3))
    return uniform_points

class BadMeshException(Exception):
    pass

class SurfacePointCloud:
    def __init__(self, mesh, points, normals=None, scans=None):
        self.mesh = mesh
        self.points = points
        self.normals = normals
        self.scans = scans

        self.kd_tree = KDTree(points)

    def get_random_surface_points(self, count, use_scans=True):
        """ if use_scans:
            indices = np.random.choice(self.points.shape[0], count)
            return self.points[indices, :]
        else:
            return self.mesh.sample(count) """
        points=torch.Tensor(self.points)
        B=1
        N=points.shape[0]
        result = torch.zeros((count * B), dtype=torch.long)
        dist = torch.zeros((B * N))
        start_idx = torch.randint(0, N - 1, (B, ), dtype=torch.long)
        fps(data=points,batch_size=B,sample_points=count,dist=dist,start_idx=start_idx,result=result)
        result=result.numpy()
        result=self.points[result]

        return result

    def get_fps_surface_points(self,count):
        points=torch.Tensor(self.points).cuda()
        B=1
        N=points.shape[0]
        result = torch.zeros((count * B), dtype=torch.long).cuda()
        dist = torch.zeros((B * N)).cuda()
        start_idx = torch.randint(0, N - 1, (B, ), dtype=torch.long).cuda()
        fps(data=points,batch_size=B,sample_points=count,dist=dist,start_idx=start_idx,result=result)
        result=result.cpu().numpy()
        result=self.points[result]
        return result 

    def get_sdf(self, query_points, use_depth_buffer=False, sample_count=11, return_gradients=False):
        if use_depth_buffer:
            distances, indices = self.kd_tree.query(query_points)
            distances = distances.astype(np.float32).reshape(-1)
            inside = ~self.is_outside(query_points)
            distances[inside] *= -1

            if return_gradients:
                gradients = query_points - self.points[indices[:, 0]]
                gradients[inside] *= -1

        else:
            distances, indices = self.kd_tree.query(query_points, k=sample_count)
            distances = distances.astype(np.float32)

            closest_points = self.points[indices]
            direction_from_surface = query_points[:, np.newaxis, :] - closest_points
            inside = np.einsum('ijk,ijk->ij', direction_from_surface, self.normals[indices]) < 0
            inside = np.sum(inside, axis=1) > sample_count * 0.5
            distances = distances[:, 0]
            distances[inside] *= -1

            if return_gradients:
                gradients = direction_from_surface[:, 0]
                gradients[inside] *= -1

        if return_gradients:
            near_surface = np.abs(distances) < math.sqrt(0.0025**2 * 3) * 3 # 3D 2-norm stdev * 3
            gradients = np.where(near_surface[:, np.newaxis], self.normals[indices[:, 0]], gradients)
            gradients /= np.linalg.norm(gradients, axis=1)[:, np.newaxis]
            return distances, gradients
        else:
            return distances

    def get_sdf_in_batches(self, query_points, use_depth_buffer=False, sample_count=11, batch_size=1000000, return_gradients=False):
        if query_points.shape[0] <= batch_size:
            return self.get_sdf(query_points, use_depth_buffer=use_depth_buffer, sample_count=sample_count, return_gradients=return_gradients)

        n_batches = int(math.ceil(query_points.shape[0] / batch_size))
        batches = [
            self.get_sdf(points, use_depth_buffer=use_depth_buffer, sample_count=sample_count, return_gradients=return_gradients)
            for points in np.array_split(query_points, n_batches)
        ]
        if return_gradients:
            distances = np.concatenate([batch[0] for batch in batches])
            gradients = np.concatenate([batch[1] for batch in batches])
            return distances, gradients
        else:
            return np.concatenate(batches) # distances

 
    def sample_sdf_near_surface(self, number_of_points=500000,ratio_surface=0.6, use_scans=True, sign_method='normal', normal_sample_count=11, min_size=0, return_gradients=False):
        query_points = []
        surface_sample_count=int(number_of_points*ratio_surface)
        surface_points = self.get_fps_surface_points(surface_sample_count)
        query_points.append(surface_points + np.random.normal(scale=0.0025, size=(surface_sample_count, 3)))
        #query_points.append(surface_points + np.random.normal(scale=0.00025, size=(surface_sample_count, 3)))
        query_points_near_surface = np.concatenate(query_points).astype(np.float32)
        
        unit_sphere_sample_count = number_of_points - query_points_near_surface.shape[0] 
        unit_sphere_points = sample_uniform_points(unit_sphere_sample_count).astype(np.float32)

        if sign_method == 'normal':
            sdf_near_surface = self.get_sdf_in_batches(query_points_near_surface, use_depth_buffer=False, sample_count=normal_sample_count, return_gradients=return_gradients)
            sdf_sphere = self.get_sdf_in_batches(unit_sphere_points, use_depth_buffer=False, sample_count=normal_sample_count, return_gradients=return_gradients)
    
        elif sign_method == 'depth':
            sdf_near_surface = self.get_sdf_in_batches(query_points_near_surface, use_depth_buffer=True, return_gradients=return_gradients)
            sdf_sphere = self.get_sdf_in_batches(unit_sphere_points, use_depth_buffer=True, sample_count=normal_sample_count, return_gradients=return_gradients)
        else:
            raise ValueError('Unknown sign determination method: {:s}'.format(sign_method))
            
        if return_gradients:
            sdf_near_surface, gradients_near_surface = sdf_near_surface
            sdf_sphere, gradients_sphere = sdf_sphere
            return query_points_near_surface, sdf_near_surface,unit_sphere_points,sdf_sphere, gradients
        else:
            return query_points_near_surface, sdf_near_surface,unit_sphere_points,sdf_sphere


    def is_outside(self, points):
        result = None
        for scan in self.scans:
            if result is None:
                result = scan.is_visible(points)
            else:
                result = np.logical_or(result, scan.is_visible(points))
        return result


def sample_from_mesh(mesh, sample_point_count=10000000, calculate_normals=True):
    if calculate_normals:
        points, face_indices = mesh.sample(sample_point_count, return_index=True)
        normals = mesh.face_normals[face_indices]
    else:
        points = mesh.sample(sample_point_count, return_index=False)

    return SurfacePointCloud(mesh, 
        points=points,
        normals=normals if calculate_normals else None,
        scans=None
    )
