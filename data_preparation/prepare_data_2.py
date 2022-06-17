import imp
import os
import trimesh
from mesh_to_sdf import get_surface_point_cloud
from pyntcloud import PyntCloud
import pandas as pd
import numpy as np


#takes way too long
def prepare_data():#data_path):
    data_path = '../../data/watertight_scaled_simplified'
    output_dir = '../datasets/ShapeNetv2_data_simplified'
    query_point_count =  50000 #config
    point_cloud_count = 2048
    i = 0
    for dir in os.listdir(data_path):
        for dir2 in os.listdir(os.path.join(*[data_path,dir])):
            file_path = os.path.join(*[data_path,dir,dir2])
            mesh = trimesh.load(file_path)
            #maybe change to scan later
            surface_pc = get_surface_point_cloud(mesh, surface_point_method='sample', sample_point_count=sample_point_count)
            pc_points = surface_pc.get_fps_surface_points(count=point_cloud_count)
            query_points_surface, sdf__surface,query_points_sphere,sdf_sphere = \
                surface_pc.sample_sdf_near_surface(number_of_points=query_point_count,ratio_surface=0.5)
            
            #save
            sub_output_dir = os.path.join(output_dir,str(i))
            os.makedirs(sub_output_dir)
            point_cloud = PyntCloud(pd.DataFrame(data={'x': pc_points[:,0],'y': pc_points[:,1],'z': pc_points[:,2]}))
            point_cloud.to_file(os.path.join(sub_output_dir,'points.ply'))
            query_point_cloud_surface = PyntCloud(pd.DataFrame(data={'x': query_points_surface[:,0],'y': query_points_surface[:,1],'z': query_points_surface[:,2], 'sdf': sdf__surface}))
            query_point_cloud_surface.to_file(os.path.join(sub_output_dir,'query_points_surface.ply'))
            query_point_cloud_sphere = PyntCloud(pd.DataFrame(data={'x': query_points_sphere[:,0],'y': query_points_sphere[:,1],'z': query_points_sphere[:,2], 'sdf': sdf_sphere}))
            query_point_cloud_sphere.to_file(os.path.join(sub_output_dir,'query_points_sphere.ply'))


if __name__ == '__main__':
    prepare_data()


