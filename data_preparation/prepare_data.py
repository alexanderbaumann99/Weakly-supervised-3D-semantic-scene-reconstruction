import imp
import os
import trimesh
from mesh_to_sdf import get_surface_point_cloud
from pyntcloud import PyntCloud
import pandas as pd
import numpy as np
import math

ShapeNet2Labels = {'04379243':'0','03001627':'1','02871439':'2','04256520':'3','02747177':'4','02933112':'5','03211117':'6','02808440':'7' }



def prepare_data(ids=True):
    data_path = '../raw/shapenet/watertight_scaled_simplified'
    output_dir = '../datasets/ShapeNetv2_data_fps2'
    sample_point_count = 50000 #config, only for testing
    point_cloud_count = 2048
    query_point_count = 50000 #config
    i = 0
    for root, dirs, files in os.walk(data_path):
        for file in files:
            
            file_path = os.path.join(root, file)
            mesh = trimesh.load(file_path)
            #maybe change to scan later
            surface_pc = get_surface_point_cloud(mesh, surface_point_method='sample', sample_point_count=sample_point_count)
            pc_points = surface_pc.get_fps_surface_points(count=point_cloud_count)
            query_points_surface, sdf_surface,query_points_sphere,sdf_sphere = surface_pc.sample_sdf_near_surface(number_of_points=query_point_count,ratio_surface=0.5)

            #save
            sub_output_dir = os.path.join(output_dir,str(i))
            os.makedirs(sub_output_dir)
            point_cloud = PyntCloud(pd.DataFrame(data={'x': pc_points[:,0],'y': pc_points[:,1],'z': pc_points[:,2]}))
            point_cloud.to_file(os.path.join(sub_output_dir,'points.ply'))
            query_point_cloud_surface = PyntCloud(pd.DataFrame(data={'x': query_points_surface[:,0],'y': query_points_surface[:,1],'z': query_points_surface[:,2], 'sdf': sdf_surface}))
            query_point_cloud_surface.to_file(os.path.join(sub_output_dir,'query_points_surface.ply'))
            query_point_cloud_sphere = PyntCloud(pd.DataFrame(data={'x': query_points_sphere[:,0],'y': query_points_sphere[:,1],'z': query_points_sphere[:,2], 'sdf': sdf_sphere}))
            query_point_cloud_sphere.to_file(os.path.join(sub_output_dir,'query_points_sphere.ply'))

            if ids:
                dic={}
                id = root.split('/')[-1] 
                dic['shapenet_id'] = id
                dic['sem_cls_id']  = ShapeNet2Labels[id] 

                #save
                np.save(os.path.join(sub_output_dir,'dict_ids'),dic)

            
            
            i += 1


if __name__ == '__main__':
    prepare_data()


