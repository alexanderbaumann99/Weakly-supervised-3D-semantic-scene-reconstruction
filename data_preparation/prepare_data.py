import os
import trimesh
from mesh_to_sdf import get_surface_point_cloud
from pyntcloud import PyntCloud
import pandas as pd

#takes way too long
def prepare_data():#data_path):
    data_path = '../../data/shapenet_examples'
    output_dir = '../datasets/ShapeNetv2_data'
    sample_point_count = 50000 #config, only for testing
    query_point_count = 50000 #config
    i = 0
    for dir in os.listdir(data_path):
        file_path = os.path.join(*[data_path,dir,'models','model_normalized.obj'])
        mesh = trimesh.load(file_path)
        #maybe change to scan later
        surface_pc = get_surface_point_cloud(mesh, surface_point_method='sample', sample_point_count=sample_point_count)
        pc_points = surface_pc.points
        query_points, sdf = surface_pc.sample_sdf_near_surface(number_of_points=query_point_count)

        #save
        sub_output_dir = os.path.join(output_dir,str(i))
        os.makedirs(sub_output_dir)
        point_cloud = PyntCloud(pd.DataFrame(data={'x': pc_points[:,0],'y': pc_points[:,1],'z': pc_points[:,2]}))
        point_cloud.to_file(os.path.join(sub_output_dir,'points.ply'))
        query_point_cloud = PyntCloud(pd.DataFrame(data={'x': query_points[:,0],'y': query_points[:,1],'z': query_points[:,2], 'sdf': sdf}))
        query_point_cloud.to_file(os.path.join(sub_output_dir,'query_points.ply'))
        #os.rmdir(dir)
        i += 1

if __name__ == '__main__':
    prepare_data()


