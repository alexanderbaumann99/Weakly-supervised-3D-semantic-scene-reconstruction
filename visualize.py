# demo file.
# author: ynie
# date: July, 2020
import trimesh
import numpy as np
from utils import pc_util
import os
import vtk
from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk
from utils.scannet.visualization.vis_for_demo import Vis_base
import argparse

def visualize(output_dir, offline):
    predicted_boxes = np.load(os.path.join(output_dir, '000000_pred_confident_nms_bbox.npz'))
    input_point_cloud = pc_util.read_ply(os.path.join(output_dir, '000000_pc.ply'))
    bbox_params = predicted_boxes['obbs']
    proposal_map = predicted_boxes['proposal_map']
    transform_m = np.array([[0, 0, -1], [-1, 0, 0], [0, 1, 0]])

    instance_models = []
    center_list = []
    vector_list = []

    for map_data, bbox_param in zip(proposal_map, bbox_params):
        mesh_file = os.path.join(output_dir, 'proposal_%d_mesh.ply' % tuple(map_data))
        ply_reader = vtk.vtkPLYReader()
        ply_reader.SetFileName(mesh_file)
        ply_reader.Update()
        # get points from object
        polydata = ply_reader.GetOutput()
        # read points using vtk_to_numpy
        obj_points = vtk_to_numpy(polydata.GetPoints().GetData()).astype(np.float64)

        '''Fit obj points to bbox'''
        center = bbox_param[:3]
        orientation = bbox_param[6]
        sizes = bbox_param[3:6]

        obj_points = obj_points - (obj_points.max(0) + obj_points.min(0))/2.
        obj_points = obj_points.dot(transform_m.T)
        obj_points = obj_points.dot(np.diag(1/(obj_points.max(0) - obj_points.min(0)))).dot(np.diag(sizes))

        axis_rectified = np.array([[np.cos(orientation), np.sin(orientation), 0], [-np.sin(orientation), np.cos(orientation), 0], [0, 0, 1]])
        obj_points = obj_points.dot(axis_rectified) + center

        points_array = numpy_to_vtk(obj_points[..., :3], deep=True)
        polydata.GetPoints().SetData(points_array)
        ply_reader.Update()

        '''draw bboxes'''
        vectors = np.diag(sizes/2.).dot(axis_rectified)

        instance_models.append(ply_reader)
        center_list.append(center)
        vector_list.append(vectors)

    scene = Vis_base(scene_points=input_point_cloud, instance_models=instance_models, center_list=center_list,
                     vector_list=vector_list)

    camera_center = np.array([0, -3, 3])
    scene.visualize(centroid=camera_center, offline=offline, save_path=os.path.join(output_dir, 'pred.png'))


def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('Visualization')
    #parser.add_argument('--offline', type=bool, default='False', help='set offline')
    parser.add_argument('--demo_path', type=str, default='demo/outputs/scene0549_00', help='Please specify the demo path.')
    return parser.parse_args()

if __name__=="__main__":
    args=parse_args()
    visualize(args.demo_path, False)
    

