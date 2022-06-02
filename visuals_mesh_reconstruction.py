import open3d as o3d
import numpy as np

n=233

pcd = o3d.io.read_point_cloud("gt"+str(n)+".pcd")
o3d.visualization.draw_geometries([pcd])

pcd2 = o3d.io.read_point_cloud("input"+str(n)+".pcd")
o3d.visualization.draw_geometries([pcd2])

mesh = o3d.io.read_triangle_mesh("item"+str(n)+".ply")
o3d.visualization.draw_geometries([mesh])



