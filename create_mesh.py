from models.iscnet.modules.prior_training import ShapePrior
import torch
import open3d as o3d
from models.iscnet.prior_dataloader import PriorDataLoader,ShapeNetDataset
from configs.config_utils import mount_external_config
from configs.config_utils import CONFIG
from net_utils.visualize_sdf import create_mesh



n=233

cfg = CONFIG('configs/config_files/ISCNet.yaml')
cfg = mount_external_config(cfg)
model=ShapePrior(cfg)
model.eval()

print("Start Testing...")
model.load_state_dict(torch.load('weights_epoch_62'))
dataset=ShapeNetDataset(cfg,"test")
item=torch.FloatTensor(dataset[n]["point_cloud"]).view(1,-1,3).cuda()
model.generate_latent(item)
pcd_full = o3d.io.read_point_cloud("datasets/ShapeNetv2_data/"+str(n)+"/points.ply")
o3d.io.write_point_cloud("gt"+str(n)+".pcd", pcd_full)
pcd=o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(dataset[n]["point_cloud"]))
o3d.io.write_point_cloud("input"+str(n)+".pcd", pcd)
create_mesh(model,"item"+str(n))