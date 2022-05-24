# Prior Dataloader of ISCNet.
# Cite: VoteNet

import torch.utils.data
from torch.utils.data import DataLoader
import numpy as np
import os
from plyfile import PlyData

import pickle


class ShapeNetDataset(torch.utils.data.Dataset):
    def __init__(self, cfg, mode):
        super(ShapeNetDataset, self).__init__(cfg, mode)
        self.num_points = cfg.config['data']['num_points']
        self.num_query_points = cfg.config['data']['num_query_points']
        self.data_path = cfg.config['data']['shapenet_path']

    def __getitem__(self, idx):
        """
        Returns a dict with following keys:
            point_cloud: (N,3)
            query_points: (M,3)
            sdf: (M,)
        """
        points_file = os.path.join(*[self.data_path, str(idx), 'points.ply'])
        with open(points_file, 'rb') as f:
            plydata = PlyData.read(f)
            num_verts = plydata['vertex'].count
            points = np.zeros(shape=[num_verts, 3], dtype=np.float32)
            points[:, 0] = plydata['vertex'].data['x']
            points[:, 1] = plydata['vertex'].data['y']
            points[:, 2] = plydata['vertex'].data['z']

        query_points_file = os.path.join(*[self.data_path, str(idx), 'query_points.ply'])
        with open(query_points_file, 'rb') as f:
            plydata = PlyData.read(f)
            num_verts = plydata['vertex'].count
            query_points = np.zeros(shape=[num_verts, 4], dtype=np.float32)
            query_points[:, 0] = plydata['vertex'].data['x']
            query_points[:, 1] = plydata['vertex'].data['y']
            query_points[:, 2] = plydata['vertex'].data['z']
            query_points[:, 3] = plydata['vertex'].data['sdf']

        #sample random pc points and query points
        points = points[np.random.choice(points.shape[0], self.num_points, replace=False)]
        query_points = query_points[np.random.choice(query_points.shape[0], self.num_query_points, replace=False)]

        ret_dict = {}
        ret_dict['point_cloud'] = points.astype(np.float32)
        ret_dict['query_points'] = query_points.astype(np.float32)[:, 0:3]
        ret_dict['sdf'] = query_points.astype(np.float32)[:, 3]
        return ret_dict



def PriorDataLoader(cfg, mode='train'):
    if cfg.config['data']['dataset'] == 'shapenet':
        dataset = ShapeNetDataset(cfg, mode)
    else:
        raise NotImplementedError

    dataloader = DataLoader(dataset=dataset,
                            num_workers=cfg.config['device']['num_workers'],
                            batch_size=cfg.config[mode]['batch_size'],
                            shuffle=(mode == 'train'))
    return dataloader