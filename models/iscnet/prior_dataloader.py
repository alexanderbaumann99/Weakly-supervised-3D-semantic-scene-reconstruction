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
        super(ShapeNetDataset, self).__init__()
        self.num_points = cfg.config['data']['num_points']
        self.num_query_points = cfg.config['data']['num_query_points']
        self.data_path = cfg.config['data']['shapenet_path']

    def __len__(self):
        return len(os.listdir(*[self.data_path]))


    def __getitem__(self, idx):
        """
        Returns a dict with following keys:
            point_cloud: (N,3)
            query_points: (M,3)
            sdf: (M,)
        """
        ret_dict = {}
    
        points_file = os.path.join(*[self.data_path, str(idx), 'points.ply'])
        with open(points_file, 'rb') as f:
            plydata = PlyData.read(f)
            num_verts = plydata['vertex'].count
            points = np.zeros(shape=[num_verts, 3], dtype=np.float32)
            points[:, 0] = plydata['vertex'].data['x']
            points[:, 1] = plydata['vertex'].data['y']
            points[:, 2] = plydata['vertex'].data['z']
        ret_dict['point_cloud'] = points.astype(np.float32)

        query_points_file = os.path.join(*[self.data_path, str(idx), 'query_points_surface.ply'])
        with open(query_points_file, 'rb') as f:
            plydata = PlyData.read(f)
            num_verts = plydata['vertex'].count
            query_points = np.zeros(shape=[num_verts, 4], dtype=np.float32)
            query_points[:, 0] = plydata['vertex'].data['x']
            query_points[:, 1] = plydata['vertex'].data['y']
            query_points[:, 2] = plydata['vertex'].data['z']
            query_points[:, 3] = plydata['vertex'].data['sdf']
        query_points_surface = query_points[np.random.choice(query_points.shape[0], int(self.num_query_points*0.5), replace=False)]
        
        query_points_file = os.path.join(*[self.data_path, str(idx), 'query_points_sphere.ply'])
        with open(query_points_file, 'rb') as f:
            plydata = PlyData.read(f)
            num_verts = plydata['vertex'].count
            query_points = np.zeros(shape=[num_verts, 4], dtype=np.float32)
            query_points[:, 0] = plydata['vertex'].data['x']
            query_points[:, 1] = plydata['vertex'].data['y']
            query_points[:, 2] = plydata['vertex'].data['z']
            query_points[:, 3] = plydata['vertex'].data['sdf']
        query_points_sphere = query_points[np.random.choice(query_points.shape[0], int(self.num_query_points*0.5), replace=False)]
        
        query_points = np.concatenate([query_points_surface,query_points_sphere],axis=0)       
        ret_dict['query_points'] = query_points.astype(np.float32)[:, 0:3]
        ret_dict['sdf'] = query_points.astype(np.float32)[:, 3]
        
        id_dic = np.load(os.path.join(*[self.data_path, str(idx), 'dict_ids.npy']),allow_pickle=True)
        ret_dict['shapenet_id'] = id_dic.item()['shapenet_id']
        ret_dict['sem_cls_id'] = id_dic.item()['sem_cls_id']


        return ret_dict



def PriorDataLoader(cfg,splits, mode='train'):
    if cfg.config['data']['dataset'] == 'shapenet':
        dataset = ShapeNetDataset(cfg, mode)
    else:
        raise NotImplementedError
 
    lengths=[round(a*len(dataset)) for a in splits]
    train,test=torch.utils.data.random_split(dataset,lengths)
    train_loader = DataLoader(dataset=train,
                            num_workers=cfg.config['device']['num_workers'],
                            batch_size=cfg.config[mode]['batch_size'],
                            shuffle=True)
    test_loader = DataLoader(dataset=test,
                            num_workers=cfg.config['device']['num_workers'],
                            batch_size=cfg.config[mode]['batch_size'],
                            shuffle=False)
    return train_loader,test_loader
