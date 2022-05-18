from turtle import forward
from models.iscnet.modules.layers import ResnetPointnet
import torch
import torch.nn as nn
import torch.nn.functional as F


class ShapeEmbedding(nn.Module):

    def __init__(self,cfg):
        super(ShapeEmbedding,self).__init__()

        self.feature_extractor=ResnetPointnet(  c_dim=cfg.config['data']['c_dim'],
                                                dim=3,
                                                hidden_dim=128)

    def _break_up_pc(self, pc):
        '''
        From skip propagation module
        '''
        xyz = pc[..., 0:3].contiguous()
        features = (
            pc[..., 3:3+self.input_feature_dim].transpose(1, 2).contiguous()
            if pc.size(-1) > 3 else None
        )

        return xyz, features   
    
    def forward(self,input_point_cloud):
        '''
        Extract point features from input pointcloud, and propagate to box xyz.
        :param input_point_cloud:   (Batch size x Num of pointcloud points x feature dim) box features.
        :return:                    shape embedding for the shape reconstruction              
        '''
        xyz, features = self._break_up_pc(input_point_cloud)
        shape_emb = self.feature_extractor(xyz)

        return shape_emb
