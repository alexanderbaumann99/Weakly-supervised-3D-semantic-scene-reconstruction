# Back propogate box features to input points.
# author: ynie
# date: March, 2020
# cite: PointNet++

from models.registers import MODULES
import torch
from torch import nn
from external.pointnet2_ops_lib.pointnet2_ops.pointnet2_modules import STN_Group
from models.iscnet.modules.layers import ResnetPointnet
from models.iscnet.modules.pointseg import PointSeg, get_loss


@MODULES.register_module
class GroupAndAlign(nn.Module):
    ''' Back-Propagte box proposal features to input points
    '''

    def __init__(self, cfg, optim_spec=None):
        super(GroupAndAlign, self).__init__()
        '''Modules'''
        self.stn = STN_Group(
            radius=1.,
            nsample=cfg.config['data']['num_box_points'],
            use_xyz=False,
            normalize_xyz=True
        )
        self.input_feature_dim = int(cfg.config['data']['use_color_completion']) * 3 + int(
            not cfg.config['data']['no_height']) * 1

    def _break_up_pc(self, pc):
        xyz = pc[..., 0:3].contiguous()
        features = (
            pc[..., 3:3 + self.input_feature_dim].transpose(1, 2).contiguous()
            if pc.size(-1) > 3 else None
        )

        return xyz, features

    def forward(self, box_xyz, box_orientations, box_feature, input_point_cloud, point_instance_labels,
                proposal_instance_labels):
        '''
        Extract point features from input pointcloud, and propagate to box xyz.
        :param box_xyz: (Batch size x N points x 3) point coordinates
        :param box_feature: (Batch size x Feature dim x Num of boxes) box features.
        :param input_point_cloud: (Batch size x Num of pointcloud points x feature dim) box features.
        :return:
        '''
        xyz, features = self._break_up_pc(input_point_cloud)
        features = torch.cat([features, point_instance_labels.unsqueeze(1)], dim=1)
        xyz, features = self.stn(xyz, features, box_xyz, box_orientations)
        # from dimension B x dim x N_proposals x N_points
        # to N_proposals x B x N_points x dim
        xyz = xyz.permute([2, 0, 3, 1]).contiguous()

        return xyz, features


