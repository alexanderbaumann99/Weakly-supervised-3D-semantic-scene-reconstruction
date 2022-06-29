from models.iscnet.modules.generator_prior import Generator3DPrior
from models.iscnet.modules.layers import ResnetPointnet, CBatchNorm1d
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.registers import MODULES
import torch.distributions as dist
from external.common import make_3d_grid

@MODULES.register_module
class ShapeRetrieval(nn.Module):
    """
    Definition of Shape Prior from DOPS paper
    Parameters:
        c_dim           : dimension of conditional latent vector
    """

    def __init__(self, cfg, optim_spec=None):
        super(ShapeRetrieval, self).__init__()

        '''Optimizer parameters used in training'''
        self.optim_spec = optim_spec
        self.cfg = cfg
        
        self.encoder = ResnetPointnet(c_dim=cfg.config['data']['c_dim_prior'],
                                      dim=3,
                                      hidden_dim=cfg.config['data']['hidden_dim'])

    def forward(self, pc):
        '''
        Returns the shape embedding of a point cloud
        Args:
            pc: point cloud of the form (N x N_P x 3)
        Returns:
            out:    shape embedding (N x c_dim)
        '''
        out = self.encoder(pc)

        return out
