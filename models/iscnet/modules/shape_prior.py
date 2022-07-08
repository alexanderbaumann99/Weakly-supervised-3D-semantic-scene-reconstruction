from models.iscnet.modules.generator import Generator3D
from models.iscnet.modules.layers import ResnetPointnet, CBatchNorm1d
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.registers import MODULES
import torch.distributions as dist
from external.common import make_3d_grid


class DecoderBlock(nn.Module):

    def __init__(self, c_dim, hidden_dim=128, leaky=False):
        super(DecoderBlock, self).__init__()

        self.fc1 = nn.Conv1d(hidden_dim, hidden_dim, 1)
        self.fc2 = nn.Conv1d(hidden_dim, hidden_dim, 1)

        self.CBatchNorm1 = CBatchNorm1d(c_dim,
                                        f_dim=hidden_dim)
        self.CBatchNorm2 = CBatchNorm1d(c_dim,
                                        f_dim=hidden_dim)
        self.act = nn.ReLU()
        if leaky:
            self.act = nn.LeakyReLU()

    def forward(self, x, condition):
        out = self.fc1(self.act(self.CBatchNorm1(x, condition)))
        out = self.fc2(self.act(self.CBatchNorm2(out, condition)))

        return out


@MODULES.register_module
class ShapePrior(nn.Module):
    """
    Definition of Shape Prior from DOPS paper
    Parameters:
        c_dim           : dimension of conditional latent vector
    """

    def __init__(self, cfg, optim_spec=None):
        super(ShapePrior, self).__init__()

        '''Optimizer parameters used in training'''
        self.optim_spec = optim_spec
        self.cfg = cfg
        self.use_cls_for_completion = cfg.config['data']['use_cls_for_completion']
        self.threshold = cfg.config['data']['threshold']
        '''Definition of the modules'''
        leaky = False
        self.encoder = ResnetPointnet(c_dim=cfg.config['data']['c_dim'],
                                      dim=3,
                                      hidden_dim=cfg.config['data']['hidden_dim'])

        hidden_dim = cfg.config['data']['c_dim']
        self.fc1 = nn.Conv1d(3, hidden_dim, 1)
        self.dblock1 = DecoderBlock(c_dim=cfg.config['data']['c_dim'],
                                    hidden_dim=hidden_dim,
                                    leaky=leaky)
        self.dblock2 = DecoderBlock(c_dim=cfg.config['data']['c_dim'],
                                    hidden_dim=hidden_dim,
                                    leaky=leaky)
        self.dblock3 = DecoderBlock(c_dim=cfg.config['data']['c_dim'],
                                    hidden_dim=hidden_dim,
                                    leaky=leaky)
        self.dblock4 = DecoderBlock(c_dim=cfg.config['data']['c_dim'],
                                    hidden_dim=hidden_dim,
                                    leaky=leaky)
        self.dblock5 = DecoderBlock(c_dim=cfg.config['data']['c_dim'],
                                    hidden_dim=hidden_dim,
                                    leaky=leaky)
        self.CBatchNorm = CBatchNorm1d(c_dim=cfg.config['data']['c_dim'],
                                       f_dim=hidden_dim)
        self.fc2 = nn.Conv1d(hidden_dim, 1, 1)
        self.act = nn.ReLU()
        if leaky:
            self.act = nn.LeakyReLU()

        '''Mount mesh generator'''
        if 'generation' in cfg.config and cfg.config['generation']['generate_mesh']:
            from models.iscnet.modules.generator import Generator3D
            self.generator = Generator3D(self,
                                              threshold=cfg.config['data']['threshold'],
                                              resolution0=cfg.config['generation']['resolution_0'],
                                              upsampling_steps=cfg.config['generation']['upsampling_steps'],
                                              sample=cfg.config['generation']['use_sampling'],
                                              refinement_step=cfg.config['generation']['refinement_step'],
                                              simplify_nfaces=cfg.config['generation']['simplify_nfaces'],
                                              preprocessor=None)

    def generate_latent(self, pc):
        '''
        Generates shape embedding of the point cloud
        Args:
            pc: point cloud of the form (N x Number of points x 3)
        Returns:
            self.latent:    shape embedding of size (N x c_dim)
        '''
        self.latent = self.encoder(pc)
        return self.latent

    def forward(self, query_points):
        '''
        Returns the signed distance of each query point to the surface
        Args:
            query_points: query points of the form (N x N_P x 3)
        Returns:
            out:    signed distance of the form  (N x N_P x 1)
        '''
        query_points = query_points.transpose(1, 2)
        out = self.fc1(query_points)
        out = self.dblock1(out, self.latent)
        out = self.dblock2(out, self.latent)
        out = self.dblock3(out, self.latent)
        out = self.dblock4(out, self.latent)
        out = self.dblock5(out, self.latent)
        out = self.act(self.CBatchNorm(out, self.latent))
        out = torch.tanh(self.fc2(out))
        out = out.transpose(1, 2)

        return out

    def set_latent(self, z):
        '''
        Sets shape embedding of the point cloud
        Args:
            z: input feature vector of size (N x c_dim)
        Returns:
            self.latent:    shape embedding of size (N x c_dim)
        '''
        self.latent = z
        return self.latent

    def compute_loss(self, object_input_features, query_points, query_points_occ,
                     cls_codes_for_completion, export_shape=False):
        '''
        Compute loss for ShapePrior
        :param object_input_features (N, c_dim): Input features generated by encoder
        :param query_points (N_B, N_P, 3): Number of bounding boxes x Number of Points x 3.
        :param query_points_occ (N_B, N_P): Corresponding occupancy values.
        :param cls_codes_for_completion (N_B, N_C): One-hot category codes.
        :param export_shape (bool): whether to export a shape voxel example.
        :return:
        '''
        self.eval()
        print(object_input_features.shape)
        self.set_latent(object_input_features)
        device = query_points.device
        batch_size = query_points.size(0)
        if self.use_cls_for_completion:
            cls_codes_for_completion = cls_codes_for_completion.to(device).float()
            input_features_for_completion = torch.cat([query_points, cls_codes_for_completion], dim=-1)

        kwargs = {}
        '''Infer latent code z.'''

        preds = self.forward(query_points)
        loss = F.mse_loss(preds.squeeze(), torch.sign(query_points_occ), reduction='mean')
        print(loss)

        if export_shape:
            shape = (16, 16, 16)
            qp = make_3d_grid([-0.5 + 1 / 32] * 3, [0.5 - 1 / 32] * 3, shape).to(device)
            qp = qp.expand(batch_size, *qp.size())
            p_r = self.forward(qp)
            occ_hat = p_r.view(batch_size, *shape)
            voxels_out = (occ_hat >= self.threshold)
        else:
            voxels_out = None

        return loss, voxels_out


