from models.iscnet.modules.layers import ResnetPointnet,CBatchNorm1d
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.registers import MODULES


class DecoderBlock(nn.Module):

    def __init__(self,c_dim,hidden_dim=128,leaky=False):
        super(DecoderBlock,self).__init__()

        self.fc1=nn.Conv1d(hidden_dim,hidden_dim,1)
        self.fc2=nn.Conv1d(hidden_dim,hidden_dim,1)

        self.CBatchNorm1=CBatchNorm1d(c_dim,
                                      f_dim=hidden_dim)
        self.CBatchNorm2=CBatchNorm1d(c_dim,
                                      f_dim=hidden_dim)
        self.act=nn.ReLU()
        if leaky:
            self.act=nn.LeakyReLU()

    def forward(self,x,condition):

        out=self.fc1(self.act(self.CBatchNorm1(x,condition)))
        out=self.fc2(self.act(self.CBatchNorm2(out,condition)))
        
        return out

@MODULES.register_module
class ShapePrior(nn.Module):
    """
    Definition of Shape Prior from DOPS paper
    Parameters:
        c_dim           : dimension of conditional latent vector
    """

    def __init__(self, cfg, optim_spec=None):
        super(ShapePrior,self).__init__()

        '''Optimizer parameters used in training'''
        self.optim_spec = optim_spec
        self.cfg = cfg
        '''Definition of the modules'''
        leaky=False
        self.encoder = ResnetPointnet(c_dim=cfg.config['data']['c_dim'],
                                      dim=3,
                                      hidden_dim=cfg.config['data']['hidden_dim'])

        hidden_dim=cfg.config['data']['c_dim']
        self.fc1=nn.Conv1d(3,hidden_dim,1)
        self.dblock1=DecoderBlock(c_dim=cfg.config['data']['c_dim'],
                                  hidden_dim=hidden_dim,
                                  leaky=leaky)
        self.dblock2=DecoderBlock(c_dim=cfg.config['data']['c_dim'],
                                  hidden_dim=hidden_dim,
                                  leaky=leaky)
        self.dblock3=DecoderBlock(c_dim=cfg.config['data']['c_dim'],
                                  hidden_dim=hidden_dim,
                                  leaky=leaky)                                                    
        self.dblock4=DecoderBlock(c_dim=cfg.config['data']['c_dim'],
                                  hidden_dim=hidden_dim,
                                  leaky=leaky)
        self.dblock5=DecoderBlock(c_dim=cfg.config['data']['c_dim'],
                                  hidden_dim=hidden_dim,
                                  leaky=leaky)                                                     
        self.CBatchNorm=CBatchNorm1d(c_dim=cfg.config['data']['c_dim'],
                                     f_dim=hidden_dim)
        self.fc2=nn.Conv1d(hidden_dim,1,1)
        self.act=nn.ReLU()
        if leaky:
            self.act=nn.LeakyReLU()

    
    def generate_latent(self,pc):
        '''
        Generates shape embedding of the point cloud
        Args: 
            pc: point cloud of the form (N x Number of points x 3) 
        Returns:
            self.latent:    shape embedding of size (N x c_dim)     
        '''
        self.latent=self.encoder(pc)
        return self.latent

    def forward(self,query_points):
        '''
        Returns the signed distance of each query point to the surface
        Args: 
            query_points: query points of the form (N x N_P x 3) 
        Returns:
            out:    signed distance of the form  (N x N_P x 1)    
        '''
        query_points=query_points.transpose(1,2)
        out=self.fc1(query_points)
        out=self.dblock1(out,self.latent)
        out=self.dblock2(out,self.latent)
        out=self.dblock3(out,self.latent)
        out=self.dblock4(out,self.latent)
        out=self.dblock5(out,self.latent)
        out=self.act(self.CBatchNorm(out,self.latent))
        out=torch.tanh(self.fc2(out))
        out=out.transpose(1,2)

        return out