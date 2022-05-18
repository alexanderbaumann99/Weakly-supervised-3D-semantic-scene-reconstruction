from models.iscnet.modules.layers import ResnetPointnet,CBatchNorm1d
import torch
import torch.nn as nn
import torch.nn.functional as F

class DecoderBlock(nn.Module):

    def __init__(self,c_dim,hidden_dim=128):
        super(DecoderBlock,self).__init__()

        self.fc1=nn.Linear(hidden_dim,hidden_dim)
        self.fc2=nn.Linear(hidden_dim,hidden_dim)

        self.CBatchNorm1=CBatchNorm1d(c_dim,
                                      f_dim=hidden_dim)
        self.CBatchNorm2=CBatchNorm1d(c_dim,
                                      f_dim=hidden_dim)
        self.act=nn.LeakyReLU()

    def forward(self,x,condition):

        out=self.fc1(self.act(self.CBatchNorm1(x,condition)))
        out=self.fc2(self.act(self.CBatchNorm2(out,condition)))
        
        return out

class ShapePrior(nn.Module):
    """
    Definition of Shape Prior from DOPS paper
    Parameters:
        c_dim           : dimension of conditional latent vector
    """

    def __init__(self,cfg):
        super(ShapePrior,self).__init__()


        self.encoder = ResnetPointnet(c_dim=cfg.config['data']['c_dim'],
                                      dim=3,
                                      hidden_dim=128)

        self.fc1=nn.Linear(3,128)
        self.dblock1=DecoderBlock(c_dim=cfg.config['data']['c_dim'],
                                          hidden_dim=128)
        self.dblock2=DecoderBlock(c_dim=cfg.config['data']['c_dim'],
                                          hidden_dim=128)
        self.dblock3=DecoderBlock(c_dim=cfg.config['data']['c_dim'],
                                          hidden_dim=128)                                  
        self.dblock4=DecoderBlock(c_dim=cfg.config['data']['c_dim'],
                                          hidden_dim=128)
        self.dblock5=DecoderBlock(c_dim=cfg.config['data']['c_dim'],
                                          hidden_dim=128)       
        self.CBatchNorm=CBatchNorm1d(c_dim=cfg.config['data']['c_dim'],
                                     f_dim=128)
        self.fc2=nn.Linear(128,1)
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

    def forward(self,query_points):
        '''
        Returns the signed distance of each query point to the surface
        Args: 
            query_points: query points of the form (batch_size_pc x batch_size_queries x 3) 
        Returns:
            out:    signed distance of the form  (batch_size_pc x batch_size_queries x 1)    
        '''
        out=self.fc1(query_points)
        out=self.dblock1(out,self.latent)
        out=self.dblock2(out,self.latent)
        out=self.dblock3(out,self.latent)
        out=self.dblock4(out,self.latent)
        out=self.dblock5(out,self.latent)
        out=self.act(self.CBatchNorm(out,self.latent))
        out=torch.tanh(self.fc2(out))

        return out