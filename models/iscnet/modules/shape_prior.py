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

        out=self.CBatchNorm1(x,condition)
        out=self.act(self.fc1(out))
        out=self.CBatchNorm2(out,condition)
        out=self.act(self.fc2(out))

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

    
    def generate_latent(self,pc):
        self.latent=self.encoder(pc)

    def forward(self,query_points):

        out=F.leaky_relu(self.fc1(query_points))
        out=self.dblock1(out,self.latent)
        out=self.dblock2(out,self.latent)
        out=self.dblock3(out,self.latent)
        out=self.dblock4(out,self.latent)
        out=self.dblock5(out,self.latent)
        out=self.CBatchNorm(out,self.latent)
        out=F.tanh(self.fc2(out))

        return out