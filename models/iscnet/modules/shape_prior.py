from models.iscnet.modules.layers import ResnetPointnet,CBatchNorm1d
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConditionalDecoder(nn.Module):

    def __init__(self,c_dim,hidden_dim=128):
        super().__init__()

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
        self.cdecoder1=ConditionalDecoder(c_dim=cfg.config['data']['c_dim'],
                                          hidden_dim=128)
        self.cdecoder2=ConditionalDecoder(c_dim=cfg.config['data']['c_dim'],
                                          hidden_dim=128)
        self.cdecoder3=ConditionalDecoder(c_dim=cfg.config['data']['c_dim'],
                                          hidden_dim=128)                                  
        self.cdecoder4=ConditionalDecoder(c_dim=cfg.config['data']['c_dim'],
                                          hidden_dim=128)
        self.cdecoder5=ConditionalDecoder(c_dim=cfg.config['data']['c_dim'],
                                          hidden_dim=128)       
        self.CBatchNorm=CBatchNorm1d(c_dim=cfg.config['data']['c_dim'],
                                     f_dim=128)
        self.fc2=nn.Linear(128,1)

    
    def generate_latent(self,pc):
        self.latent=self.encoder(pc)

    def forward(self,query_points):

        out=F.leaky_relu(self.fc1(query_points))
        out=self.cdecoder1(out,self.latent)
        out=self.cdecoder2(out,self.latent)
        out=self.cdecoder3(out,self.latent)
        out=self.cdecoder4(out,self.latent)
        out=self.cdecoder5(out,self.latent)
        out=self.CBatchNorm(out,self.latent)
        out=F.tanh(self.fc2(out))

        return out