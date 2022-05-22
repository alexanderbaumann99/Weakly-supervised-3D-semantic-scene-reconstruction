from models.iscnet.modules.shape_prior import DecoderBlock
from models.iscnet.modules.layers import ResnetPointnet, CBatchNorm1d
import torch.nn.functional as F
import torch.nn as nn
import torch



class ShapePrior(nn.Module):
    """
    Definition of Shape Prior from DOPS paper
    Parameters:
        c_dim           : dimension of conditional latent vector
    """

    def __init__(self):
        super(ShapePrior,self).__init__()


        '''Definition of the modules'''
        leaky=False
        self.encoder = ResnetPointnet(c_dim=128,
                                      dim=3,
                                      hidden_dim=512)

        hidden_dim=128
        self.fc1=nn.Conv1d(3,hidden_dim,1)
        self.dblock1=DecoderBlock(c_dim=128,
                                  hidden_dim=hidden_dim,
                                  leaky=leaky)
        self.dblock2=DecoderBlock(c_dim=128,
                                  hidden_dim=hidden_dim,
                                  leaky=leaky)
        self.dblock3=DecoderBlock(c_dim=128,
                                  hidden_dim=hidden_dim,
                                  leaky=leaky)                                                    
        self.dblock4=DecoderBlock(c_dim=128,
                                  hidden_dim=hidden_dim,
                                  leaky=leaky)
        self.dblock5=DecoderBlock(c_dim=128,
                                  hidden_dim=hidden_dim,
                                  leaky=leaky)                                                     
        self.CBatchNorm=CBatchNorm1d(c_dim=128,
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

    def training_epoch(self,loader,optim):
        '''
        Pre-train procedure of the Shape Prior on the ShapeNet dataset
            Args of Loader:
                shape_net:      batches of point clouds (N x N_P x 3)
                query_points:   query points            (N x M_P x 3)  
                gt_val:         Ground truth SDF values (N x M_P x 1)
        '''

        running_loss=0
        for i, data in enumerate(loader):
            shape_net,query_points,gt_val=data
            optim.zero_grad()
            c=self.generate_latent(shape_net)
            preds=self.forward(query_points,c)
            loss=F.mse_loss(preds,gt_val,reduction='mean')
            loss.backward()
            optim.step()

            running_loss+=loss.item()
        epoch_mean=running_loss/(i+1)

        return epoch_mean
    


    
    