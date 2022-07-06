from models.iscnet.modules.shape_prior import DecoderBlock
from models.iscnet.modules.layers import ResnetPointnet, CBatchNorm1d
import torch.nn.functional as F
import torch.nn as nn
import torch
from net_utils.sampling import farthest_point_sampler_batch
import numpy as np



class ShapePrior(nn.Module):
    """
    Definition of Shape Prior from DOPS paper
    Parameters:
        c_dim           : dimension of conditional latent vector
    """

    def __init__(self,cfg,device):
        super(ShapePrior,self).__init__()
        
        self.cfg=cfg
        self.clamp_dist = cfg.config['model']['shape_prior']["ClampingDistance"]
        self.device = device
        '''Definition of the modules'''
        leaky=False
        self.encoder = ResnetPointnet(c_dim=cfg.config['data']['c_dim'],
                                      dim=3,
                                      hidden_dim=512)

        hidden_dim=128
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
        #return self.latent

    def forward(self,query_points,pc):
        '''
        Encodes the point cloud and computes the signed distance of each query point
        Args: 
            query_points: query points of the form (N x N_P x 3) 
            pc:           point cloud of the form (N x N_P x 3) 
        Returns:
            out:    signed distance of the form  (N x N_P x 1)    
        '''
        self.latent=self.encoder(pc)
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
    
    def compute_sdf(self,query_points):
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
    
    def save_shape_embedding(self,loader):

        num_cats=8
        emb_per_cat = torch.zeros((num_cats,self.cfg.config['data']['c_dim_prior'])).to(self.device)
        n_obj_per_cat = torch.zeros((num_cats,)).to(self.device)

        for i, data in enumerate(loader):
            point_cloud = data['point_cloud'].to(self.device)
            cat = data['sem_cls_id']
            with torch.no_grad():
                shape_embs=self.encoder(point_cloud)
            for j in range(shape_embs.shape[0]):
                emb_per_cat[int(cat[j]),:]+=shape_embs[j,:]
                n_obj_per_cat[int(cat[j])]+=1
        
        print(n_obj_per_cat)
        print(emb_per_cat)
        for j in range(n_obj_per_cat.shape[0]): 
            emb_per_cat[j] /= n_obj_per_cat[j]
        torch.save(emb_per_cat,self.cfg.config['data']['embedding_path'])


def training_epoch(model,loader,optim,epoch,device,cfg):
    '''
    Pre-train procedure of the Shape Prior on the ShapeNet dataset
        Args of Loader:
            shape_net:      batches of point clouds (N x N_P x 3)
            query_points:   query points            (N x M_P x 3)  
            gt_val:         Ground truth SDF values (N x M_P x 1)
    '''

    running_loss=0
    for i, data in enumerate(loader):
        point_cloud = data['point_cloud'].to(device)
        query_points = data['query_points'].to(device)
        gt_sdf = data['sdf'].to(device)
        '''
        point_cloud = data['point_cloud'].to(self.device)
        query_points_surface = data['query_points_surface'].to(self.device)
        query_points_sphere = data['query_points_sphere'].to(self.device)
        sdf_surface = data['sdf_surface'].to(self.device)
        sdf_sphere = data['sdf_sphere'].to(self.device)
        query_points,gt_sdf = self.get_batch(query_points_surface,query_points_sphere,sdf_surface,sdf_sphere,1024,1024)
        '''

        optim.zero_grad()
        preds=model(query_points,point_cloud)
        preds=preds.squeeze()
        loss=F.mse_loss(preds,torch.sign(gt_sdf),reduction='mean')
        #preds = torch.clamp(preds, -self.clamp_dist, self.clamp_dist)
        #loss = F.l1_loss(preds,gt_sdf,reduction='mean') 
        loss.backward()
        optim.step()
        

        running_loss+=loss.item()
        if (i+1)%10==0:
            cfg.log_string("EPOCH %d\t ITER %d\t LOSS %.3f" %(epoch+1,i+1,running_loss/(i+1)))
    epoch_mean=running_loss/(i+1)

    return epoch_mean

def testing_epoch(model,loader,epoch,device,cfg):
    '''
    Pre-train procedure of the Shape Prior on the ShapeNet dataset
        Args of Loader:
            shape_net:      batches of point clouds (N x N_P x 3)
            query_points:   query points            (N x M_P x 3)  
            gt_val:         Ground truth SDF values (N x M_P x 1)
    '''
    running_loss=0
    for i, data in enumerate(loader):
        point_cloud = data['point_cloud'].to(device)
        query_points = data['query_points'].to(device)
        gt_sdf = data['sdf'].to(device)
        '''
        point_cloud = data['point_cloud'].to(self.device)
        query_points_surface = data['query_points_surface'].to(self.device)
        query_points_sphere = data['query_points_sphere'].to(self.device)
        sdf_surface = data['sdf_surface'].to(self.device)
        sdf_sphere = data['sdf_sphere'].to(self.device)
        query_points,gt_sdf = self.get_batch(query_points_surface,query_points_sphere,sdf_surface,sdf_sphere,1024,1024)
        '''

        with torch.no_grad():
            preds=model(query_points,point_cloud)
            preds=preds.squeeze()
            loss=F.mse_loss(preds,torch.sign(gt_sdf),reduction='mean')
            #preds = torch.clamp(preds, -self.clamp_dist, self.clamp_dist)
            #loss = F.l1_loss(preds,gt_sdf,reduction='mean') 
        running_loss+=loss.item()
        if (i+1)%10==0:
            cfg.log_string("VALIDATION \t EPOCH %d\t ITER %d\t LOSS %.3f" %(epoch+1,i+1,running_loss/(i+1)))
    epoch_mean=running_loss/(i+1)

    return epoch_mean


    


    
    
