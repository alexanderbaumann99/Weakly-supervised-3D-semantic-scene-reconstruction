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

    def __init__(self,cfg):
        super(ShapePrior,self).__init__()
        
        self.cfg=cfg
        self.clamp_dist = cfg.config['model']['shape_prior']["ClampingDistance"]
        cuda=True
        if cuda:
            self.device=torch.device('cuda')
        '''Definition of the modules'''
        leaky=False
        self.encoder = ResnetPointnet(c_dim=cfg.config['data']['c_dim'],
                                      dim=3,
                                      hidden_dim=512).to(self.device)

        hidden_dim=128
        self.fc1=nn.Conv1d(3,hidden_dim,1).to(self.device)
        self.dblock1=DecoderBlock(c_dim=cfg.config['data']['c_dim'],
                                  hidden_dim=hidden_dim,
                                  leaky=leaky).to(self.device)
        self.dblock2=DecoderBlock(c_dim=cfg.config['data']['c_dim'],
                                  hidden_dim=hidden_dim,
                                  leaky=leaky).to(self.device)
        self.dblock3=DecoderBlock(c_dim=cfg.config['data']['c_dim'],
                                  hidden_dim=hidden_dim,
                                  leaky=leaky).to(self.device)                                                   
        self.dblock4=DecoderBlock(c_dim=cfg.config['data']['c_dim'],
                                  hidden_dim=hidden_dim,
                                  leaky=leaky).to(self.device)
        self.dblock5=DecoderBlock(c_dim=cfg.config['data']['c_dim'],
                                  hidden_dim=hidden_dim,
                                  leaky=leaky).to(self.device)                                                     
        self.CBatchNorm=CBatchNorm1d(c_dim=cfg.config['data']['c_dim'],
                                     f_dim=hidden_dim).to(self.device)
        self.fc2=nn.Conv1d(hidden_dim,1,1).to(self.device)
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

    def training_epoch(self,loader,optim,epoch):
        '''
        Pre-train procedure of the Shape Prior on the ShapeNet dataset
            Args of Loader:
                shape_net:      batches of point clouds (N x N_P x 3)
                query_points:   query points            (N x M_P x 3)  
                gt_val:         Ground truth SDF values (N x M_P x 1)
        '''

        running_loss=0
        for i, data in enumerate(loader):
            """ point_cloud = data['point_cloud'].to(self.device)
            query_points = data['query_points'].to(self.device)
            gt_sdf = data['sdf'].to(self.device) """
            point_cloud = data['point_cloud'].to(self.device)
            query_points_surface = data['query_points_surface'].to(self.device)
            query_points_sphere = data['query_points_sphere'].to(self.device)
            sdf_surface = data['sdf_surface'].to(self.device)
            sdf_sphere = data['sdf_sphere'].to(self.device)
            query_points,gt_sdf = self.get_batch(query_points_surface,query_points_sphere,sdf_surface,sdf_sphere,1024,1024)
         
            query_points = query_points
            gt_sdf = gt_sdf

            optim.zero_grad()
            self.generate_latent(point_cloud)
            preds=self.forward(query_points)
            preds=preds.squeeze()
            loss=F.mse_loss(preds,torch.sign(gt_sdf),reduction='mean')
            #preds = torch.clamp(preds, -self.clamp_dist, self.clamp_dist)
            #loss = F.l1_loss(preds,gt_sdf,reduction='mean') 
            loss.backward()
            optim.step()

            running_loss+=loss.item()
            if (i+1)%10==0:
                self.cfg.log_string("EPOCH %d\t ITER %d\t LOSS %.3f" %(epoch+1,i+1,running_loss/(i+1)))
        epoch_mean=running_loss/(i+1)

        return epoch_mean

    def testing_epoch(self,loader,epoch):
        '''
        Pre-train procedure of the Shape Prior on the ShapeNet dataset
            Args of Loader:
                shape_net:      batches of point clouds (N x N_P x 3)
                query_points:   query points            (N x M_P x 3)  
                gt_val:         Ground truth SDF values (N x M_P x 1)
        '''
        running_loss=0
        for i, data in enumerate(loader):
            """ point_cloud = data['point_cloud'].to(self.device)
            query_points = data['query_points'].to(self.device)
            gt_sdf = data['sdf'].to(self.device) """
            point_cloud = data['point_cloud'].to(self.device)
            query_points_surface = data['query_points_surface'].to(self.device)
            query_points_sphere = data['query_points_sphere'].to(self.device)
            sdf_surface = data['sdf_surface'].to(self.device)
            sdf_sphere = data['sdf_sphere'].to(self.device)
            query_points,gt_sdf = self.get_batch(query_points_surface,query_points_sphere,sdf_surface,sdf_sphere,1024,1024)

            query_points = query_points
            gt_sdf = gt_sdf
            
            with torch.no_grad():
                self.generate_latent(point_cloud)
                preds=self.forward(query_points)
                preds=preds.squeeze()
                loss=F.mse_loss(preds,torch.sign(gt_sdf),reduction='mean')
                #preds = torch.clamp(preds, -self.clamp_dist, self.clamp_dist)
                #loss = F.l1_loss(preds,gt_sdf,reduction='mean') 
            running_loss+=loss.item()
            if (i+1)%10==0:
                self.cfg.log_string("VALIDATION \t EPOCH %d\t ITER %d\t LOSS %.3f" %(epoch+1,i+1,running_loss/(i+1)))
        epoch_mean=running_loss/(i+1)

        return epoch_mean

    def get_batch(self,query_points_surface,query_points_sphere,sdf_surface,sdf_sphere,n_surface,n_sphere):
        batch_size=query_points_surface.shape[0]

        idx=farthest_point_sampler_batch(query_points_surface,n_surface)
        query_points_surface_sampled=torch.empty((batch_size,n_surface,3)).to(self.device)
        sdf_surface_sampled=torch.empty((batch_size,n_surface)).to(self.device)
        for j in range(batch_size):
            query_points_surface_sampled[j,:,:]=query_points_surface[j,idx[j,:],:]
            sdf_surface_sampled[j,:]=sdf_surface[j,idx[j,:]]
        
        idx = np.random.choice(query_points_sphere.shape[1], (query_points_sphere.shape[0],n_sphere), replace=False)
        query_points_sphere_sampled=torch.empty((batch_size,n_sphere,3)).to(self.device)
        sdf_sphere_sampled=torch.empty((batch_size,n_surface)).to(self.device)
        for j in range(batch_size):
            query_points_sphere_sampled[j,:,:]=query_points_sphere[j,idx[j,:],:]
            sdf_sphere_sampled[j,:]=sdf_sphere[j,idx[j,:]]

        query_points = torch.cat([query_points_surface_sampled,query_points_sphere_sampled],axis=1)
        sdf = torch.cat([sdf_surface_sampled,sdf_sphere_sampled],axis=1)
        
        return query_points,sdf

    


    
    
