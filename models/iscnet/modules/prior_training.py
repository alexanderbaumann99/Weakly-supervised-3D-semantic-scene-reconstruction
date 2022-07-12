# Shape Prior for training.
# authors: Alexander Baumann, Sophia Wagner
# date: Jul, 2022
from models.iscnet.modules.shape_prior import DecoderBlock
from models.iscnet.modules.layers import ResnetPointnet, CBatchNorm1d
from models.iscnet.modules.generator import Generator3D
from models.loss import chamfer_func

import torch.nn.functional as F
import torch.nn as nn
import torch

ScanNet2Cat = {'0':'table','1':'chair','2':'bookshelf','3':'sofa','4':'trash bin','5':'cabinet','6':'display','7':'bathtub' }


class ShapePrior(nn.Module):
    """
    Definition of Shape Prior from DOPS paper
    Parameters:
        c_dim           : dimension of conditional latent vector
    """

    def __init__(self,cfg,device):
        super(ShapePrior,self).__init__()
        
        self.cfg=cfg
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
            
        self.generator = Generator3D(self,    threshold=cfg.config['data']['threshold'],
                                              resolution0=cfg.config['generation']['resolution_0'],
                                              upsampling_steps=cfg.config['generation']['upsampling_steps'],
                                              sample=cfg.config['generation']['use_sampling'],
                                              refinement_step=cfg.config['generation']['refinement_step'],
                                              simplify_nfaces=cfg.config['generation']['simplify_nfaces'],
                                              preprocessor=None)
    
    def generate_latent(self,pc):
        '''
        Generates shape embedding of the point cloud
        Args: 
            pc: point cloud of the form (N x Number of points x 3) 
        Returns:
            self.latent:    shape embedding of size (N x c_dim)     
        '''
        self.latent=self.encoder(pc)
    
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
    
    def save_mean_embeddings(self,loader):

        num_cats=8
        emb_per_cat = torch.zeros((num_cats,self.cfg.config['data']['c_dim'])).to(self.device)
        n_obj_per_cat = torch.zeros((num_cats,)).to(self.device)

        for i, data in enumerate(loader):
            point_cloud = data['point_cloud'].to(self.device)
            cat = data['sem_cls_id']
            with torch.no_grad():
                shape_embs=self.encoder(point_cloud)
            for j in range(shape_embs.shape[0]):
                emb_per_cat[int(cat[j]),:]+=shape_embs[j,:]
                n_obj_per_cat[int(cat[j])]+=1
        
        for j in range(n_obj_per_cat.shape[0]): 
            emb_per_cat[j] /= n_obj_per_cat[j]
        torch.save(emb_per_cat,self.cfg.config['data']['mean_embedding_path'])

    def save_all_embeddings(self,loader):

        num_cats = 8
        embeddings = []
        for _ in range(num_cats):
            embeddings.append([])

        for i, data in enumerate(loader):
            point_cloud = data['point_cloud'].to(self.device)
            cat = data['sem_cls_id']
            with torch.no_grad():
                shape_embs=self.encoder(point_cloud)
            for j in range(shape_embs.shape[0]):
                embeddings[int(cat[j])].append(shape_embs[j,:])
        
        torch.save(embeddings,self.cfg.config['data']['all_embedding_path'])


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

        optim.zero_grad()
        preds=model(query_points,point_cloud)
        preds=preds.squeeze()
        loss=F.mse_loss(preds,torch.sign(gt_sdf),reduction='mean')
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

        with torch.no_grad():
            preds=model(query_points,point_cloud)
            preds=preds.squeeze()
            loss=F.mse_loss(preds,torch.sign(gt_sdf),reduction='mean')
        running_loss+=loss.item()
        if (i+1)%10==0:
            cfg.log_string("VALIDATION \t EPOCH %d\t ITER %d\t LOSS %.3f" %(epoch+1,i+1,running_loss/(i+1)))
    epoch_mean=running_loss/(i+1)

    return epoch_mean

def evaluation_epoch(model,loader,device,cfg):
    '''
    Pre-train procedure of the Shape Prior on the ShapeNet dataset
        Args of Loader:
            shape_net:      batches of point clouds (N x N_P x 3)
            query_points:   query points            (N x M_P x 3)  
            gt_val:         Ground truth SDF values (N x M_P x 1)
    '''

    max_obj_points = 50000
    loss_per_cat = torch.zeros((8,2)).to(device)
    total_loss = 0
    n=0
    
    for i, data in enumerate(loader):
        point_cloud = data['point_cloud'].to(device)
        query_points = data['query_points'].to(device)
        gt_sdf = data['sdf'].to(device)
        cat = data['sem_cls_id']
        
        obj_points_matrix = torch.zeros((point_cloud.shape[0],max_obj_points, 3)).to(device)
        with torch.no_grad():
            embeddings=model.module.encoder(point_cloud)
            meshes = model.module.generator.generate_mesh(embeddings)
            
        points_of_meshes = [torch.Tensor(mesh.vertices).to(device) for mesh in meshes]
        for j,verts in enumerate(points_of_meshes):
            obj_points_matrix[j,:verts.shape[0],:] = verts

        d1,d2 = chamfer_func(obj_points_matrix,point_cloud) 
        loss = torch.mean(d2,axis=1)
        total_loss += torch.sum(loss).item()
        n+=loss.shape[0]
        
        for batch_id in range(loss.shape[0]):
            loss_per_cat[int(cat[batch_id]),0] += loss[batch_id]
            loss_per_cat[int(cat[batch_id]),1] += 1

    loss_per_cat[:,0] /= loss_per_cat[:,1]

    cfg.log_string('---average chamfer distances per category---') 
    for i in range(8):
        cfg.log_string(f'{ScanNet2Cat[str(i)]:15} : {loss_per_cat[i,0]:10f}')
    cfg.log_string('') 
    cfg.log_string(f'{"Total average":15} : {total_loss/(n):10f}')
    
    


    
    
