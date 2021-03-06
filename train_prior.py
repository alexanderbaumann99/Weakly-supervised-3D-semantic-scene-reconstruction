# training of the shape prior.
# authors: Alexander Baumann, Sophia Wagner
# date: Jul, 2022

from models.iscnet.modules.prior_training import ShapePrior,testing_epoch,training_epoch,evaluation_epoch
import torch
from torch.utils.tensorboard import SummaryWriter
from models.iscnet.prior_dataloader import PriorDataLoader,ShapeNetDataset
from models.optimizers import load_optimizer,load_bnm_scheduler,load_scheduler
from configs.config_utils import mount_external_config
from configs.config_utils import CONFIG

cfg = CONFIG('configs/config_files/ISCNet_prior.yaml')
cfg = mount_external_config(cfg)
writer = SummaryWriter(log_dir=cfg.save_path)

if cfg.config['device']['use_gpu']:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
    
cfg.log_string('Load data...')
train_loader,val_loader = PriorDataLoader(cfg,splits=[1.0,0.0])

cfg.log_string('Load model...')
model=ShapePrior(cfg,device)
if cfg.config['weight_prior'] is not None:
    model.load_state_dict(torch.load(cfg.config['weight_prior']))
model = torch.nn.DataParallel(model).to(device)

if 'train' in cfg.config['modes']:
    optimizer=load_optimizer(cfg.config,model)
    scheduler = load_scheduler(config=cfg.config, optimizer=optimizer)
    bnm_scheduler = load_bnm_scheduler(cfg=cfg, net=model, start_epoch=scheduler.last_epoch)
    
    cfg.log_string('Start Training...')
    max_epochs=cfg.config['train']['epochs']
    for epoch in range(max_epochs):
        lrs = [optimizer.param_groups[i]['lr'] for i in range(len(optimizer.param_groups))]
        cfg.log_string('Current learning rates are: ' + str(lrs) + '.')
        bnm_scheduler.show_momentum()
        epoch_loss_train=training_epoch(model,train_loader,optimizer,epoch,device,cfg)
        cfg.log_string("TRAINING\t EPOCH %d\t LOSS %.5f" %(epoch+1,epoch_loss_train))
        scheduler.step(epoch_loss_train)
        bnm_scheduler.step()
        writer.add_scalar("Loss/train", epoch_loss_train, epoch+1)
        torch.save(model.module.state_dict(), cfg.save_path + "/weights_epoch_last") 
        
if 'save' in cfg.config['modes']:
    model.eval()
    if cfg.config['data']['mean_embeddings']:
        model.module.save_mean_embeddings(train_loader)
    else:
        model.module.save_all_embeddings(train_loader)

if 'eval' in cfg.config['modes']:
    model.eval()
    evaluation_epoch(model,train_loader,device,cfg)
    
cfg.write_config()
