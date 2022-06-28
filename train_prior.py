from models.iscnet.modules.prior_training import ShapePrior
import torch
from torch.utils.tensorboard import SummaryWriter
#from models.iscnet.prior_dataloader import PriorDataLoader,ShapeNetDataset
from models.iscnet.prior_dataloader import PriorDataLoader,ShapeNetDataset
from models.optimizers import load_optimizer,load_bnm_scheduler,load_scheduler
from configs.config_utils import mount_external_config
from configs.config_utils import CONFIG

cfg = CONFIG('configs/config_files/ISCNet.yaml')
cfg = mount_external_config(cfg)
writer = SummaryWriter(log_dir=cfg.save_path)

cfg.log_string('Load data...')
train_loader,val_loader = PriorDataLoader(cfg,splits=[0.75,0.25])

cfg.log_string('Load model...')
model=ShapePrior(cfg)
if cfg.config['resume']: 
    model.load_state_dict(torch.load(cfg.config['weight_prior']))
    print("... model loaded")
optimizer=load_optimizer(cfg.config,model)
scheduler = load_scheduler(config=cfg.config, optimizer=optimizer)
'''BN momentum scheduler'''
bnm_scheduler = load_bnm_scheduler(cfg=cfg, net=model, start_epoch=scheduler.last_epoch)


cfg.log_string('Start Training...')
max_epochs=cfg.config['train']['epochs']
for epoch in range(max_epochs):
    lrs = [optimizer.param_groups[i]['lr'] for i in range(len(optimizer.param_groups))]
    cfg.log_string('Current learning rates are: ' + str(lrs) + '.')
    bnm_scheduler.show_momentum()
    epoch_loss_train=model.training_epoch(train_loader,optimizer,epoch)
    cfg.log_string("TRAINING\t EPOCH %d\t LOSS %.5f" %(epoch+1,epoch_loss_train))
    epoch_loss_val=model.testing_epoch(val_loader,epoch)
    cfg.log_string("VALIDATION \t EPOCH %d\t LOSS %.5f" %(epoch+1,epoch_loss_val))
    scheduler.step(epoch_loss_val)
    bnm_scheduler.step()
    writer.add_scalar("Loss/train", epoch_loss_train, epoch+1)
    writer.add_scalar("Loss/val",  epoch_loss_val, epoch+1)
    if (epoch+1)%2==0:
        torch.save(model.state_dict(), cfg.save_path + "/weights_epoch_"+str(epoch+1))
cfg.write_config()


#model.save_shape_embedding(train_loader)