from models.iscnet.modules.prior_training import ShapePrior
import torch
from torch.utils.tensorboard import SummaryWriter
from models.iscnet.prior_dataloader import PriorDataLoader
from models.optimizers import load_optimizer
from configs.config_utils import mount_external_config
from configs.config_utils import CONFIG

cfg = CONFIG('configs/config_files/ISCNet.yaml')
cfg = mount_external_config(cfg)
writer = SummaryWriter(log_dir=cfg.save_path)

cfg.log_string('Load data...')
loader = PriorDataLoader(cfg)

cfg.log_string('Load model...')
model=ShapePrior(cfg)
optimizer=load_optimizer(cfg.config,model)

max_epochs=100
cfg.log_string('Start Training...')
for epoch in range(max_epochs):
    epoch_loss=model.training_epoch(loader,optimizer,epoch)
    cfg.log_string("EPOCH %d\t LOSS %.5f" %(epoch+1,epoch_loss))
    writer.add_scalar("Loss/train", epoch_loss, epoch+1)
    if (epoch+1)%10==0:
        torch.save(model.state_dict(), cfg.save_path + "/weights_epoch_"+str(epoch+1))
    
cfg.write_config()
