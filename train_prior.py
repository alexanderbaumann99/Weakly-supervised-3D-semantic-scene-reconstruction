from models.iscnet.modules.prior_training import ShapePrior
import torch
from torch.utils.tensorboard import SummaryWriter
from models.iscnet.modules.prior_dataloader import PriorDataLoader
from configs.config_utils import mount_external_config
from configs.config_utils import CONFIG

cfg = CONFIG('configs/config_files/ISCNet.yaml')
cfg = mount_external_config(cfg)

loader = PriorDataLoader(cfg)
writer = SummaryWriter()
model=ShapePrior()
optimizer = torch.optim.Adam(model.parameters(),lr=1e-4, weight_decay=0)
max_epochs=100

for epoch in range(max_epochs):
    epoch_loss=model.training_epoch(loader,optimizer)
    writer.add_scalar(("Loss/train", epoch_loss, epoch))
    if epoch%10==0:
        torch.save(model.state_dict(), "weights_epoch_"+str(epoch))
    
