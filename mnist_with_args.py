import os
import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl

from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import loggers as pl_loggers

from datetime import datetime
from argparse import ArgumentParser
from omegaconf import OmegaConf

class SimpleNet(nn.Module):
    def __init__(self, ic, hc, oc):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(ic, hc), nn.ReLU(),
            nn.Linear(hc, hc), nn.ReLU(),
            nn.Linear(hc, oc)
        )
    
    def forward(self, x):
        return self.net(x.reshape(-1, 28*28))
        

class BoilerNet(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.save_hyperparameters(args)

        cfg = OmegaConf.load(args.config)
        self.cfg = cfg
        
        self.model = SimpleNet(cfg.model.ic, cfg.model.hc, cfg.model.oc)
        self.loss = nn.CrossEntropyLoss()
    
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        pred = self(x)
        loss = self.loss(pred, y)

        output = {"loss": loss}
        
        self.logger.experiment.add_scalar("Training Loss", loss.item(), self.global_step)
        
        return output
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        pred = self(x)
        loss = self.loss(pred, y)
        
        acc = (pred.argmax(1) == y).sum() / float(x.shape[0])

        output = {
            "batch_val_loss": loss,
            "batch_val_acc": acc
        }
        
        return output
    
    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([o["batch_val_loss"] for o in outputs]).mean()
        avg_acc = torch.stack([o["batch_val_acc"] for o in outputs]).mean()
        
        output = {
            "val_loss": avg_loss,
            "val_acc": avg_acc
        }
        
        self.logger.experiment.add_scalar("Validation Loss", avg_loss.item(), self.global_step)
        self.logger.experiment.add_scalar("Validation Acc", avg_acc.item(), self.global_step)
        
        return output
    
    def configure_optimizers(self):
        lr = self.cfg.training.lr
        beta1 = self.cfg.training.beta1
        beta2 = self.cfg.training.beta2
        
        opt = optim.Adam(self.model.parameters(), lr=lr, betas=(beta1, beta2))
        return opt
    
    def train_dataloader(self):
        num_gpus = self.hparams.num_gpus
        num_nodes = self.hparams.num_nodes
        dist_mode = self.hparams.dist_mode
        grad_acc = self.hparams.grad_acc
        bs, n_workers = self.cfg.training.bs, self.cfg.training.n_workers
        bs = inv_effective_bs(bs, num_gpus, num_nodes, dist_mode, grad_acc)

        tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
        ds = datasets.MNIST(self.cfg.data.train_dir, train=True, download=True, transform=tf)
        return DataLoader(ds, batch_size=bs, num_workers=n_workers)
    
    def val_dataloader(self):
        num_gpus = self.hparams.num_gpus
        num_nodes = self.hparams.num_nodes
        dist_mode = self.hparams.dist_mode
        grad_acc = self.hparams.grad_acc
        bs, n_workers = self.cfg.training.bs, self.cfg.training.n_workers
        bs = inv_effective_bs(bs, num_gpus, num_nodes, dist_mode, grad_acc)

        tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
        ds = datasets.MNIST(self.cfg.data.val_dir, train=False, download=True, transform=tf)
        return DataLoader(ds, batch_size=bs, num_workers=n_workers)



def effective_bs(bs, num_gpus, num_nodes, dist_mode, grad_acc):
    if dist_mode == 'dp':
        eff_bs = bs
    elif dist_mode == 'ddp' or dist_mode == 'horovod':
        eff_bs = bs * num_gpus * num_nodes
    elif dist_mode == 'ddp2':
        eff_bs = bs * num_nodes
    
    eff_bs *= grad_acc
    return eff_bs

def inv_effective_bs(eff_bs, num_gpus, num_nodes, dist_mode, grad_acc):
    if dist_mode == 'dp':
        bs = eff_bs
    elif dist_mode == 'ddp' or dist_mode == 'horovod':
        bs = eff_bs // num_gpus // num_nodes
    elif dist_mode == 'ddp2':
        bs = eff_bs // num_nodes
    
    bs //= grad_acc
    return bs

parser = ArgumentParser()
parser.add_argument('-c', '--config', type=str, default='config.yaml', help='.yaml config file')
parser.add_argument('-g', '--num-gpus', type=int, default=2, help='gpus')
parser.add_argument('-n', '--num-nodes', type=int, default=1, help='nodes')
parser.add_argument('-d', '--dist-mode', type=str, default='ddp', help='distributed modes')
parser.add_argument('-a', '--grad-acc', type=int, default=1, help='accumulated gradients')

parser.add_argument('-m', '--model-path-ckpt', type=str, help='model checkpoint path')
parser.add_argument('-r', '--resume-path-ckpt', type=str, default=None, help='resume training checkpoint path')
parser.add_argument('-e', '--experiment-name', type=str, default=None, help='experiment name')
parser.add_argument('-t', '--top-k-save', type=int, default=5, help='save top k')
parser.add_argument('-f', '--fast-dev-run', action='store_true', help='perform fast dev run')


args = parser.parse_args()
model = BoilerNet(args)
cfg = OmegaConf.load(args.config)

experiment_name = args.experiment_name
if experiment_name is None:
    experiment_name = datetime.now().strftime("%m%d%Y-%H:%M:%S")

ckpt_pth = os.path.join(args.model_path_ckpt, experiment_name)
log_dir = os.path.join(cfg.logging.log_dir, experiment_name)

os.makedirs(ckpt_pth, exist_ok=True)
ckpt_callback = ModelCheckpoint(
    filepath=ckpt_pth,
    monitor='val_loss',
    verbose=True,
    save_top_k=args.top_k_save
)

trainer = Trainer(
    logger=pl_loggers.TensorBoardLogger(log_dir),
    checkpoint_callback=ckpt_callback,
    weights_save_path=ckpt_pth,
    gpus=args.num_gpus,
    accumulate_grad_batches=args.grad_acc,
    distributed_backend='ddp',
    resume_from_checkpoint=args.resume_path_ckpt,
    gradient_clip_val=cfg.training.gradient_clip,
    fast_dev_run=args.fast_dev_run,
    max_epochs=cfg.training.epoch_num
)

trainer.fit(model)