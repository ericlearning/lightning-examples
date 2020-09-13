import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
from argparse import ArgumentParser

from pytorch_lightning import TrainResult, EvalResult
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.core import LightningModule
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning import seed_everything

# define architecture
class ConvNet(nn.Module):
    def __init__(self, ic, oc):
        super(ConvNet, self).__init__()
        self.m = nn.Sequential(
            nn.Conv2d(ic, 16, 3, 1, 1),
            nn.BatchNorm2d(16), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, 1, 1),
            nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.cls = nn.Linear(7*7*32, 10)
    
    def forward(self, x):
        bs = x.shape[0]
        o = self.m(x).reshape(bs, -1)
        o = self.cls(o)
        return o

# define model
class Model(LightningModule):
    def __init__(self, ic, oc, lr, batch_size, data_pth, num_workers, **kwargs):
        super(Model, self).__init__()
        print(ic, oc, lr, batch_size, data_pth, num_workers)
        self.save_hyperparameters()
        self.ic = ic
        self.oc = oc
        self.lr = lr
        self.batch_size = batch_size
        self.data_pth = data_pth
        self.num_workers = num_workers
        self.model = ConvNet(self.ic, self.oc)
    
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        x, t = batch
        p = self(x)
        loss = F.cross_entropy(p, t)
        
        train_dict = {'train_loss': loss}
        result = TrainResult(minimize=loss)
        result.log_dict(train_dict)
        return result
    
    def validation_step(self, batch, batch_idx):
        x, t = batch
        p = self(x)
        loss = F.cross_entropy(p, t)
        acc = (p.argmax(-1) == t).sum() / float(x.shape[0])
        
        val_dict = {'val_loss': loss, 'val_acc': acc}
        result = EvalResult()
        result.log_dict(val_dict)
        return result
    
    def validation_epoch_end(self, outputs):
        acc_end = outputs.val_acc.mean()
        result_end = EvalResult(checkpoint_on=acc_end)
        return result_end
        
    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.lr)
    
    def train_dataloader(self):
        tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
        train_ds = datasets.MNIST(root=self.data_pth, train=True, transform=tf, download=True)
        train_dl = DataLoader(train_ds, self.batch_size, shuffle=True, num_workers=self.num_workers)
        return train_dl
    
    def val_dataloader(self):
        tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
        val_ds = datasets.MNIST(root=self.data_pth, train=False, transform=tf, download=True)
        val_dl = DataLoader(val_ds, self.batch_size, shuffle=False, num_workers=self.num_workers)
        return val_dl
    
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser])
        parser.add_argument('--ic', default=1, type=int)
        parser.add_argument('--oc', default=10, type=int)
        parser.add_argument('--lr', default=0.01, type=float)
        parser.add_argument('--batch_size', default=16, type=int)
        parser.add_argument('--data_pth', default='data/', type=str)
        parser.add_argument('--num-workers', default=10, type=int)
        return parser

# logger & callbacks
logger = pl_loggers.TensorBoardLogger('exp_logs/')
callback = ModelCheckpoint(
    save_top_k=1,
    mode='max',
    save_last=True
)

# parse arguments
parser = ArgumentParser(add_help=False)
parser = Trainer.add_argparse_args(parser)
parser.add_argument('--seed', default=1, type=int)
parser = Model.add_model_specific_args(parser)
parser.set_defaults(
    logger=logger,
    max_epochs=20,
    gpus=1
)
args = parser.parse_args()
# print(args)


# set seed
if args.seed is not None:
    seed_everything(args.seed)

# scale batchsize
if args.distributed_backend == 'ddp':
    args.batch_size = args.batch_size // (max(args.gpus, 1) * max(args.num_nodes, 1))
    args.num_workers = args.num_workers // (max(args.gpus, 1) * max(args.num_nodes, 1))
elif args.distributed_backend == 'ddp2':
    args.batch_size = args.batch_size // (max(args.gpus, 1) * max(args.num_nodes, 1))
    args.num_workers = args.num_workers // max(args.num_nodes, 1)

# initialize model and trainer
model = Model(**vars(args))
print(args.checkpoint_callback)
trainer = Trainer.from_argparse_args(args, checkpoint_callback=callback)

# fit the model
trainer.fit(model)