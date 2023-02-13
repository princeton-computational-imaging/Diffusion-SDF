import math
import copy
import torch
from torch import nn, einsum
import torch.nn.functional as F
from inspect import isfunction
from collections import namedtuple
from functools import partial

from torchvision.utils import save_image
from PIL import Image

from torch.utils.data import Dataset, DataLoader

from pathlib import Path
from torch.optim import Adam
from torchvision import transforms as T, utils

from einops import rearrange, reduce
from einops.layers.torch import Rearrange

from tqdm.auto import tqdm

#from accelerate import Accelerator

from model import * 
from diffusion import * 
from utils.helpers import * 
#from sdf_utils import *

#from sdf_model.model import *
import json
import numpy as np
import pandas as pd 
import os
from statistics import mean

from torch.utils.tensorboard import SummaryWriter

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Dataset(Dataset):
    def __init__(self, data_path, split_file, pc_path=None, pc_size=None):
        super().__init__()

        self.cond = pc_path is not None

        if not self.cond:
            self.modulations = unconditional_load_modulations(data_path, split_file)
        else:
            self.modulations, pc_paths = load_modulations(data_path, pc_path, split_file)

        #self.modulations = self.modulations[0:8]
        #pc_paths = pc_paths[0:8]

        print("data shape: ", self.modulations[0].shape)
        print("dataset len: ", len(self.modulations))
        assert args.batch_size <= len(self.modulations)
    
        if self.cond:
            print("loading ground truth point clouds...")            
            lst = []
            with tqdm(pc_paths) as pbar:
                for i, f in enumerate(pc_paths):
                    pbar.set_description("Point clouds loaded: {}/{}".format(i, len(pc_paths)))
                    lst.append(sample_pc(f, pc_size))
            self.point_clouds = lst

            assert len(self.point_clouds) == len(self.modulations)
        
        
    def __len__(self):
        return len(self.modulations)

    def __getitem__(self, index):
        if self.cond:
            return self.modulations[index], self.point_clouds[index]
        else:
            return self.modulations[index], False
    


# trainer class

class Trainer(object):
    def __init__(
        self,
        data_path, split_file, diffusion_specs, model_specs, 
        train_lr = 1e-5, training_iters = 100000, desc='',
        save_and_sample_every = 10000, save_model=True, print_freq=1000,
        pc_path = None, total_pc_size = None, sample_pc_size = None, perturb_pc=None, crop_percent=0.25,
    ):
        super().__init__()

        print("description: ", desc)
        self.model = GaussianDiffusion(
                        model=DiffusionNet(**model_specs).to(device),
                        **diffusion_specs
                        ).to(device)

        self.save_and_sample_every = save_and_sample_every
        self.save_model = save_model
        self.print_freq = print_freq

        self.has_cond = pc_path is not None 

        # total pc size is pc size loaded into ram (e.g. 10000) so we can sample diff points each iter
        # sample pc size is pc size sampled every iteration for training (e.g. 1024)
        self.pc_size = sample_pc_size
        self.perturb_pc = perturb_pc 
        assert self.perturb_pc in [None, "partial", "noisy"]
        #print("perturb pc: ", self.perturb_pc, crop_percent)
        self.crop_percent = crop_percent

        self.batch_size = args.batch_size

        self.training_iters = training_iters

        # optimizer
        self.opt = Adam(self.model.parameters(), lr = train_lr)
      
        self.results_folder = os.path.join(args.exp_dir, "results")
        os.makedirs(self.results_folder, exist_ok=True)
        self.resume = os.path.join(self.results_folder, "{}.pt".format(args.resume)) if args.resume is not None else None

        # step counter state
        self.step = 0

        if self.resume:
            self.step, self.model, self.opt, loss = load_model(self.model, self.opt, self.resume)
            
        
        #num_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        #print("num params: ",num_params)
        #print(self.model)
        
        # dataset and dataloader
        self.ds = Dataset(data_path, split_file, pc_path, total_pc_size)
        
        dl = DataLoader(self.ds, batch_size=self.batch_size, shuffle=True, pin_memory=True, num_workers=args.workers, drop_last=True)
        self.dl = cycle(dl)
        save_code_to_conf(args.exp_dir)
        self.train()



    def train(self):

        writer = SummaryWriter(log_dir=os.path.join("tensorboard_logs", args.exp_dir))

        with tqdm(initial = self.step, total = self.training_iters) as pbar:


            # diff_100 and 1000 loss refers to the losses when t<100 and 100<t<1000, respectively 
            # typically diff_100 approaches 0 while diff_1000 can still be relatively high
            current_loss = 0
            loss_100, loss_500, loss_1000  = [0], [0], [0]
            while self.step < self.training_iters:
                self.model.train()
                data, pc = next(self.dl)
                data = data.to(device)

                if self.has_cond:
                    pc = perturb_point_cloud(pc, self.perturb_pc, self.pc_size, self.crop_percent).cuda()
                else:
                    pc = None

                # sample time 
                t = torch.randint(0, self.model.num_timesteps, (self.batch_size,), device=device).long()
                loss, xt, target, pred, unreduced_loss = self.model(data, t, ret_pred_x=True, cond=pc)

                writer.add_scalar("Train loss", loss, self.step)

                loss.backward()

                pbar.set_description(f'loss: {loss.item():.4f}')

                if self.step%self.print_freq==0:
                    print("avg loss at {} iters: {}".format(self.step, current_loss / self.print_freq))
                    print("losses per time at {} iters: {}, {}, {}".format(self.step, mean(loss_100),mean(loss_500),mean(loss_1000)))
                    writer.add_scalar("loss 100", mean(loss_100), self.step)
                    writer.add_scalar("loss 500", mean(loss_500), self.step)
                    writer.add_scalar("loss 1000", mean(loss_1000), self.step)
                    current_loss = 0
                    loss_100, loss_500, loss_1000  = [0], [0], [0]

                current_loss += loss.detach().item()

                loss_100.extend( unreduced_loss[t<100].cpu().numpy())
                loss_500.extend( unreduced_loss[t<500].cpu().numpy())
                loss_1000.extend( unreduced_loss[t>500].cpu().numpy())

                self.opt.step()
                self.opt.zero_grad()
                self.step += 1
                pbar.update(1)

                if self.step != 0 and self.step % self.save_and_sample_every == 0:
                    save_model(self.step, self.model, self.opt, loss.detach(), "{}/{}.pt".format(self.results_folder, self.step))
                    writer.flush()
        writer.close()

            
if __name__ == "__main__":

    import argparse

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        "--exp_dir", "-e",
        required=True,
        help="This directory should include experiment specifications in 'specs.json,' and logging will be done in this directory as well.",
    )
    arg_parser.add_argument(
        "--resume", "-r",
        default=None,
        help="continue from previous saved logs, integer value or 'last'",
    )

    arg_parser.add_argument(
        "--batch_size", "-b",
        default=32, type=int
    )

    arg_parser.add_argument(
        "--workers", "-w",
        default=0, type=int
    )

    args = arg_parser.parse_args()
    specs = json.load(open(os.path.join(args.exp_dir, "specs.json")))

    Trainer(**specs)

