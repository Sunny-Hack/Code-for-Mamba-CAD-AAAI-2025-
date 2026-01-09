import os
import numpy as np
import torch
import torch.autograd as autograd
import torch.optim as optim
from tqdm import tqdm
from .base import BaseTrainer
from model.latentflow import RealNVP, RealNVPLayer
from utils import cycle


class TrainerLatentFLOW(BaseTrainer):
    def __init__(self, cfg):
        super(TrainerLatentFLOW, self).__init__(cfg)
        self.batch_size = cfg.batch_size
        self.save_frequency = cfg.save_frequency
        self.n_iters = cfg.n_iters
        self.n_dim = cfg.n_dim
        self.hidden_dim = cfg.hidden_dim
        self.num_features = cfg.num_features
        self.set_optimizer(cfg)
        self.build_net(cfg)
    
    def build_net(self, cfg):
        self.net = RealNVP(cfg.num_features, cfg.hidden_dim).cuda()
    
    def eval(self):
        self.net.eval()

    def set_optimizer(self, cfg):
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=0.001)
    ###save ckpt####
    def save_ckpt(self, name=None):
        """save checkpoint during training for future restore"""
        if name is None:
            save_path = os.path.join(self.model_dir, "ckpt_epoch{}.pth".format(self.clock.step))
            print("Saving checkpoint epoch {}...".format(self.clock.step))
        else:
            save_path = os.path.join(self.model_dir, "{}.pth".format(name))

        torch.save({
            'clock': self.clock.make_checkpoint(),
            'net_state_dict': self.net.cpu().state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, save_path)

        self.net.cuda()   
    ###save ckpt####
    ###load ckpt#####
    def load_ckpt(self, name=None):
        name = name if name == 'latest' else "ckpt_epoch{}".format(name)
        load_path = os.path.join(self.model_dir, "{}.pth".format(name))
        if not os.path.exists(load_path):
            raise ValueError("Checkpoint {} not exists.".format(load_path))

        checkpoint = torch.load(load_path)
        print("Loading checkpoint from {} ...".format(load_path))
        self.net.load_state_dict(checkpoint['net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.clock.restore_checkpoint(checkpoint['clock'])
    ###load ckpt######
    
    def train(self, dataloader):
        data = cycle(dataloader)
        pbar = tqdm(range(self.clock.step, self.n_iters))
        for iteration in pbar:
            train_data = next(data)
            train_data = train_data.cuda()
            self.optimizer.zero_grad()
            predict = self.net(train_data)
            log_det_jacobian = sum(layer.scale.sum() for layer in self.net.layers)
            log_likelihood = -0.5 * torch.sum(predict**2, dim=1) - 0.5 * train_data.shape[1] * torch.log(torch.tensor(2 * torch.pi)) + log_det_jacobian
            loss = -log_likelihood.mean()
            loss.backward()
            self.optimizer.step()

            pbar.set_postfix({"loss": loss.item()})
            self.train_tb.add_scalars("loss", {"D_loss": loss.item()}, global_step=self.clock.step)

            #save model
            self.clock.tick()
            if self.clock.step % self.save_frequency == 0:
                self.save_ckpt()

    def generate(self, n_samples):
        self.eval()

        chunk_num = n_samples // self.batch_size
        generated_z = []
        for i in range(chunk_num):
            noise = torch.randn(self.batch_size, self.n_dim, 128).cuda()
            with torch.no_grad():
                fake_z = self.net(noise, reverse=True)
            fake_z = fake_z.detach().cpu().numpy()
            generated_z.append(fake_z)
            print("chunk {} finished.".format(i))
        remains = n_samples - self.batch_size * chunk_num
        noise = torch.randn(remains, self.n_dim, 128).cuda()
        with torch.no_grad():
            fake_z = self.net(noise, reverse=True)
            fake_z = fake_z.detach().cpu().numpy()
        generated_z.append(fake_z)
        return generated_z
