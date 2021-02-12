from cs285.infrastructure import pytorch_util as ptu
from .base_exploration_model import BaseExplorationModel
import torch.optim as optim
from torch import nn
import torch
import numpy as np

def init_method_1(model):
    model.weight.data.uniform_()
    model.bias.data.uniform_()

def init_method_2(model):
    model.weight.data.normal_()
    model.bias.data.normal_()


class RNDModel(nn.Module, BaseExplorationModel):
    def __init__(self, hparams, optimizer_spec, **kwargs):
        super().__init__(**kwargs)
        self.ob_dim = hparams['ob_dim']
        self.output_size = hparams['rnd_output_size']
        self.n_layers = hparams['rnd_n_layers']
        self.size = hparams['rnd_size']
        self.optimizer_spec = optimizer_spec

        # TODO: Create two neural networks:
        # 1) f, the random function we are trying to learn
        # 2) f_hat, the function we are using to learn f
        # WARNING: Make sure you use different types of weight 
        #          initializations for these two functions

        # HINT 1) Check out the method ptu.build_mlp
        # HINT 2) There are two weight init methods defined above

        self.f = ptu.build_mlp(self.ob_dim, self.output_size, self.n_layers, self.size, init_method=init_method_1)
        self.f_hat = ptu.build_mlp(self.ob_dim, self.output_size, self.n_layers, self.size, init_method=init_method_2)
        
        self.optimizer = self.optimizer_spec.constructor(
            self.f_hat.parameters(),
            **self.optimizer_spec.optim_kwargs
        )
        self.learning_rate_scheduler = optim.lr_scheduler.LambdaLR(
            self.optimizer,
            self.optimizer_spec.learning_rate_schedule,
        )

        self.f.to(ptu.device)
        self.f_hat.to(ptu.device)

    def forward(self, ob_no):
        # TODO: Get the prediction error for ob_no
        # HINT: Remember to detach the output of self.f!
        f_out = self.f(ob_no).detach()
        f_hat_out = self.f_hat(ob_no)
        error = torch.norm(f_out - f_hat_out, dim=1)  # mean error over ob_dim for each item in the batch
        return error

    def forward_np(self, ob_no):
        ob_no = ptu.from_numpy(ob_no)
        error = self(ob_no)
        return ptu.to_numpy(error)

    def update(self, ob_no):
        # TODO: Update f_hat using ob_no
        # Hint: Take the mean prediction error across the batch
        ob_no = ptu.from_numpy(ob_no)
        error = self.forward(ob_no)
        loss = torch.mean(error)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()


class DistanceModel(nn.Module, BaseExplorationModel):
    def __init__(self, hparams, batch_size):
        self.ob_dim = hparams['ob_dim']
        self.mean = np.zeros((1, self.ob_dim))
        self.time_discount = 0.8

    def forward_np(self, ob_no):
        return np.linalg.norm(ob_no-self.mean, axis=1)

    def update(self, ob_no):
        new_mean = self.mean*self.time_discount + ob_no.mean(0)*(1-self.time_discount)
        mean_difference = np.linalg.norm(new_mean - self.mean)
        self.mean = new_mean
        assert self.mean.shape == (1,self.ob_dim)
        return mean_difference


