import os
from collections import defaultdict
import logging
import importlib.util
from tqdm.auto import trange
from tqdm.contrib.logging import logging_redirect_tqdm

import numpy as np
import pandas as pd
import torch
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter

import pyro
from pyro.contrib.autoguide import AutoDelta
from pyro import poutine
from pyro.infer import SVI
from pyro.infer import TraceEnum_ELBO, JitTraceEnum_ELBO
from pyro.optim import Adam
if importlib.util.find_spec("wandb") is not None:
    import wandb

from ..models import MDN


logger = logging.getLogger(__name__)

torch.set_num_threads(max(1, torch.get_num_threads()-2))

class Pretrainer:
    def __init__(self, args):
        self.args = args
        self.param = None

    def load_model(self,):
        pyro.clear_param_store()
        self.premodel = MDN(
            n_input=self.args.coln,
            n_hidden=self.args.premodel_hidden,
            n_gaussians=self.args.n_gaussians)
        logger.info(self.premodel)

    def load_svi(self,):
        adam_params = {"lr": 0.001, "betas": (0.9, 0.999)}
        self.preoptimizer = Adam(adam_params)
        self.guide = AutoDelta(
            poutine.block(self.premodel.model,
                          hide=['assign', 'obs'])
        )
        if self.args.predevice == 'cpu':
            self.elbo = JitTraceEnum_ELBO(max_plate_nesting=1)
        else:
            # self.elbo = JitTraceEnum_ELBO(max_plate_nesting=1)
            self.elbo = TraceEnum_ELBO(max_plate_nesting=1)
        self.svi = SVI(
            self.premodel.model,
            self.guide,
            self.preoptimizer,
            loss=self.elbo)

    def get_dataloader(self, df: pd.DataFrame, true: bool = True):
        if true:
            if self.param is None:
                self.param = {"mean": df[self.args.z_column].mean(), "std": df[self.args.z_column].mean()}
            true_data = ((df[self.args.z_column] - self.param["mean"]) / self.param["std"]).values
        else:
            true_data = np.array([np.nan])
        exp_data = df[self.args.premodel_using_data].values

        # change data type and shape, move from numpy to torch
        exp = Variable(torch.from_numpy(exp_data).float(), requires_grad=False)
        true = Variable(torch.from_numpy(true_data).float())
        return [[exp, true]]

    def train(self, loader, eval_loader=None):
        n_samples = loader[0][0].shape[0]
        if eval_loader is not None:
            eval_n_samples = eval_loader[0][0].shape[0]
        grad_norms = defaultdict(list)
        for name, value in pyro.get_param_store().named_parameters():
            value.register_hook(lambda g, name=name: grad_norms[name].append(g.norm().item()))
        losses = []
        writer = SummaryWriter(log_dir=(self.args.prelog_dir).format(self.fold))
        self.premodel.train()
        self.premodel.to(self.args.predevice)
        with logging_redirect_tqdm():
            for epoch in trange(self.args.preepochs):
                tot_loss = 0
                for e, t in loader:
                    t, e = t.to(self.args.predevice), e.to(self.args.predevice)
                    loss = self.svi.step(t, e)
                    tot_loss += loss
                if (epoch % 5) == 0:
                    writer.add_scalar('PreTrain/loss', tot_loss / n_samples, epoch)
                    if self.args.wandb:
                        wandb.log({'PreTrain/loss': tot_loss / n_samples}, step=epoch)
                    for name, g_norm in grad_norms.items():
                        writer.add_scalar(f'PreTrain/{name}', g_norm[epoch], epoch)
                        if self.args.wandb:
                            wandb.log({f'PreTrain/{name}': g_norm[epoch]}, step=epoch)
                losses.append((loss / n_samples))
                if eval_loader is not None:
                    self.premodel.eval()
                    tot_loss = 0
                    for e, t in eval_loader:
                        t, e = t.to(self.args.predevice), e.to(self.args.predevice)
                        eval_loss = self.svi.evaluate_loss(t, e)
                        tot_loss += eval_loss
                    if (epoch % 5) == 0:
                        writer.add_scalar('PreTrain_eval/loss', tot_loss / eval_n_samples, epoch)
                        if self.args.wandb:
                            wandb.log({'PreTrain_eval/loss': tot_loss / eval_n_samples}, step=epoch)
                    self.premodel.train()
        writer.close()
        return losses

    def pred(self, loader):
        self.premodel.eval()
        self.premodel.to(self.args.predevice)
        pred = []
        with torch.no_grad():
            for e, t in loader:
                e = e.to(self.args.predevice)
                pred.append(
                    torch.cat(self.premodel.forward(e),
                              axis=1).cpu().detach().numpy()
                )
        return np.concatenate(pred, axis=0)

    def start(self,train_X, val_X, test_X, fold: int):
        self.load_model()
        self.fold = fold
        ### pretrain
        if self.args.pretrain_flg:
            if self.args.wandb:
                wandb.init(project="PCFNet_MDN", config=self.args)
                wandb.watch(self.premodel, log="all")
            self.load_svi()
            train_loader = self.get_dataloader(train_X)
            val_loader = self.get_dataloader(val_X)
            loss = self.train(train_loader, val_loader)
            logger.info(f'loss: {loss[-1]}')
            model_file_path = (self.args.premodel_name).format(fold)
            if not os.path.exists(os.path.dirname(model_file_path)):
                os.makedirs(os.path.dirname(model_file_path))
            torch.save(
                self.premodel.state_dict(),
                model_file_path)
            logger.info(f'save model: {model_file_path}')
            if self.args.wandb:
                wandb.finish()
        else:
            logger.info(f'load pretrained model: {(self.args.pretrainedmodel_name).format(fold)}')
            self.premodel.load_state_dict(
                torch.load(
                    (self.args.pretrainedmodel_name).format(fold)
                ))
        
        ### predict
        train_loader = self.get_dataloader(train_X)
        val_loader = self.get_dataloader(val_X)
        test_loader = self.get_dataloader(test_X)
        col_name = list(np.array(
            [[f"pi_{i}", f"sigma_{i}", f"mu_{i}"]
             for i in range(self.args.n_gaussians)]
        ).T.reshape(-1))

        train_X[col_name] = self.pred(train_loader)
        val_X[col_name] = self.pred(val_loader)
        test_X[col_name] = self.pred(test_loader)

        return train_X, val_X, test_X
    
    def obs_pred(self, obs, fold: int):
        self.load_model()
        self.fold = fold
        
        logger.info(f'load model: {(self.args.premodel_name).format(fold)}')
        self.premodel.load_state_dict(
            torch.load(
                (self.args.premodel_name).format(fold)
            ))
        obs_loader = self.get_dataloader(obs, true=False)
        col_name = list(np.array(
            [[f"pi_{i}", f"sigma_{i}", f"mu_{i}"]
             for i in range(self.args.n_gaussians)]
        ).T.reshape(-1))
        obs[col_name] = self.pred(obs_loader)
        return obs