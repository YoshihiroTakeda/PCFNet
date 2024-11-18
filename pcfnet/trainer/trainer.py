import logging
import importlib.util
from tqdm.auto import tqdm, trange
from tqdm.contrib.logging import logging_redirect_tqdm

import numpy as np
from sklearn import metrics
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
if importlib.util.find_spec("wandb") is not None:
    import wandb
    from wandb.plot.viz import CustomChart

from .pretrainer import Pretrainer
from ..models import PCFNet
from ..data_processor import DataProcessor
from ..util import util

logger = logging.getLogger(__name__)


def feature_transform_regularizer(trans: torch.Tensor):
    d = trans.size()[1]
    device = trans.device
    I = torch.eye(d, device=device)[None, :, :]
    loss = torch.mean(torch.norm(torch.bmm(trans, trans.transpose(2,1)) - I, dim=(1,2)))
    return loss


class Trainer:
    def __init__(self, args: any):
        self.args = args
        self.criterion = nn.NLLLoss()
        self.batch_size = self.args.batch_size

    def load_model(self):
        self.model = PCFNet(
            **self.args.model_args
        ).to(self.args.device)
        logger.info(self.model)

    def load_optimizer(self):
        self.optimizer = optim.Adam(self.model.parameters(),lr=self.args.lr)
        self.scheduler = ReduceLROnPlateau(self.optimizer, 'min', patience = self.args.patience)

    def train(self,
              dataloader_train, 
              dataloader_valid, 
              n_epochs: int, 
              log_dir: str = "./logs", 
              early_stop_epoch: int = 10,
              start_epoch: int = 0, 
              early_stopping: bool = True, 
              early_stoping_criteria: str = "pr",
              warmup: int = 10, 
              verpose: int = 1, 
              regularization_factor: float = 0.001):

        writer = SummaryWriter(log_dir=log_dir)

        model_epoch = []
        pr_epoch = []
        max_pr_epoch = start_epoch + warmup
        factor = 1
        with logging_redirect_tqdm():
            for epoch in tqdm(range(start_epoch, start_epoch+n_epochs)):
                self.model.train()
                acc_train = 0
                roc_auces_train = []
                pr_auces_train = []
                losses_train = []
                losses_valid = []
                for i, batch in tqdm(enumerate(dataloader_train), total=len(dataloader_train)):
                    xc, m, t = batch
                    self.model.zero_grad()
                    xc = xc.transpose(2,1).squeeze(1).to(self.args.device, non_blocking=True)
                    m = m.transpose(2,1).to(self.args.device, non_blocking=True)
                    t = t.to(self.args.device)

                    if xc.shape[2]>3:
                        x, c  = xc[:,:3,:], xc[:,3:,:]
                    else:
                        x, c = xc, None

                    output, trans, trans_feat = self.model(x, c, m)

                    error = self.criterion(output, t)
                    if trans is not None:
                        error += feature_transform_regularizer(trans) * regularization_factor
                    if trans_feat is not None:
                        error += feature_transform_regularizer(trans_feat) * regularization_factor
                    self.optimizer.zero_grad()
                    error.backward()
                    self.optimizer.step()
                    
                    with torch.no_grad():
                        pred = np.argmax(output.cpu().numpy(), axis=1)
                        acc_train += (pred == t.cpu().numpy().reshape(-1,).astype(int)).sum().item()
                        fpr, tpr, thresholds = metrics.roc_curve(
                            t.cpu().numpy().reshape(-1,).astype(int),
                            np.exp(output.detach().cpu().numpy()[:,1]))
                        precision, recall, thresholds = metrics.precision_recall_curve(
                            t.cpu().numpy().reshape(-1,).astype(int),
                            np.exp(output.detach().cpu().numpy()[:,1]))
                        roc_auces_train.append(metrics.auc(fpr, tpr))
                        pr_auces_train.append(metrics.auc(recall, precision))
                        losses_train.append(error.item())

                self.model.eval()
                acc_val = 0
                roc_auces_val = []
                pr_auces_val = []
                y_pred = []
                y_true = []
                for batch in dataloader_valid:
                    xc, m, t = batch
                    xc = xc.transpose(2,1).squeeze(1).to(self.args.device, non_blocking=True)
                    m = m.transpose(2,1).to(self.args.device, non_blocking=True)
                    t = t.to(self.args.device)
                    if xc.shape[2]>3:
                        x, c  = xc[:,:3,:], xc[:,3:,:]
                    else:
                        x, c = xc, None

                    y, trans, trans_feat = self.model(x, c, m) 

                    error = self.criterion(y, t)
                    if trans is not None:
                        error += feature_transform_regularizer(trans) * regularization_factor
                    if trans_feat is not None:
                        error += feature_transform_regularizer(trans_feat) * regularization_factor
                    with torch.no_grad():
                        pred = np.argmax(y.cpu().numpy(), axis=1)
                        acc_val += (pred == t.cpu().numpy().reshape(-1,).astype(int)).sum().item()
                        fpr, tpr, thresholds = metrics.roc_curve(
                            t.cpu().numpy().reshape(-1,).astype(int),
                            np.exp(y.detach().cpu().numpy()[:,1]))
                        precision, recall,thresholds = metrics.precision_recall_curve(
                            t.cpu().numpy().reshape(-1,).astype(int),
                            np.exp(y.detach().cpu().numpy()[:,1]))
                        y_pred.extend(np.exp(y.detach().cpu().numpy()[:,1]))
                        y_true.extend(t.cpu())
                        roc_auces_val.append(metrics.auc(fpr, tpr))
                        pr_auces_val.append(metrics.auc(recall, precision))
                        losses_valid.append(error.item())

                roc_auces_train = np.array(roc_auces_train)[~np.isnan(roc_auces_train)]
                pr_auces_train = np.array(pr_auces_train)[~np.isnan(pr_auces_train)]
                roc_auces_val = np.array(roc_auces_val)[~np.isnan(roc_auces_val)]
                pr_auces_val = np.array(pr_auces_val)[~np.isnan(pr_auces_val)]
                loss_train = np.mean(losses_train)
                roc_auc_train = np.mean(roc_auces_train)
                pr_auc_train = np.mean(pr_auces_train)
                loss_valid = np.mean(losses_valid)
                roc_auc_valid = np.mean(roc_auces_val)
                pr_auc_valid = np.mean(pr_auces_val)

                self.scheduler.step(loss_valid)

                if epoch % verpose == verpose-1:
                    logger.info('EPOCH: {}, Train [Loss: {:.3f}, ROC_AUC: {:.3f}, PR_AUC: {:.3f}]\n Valid [Loss: {:.3f}, ROC_AUC: {:.3f}, PR_AUC: {:.3f}]'.format(
                        epoch,
                        loss_train,
                        roc_auc_train,
                        pr_auc_train,
                        loss_valid,
                        roc_auc_valid,
                        pr_auc_valid
                    ))
                writer.add_scalar("loss/train_loss", loss_train, epoch)
                writer.add_scalar("ROC_AUC/train_ROC_AUC", roc_auc_train, epoch)
                writer.add_scalar("PR_AUC/train_PR_AUC", pr_auc_train, epoch)
                writer.add_scalar("loss/valid_loss", loss_valid, epoch)
                writer.add_scalar("ROC_AUC/valid_ROC_AUC", roc_auc_valid, epoch)
                writer.add_scalar("PR_AUC/valid_PR_AUC", pr_auc_valid, epoch)
                writer.add_pr_curve('pr_curve', np.array(y_true), np.array(y_pred).reshape(-1), epoch)
                if self.args.wandb:
                    precision, recall, thresholds = metrics.precision_recall_curve(np.array(y_true), np.array(y_pred).reshape(-1))
                    sampling_rate = int(len(thresholds)/500)
                    pr_table = wandb.Table(
                        data=[list(x) for x in zip(
                            thresholds[::sampling_rate],
                            precision[:-1][::sampling_rate],
                            recall[:-1][::sampling_rate])], 
                        columns=["threshold","precision", "recall"])
                    chart = CustomChart(
                        id="astro_takeda/pr_curve_with_slider",
                        data=pr_table,
                        fields={
                            "x-axis": "recall",
                            "y-axis": "precision",
                            "threshold": "threshold",
                        },
                        string_fields={}
                    )
                    wandb.log({
                        "loss/train_loss": loss_train,
                        "ROC_AUC/train_ROC_AUC": roc_auc_train,
                        "PR_AUC/train_PR_AUC": pr_auc_train,
                        "loss/valid_loss": loss_valid,
                        "ROC_AUC/valid_ROC_AUC": roc_auc_valid,
                        "PR_AUC/valid_PR_AUC": pr_auc_valid,
                        "PR_curve": chart,
                        # "PR_curve": pr_table,
                        "epoch": epoch
                        }, step=epoch)

                if early_stopping:
                    if epoch>=warmup:
                        if early_stoping_criteria=="roc":
                            this_score = np.mean(roc_auc_valid)
                        elif early_stoping_criteria=="loss":
                            factor=-1
                            this_score = np.mean(-loss_valid)
                        else:
                            this_score = np.mean(pr_auc_valid)
                        pr_epoch.append(this_score)
                        model_epoch.append(self.model.state_dict())
                        if pr_epoch[max_pr_epoch-start_epoch-warmup] <= this_score:
                            max_pr_epoch = epoch
                        if epoch - max_pr_epoch >= early_stop_epoch:
                            logger.info("early stop!")
                            break
        if early_stopping:
            self.model.load_state_dict(model_epoch[max_pr_epoch-start_epoch-warmup])
            logger.info(
                f"best epoch:{max_pr_epoch}, "\
                f"best {early_stoping_criteria}: {factor*pr_epoch[max_pr_epoch-start_epoch-warmup]}")
        writer.close()

    def eval(self, dataloader_test):
        self.model.eval()
        y_pred = []
        y_true = []
        with logging_redirect_tqdm():
            for batch in tqdm(dataloader_test):
                xc, m, t = batch
                xc = xc.transpose(2,1).to(self.args.device)
                m = m.transpose(2,1).to(self.args.device)
                t = t.to(self.args.device)
                if xc.shape[2]>3:
                    x, c  = xc[:,:3,:], xc[:,3:,:]
                else:
                    x, c = xc, None
                y, _, _ = self.model(x, c, m)
                y_true.extend(t.detach().cpu().numpy())
                y_pred.extend(y.detach().cpu().numpy()[:,1])
        return np.array(y_pred).reshape(-1,), y_true
    
    def predict(self, dataloader_obs):
        self.model.eval()
        y_pred = []
        with logging_redirect_tqdm():
            for batch in tqdm(dataloader_obs):
                xc, m = batch
                xc = xc.transpose(2,1).to(self.args.device)
                m = m.transpose(2,1).to(self.args.device)
                if xc.shape[2]>3:
                    x, c  = xc[:,:3,:], xc[:,3:,:]
                else:
                    x, c = xc, None
                y, _, _ = self.model(x, c, m)
                y_pred.extend(y.detach().cpu().numpy()[:,1])
        return np.array(y_pred).reshape(-1,)
    
    def start(self):
        data_processor = DataProcessor(
            file_name=self.args.file_name,
            min_pc_member=self.args.min_pc_member,
            min_completeness=self.args.min_completeness,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            use_columns=self.args.using_data,
            n_gaussians=self.args.n_gaussians,
            flg_column=self.args.flg_column,
        )
        train_ids, val_ids, test_ids = data_processor.assign_ids(self.args.val_ids, self.args.test_ids)
        param = util.param_extend(self.args.param, self.args.model_args["dim"])


        for fold in trange(data_processor.fold_num):
            train_X, train_index, train_y = data_processor.data_selection(train_ids[fold])  
            val_X, val_index, val_y = data_processor.data_selection(val_ids[fold])
            test_X, test_index, test_y = data_processor.data_selection(test_ids)              

            if self.args.premodel_flg:
                preprocess = Pretrainer(self.args)
                train_X, val_X, test_X = preprocess.start(train_X, val_X, test_X, fold)
                gaussian_cols = list(np.array(
                    [[f"pi_{i}", f"sigma_{i}", f"mu_{i}"]
                    for i in range(self.args.n_gaussians)]
                ).T.reshape(-1))
                self.args.model_using_data = self.args.model_using_data + gaussian_cols
                
            train_X = train_X[self.args.model_using_data].to_numpy()
            val_X = val_X[self.args.model_using_data].to_numpy()
            test_X = test_X[self.args.model_using_data].to_numpy()

            dataloader_train = data_processor.get_dataloader(train_X, train_index, train_y, param)
            dataloader_valid = data_processor.get_dataloader(val_X, val_index, val_y, param, train=False)
            dataloader_test = data_processor.get_dataloader(test_X, test_index, test_y, param, train=False)

            #reset model
            self.load_model()
            self.load_optimizer()

            if self.args.train_flg:
                if self.args.wandb:
                    wandb.init(project="PCFNet", config=self.args)
                    wandb.watch(self.model, log="all")
                self.train(
                    dataloader_train, dataloader_valid,
                    self.args.epochs, 
                    log_dir=(self.args.log_dir).format(fold),
                    early_stop_epoch=self.args.early_stop_epoch,
                    start_epoch=self.args.start_epoch,
                    early_stoping_criteria=self.args.early_stoping_criteria,
                    warmup=self.args.warmup,
                    verpose=self.args.verpose,
                    regularization_factor=self.args.regularization_factor)
                torch.save(self.model.state_dict(),
                    (self.args.model_name).format(fold))
                if self.args.wandb:
                    wandb.finish()
            else:
                self.model.load_state_dict(
                    torch.load((self.args.trainedmodel_name).format(fold)))

            test_y_pred, test_y_true = self.eval(dataloader_test)
            test_df = data_processor.prediction_arrange(test_ids, test_y_pred, test_y_true)
            data_processor.save_pred(test_df, self.args.result_pred_name, fold)

