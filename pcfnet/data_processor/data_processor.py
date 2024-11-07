import logging
from typing import Optional
import numpy as np
import pandas as pd
import h5py
import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import torchvision.transforms as transforms
from .feeder import Feeder
from ..util import util

logger = logging.getLogger(__name__)


def pad_collate(batch):
    (xx, yy) = zip(*batch)

    xx_pad = pad_sequence(xx, batch_first=True, padding_value=0)
    xx_len = np.array([len(i) for i in xx])
    mask = np.arange(xx_pad.shape[1])[np.newaxis,:,np.newaxis]
    mask = np.where(mask<xx_len[:,np.newaxis, np.newaxis], 1.,0.)
    mask = torch.from_numpy(mask).float()
    if yy[0] is not None:
        yy = torch.stack(yy)
        return xx_pad, mask, yy
    else:
        return xx_pad, mask


class DataProcessor:
    """DaraProcessor
    
    DataProcessor for PCFNet.
    """
    def __init__(self,
                 file_name: str = 'output/preprocess/lightcones.h5',
                 min_pc_member: int = 3, 
                 min_completeness: float = 0.5,
                 batch_size: int = 32,
                 num_workers: int = 4,
                 use_columns: list[str] = ["ra", "dec",],
                 column_mapping: Optional[dict] = None,
                 n_gaussians: int = 3,
                 ):
        self.file_name = file_name
        self.min_pc_member = min_pc_member
        self.min_completeness = min_completeness
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.use_columns = use_columns
        self.column_mapping = column_mapping
        self.n_gaussians = n_gaussians
        self.data_load()

    def data_load(self,):
        self.data = {}
        self.targets = {}
        self.neighbors = {}
        with h5py.File(self.file_name, 'r') as f:
            self.data_ids = list(f.keys())
            for i in self.data_ids:
                self.targets[i] = f[i+"/targets"][:]
                self.neighbors[i] = f[i+"/neighbors"][:]
        for i in self.data_ids:
            self.data[i] = pd.read_hdf(self.file_name, key=i+"/data")
        if self.column_mapping is not None:
            if len(self.column_mapping.keys())>0:
                self.data_rename()
                
    def prediction_arrange(self,
                           lightcone_ids: list[any],
                           pred: np.array,
                           true: Optional[np.array] = None):
        pred_df = pd.DataFrame([[i, idx]  for i in lightcone_ids for idx in self.targets[i]], columns=["lightcone_id", "index"])
        pred_df["pred"] = np.exp(pred)
        if true is not None:
            pred_df["true"] = true
        return pred_df

    def save_pred(self, pred_df: pd.DataFrame, file_name: str, fold: Optional[int] = None):
        for i in self.data_ids:
            if fold is not None:
                key = f"{fold}/"+i+"/pred"
            else:
                key = i+"/pred"
            pred_df.query("lightcone_id==@i").to_hdf(file_name, key=key, mode='a')

    def data_rename(self):
        for name in self.data:
            self.data[name] = self.data[name].rename(columns=self.column_mapping)

    def assign_ids(self, val_ids: list[list[any]], test_ids: list[any]):
        '''
        Assign ids for train, validation, and test.
        
        output: 
            - train_ids: 2d list
            - val_ids: 2d list
            - test_ids: 1d list
        '''
        # test_ids check
        if not set(test_ids).issubset(self.data_ids):
            logger.error(
                f'test_ids has to be in dataset ids.'\
                f' Please check your test_ids: {test_ids}')
            return
        if len(test_ids)==0:
            logger.error('Please input test_ids.')
            return
        test_ids = set(test_ids)

        # val_ids check
        if len(val_ids)==0:
            logger.error('Please input val_ids.')
            return
        train_ids = []
        for folder in val_ids:
            if not set(folder).issubset(self.data_ids):
                logger.error(
                    f'val_ids has to be in dataset ids.'\
                    f' Please check your test_ids: {val_ids}')
                return
            if test_ids.issubset(folder):
                logger.error('val_ids has same ids in test_ids.')
                return
            train_ids.append(list(set(self.data_ids) - set(folder) - test_ids))
        test_ids = test_ids

        self.fold_num = len(train_ids)
        
        logger.info(
            f'train_ids:{train_ids}, '
            f'val_ids:{val_ids}, '
            f'test_ids:{test_ids}, '
        )
        return train_ids, val_ids, test_ids

    def data_selection(self, ids: list[any]):
        length_df = [0] + [self.data[i].shape[0] for i in ids]
        cum_length_df = np.cumsum(length_df)
        X = pd.concat([self.data[i][self.use_columns] for i in ids]).reset_index(drop=True)
        index = [[c + cum_length_df[num], self.neighbors[i][j] + cum_length_df[num]]
                 for num, i in enumerate(ids) for j, c in enumerate(self.targets[i])]
        if "flg" in self.data[list(ids)[0]].columns:
            y = pd.concat([self.data[i][["flg"]] for i in ids]).to_numpy().reshape(-1)
            return X, index, y
        else:
            return X, index

    def get_dataloader(self,
                       X, 
                       index, 
                       y, 
                       param: dict[str, float]={"mean":0., "std":5.}, 
                       train: bool = True):
        trans = transforms.Compose([
                util.rotate_radec(), 
                util.normalize(param["mean"], param["std"])
                ])
        dataset = Feeder(X, index, y=y, transform=trans, n_gaussians=self.n_gaussians)

        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            collate_fn=pad_collate,
            shuffle=train,
            num_workers = self.num_workers,
        )
        return dataloader