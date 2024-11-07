import numpy as np
from tqdm.auto import trange
import torch

from ..data_processor import DataProcessor
from .trainer import Trainer
from .pretrainer import Pretrainer
from ..util import util


class Predictor(Trainer):
    def __init__(self, args):
        super(Predictor, self).__init__(args)

    def start(self):
        self.args.using_data.remove('comoving_z')
        data_processor = DataProcessor(
            use_columns=self.args.using_data,
            file_name=self.args.obs_file_name,
            column_mapping=self.args.column_mapping_sim2obs,
            n_gaussians=self.args.n_gaussians,
            )
        
        self.fold_num = len(self.args.val_ids)
        # obs_pred = []
        param = self.args.param
        param = util.param_extend(self.args.param, self.args.model_args["dim"])


        for fold in trange(self.fold_num):
            obs_X, obs_index = data_processor.data_selection(self.args.obs_ids)

            if self.args.premodel_flg:
                preprocess = Pretrainer(self.args)
                obs_X = preprocess.obs_pred(obs_X, fold)
                gaussian_cols = list(np.array(
                    [[f"pi_{i}", f"sigma_{i}", f"mu_{i}"]
                    for i in range(self.args.n_gaussians)]
                ).T.reshape(-1))
                self.args.model_using_data = self.args.model_using_data + gaussian_cols
            obs_X = obs_X[self.args.model_using_data].to_numpy()
                
            dataloader_obs = data_processor.get_dataloader(obs_X, obs_index, None, param, train=False)
            #reset model
            self.load_model()
            self.load_optimizer()

            self.model.load_state_dict(
                torch.load((self.args.trainedmodel_name).format(fold)))
            pred = self.predict(dataloader_obs)
            pred_df = data_processor.prediction_arrange(self.args.obs_ids, pred)
            data_processor.save_pred(pred_df, self.args.obs_pred_name, fold)
