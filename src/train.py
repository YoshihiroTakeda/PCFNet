import os
import logging
import argparse
import yaml
import importlib

from pcfnet.trainer import Trainer

logger = logging.getLogger(__name__)

def get_parser():
    parser = argparse.ArgumentParser(
        description="Train PCFNet"
    )
    parser.add_argument(
        "--config", type=str, required=True, help="configuration file *.yaml"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="the indexes of GPUs for training or testing",
    )
    return parser


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    if args.config is not None:
        opt = vars(args)
        args = yaml.load(open(args.config), Loader=yaml.FullLoader)
        args.update(opt)
        args = argparse.Namespace(**args)
    
    if args.wandb and (importlib.util.find_spec('wandb') is not None):
        import wandb
        wandb.login()

    if not os.path.exists(os.path.dirname(args.logfile)):
        os.makedirs(os.path.dirname(args.logfile))
    flh = logging.FileHandler(args.logfile)
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.DEBUG,
                    handlers=[flh])
    logger.info(f'load config file: {args.config}')
    logger.info(args)

    processor = Trainer(args)
    processor.start()