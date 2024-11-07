import argparse
import numpy as np
import pandas as pd
import scipy.spatial as ss    
from tqdm.auto import tqdm

from pcfnet.util import util


def get_parser():
    parser = argparse.ArgumentParser(
        description="Preprocess to convert observational data for PCFNet"
    )
    parser.add_argument(
        "--config", type=str, required=True, help="configuration file *.yaml"
    )
    return parser

if __name__ == '__main__':
    parser = get_parser()
    args = util.load_args_from_command_and_yaml(parser)

    obsdata = {}
    maskdata = {}
    for name, path in args.obs_files.items():
        obsdata[name] = pd.read_csv(path)
    for name, path in args.obsmask_files.items():
        maskdata[name] = pd.read_csv(path)
    
    ### target selection
    targets = {}
    for name in tqdm(maskdata):
        x, y, z = util.spherical2cartesian(maskdata[name].ra_arcmin.values, maskdata[name].dec_arcmin.values, 1)
        xyz = np.vstack([x,y,z]).T
        tree = ss.KDTree(xyz, leafsize=10)
        x_c, y_c, z_c = util.spherical2cartesian(obsdata[name].ra_arcmin.values, obsdata[name].dec_arcmin.values, 1)
        xyz_c = np.vstack([x_c,y_c,z_c]).T
        r_arcmin = args.fov_r_arcmin
        r = np.linalg.norm(np.array(util.spherical2cartesian(0,r_arcmin,1)) - np.array(util.spherical2cartesian(0,0,1)))
        randoms = tree.query_ball_point(xyz_c, r=r, workers=4)
        del tree
        obsdata[name]["n_random"] = [len(j) for j in randoms]
        mean_random = np.pi * r_arcmin**2 * 100
        th_random = mean_random * 0.5
        targets[name] = obsdata[name].query('n_random >= @th_random')

    neighbors = {}
    for name in tqdm(obsdata):
        neighbors[name] = util.neighbors_search(obsdata[name], targets[name], r_arcmin=args.fov_r_arcmin)
        
    util.save_data4pcfnet(args.obs_file_name, targets, neighbors, obsdata, maskdata)

    print('done!')