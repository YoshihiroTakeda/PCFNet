import argparse
from tqdm import tqdm

import h5py
import yaml
import numpy as np
import pandas as pd
from pandarallel import pandarallel
from scipy import interpolate
from astropy import cosmology as cosmo

from pcfnet.util import util

np.random.seed(12345)
pandarallel.initialize()

def get_parser():
    parser = argparse.ArgumentParser(
        description="Preprocess to convert PCcone data for PCFNet"
    )
    parser.add_argument(
        "--config", type=str, required=True, help="configuration file *.yaml"
    )
    return parser


def load_obs(filename):
    with h5py.File(filename, 'r') as f:
        keys = list(f.keys())
    obsdata = {}
    maskdata = {}
    for key in keys:
        obsdata[key] = pd.read_hdf(filename, key=key+'/data')
        maskdata[key] = pd.read_hdf(filename, key=key+'/mask')
    return obsdata, maskdata


if __name__ == '__main__':
    parser = get_parser()
    args = util.load_args_from_command_and_yaml(parser)
    
    Planck1 = cosmo.FlatLambdaCDM(H0=67.3, Om0=0.315, Ob0=0.049, Tcmb0=2.725, name='Planck1')
    print(Planck1)
    lim_mag5 = yaml.safe_load(open(args.limit_mag5))
    halo_all = pd.read_csv(args.halodata_path)

    ### PCcone columns
    column_name = [
    'GalaxyID', 'HaloId', 'x', 'y', 'z', 'vel_x', 'vel_y', 'vel_z',
    'redshift_snapshot', 'snapshot', 'sfr', 'stellarmass',
    'coldgasmass', 'metals_coldgasmass', 'radius_coldgas', 'sfh_index_x', 'sfh_index_y',
    'ra', 'dec', 'z_geo', 'comoving_distance', 'z_app',
    'HSC_g_new', 'HSC_g_err',
    'HSC_r_new', 'HSC_r_err',
    'HSC_i_new', 'HSC_i_err',
    'HSC_z_new', 'HSC_z_err',
    'HSC_y_new', 'HSC_y_err',
    ]
    
    # gdropout selection criteria
    # Ono+18
    query = '''23. < HSC_i_cut < 26.
    and HSC_r_err < 0.5 and HSC_i_err < 0.5
    and HSC_g_cut - HSC_r_cut > 1
    and -1 < HSC_r_cut - HSC_i_cut < 1
    and 1.5*(HSC_r_cut - HSC_i_cut) < HSC_g_cut - HSC_r_cut - 0.8
    '''.replace('\n',' ')
    
    pc = {}
    for key, path in tqdm(args.PCcone.items()):
        hf = h5py.File(path, "r").get("data")[:]
        df = pd.DataFrame(hf, columns=column_name).astype({'GalaxyID':int, 'HaloId':int})
        # replace with 2\sigma
        for band in ['g', 'r','i', 'z', 'y']:
            df['HSC_'+band+'_cut'] = np.where(
                df['HSC_'+band+'_new']>lim_mag5['COSMOS'][band]+2.5*np.log10(5/2),
                lim_mag5['COSMOS'][band]+2.5*np.log10(5/2),
                df['HSC_'+band+'_new'])
        # truncate g band to 2\sigma limit
        df = df[df['HSC_g_new']<=lim_mag5['COSMOS']['g']+2.5*np.log10(5/2)]
        pc[key] = df.query(query).copy()
        pc[key]['g-i'] = pc[key].eval('HSC_g_cut - HSC_i_cut')
        pc[key]['ra'] = (pc[key]['ra']+180.) % 360 - 180.
        pc[key]['ra_arcmin'] = pc[key]['ra']*60.
        pc[key]['dec_arcmin'] = pc[key]['dec']*60.
        pc[key]['radius_arcmin'] = pc[key][['ra_arcmin', 'dec_arcmin']].parallel_apply(lambda x:util.angle_b_two_v(*x, 0,0), axis='columns')
        pc[key]['comoving_z'] = pc[key]['z_app'].parallel_apply(lambda x:Planck1.comoving_distance(z=x).value).values
        pc[key][['X', 'Y', 'Z']] = np.array(util.spherical2cartesian(pc[key].ra_arcmin, pc[key].dec_arcmin, pc[key].comoving_z)).T
        pc[key] = pd.merge(pc[key], halo_all, on = "HaloId", how="left")
        pc[key] = pd.merge(pc[key], pc[key].groupby("z0cen_HaloId")["GalaxyID"].count().rename("Halo_num"), on = "z0cen_HaloId", how="left")
    pc_all = pd.concat([df for df in pc.values()])
    
    if args.downsampling:
        obsdata, maskdata = load_obs(args.obs_file_name)
        obs_all = pd.concat([df for df in obsdata.values()])
        mask_all = pd.concat([df for df in maskdata.values()])
        alpha = (pc_all.shape[0]/len(pc)/np.pi) / (obs_all.shape[0]/mask_all.shape[0]*100.*3600.)  # downsampling rate
        rng = np.random.default_rng(100)
        for name in pc:
            pc[name] = pc[name].sample(
                int(pc[name].shape[0]/alpha),
                random_state=rng).reset_index(drop=True)

    #completeness
    dist, zz = util.get_mean_n(pc_all, grid_size=40)
    curve = interpolate.interp1d(zz, dist, kind="cubic", fill_value=0., bounds_error=False)
    for i, df in pc.items():
        pc[i]["completeness"] = curve(df["Z"])
    
    #selected columns
    delta2d_dfs = {}
    df_all = pd.concat([dg for dg in pc.values()])
    mean_2d_density = df_all.shape[0] / (60.**2 * np.pi)/ len(pc)  # /arcmin^2
    for i, df in tqdm(pc.items()):
        pc[i] = util.PCflag(df, dth=args.dth, mth=args.mth)
        delta2d_dfs[i] = util.cal_delta_density2d(
                pc[i][['ra_arcmin', 'dec_arcmin']],
                pc[i][['ra_arcmin', 'dec_arcmin']],
                args.surface_density_radius,
                mean_2d_density
        )
        pc[i] = pd.concat([pc[i], delta2d_dfs[i]], axis="columns")
        ### add label
        pc[i]['flg'] = np.where(
            (pc[i].corrected_neighbor_num>=args.min_pc_member)&\
            (pc[i].neighbor_completeness>=args.min_completeness),
            1, 0)
    
    targets = {}
    neighbors = {}
    for i in tqdm(pc):
        targets[i] = pc[i].query(f"radius_arcmin <= 60 - {args.fov_r_arcmin}")
        neighbors[i] = util.neighbors_search(pc[i], targets[i], r_arcmin=args.fov_r_arcmin)
        
    util.save_data4pcfnet(args.file_name, targets, neighbors, pc)
    
    print('done!')