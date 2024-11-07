import argparse

import pandas as pd
import numpy as np
import h5py
import yaml

import scipy.spatial as ss                                      
import dask.array as da

def spherical2cartesian(ra, dec, comoving_z):
    """spherical2cartesian
    
    return: x, y, z
    _____________
    radecz -> xyz
    """
    arcmin2rad = 0.0002908882086657216
    phi = ra * arcmin2rad
    theta = np.pi/2 - dec * arcmin2rad  # Caution on the definition of theta and dec
    r = comoving_z
    
    z = r * np.sin(theta) * np.cos(phi)
    x = r * np.sin(theta) * np.sin(phi)
    y = r * np.cos(theta)
    
    return x, y, z


def cartesian2spherical(x, y, z):
    """cartesian2spherical
    
    return: ra, dec, comoving_z
    _____________
    xyz -> radecz
    """
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arccos(y/r)
    phi = np.arctan(x/z)
    
    arcmin2rad = 0.0002908882086657216
    ra = phi / arcmin2rad
    dec = (np.pi/2 - theta) / arcmin2rad  # Caution on the definition of theta and dec
    comoving_z = r

    return ra, dec, comoving_z


def change_origin(x,c):
    """change_origin
    
    Change the origin of the coordinate system to c.
    """
    s = x.shape
    if len(s)==1:
        arcmin2rad = 0.0002908882086657216
        new = x - c
        new[0] = new[0] * np.sqrt(np.cos(x[1]*arcmin2rad) * np.cos(c[1]*arcmin2rad))
        return new
    elif len(s)==2:
        arcmin2rad = 0.0002908882086657216
        new = x - c
        new[:,0] = new[:,0] * np.sqrt(np.cos(x[:,1]*arcmin2rad) * np.cos(c[:,1]*arcmin2rad))
        return new
    elif len(s)==3:
        arcmin2rad = 0.0002908882086657216
        new = x - c
        new[:,:,0] = new[:,:,0] * np.sqrt(np.cos(x[:,:,1]*arcmin2rad) * np.cos(c[:,:,1]*arcmin2rad))
        return new     


def angle_b_two_v(ra1, dec1, ra2, dec2):
    '''angle_b_two_v
    
    Calculate the angular distance between two points on the celestial sphere.
    https://ja.wikipedia.org/wiki/%E5%A4%A7%E5%86%86%E8%B7%9D%E9%9B%A2
    '''
    arcmin2rad = 0.0002908882086657216
    alpha1 = ra1*arcmin2rad
    delta1 = dec1*arcmin2rad
    alpha2 = ra2*arcmin2rad
    delta2 = dec2*arcmin2rad

    d_alpha = alpha2 - alpha1
    d_delta = delta2 - delta1
    hav = np.sin(d_delta/2)**2 + np.cos(delta2)*np.cos(delta1)*np.sin(d_alpha/2)**2
    
    d_sigma = 2 * np.arcsin(np.sqrt(hav))

    return d_sigma/arcmin2rad


def cal_delta_density2d(center, galaxy, radius, mean_2d_density):
    arcmin2rad = 0.0002908882086657216

    gal_ra = da.from_array(galaxy["ra_arcmin"].values[:,np.newaxis], chunks=3000)
    gal_dec = da.from_array(galaxy["dec_arcmin"].values[:,np.newaxis], chunks=3000)
    center_ra = da.from_array(center["ra_arcmin"].values[np.newaxis,:], chunks=3000)
    center_dec = da.from_array(center["dec_arcmin"].values[np.newaxis,:], chunks=3000)
    r_da = angle_b_two_v(gal_ra, gal_dec, center_ra, center_dec)
    a_da = da.from_array([[radius]])
    circ = 2 * np.pi * (1-np.cos(a_da*arcmin2rad)) * (180*60/np.pi)**2
    
    df = pd.DataFrame((((r_da[:,:,np.newaxis] < a_da).sum(0)/ circ[0,:,:] - mean_2d_density) / mean_2d_density).compute(),
                      columns=[f"delta2d_{a}".replace('.', 'p') for a in radius])
    del r_da, a_da
        
    return df


def PCflag(df, dth=5, mth=10**4):
    mainhalo = df.query("mainhalo_bit==1 and z0cen_m_tophat>=@mth and z_app>2.5").reset_index(drop=True)
    # mainhalo = df.query("z0cen_m_tophat>=@mth and z_app>2.5").drop_duplicates("z0cen_HaloId").reset_index(drop=True)
    tree = ss.KDTree(mainhalo[["X", "Y", "Z"]].values, leafsize=10)
    d, i = tree.query(df[["X", "Y", "Z"]], workers=8)
    df["neighbor_HaloId"] = np.where(d<=dth, mainhalo["HaloId"][i], -1)
    df["neighbor_z0cen_HaloId"] = np.where(d<=dth, mainhalo["z0cen_HaloId"][i], -1)
    df["neighbor_z0cen_m_tophat"] = np.where(d<=dth, mainhalo["z0cen_m_tophat"][i], -1)
    df["neighbor_z0cen_m_crit200"] = np.where(d<=dth, mainhalo["z0cen_m_crit200"][i], -1)
    # df["neighbor_completeness"] = np.where(d<=dth, mainhalo["completeness"][i], -1)
    c = df.query("z0cen_HaloId == neighbor_z0cen_HaloId").groupby("neighbor_z0cen_HaloId")["completeness"].mean().rename("neighbor_completeness")
    df = pd.merge(df, c, on="neighbor_z0cen_HaloId",how='left')
    df["neighbor_completeness"] = df["neighbor_completeness"].fillna(-1)
    num = df.groupby(["neighbor_z0cen_HaloId"])["GalaxyID"].count().rename("neighbor_num")
    df = pd.merge(df, num, on="neighbor_z0cen_HaloId",how='left')
    df["neighbor_num"] = np.where(df["neighbor_z0cen_HaloId"]!=-1, df["neighbor_num"], 0)
    df["corrected_neighbor_num"] = df["neighbor_num"] / df["neighbor_completeness"]
    return df


def mean_n(df_all, grid_size=40, zmin = 6000, zmax = 8000):
    """
    """
    xmin = -75
    xmax = 75
    ymin = -75
    ymax = 75
    zn = (zmax-zmin)//grid_size
    z_bins = np.linspace(zmin, zmax, zn+1)
    z_bins_center = [(z_bins[i]+z_bins[i+1])/2 for i in range(zn)]

    mean_dist ,_ =  np.histogram(
        df_all.query(f"{xmin}< X < {xmax} and {ymin} <Y < {ymax}").Z, 
        bins=z_bins,) 
    return mean_dist, z_bins_center


def get_mean_n(df_all, grid_size=40, zmin=6000, zmax=8000):
    mean_dist, z_bins_center = mean_n(df_all, grid_size=grid_size, zmin=zmin, zmax=zmax)
    dist = mean_dist / mean_dist.max()
    return dist, z_bins_center


class rotate_radec(object):
    def __init__(self):
        pass
    def __call__(self, x):
        theta = np.random.rand()*2*np.pi
        trans_x = x.copy()
        trans_x[:, 0] = x[:, 0]*np.cos(theta)-x[:, 1]*np.sin(theta)
        trans_x[:, 1] = x[:, 0]*np.sin(theta)+x[:, 1]*np.cos(theta)
        return trans_x


class normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, x):
        trans = x.copy()
        trans = (x - self.mean) / self.std
        return trans


def param_extend(param, dim):
    param["mean"] = param["mean"] + [0.0 for _ in range(dim-len(param["mean"]))]
    param["std"] = param["std"] + [1.0 for _ in range(dim-len(param["std"]))]
    return param
    

def neighbors_search(data, targets, r_arcmin=5, workers=4, leafsize=10):
    x, y, z = spherical2cartesian(
        data["ra_arcmin"].values,
        data["dec_arcmin"].values,
        1)
    xyz = np.vstack([x,y,z]).T
    tree = ss.KDTree(xyz, leafsize=leafsize)
    x_c, y_c, z_c = spherical2cartesian(
        targets["ra_arcmin"].values,
        targets["dec_arcmin"].values,
        1)
    xyz_c = np.vstack([x_c,y_c,z_c]).T
    r = np.linalg.norm(np.array(spherical2cartesian(0,r_arcmin,1)) - np.array(spherical2cartesian(0,0,1)))
    neighbors = tree.query_ball_point(xyz_c, r=r, workers=workers)
    return neighbors
    
    
def save_data4pcfnet(h5filename, targets, neighbors, obsdata, maskdata=None):
    h5file = h5py.File(h5filename, 'w')
    for i in targets:
        h5file.create_group(i)
        dt = h5py.vlen_dtype(np.dtype('int32'))
        h5file.create_dataset(i + "/targets", data=np.array(targets[i].index))
        f = h5file.create_dataset(i + "/neighbors", (len(neighbors[i])),dtype=dt)
        for j, v in enumerate(neighbors[i]):
            f[j] = np.array(v)
    h5file.close()
    for i in obsdata:        
        obsdata[i].to_hdf(h5filename, key=i+'/data')
        if maskdata is not None:
            maskdata[i].to_hdf(h5filename, key=i+'/mask')


def load_args_from_command_and_yaml(parser):
    args = parser.parse_args()
    if args.config is not None:
        opt = vars(args)
        args = yaml.load(open(args.config), Loader=yaml.FullLoader)
        args.update(opt)
        args = argparse.Namespace(**args)
    return args