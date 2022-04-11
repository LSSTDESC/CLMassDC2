import numpy as np
import os
import pickle
import astropy
from astropy.table import Table, vstack, QTable
from astropy import units as u
from astropy.coordinates import SkyCoord

def load(filename, **kwargs):
    """Loads GalaxyCluster object to filename using Pickle"""
    with open(filename, 'rb') as fin:
        return pickle.load(fin, **kwargs)

tract_list = load('/pbs/throng/lsst/users/cpayerne/CLMassDC2/data/data_extraction/tract_dc2dr6_coord.pkl')

def neigboring_tracts(ra_deg = 1, dec_deg = 1):
    tract_coord = SkyCoord(ra=tract_list['ra']*u.degree, dec=tract_list['dec']*u.degree)
    coord = SkyCoord(ra=ra_deg*u.degree, dec=dec_deg*u.degree)
    idx, d2d, d3d = astropy.coordinates.match_coordinates_sky(coord, tract_coord, nthneighbor=1, storekdtree='kdtree_sky')
    t = tract_list[idx]
    arr, index = np.unique(t['tract_id'], return_index = True)
    #print(index)
    #t = t[index]
    return arr

