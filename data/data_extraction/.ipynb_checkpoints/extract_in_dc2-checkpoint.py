import numpy as np
from astropy.table import QTable, Table, vstack, hstack
import clmm
import matplotlib.pyplot as plt
from GCR import GCRQuery
from clmm.galaxycluster import GalaxyCluster
import numpy as np
import random
import DC2_tract_coordinates as c
import lens_data as run
import convert_shapes as shape
import photoz_utils

def combinatory(X):
    dimension = len(X)
    mesh = np.array(np.meshgrid(*X))
    combinations = mesh.T.reshape(-1, dimension)
    Xcomb = combinations.T
    return Xcomb

def angular_distance(ra_cl, dec_cl, ra, dec):
    r"""anbgular dustance between lens and gal"""
    return np.sqrt((ra_cl - ra)**2*np.cos(dec_cl*np.pi/180)**2 + (dec - dec_cl)**2)

def neighboring_tract(lens_ra, lens_dec, theta_max):

    ra = lens_ra + np.random.random(100)*(theta_max + theta_max) - theta_max
    dec = lens_dec + np.random.random(100)*(theta_max + theta_max) - theta_max
    return c.neigboring_tracts(ra_deg = ra, dec_deg = dec)