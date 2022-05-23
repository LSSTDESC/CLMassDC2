import pickle
from astropy.table import Table, QTable, hstack, vstack,join
import sys
import numpy as np
from iminuit import Minuit
import emcee
import astropy.units as u
from astropy.coordinates import SkyCoord, match_coordinates_3d, match_coordinates_sky
from astropy.cosmology import FlatLambdaCDM
cosmo_astropy = FlatLambdaCDM(H0=71.0, Om0=0.265, Ob0 = 0.0448)
from clevar.catalog import ClCatalog
from clevar.match import ProximityMatch
from clevar.cosmology import AstroPyCosmology
from clevar.match import get_matched_pairs
import CL_DATAOPS_match_catalogs as match
import CL_Likelihood_for_Mass_richness_relation as mass_richness
import analysis_Mass_Richness_relation as analysis

def theta(ra1,dec1, ra2, dec2):
    r"""
    Attributes:
    -----------
    ra1, dec1: float, float
        position of object 1
    ra2, dec2: float, float
        position of object 2
    Returs:
    -------
    t: float
        angular distance between 1 & 2
    """
    t = np.sqrt((ra1 - ra2)**2*np.cos(dec1*np.pi/180)**2 + (dec1 - dec2)**2)*(np.pi/180)
    return t

def match1to2(cat1, cat2, cat1_SkyCoord, cat2_SkyCoord, label1='_1', label2='_2', deltaz=.05):
    r"""
    cat1: Table
    cat2: Table
    cat1_SkyCoord: SkyCoord catalog
    cat2_SkyCoord: SkyCoord catalog
    label1: str
    label2: str
    deltaz: float
    """
    idx, sep2d, sep3d = match_coordinates_sky(cat1_SkyCoord,cat2_SkyCoord,nthneighbor=1,storekdtree='kdtree_3d')
    cat_base=Table(np.copy(cat1))
    cat_target=Table(np.copy(cat2))[idx]
    for name in cat_base.colnames: cat_base.rename_column(name, name+label1)
    for name in cat_target.colnames: cat_target.rename_column(name, name+label2)
    tab12=hstack([cat_base, cat_target])
    angsep12=theta(tab12['ra'+label1],tab12['dec'+label1], tab12['ra'+label2], tab12['dec'+label2])
    distance12=cosmo_astropy.angular_diameter_distance(tab12['redshift'+label1]).value * angsep12
    dz12=abs(tab12['redshift'+label1]-tab12['redshift'+label2])
    maskz = dz12 < deltaz#*(1 + tab12['redshift'+label1])
    maskr = distance12 < 1.
    cat_target_cut=cat_target
    return tab12[maskz*maskr]

def match_catalog(cat1, cat2, clevar=False, deltaz=.05):
    r"""
    cat1: Table
    cat2: Table
    clevar: boolean
    deltaz: float
    """
    # Matching catalogs
    if clevar==False:
        cat1_SkyCoord = SkyCoord(ra=cat1['ra']*u.deg, dec=cat1['dec']*u.deg, 
                            distance=cosmo_astropy.angular_diameter_distance(cat1['redshift']))
        cat2_SkyCoord = SkyCoord(ra=cat2['ra']*u.deg, dec=cat2['dec']*u.deg, 
                            distance=cosmo_astropy.angular_diameter_distance(cat2['redshift']))
        #matching each 1 to 2
        match12 = match1to2(cat1, cat2, cat1_SkyCoord, cat2_SkyCoord, label1='_1', label2='_2',deltaz=deltaz)
        #matching each 2 to 1
        match21 = match1to2(cat2, cat1, cat2_SkyCoord, cat1_SkyCoord, label1='_2', label2='_1',deltaz=deltaz)
        for name in match12.colnames:
            match12.rename_column(name, name + '_1_as_base')
        for name in match21.colnames:
            match21.rename_column(name, name + '_2_as_base')
        #rename
        match12['id_to_match']=match12['id_1_1_as_base']
        match21['id_to_match']=match21['id_1_2_as_base']
        #ensure bijective matching
        join1 = join(match12,match21, keys='id_to_match')
        maskid2=(join1['id_2_1_as_base']==join1['id_2_2_as_base'])
        join12=join1[maskid2]
        res = Table()
        names=cat1.colnames
        for n in names:
            res[n+'_1']=join12[n+'_1_1_as_base']
            res[n+'_2']=join12[n+'_2_1_as_base']
        return res

    elif clevar==True:
        #use clevar
        cosmo = AstroPyCosmology()
        input1 = Table({ 'ID': [f'CL{i}' for i in range(len(cat1['id']))],
                        'RA': cat1['ra'],
                        'DEC': cat1['dec'],
                        'Z': cat1['redshift'],
                        'MASS': cat1['obs'],
                        'RADIUS_ARCMIN': np.zeros(len(cat1['id']))})
        input2 = Table({'ID': [f'CL{i}' for i in range(len(cat2['id']))],
                        'RA': cat2['ra'],
                        'DEC': cat2['dec'],
                        'Z': cat2['redshift'],
                        'MASS': cat2['obs'],
                        'RADIUS_ARCMIN': np.zeros(len(cat2['redshift']))})
        c1 = ClCatalog('Cat1', id=input1['ID'], ra=input1['RA'], dec=input1['DEC'], z=input1['Z'], mass=input1['MASS'])
        c2 = ClCatalog('Cat2', id=input2['ID'], ra=input2['RA'], dec=input2['DEC'], z=input2['Z'], mass=input2['MASS'])
        #c1['zmin']=c1['z']-deltaz*(1+c1['z'])
        #c1['zmax']=c1['z']+deltaz*(1+c1['z'])
        c2['zmin']=c2['z']-deltaz*(1+c2['z'])
        c2['zmax']=c2['z']+deltaz*(1+c2['z'])
        match_config = {'type': 'cross', # options are cross, cat1, cat2
                         'which_radius': 'max', # Case of radius to be used, can be: cat1, cat2, min, max
                         'preference': 'angular_proximity', # options are more_massive, angular_proximity or redshift_proximity
                         'catalog1': {'delta_z':deltaz#*(1.+c1['z'])
                                     ,'match_radius': '1 mpc'},
                         'catalog2': {'delta_z':deltaz
                                     ,'match_radius': '1 mpc'}}
#         mt_config1 = {'delta_z':deltaz*(1+c1['z'])
#                         ,'match_radius': '1 mpc',
#                         'cosmo':cosmo}
#         mt_config2 = {'delta_z':deltaz#*(1+c2['z']),
#                     ,'match_radius': '1 mpc',
#                      'cosmo':cosmo}
        mt=ProximityMatch()
        #mt.prep_cat_for_match(c1, **mt_config1)
        #print(c1.mt_input)
        #mt.prep_cat_for_match(c2, **mt_config2)
        #mt.multiple(c1, c2)
        #mt.multiple(c2, c1)
        mt.match_from_config(c1, c2, match_config, cosmo=cosmo)
        mt1, mt2 = get_matched_pairs(c1, c2, 'cross')
        mt1_table = Table(mt1.data)
        mt1_table['halo_id'] = mt1_table['id']
        mt2_table = Table(mt2.data)
        mt2_table['halo_id']=[str(x) for x in mt2_table['mt_cross']]
        matched_table = join(mt1_table,mt2_table, keys='halo_id')
        return matched_table
    
def make_binned(mass, richness, redshift, Z_bin = None, Richness_bin = None):
    r"""make binned mass-richness relation"""
    r"""
    Attributes:
    -----------
    match: Table
        matched catalog
    Z_bin: array
        redshift bins
    Richness_bin: array
        richness bins
    Returns:
    --------
    ml: dict
        binned catalog
    """
    ml = {'Z_bin':[],'Obs_bin':[], 'z_mean' : [], 'logrichness' : [], 'richness_err' : [], 'm200' : [],'m200_err' : [], 'n_stack' : [], 
          'logrichness_in_bin':[], 'redshift_in_bin':[],'M200c_in_bin':[], 'logrichness_err_in_bin':[], 'redshift_err_in_bin':[]}
    for z_bin in Z_bin:
        ml['Z_bin'].append(z_bin)
        for l_bin in Richness_bin:
            ml['Obs_bin'].append(l_bin)
            mask_richness = (richness > l_bin[0])*(richness < l_bin[1])
            mask_z = (redshift > z_bin[0])*(redshift < z_bin[1])
            mask = mask_richness * mask_z
            if len(mask[mask == True]) == 0: continue
            if len(mask[mask == True]) > 1:
                ml['logrichness_in_bin'].append(np.log10(np.array(richness)[mask]))
                ml['M200c_in_bin'].append(mass[mask])
                ml['redshift_in_bin'].append(redshift[mask])
                ml['z_mean'].append(np.average(redshift[mask], weights = None))
                ml['logrichness'].append(np.log10(np.average(richness[mask], weights = None)))
                ml['m200'].append(np.mean(mass[mask]))
                err_m = np.std(mass[mask])
                ml['m200_err'].append(err_m/np.sqrt(len(mass[mask])))
                ml['n_stack'].append(len(mass[mask]))
            if len(mask[mask == True]) == 1: continue   
    return ml

def constrain_fiducial(mass_av, mass_av_err, richness_av, z_av, logrichness_ind, z_ind):

    npath = 100
    nwalkers = 100
    initial_binned = [14.15,0,0.75]
    pos_binned = initial_binned + 0.01 * np.random.randn(npath, len(initial_binned))
    nwalkers, ndim = pos_binned.shape
    sampler_binned_true = emcee.EnsembleSampler(nwalkers, 
                                                ndim, 
                                                mass_richness.lnL_validation_binned, 
                                                args = (mass_av, 
                                                         mass_av_err, 
                                                         logrichness_ind, 
                                                         z_ind, 
                                                         analysis.z0, 
                                                         analysis.richness0))
    sampler_binned_true.run_mcmc(pos_binned, nwalkers, progress=True)
    flat_sampler_binned_true = sampler_binned_true.get_chain(discard=90, flat=True)
    return flat_sampler_binned_true#np.mean(flat_sampler_binned_true, axis = 0)
