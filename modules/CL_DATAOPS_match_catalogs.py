import numpy as np
from astropy.coordinates import SkyCoord, match_coordinates_3d, match_coordinates_sky
from astropy.table import Table, QTable, hstack, vstack
import astropy.units as un
from astropy.cosmology import FlatLambdaCDM
cosmo_astropy = FlatLambdaCDM(H0=71.0, Om0=0.265, Ob0 = 0.0448)

def theta(ra1,dec1, ra2, dec2):
    t = np.sqrt((ra1 - ra2)**2*np.cos(dec1*np.pi/180)**2 + (dec1 - dec2)**2)*(np.pi/180)
    return t

def match_nearest_neghbor(base_catalog = 1, target_catalog =1, label_base = '', label_target = '', id_base = '', id_target = ''):
    r"""match to nearest neighbor"""
    ml = {'z_mean' : [], 'richness' : [], 'richness_err' : [], 'logm200' : [],'logm200_err' : [], 'n' : []}
    dat_tot = Table()
    base_SkyCoord = SkyCoord(ra=base_catalog['ra']*un.deg, dec=base_catalog['dec']*un.deg, distance=cosmo_astropy.angular_diameter_distance(base_catalog['redshift']))
    target_SkyCoord = SkyCoord(ra=target_catalog['ra']*un.deg, dec=target_catalog['dec']*un.deg, distance=cosmo_astropy.angular_diameter_distance(target_catalog['redshift']))
    idx, sep2d, sep3d = match_coordinates_sky(base_SkyCoord,target_SkyCoord, nthneighbor=1,storekdtree='kdtree_3d')
    base_cut = base_catalog
    target_cut = target_catalog[idx]
    for name in base_catalog.colnames:
        name_match = name + label_base
        dat_tot[name_match] = base_cut[name]
    for name in target_catalog.colnames:
        name_match = name + label_target
        dat_tot[name_match] = target_cut[name]
    return dat_tot

def selection_cut(match = 1, label_base = '', label_target = ''):
    r"""apply selection cut"""
    angsep = theta(match['ra' + label_base],match['dec' + label_base], match['ra' + label_target], match['dec' + label_target])
    rsep = angsep*cosmo_astropy.angular_diameter_distance(match['redshift' + label_base]).value
    match['distance'] = rsep
    z_selection = abs(match['redshift' + label_base] - match['redshift' + label_target]) < 0.05
    RLambda = ((match['richness'+label_base]/100)**0.2)/0.71
    r_selection = (match['distance'] < RLambda)
    selection = z_selection * r_selection
    match['R_lambda'] = RLambda
    return match[selection]

def find_repetition(match = 1, label_base = '', label_target = '', id_base = '', id_target = ''):
    r"""check if repeated matchs"""
    choose_base_id = []
    list_unique_target_id = np.unique(match[id_target + label_target])
    for target_key in list_unique_target_id:
        mask = np.isin(match[id_target + label_target], target_key)
        cut = match[mask]
        if len(cut) == 1: choose_base_id.append(cut[id_base + label_base][0])
        else: 
            index = np.argsort(cut['distance'])
            choose_base_id.append(cut[index][id_base + label_base][0])
    index = np.arange(len(match))
    ide = match[id_base + label_base]
    mask = np.isin(ide, np.array(choose_base_id))
    return match[mask]

def make_binned(match = 1, Z_bin = '', Richness_bin = ''):
    ml = {'z_mean' : [], 'logrichness' : [], 'richness_err' : [], 'm200' : [],'m200_err' : [], 'n_stack' : [], 
          'logrichness_in_bin':[], 'redshift_in_bin':[],'M200c_in_bin':[], 'logrichness_err_in_bin':[], 'redshift_err_in_bin':[]}
    for z_bin in Z_bin:
        for l_bin in Richness_bin:
            mask_richness = (match['richness_RedMapper'] > l_bin[0])*(match['richness_RedMapper'] < l_bin[1])
            mask_z = (match['redshift_RedMapper'] > z_bin[0])*(match['redshift_RedMapper'] < z_bin[1])
            mask = mask_richness * mask_z
            if len(mask[mask == True]) == 0: continue
            if len(mask[mask == True]) > 1:
                
                ml['logrichness_in_bin'].append(np.log10(np.array(match['richness_RedMapper'][mask])))
                ml['M200c_in_bin'].append(np.array(match['M200c_cosmoDC2'][mask]))
                ml['logrichness_err_in_bin'].append(np.array(match['richness_err_RedMapper'][mask])/(np.log(10) * np.array(match['richness_err_RedMapper'][mask])))
                ml['redshift_in_bin'].append(np.array(match['redshift_RedMapper'][mask]))
                ml['redshift_err_in_bin'].append(np.array(match['redshift_err_RedMapper'][mask]))
                
                weight_z = 1/np.array(match['redshift_err_RedMapper'][mask]**2)
                ml['z_mean'].append(np.average(match['redshift_RedMapper'][mask], weights = None))
                
                weight_richness = 1./(np.array(match['richness_err_RedMapper'][mask])/(np.log(10) * np.array(match['richness_RedMapper'][mask]))**2)
                ml['logrichness'].append(np.log10(np.average(match['richness_RedMapper'][mask], 
                                                    weights = None)))
                err_richness = np.std(match['richness_RedMapper'][mask])
                ml['richness_err'].append(err_richness/np.sqrt(len(match['richness_RedMapper'][mask])))
                ml['m200'].append(np.mean(match['M200c_cosmoDC2'][mask]))
                
                err_m = np.std(match['M200c_cosmoDC2'][mask])
                ml['m200_err'].append(err_m/np.sqrt(len(match['M200c_cosmoDC2'][mask])))
                ml['n_stack'].append(len(match['M200c_cosmoDC2'][mask]))
            if len(mask[mask == True]) == 1: continue
               
    return ml
