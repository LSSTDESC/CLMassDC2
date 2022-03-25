import sys, os, glob, fnmatch
import numpy as np
from astropy.table import Table, vstack, join
import pickle 
import clmm
import clmm.dataops
from clmm.dataops import compute_tangential_and_cross_components, make_radial_profile, make_bins
from clmm.galaxycluster import GalaxyCluster
import clmm.utils as u
from clmm import Cosmology
from clmm.support import mock_data as mock
cosmo = Cosmology(H0 = 71.0, Omega_dm0 = 0.265 - 0.0448, Omega_b0 = 0.0448, Omega_k0 = 0.0)
from scipy import interpolate

def load(filename, **kwargs):
    with open(filename, 'rb') as fin:
        return pickle.load(fin, **kwargs)

cosmo = cosmo
moo = clmm.Modeling(massdef = 'mean', delta_mdef = 200, halo_profile_model = 'nfw')
moo.set_cosmo(cosmo)

class MetaCatalog():
    r"""
    A class for creating stacked background galaxy catalog and deriving shear profile using stacking estimation.
    """
    def __init__(self, cosmo = 0):  
        self.cosmo = cosmo
            
    def _check_available_catalogs(self, dc2_infos = 1, redshift_def ='z_RM', obs_def = 'halo_mass', z_bin = [1,1], obs_bin = [1,1], cluster_name = 'cluster_id', cluster_key = 'cluster_id_RedMapper_', where_source = '1', where_weights_photoz = '1'):

        r"""
        Attributes:
        ----------
        bin_def : string
            the definition for binning the available individual background galaxy catalogs
        z_bin : list
            the bin in redshift [z_low, z_up]
        obs_bin : list
            the bin in the choosen observable 'bin_def' [obs_low, obs_up]
        where_source : string
            the directory of individual cluster catalogs
        r_lim : array
            check completeness of individual catalogs up to r_lim [Mpc]
        Returns:
        -------
        the list of corresponding files for stacking

        """
        file_where_source = np.array(glob.glob(where_source + cluster_key + '*'))
        print(len(file_where_source))
        self.z_bin, self.obs_bin = z_bin, obs_bin
        self.where_source = where_source
        self.cluster_key = cluster_key
        self.cluster_name = cluster_name
        self.redshift_def = redshift_def
        self.where_weights_photoz = where_weights_photoz
        self.dc2_infos = dc2_infos
        halo_id, redshift = dc2_infos[cluster_name], dc2_infos[redshift_def]
        obs =  dc2_infos[obs_def]
        mask_z = (redshift > self.z_bin[0]) * (redshift < self.z_bin[1])
        mask_obs = (obs > self.obs_bin[0]) * (obs < self.obs_bin[1])
        mask_tot = mask_z * mask_obs
        self.selected_halo_id = halo_id[mask_tot]
        self.file_in_bin = np.array([where_source + cluster_key + str(h_id) + '.pkl' for h_id in self.selected_halo_id])
        mask_in = np.isin(self.file_in_bin, file_where_source)
        self.file_in_bin = self.file_in_bin[mask_in]
        self.selected_halo_id = self.selected_halo_id[mask_in]
        mask = np.isin(dc2_infos[cluster_name], self.selected_halo_id)
        self.z_in_bin = dc2_infos[redshift_def][mask]
        self.obs_in_bin = dc2_infos[obs_def][mask]
        self.z, self.z_rms = np.mean(self.z_in_bin), np.std(self.z_in_bin)
        self.obs, self.obs_rms = np.mean(self.obs_in_bin), np.std(self.obs_in_bin) 
        
    def make_GalaxyCluster_catalog(self, z_gal_name = 'z', cluster_ra = 'ra', cluster_dec = 'dec', shape_component1_in = 'shear1', shape_component2_in = 'shear2', tan_component_out = 'et', cross_component_out = 'ex', column_to_extract = 1, rmax = 1, modify_quantity = True, quantity_modifier = 1):
        
        tan_component, cross_component = 'et', 'ex'
        """
        Make GalaxyCluster catalogs with corresponding r, phi, ra, dec, e1, e2, et, ex, z_gal, sigma_c
        galaxy related quantities
        
        Returns:
        -------
        cl : GalaxyCluster
            Galaxy cluster catalog stacked along all individual clusters
        """ 
        halo_id = []
        column_to_extract = column_to_extract + ['r'] + [z_gal_name] + ['z_cluster']
        catalog = {name:[] for name in column_to_extract}
        for i, halo in enumerate(self.selected_halo_id):
            cl_source = load(self.where_source + self.cluster_key + str(halo) + '.pkl')
            cl_source_tab = cl_source.galcat
            ra = self.dc2_infos[self.dc2_infos[self.cluster_name] == halo][cluster_ra][0]
            dec = self.dc2_infos[self.dc2_infos[self.cluster_name] == halo][cluster_dec][0]
            z = self.dc2_infos[self.dc2_infos[self.cluster_name] == halo][self.redshift_def][0]
            cl_source = clmm.GalaxyCluster('Stack', ra, dec, z, cl_source_tab)
            cl_source.galcat['sigma_c_1_TRUE'] = 1./self.cosmo.eval_sigma_crit(cl_source.z, cl_source.galcat[z_gal_name])
            if modify_quantity == True: cl_source = quantity_modifier(cl_source)
            if len(cl_source.galcat) == 0: continue
            cl_source.compute_tangential_and_cross_components(geometry = "flat",
                                                              shape_component1 = shape_component1_in,
                                                              shape_component2 = shape_component2_in,
                                                              tan_component = tan_component_out, cross_component = cross_component_out, 
                                                              is_deltasigma = False)
            cl_source.galcat['z_cluster'] = cl_source.z
            tab_source = cl_source.galcat
            tab_source['r'] = cosmo.eval_da(cl_source.z)*tab_source['theta']
            mask = (tab_source['r'] < rmax)*(tab_source[z_gal_name] > cl_source.z + 0.1)
            tab_source_cut = tab_source[mask]
            if len(tab_source_cut) == 0: continue 
            tab_source_cut['halo_id'] = halo
            for name in column_to_extract:
                try : 
                    catalog[name].extend(tab_source_cut[name])
                except: a = 1
                    
        t = Table()
        for name in column_to_extract: 
            try: t[name] = np.array(catalog[name])
            except: a = 1
            
        t['galaxy_id'] = np.arange(len(t))
        data = clmm.GCData(t)
        self.cl = clmm.GalaxyCluster('Stack', 0, 0, 0, data)
        self.cl.n_stacked_catalogs = len(self.file_in_bin)
        self.dic = catalog
    
    def compute_signal(self, sigma_c_1_in = '1', tan_in = '1', cross_in = '1', tan_out = '1', cross_out = '1', catalog = 1):
    
        catalog.galcat[tan_out] = catalog.galcat[sigma_c_1_in]**(-1.)*catalog.galcat[tan_in]
        catalog.galcat[cross_out] = catalog.galcat[sigma_c_1_in]**(-1.)*catalog.galcat[cross_in]
        return catalog
    
    def compute_weights(self, err_shape_in = '1', sigma_c_1_in = '1', weight_out = '1', catalog = 1):
        
        if err_shape_in == None:
            w_ls_shape = 1
        else: w_ls_shape =  1./(catalog.galcat[err_shape_in]**2/2 + np.std(catalog.galcat['et'])**2)
        catalog.galcat[weight_out] = w_ls_shape * catalog.galcat[sigma_c_1_in]**(2.)
        return catalog
    
    def remove_columns(self, columns = ['a','b']):
        
        cl_tab = Table(self.cl.galcat)
        cl_tab.remove_columns(columns)
        data = clmm.GCData(cl_tab)
        self.cl = clmm.GalaxyCluster('Stack', 0, 0, self.cl.z, data)
        
        return self.cl

        

