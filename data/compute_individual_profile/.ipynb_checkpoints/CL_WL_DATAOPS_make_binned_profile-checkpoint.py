import sys
import os
import numpy as np
from astropy.table import Table
import fnmatch
import pickle 
import random
import glob
from scipy import interpolate

class Shear():
    
    r"""
    A class for creating stacked background galaxy catalog and deriving shear profile using stacking estimation.
    """
    
    def __init__(self, is_deltasigma = True, cosmo = 0):
        
        self.is_deltasigma = is_deltasigma
        
        self.cosmo = cosmo
        
   
    def make_binned_profile(self, metacatalog = 1, tan_in = 'st', cross_in = 'sx', weights = 'w_ls', tan_out = 'gt', cross_out = 'gx',  add_columns_to_bin = [], bin_edges = 1):
        
        cl = metacatalog
        
        col_to_bin = {add_columns_to_bin[i] : [] for i in range(len(add_columns_to_bin))}
        
        """
        Attributes:
        ----------
        cl : GalaxyCluster catalog (clmm)
            the background galaxy cluster catalog where weights are computed, and radial distance to cluster center
        bin_edges: array
            edges of radial bins for making binned profile
        Returns:
        -------
        profile : Astropy Table
            table containing shear estimation information
        """
        
        radial_bins = [[bin_edges[i], bin_edges[i + 1]] for i in range(len(bin_edges) - 1)]
        
        """stacked signal quantities"""
                  
        radius, s_t, s_x, count = [], [], [], []
        
        """assigned value to locate individual galaxies """
        
        n_gal, gal_id, r_to_center, halo_id, wls  = [], [], [], [], [] 
        
        """individual weak lensing quantities"""
        
        e_tangential, e_cross = [], []
                  
        for radial_bin in radial_bins:
                  
            mask_down, mask_up = (cl.galcat['r'] >= radial_bin[0]),  (cl.galcat['r'] < radial_bin[1])
            
            mask = mask_down * mask_up
                
            r = cl.galcat['r'][mask]
                  
            et = cl.galcat[tan_in][mask]
                  
            ex = cl.galcat[cross_in][mask]
            
            w_ls = cl.galcat[weights][mask]
            
            n_gal = len(w_ls)
            
            if len(mask[mask == True]) == 0: radius.append(0), s_t.append(0), s_x.append(0), count.append(0)
                
            else:
                  
                radius.append(np.sum(r * w_ls)/np.sum(w_ls))

                s_t.append(np.sum(et * w_ls)/np.sum(w_ls))

                s_x.append(np.sum(ex * w_ls)/np.sum(w_ls))
                
                count.append(n_gal)
            
            for name in add_columns_to_bin:
                
                col_to_bin[name].append(cl.galcat[name][mask])
            
        profile = Table()
        
        profile[tan_out] = np.array(s_t, dtype=object)
        
        profile[cross_out] = np.array(s_x, dtype=object)
                  
        profile['radius'] = np.array(radius, dtype=object)
        
        profile['n_gal'] = np.array(count, dtype=object)
        
        for name in add_columns_to_bin:
        
            profile[name] = np.array(col_to_bin[name], dtype=object)
        
        self.profile = profile
                       
        return profile
    
    def make_binned_average(self, profile = 1, v_in = 'st', weights = 'w_ls', v_out = 'gt'):
             
        quant = []

        for i, r in enumerate(profile['radius']):
            
            if v_in == None:
                
                quantity_to_average = 1. + np.zeros(len(profile['radius'][i]))
                  
            else : quantity_to_average = profile[v_in][i]
            
            w_ls = profile[weights][i]
            
            if len(w_ls) == 0: quant.append(0)
                
            else: quant.append(np.sum(quantity_to_average * w_ls)/np.sum(w_ls))
        
        profile[v_out] = np.array(quant, dtype=object)
                       
        return profile
    
    def make_binned_sum(self, profile = 1, v_in = 'st', v_out = 'gt'):
            
        quant = []
           
        for i, r in enumerate(profile['radius']):
            
            if v_in == None:
                
                quantity_to_average = 1. + np.zeros(len(profile['radius'][i]))
                  
            else : quantity_to_sum = profile[v_in][i]
                  
            quant.append(np.sum(quantity_to_sum))
        
        profile[v_out] = np.array(quant, dtype=object)
                       
        return profile
    
    
    def calculate_binned_selection_response(self, profile = 1,  uncal_e1 = 'e1', uncal_e2 = 'e2', R_s_out = '1', s2n_cut = 10):
        
        av_RS = []
        
        for i, r in enumerate(profile['radius']):
            
            sample = profile[i]
        
            delta_gamma = 0.02

            sel_1p = sample['mcal_s2n_1p'] > s2n_cut

            sel_1m = sample['mcal_s2n_1m'] > s2n_cut

            sel_2p = sample['mcal_s2n_2p'] > s2n_cut

            sel_2m = sample['mcal_s2n_2m'] > s2n_cut
            
            try:
                
                S_11 = (sample[uncal_e1][sel_1p].mean() - sample[uncal_e1][sel_1m].mean()) / delta_gamma
                #S_12 = (tab_cut[e_1][sel_2p].mean() - tab_cut[e_1][sel_2m].mean()) / delta_gamma
                #S_21 = (tab_cut[e_2][sel_1p].mean() - tab_cut[e_2][sel_1m].mean()) / delta_gamma
                S_22 = (sample[uncal_e2][sel_2p].mean() - sample[uncal_e2][sel_2m].mean()) / delta_gamma

                av_RS.append(0.5*(S_11 + S_22))
                
            except:
                
                av_RS.append(0)
                

        profile[R_s_out] = np.array(av_RS)
    
    def bootstrap_resampling_with_numpy(self, binned_profile = 1, tan_in = 'st', cross_in = 'sx', weights = 'w_ls', metacatalog = 1, n_boot = 1):
        
        """
        Method:
        ------
        Calculates the bootstrap covariance matrix from true shear measurements
        Attributes:
        ----------
        binned_profile : Astropy Table
            Table containing meta data and binned profile
        catalog : GalaxyCluster catalog
            meta data catalog
        n_boot : int
            the number of bootstrap resampling
        Returns:
        -------
        cov_t_boot, cov_x_boot : array, array
            the covariance matrices respectively for tangential and cross shear
        """
        
        profile = binned_profile
        
        names = []
        
        for i, r in enumerate(profile):
            
            names.append('x' + str(i))
        
        t,x = Table(names = names), Table(names = names)
        
        indexes = np.unique(metacatalog.galcat['halo_id'])
        
        for n in range(n_boot):

            choice_halo_id = np.array(np.random.choice(indexes, metacatalog.n_stacked_catalogs))
   
            unique_halo_id, n_repeated = np.unique(choice_halo_id, return_counts = True)

            signal_t, signal_x = [], []

            index = np.argsort(unique_halo_id)

            unique_id, repetition = unique_halo_id[index], n_repeated[index]
            
            if len(unique_id) == 1:
                
                def f(x): return repetition[0]
            
            else: f = interpolate.interp1d(unique_id, repetition)

            for i, r in enumerate(profile['radius']):

                mask = np.isin(profile['halo_id'][i], unique_halo_id)

                halo_id = np.array(profile['halo_id'][i][mask])

                et, ex = profile[tan_in][i][mask], profile[cross_in][i][mask]

                r = f(halo_id)

                wls = profile[weights][i][mask] * r

                signal_t.append(np.sum(et*wls)/np.sum(wls))

                signal_x.append(np.sum(ex*wls)/np.sum(wls))

            t.add_row(signal_t), x.add_row(signal_x)
            
        listet, listex = [], []
        
        for n in names:
            
            listet.append(t[n]), listex.append(x[n])
            
        Xt, Xx = np.stack((listet), axis=0), np.stack((listex), axis=0)
        
        covt, covx = np.cov(Xt), np.cov(Xx)
        
        return covt, covx
    
    def bootstrap_resampling_cluster_with_calibration(self, binned_profile = 1, tan_in = 'st', cross_in = 'sx', weights = 'w_ls', shear_response = 'Rg', n_boot = 1, halo_id = 1):
            
        halo_id_sample = halo_id
        
        list_st, list_sx = [], []
        
        names = []
        
        for i, r in enumerate(binned_profile['radius']):
            
            names.append('x' + str(i))
        
        t,x = Table(names = names), Table(names = names)
        
        for k in range(n_boot):
            
            random.shuffle(halo_id_sample)
            
            halo_id_bootstrap = np.random.choice(halo_id_sample, len(halo_id))
            
            unique_halo_id, n_repeated = np.unique(halo_id_bootstrap, return_counts = True)
            
            a = np.unique(halo_id_bootstrap, return_counts = True)

            index = np.argsort(unique_halo_id)

            unique_id, repetition = unique_halo_id[index], n_repeated[index]
            
            if len(unique_id) == 1:
                
                def f(x): return repetition[0]
                
            else: f = interpolate.interp1d(unique_id, repetition)
                
            signal_t, signal_x = [], []
             
            profile_boot = Table()
                
            for i, r in enumerate(binned_profile['radius']):

                mask = np.isin(binned_profile['halo_id'][i], unique_halo_id)

                halo_id_selected = np.array(binned_profile['halo_id'][i][mask])

                st, sx = binned_profile[tan_in][i][mask], binned_profile[cross_in][i][mask]
                
                if shear_response != None:
                
                    Rg = binned_profile[shear_response][i][mask]
                    
                else: Rg = 1. + np.zeros(len(st))

                r = f(halo_id_selected)

                wls = binned_profile[weights][i][mask] * r

                av_st = np.sum((st*wls)/np.sum(wls))

                av_sx = np.sum((sx*wls)/np.sum(wls))
                
                av_Rg = np.sum((Rg*wls)/np.sum(wls))
                
                signal_t.append(av_st/av_Rg), signal_x.append(av_sx/av_Rg)

            t.add_row(np.array(signal_t)), x.add_row(np.array(signal_x))
            
            listet, listex = [], []
        
        for n in names:
            
            listet.append(t[n]), listex.append(x[n])
            
        Xt, Xx = np.stack((listet), axis=0), np.stack((listex), axis=0)
        
        covt, covx = np.cov(Xt), np.cov(Xx)
        
        return covt, covx, Xt, Xx
    
    
    def bootstrap_resampling_galaxy(self, catalog = 1, tan_in = 'st', cross_in = 'sx', tan_out = 'gt', cross_out = 'gx', weights = 'w_ls', bin_edges = 1, n_boot = 1):
        
        list_st, list_sx = [], []
        
        names = []
        
        for i in range(len(bin_edges) - 1):
            
            names.append('x' + str(i))
        
        t,x = Table(names = names), Table(names = names)
        
        for k in range(n_boot):
                
            signal_t, signal_x = [], []
                
            mask = np.random.choice(np.arange(len(catalog.galcat)), len(catalog.galcat))
                                    
            catalog_boot = clmm.GalaxyCluster('Stack', 0, 0, 0, catalog.galcat[mask])
                                    
            profile_boot = self.make_binned_profile(metacatalog = catalog_boot, tan_in = tan_in, cross_in = cross_in, weights = weights, tan_out = tan_out, cross_out = cross_out, bin_edges = bin_edges)

            st, sx = profile_boot[tan_out], profile_boot[cross_out]

            t.add_row(np.array(st)), x.add_row(np.array(sx))

        listet, listex = [], []

        for n in names:

            listet.append(t[n]), listex.append(x[n])

        Xt, Xx = np.stack((listet), axis=0), np.stack((listex), axis=0)

        covt, covx = np.cov(Xt), np.cov(Xx)

        return covt, covx, Xt, Xx

    def jacknife_resampling(self, binned_profile = 1, catalog = 1, n_jk = 1):
        
        """
        Method:
        ------
        Calculates the jacknife covariance matrix from true shear measurements
        
        Attributes:
        ----------
        binned_profile : Astropy Table
            Table containing meta data and binned profile
        catalog : GalaxyCluster catalog
            meta data catalog
        n_jk : int
            the number of jacknife resampling
            
        Returns:
        -------
        cov_t_jk, cov_x_jk : array, array
            the covariance matrices respectively for tangential and cross shear
        """
    
        profile = binned_profile
        
        cl = catalog
        
        n_bins = len(profile['radius'])

        Stat_t, Stat_x = stat.Statistics(n_bins), stat.Statistics(n_bins)

        indexes = np.arange(cl.n_stacked_catalogs)

        indexes_cut = np.array_split(indexes, n_jk)

        for i in range(n_jk):

            unique_halo_id = indexes_cut[i]

            signal_t, signal_x = [], []

            for i, R in enumerate(profile['radius']):

                mask_is_in = np.isin(profile['halo_id'][i], unique_halo_id)

                mask = np.invert(mask_is_in)

                halo_id = np.array(profile['halo_id'][i][mask])

                et, ex = profile['et'][i][mask], profile['ex'][i][mask]

                wls = profile['w_ls'][i][mask] 

                signal_t.append(np.sum(et*wls)/np.sum(wls))

                signal_x.append(np.sum(ex*wls)/np.sum(wls))

            Stat_t._add_realization(signal_t), Stat_x._add_realization(signal_x)

        Stat_t.estimate_covariance(), Stat_x.estimate_covariance()

        coeff = 1

        cov_t_jk = Stat_t.covariance_matrix  * ((n_jk - 1)/(n_jk)) * coeff

        cov_x_jk = Stat_x.covariance_matrix  * ((n_jk - 1)/(n_jk)) * coeff

        return cov_t_jk, cov_x_jk







                  

                  





