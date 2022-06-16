import numpy as np
import cosmodc2
import lens_data as lens
import GCRCatalogs
import healpy
import clmm
import pickle,sys
import utils
sys.path.append('/pbs/throng/lsst/users/cpayerne/LikelihoodsClusterAbundance/modules/')
import edit

from clmm import Cosmology
from scipy.integrate import simps

#cosmoDC2 cosmology
cosmo = Cosmology(H0 = 71.0, Omega_dm0 = 0.265 - 0.0448, Omega_b0 = 0.0448, Omega_k0 = 0.0)

#connection with qserv
import mysql
from mysql.connector import Error
conn = mysql.connector.connect(host='ccqserv201', user='qsmaster', port=30040)
cursor = conn.cursor(dictionary=True, buffered=True)

start, end = int(sys.argv[1]), int(sys.argv[2])

#select galaxy clusters
lens_catalog_name='/pbs/throng/lsst/users/cpayerne/CLMassDC2/data/lens_catalog_cosmoDC2_v1.1.4_redmapper_v0.8.1.pkl'
#lens_catalog_name='/pbs/throng/lsst/users/cpayerne/CLMassDC2/data/lens_catalog_SkySim5000.pkl'

lens_catalog=edit.load_pickle(lens_catalog_name)
where_to_save='/sps/lsst/users/cpayerne/CLMassDC2/cosmoDC2/redmapper_clusters/'
mask=np.arange(start, end)
lens_catalog_truncated=lens_catalog[mask]

#load source catalogs
gc_flex = GCRCatalogs.load_catalog("cosmoDC2_v1.1.4_image_with_photozs_flexzboost_v1")
gc_bpz  = GCRCatalogs.load_catalog("cosmoDC2_v1.1.4_image_with_photozs_v1")

#list of healpix in cosmoDC2
healpix_dc2 = GCRCatalogs.load_catalog("cosmoDC2_v1.1.4_image").get_catalog_info()['healpix_pixels']
z_bins  = gc_flex.photoz_pdf_bin_centers
z_bins[0] = 1e-7
photoz_label=['_bpz', '_flex']
photoz_gc=[gc_bpz, gc_flex]

def qserv_query(lens_z, lens_distance, ra, dec, rmax = 10):
    r"""
    quantities wanted + cuts for qserv
    Attributes:
    -----------
    z: float
        lens redshift
    lens_distance: float
        distance to the cluster
    ra: float
        lens right ascension
    dec: float
        lens declinaison
    rmax: float
        maximum radius
    """
    zmax = 3.
    zmin = lens_z + .1
    theta_max = (rmax/lens_distance) * (180./np.pi)
    query = "SELECT data.coord_ra as ra, data.coord_dec as dec, data.redshift as z, "
    query += "data.galaxy_id as galaxy_id, "
    query += "data.mag_i, data.mag_r, data.mag_y, "
    query += "data.shear_1 as shear1, data.shear_2 as shear2, data.convergence as kappa, "
    query += "data.ellipticity_1_true as e1_true, data.ellipticity_2_true as e2_true " 
    query += "FROM cosmoDC2_v1_1_4_image.data as data "
    query += f"WHERE data.redshift >= {zmin} AND data.redshift < {zmax} "
    query += f"AND scisql_s2PtInCircle(coord_ra, coord_dec, {ra}, {dec}, {theta_max}) = 1 "
    query += f"AND data.mag_i <= 24.5 "
    query += ";" 
    return query

for n, lens in enumerate(lens_catalog_truncated):
    
    #cluster metadata
    z, ra, dec=lens['redshift'], lens['ra'], lens['dec']
    cluster_id=lens['cluster_id']
    lens_distance=cosmo.eval_da(z)
    
    #extract background galaxies with qserv (only true photoz)
    print('extracting true redshift infos (Qserv)')
    bckgd_galaxy_catalog=cosmodc2.extract(qserv_query = qserv_query(z, lens_distance, ra, dec, rmax = 1),
                                        conn_qserv=conn, cosmo=cosmo)
    bckgd_galaxy_catalog.ra = ra
    bckgd_galaxy_catalog.dec = dec
    bckgd_galaxy_catalog.z = z
    bckgd_galaxy_catalog.id = cluster_id
    
    #extract photometric redshifts with GCRCatalogs
    print('extracting photoz redshift infos (GCRCatalogs)')
    id_gal=bckgd_galaxy_catalog.galcat['galaxy_id']
    ras=bckgd_galaxy_catalog.galcat['ra']
    decs=bckgd_galaxy_catalog.galcat['dec']
    
    #find all different healpix pixels
    healpix = np.unique(healpy.ang2pix(32, ras, decs, nest=False, lonlat=True))
    healpix = healpix[np.isin(healpix, healpix_dc2)]
    for k, photoz_gc_ in enumerate(photoz_gc):
        
        #load photoz_data
        dat_photoz=cosmodc2.extract_photoz(id_gal, healpix=healpix, GCRcatalog=photoz_gc_)
        name_dat_photoz_to_save=['photoz_zbins', 'photoz_pdf', 'photoz_mean', 'photoz_odds', 'photoz_mode']
        
        #store data to save
        pzbins_table=np.array([z_bins for i in range(len(dat_photoz['photoz_pdf'].data))])
        dat_to_save=[pzbins_table,
                     dat_photoz['photoz_pdf'].data,
                     dat_photoz['photoz_mean'].data,
                     dat_photoz['photoz_odds'].data,
                     dat_photoz['photoz_odds'].data]
        
        #save data
        for s, dat in enumerate(dat_to_save):
            name=name_dat_photoz_to_save[s]
            bckgd_galaxy_catalog.galcat[name+photoz_label[k]]=dat
        
        #compute effective critical surface density (compare to clmm)
        sigma_c_my = utils.compute_photoz_sigmac(z, bckgd_galaxy_catalog.galcat['photoz_pdf'+ photoz_label[k]], 
                                                 pzbins_table, cosmo=cosmo, use_clmm=False)
        
        sigma_c_CLMM = utils.compute_photoz_sigmac(z, bckgd_galaxy_catalog.galcat['photoz_pdf'+ photoz_label[k]], 
                                                   pzbins_table, cosmo=cosmo, use_clmm=True)
        
        #compute background probability
        p_background = utils.compute_p_background(z, bckgd_galaxy_catalog.galcat['photoz_pdf'+ photoz_label[k]], 
                                                  pzbins_table, use_clmm=False)
        
        bckgd_galaxy_catalog.galcat['sigma_c_my' + photoz_label[k]] = np.array(sigma_c_my)
        bckgd_galaxy_catalog.galcat['sigma_c_CLMM' + photoz_label[k]] = np.array(sigma_c_CLMM)
        bckgd_galaxy_catalog.galcat['p_background' + photoz_label[k]] = np.array(p_background)
        
        #remove photoz_pdf from table (too laege to save)
        #bckgd_galaxy_catalog.galcat.remove_column('photoz_zbins'+ photoz_label[k])
        #bckgd_galaxy_catalog.galcat.remove_column('photoz_pdf'+ photoz_label[k])
    
    #create GalaxyCluster object                                                                  
    cl = clmm.GalaxyCluster('cosmodc2', ra, dec, z, clmm.GCData(bckgd_galaxy_catalog.galcat))
            
    edit.save_pickle(cl, where_to_save + 'test_bckgd_source_cat_of_cluster_' + str(cluster_id) + '.pkl')