import numpy as np
import cosmodc2
import lens_data as lens
import matplotlib.pyplot as plt
import GCRCatalogs
import healpy
import clmm
import pickle,sys

sys.path.append('/pbs/throng/lsst/users/cpayerne/LikelihoodsClusterAbundance/modules/')
import edit

from clmm.dataops import compute_galaxy_weights
from clmm import Cosmology
from scipy.integrate import simps
cosmo = Cosmology(H0 = 71.0, Omega_dm0 = 0.265 - 0.0448, Omega_b0 = 0.0448, Omega_k0 = 0.0)

#connection with qserv
import mysql
from mysql.connector import Error
conn = mysql.connector.connect(host='ccqserv201', user='qsmaster', port=30040)
cursor = conn.cursor(dictionary=True, buffered=True)

start, end = int(sys.argv[1]), int(sys.argv[2])

#select galaxy clusters
lens_catalog_name='/pbs/throng/lsst/users/cpayerne/CLMassDC2/data/lens_catalog_cosmoDC2_v1.1.4_redmapper_v0.8.1.pkl'
lens_catalog=edit.load_pickle(lens_catalog_name)
where_to_save='/sps/lsst/users/cpayerne/CLMassDC2/DC2/'
mask=np.arange(start, end)
lens_catalog_truncated=lens_catalog[mask]

#load source catalogs
gc_flex = GCRCatalogs.load_catalog("cosmoDC2_v1.1.4_image_with_photozs_flexzboost_v1")
gc_bpz  = GCRCatalogs.load_catalog("cosmoDC2_v1.1.4_image_with_photozs_v1")
healpix_dc2 = GCRCatalogs.load_catalog("cosmoDC2_v1.1.4_image").get_catalog_info()['healpix_pixels']
z_bins  = gc_flex.photoz_pdf_bin_centers
z_bins[0] = 1e-7
photoz_label=['bpz', 'flex']
photoz_gc=[gc_bpz, gc_flex]

def qserv_query(lens_z, lens_distance, ra, dec, rmax = 10):
    r"""
    quantities wanted + cuts
    Attributes:
    -----------
    z: float
        lens redshift
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
    query += f"AND data.mag_i <= 25 "
    query += ";" 
    return query

for n, lens in enumerate(lens_catalog_truncated):
    
    z, ra, dec=lens['redshift'], lens['ra'], lens['dec']
    cluster_id=lens['cluster_id']
    lens_distance=cosmo.eval_da(z)
    
    #extract background galaxies with qserv (only true photoz)
    print('extracting true redshift infos (Qserv)')
    bckgd_galaxy_catalog=cosmodc2.extract(qserv_query = qserv_query(z, lens_distance, ra, dec, rmax = 5),
                                        conn_qserv=conn, cosmo=cosmo)
    bckgd_galaxy_catalog.ra = ra
    bckgd_galaxy_catalog.dec = dec
    bckgd_galaxy_catalog.z = z
    
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
        dat_to_save=[np.array([z_bins for i in range(len(dat_photoz['photoz_pdf'].data))]),
                     dat_photoz['photoz_pdf'].data,
                     dat_photoz['photoz_mean'].data,
                     dat_photoz['photoz_odds'].data,
                     dat_photoz['photoz_odds'].data]
        
        #save data
        for s, dat in enumerate(dat_to_save):
            name=name_dat_photoz_to_save[s]
            bckgd_galaxy_catalog.galcat[name + photoz_label[k]]=dat
    
    cl = clmm.GalaxyCluster('cosmodc2', ra, dec, z, clmm.GCData(bckgd_galaxy_catalog.galcat))
            
    #edit.save_pickle(cl, where_to_save + name)