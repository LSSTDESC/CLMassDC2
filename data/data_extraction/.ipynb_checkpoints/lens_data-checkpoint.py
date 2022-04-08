import numpy as np
import pickle
filename_redmapper = '/pbs/throng/lsst/users/cpayerne/CLMassDC2/data/lens_catalog_redMaPPer.pkl'
filename_cosmodc2 = '/pbs/throng/lsst/users/cpayerne/CLMassDC2/data/lens_catalog_redMaPPer.pkl'
lens_catalog_redMaPPer = pickle.load(open(filename_redmapper, 'rb'))
lens_catalog_cosmoDC2 = pickle.load(open(filename_cosmodc2, 'rb'))

save_key_redMaPPer = 'redMaPPer'
save_key_cosmoDC2 = 'cosmoDC2'

where_to_save_redMaPPer = '...'
where_to_save_cosmoDC2 = '...'
save_key_cosmodc2 = '...'
save_key_redmapper = '...'

#analysis
lens_catalog = lens_catalog_redMaPPer
save_key = save_key_redmapper
where_to_save = where_to_save_redMaPPer
redshift_name = 'redshift'
ra_name = 'ra'
dec_name = 'dec'

