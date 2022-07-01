import numpy as np
import clmm
from astropy.table import Table

def check(table, ra_cl, dec_cl, z_cl, cosmo):
    
    print(table.colnames)
    
    cl = clmm.galaxycluster.GalaxyCluster('id', ra_cl, dec_cl, z_cl, clmm.gcdata.GCData(Table(table)))
    theta1, g_t, g_x = cl.compute_tangential_and_cross_components(is_deltasigma=False, cosmo=cosmo)
    ld = cosmo.eval_da_z1z2(0,z_cl)
    print('cluster: ra, dec = ' + str(ra_cl) + ',' + str(dec_cl))
    print('cluster: z = ' + str(z_cl) )
    print('----')
    print('galaxies: maximum distance = ' + str(max( ld * theta1)))
    print('galaxies: minimum redshift = ' + str(min( cl.galcat['z'])))
    print('galaxies: minimum ra, maximum ra = ' + str(min( cl.galcat['ra'])) + ',' + str(max( cl.galcat['ra'])))
    print('galaxies: minimum dec, maximum dec = ' + str(min( cl.galcat['dec'])) + ',' + str(max( cl.galcat['dec'])))