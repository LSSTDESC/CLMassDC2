import numpy as np
from astropy.table import Table
from scipy.integrate import simps
from clmm.theory import compute_critical_surface_density
from scipy.integrate import simps
from scipy.interpolate import interp1d

def cdf_from_pdf(pdf, z_array):
    r"""
    Attributes:
    -----------
    pdf: array
        tabulated photometric distribution
    z_array: array
        tabulated redshift axis
    Returns:
    --------
    cdf: array
        tabulated photometric cumulative distribution
    """
    cdf = np.array([simps(pdf[np.arange(j+1)], 
                    z_array[np.arange(j+1)]) 
                    for j in range(len(z_array))])
    return cdf/max(cdf)

def inverse_cdf_fct(cdf, z_array):
    r"""
    Attributes:
    -----------
    cdf: array
        tabulated photometric cumulative distribution
    z_array: array
        tabulated redshift axis
    Returns:
    --------
    cdf_1: fct
        inverse cumulative distribution (interpolated)
    """
    return interp1d(cdf, z_array, kind='linear')
    
def draw_z_from_inverse_cdf(cdf_1, n_samples=1):
    r"""
    Attributes:
    -----------
    cdf_1: fct
        inverse cumulative distribution (interpolated)
    n_samples: int
        number of samples from the pdf
    Returns:
    --------
    z_samples: array
        sampled redshifts
    """
    z_samples = cdf_1(np.random.random(n_samples) * 1)
    return z_samples
    
def draw_z_from_pdf(pdf, pzbins, n_samples=1, use_clmm=False):
    r"""
    Attributes:
    -----------
    pdf: array
        tabulated photometric distribution
    pzbins: array
        tabulated redshift axis
    n_samples: int
        number of samples from the pdf
    use_clmm: Bool
        use clmm or not
    Returns:
    --------
    z_sample: array
        redshift samples from pdfs
    """
        
    n_pdf = len(pdf)
    z_array=pzbins[0]
    z_sample = np.zeros([n_pdf, n_samples])
    
    for i in range(n_pdf):
        
        #create cdf from pdf
        cdf = cdf_from_pdf(pdf[i], z_array)
        #create inverse cdf from cdf
        cdf_1 = inverse_cdf_fct(cdf, z_array)
        #draw sample from cdf_1 (inverse method)
        z_sample[i,:]=np.array(draw_z_from_inverse_cdf(cdf_1, 
                                                       n_samples=n_samples))
    
    return z_sample
        
def compute_photoz_sigmac(z_lens, pdf, pzbins, cosmo=None, use_clmm=False):
    r"""
    Attributes:
    -----------
    z_lens: float
        lens redshift
    pdf: array
        photoz distrib
    pzbins: array
        z_bin centers
    Returns:
    --------
    sigma_c: array
        photometric weak lensing sigma_c
    """
    if pdf.shape!=(len(pdf), len(pzbins)):
        pdf_new=np.zeros([len(pdf), len(pzbins[0])])
        for i, pdf_ in enumerate(pdf):
            pdf_new[i,:] = pdf_
        pdf = pdf_new
    norm=simps(pdf, pzbins, axis=1)
    pdf_norm=(pdf.T*(1./norm)).T
    x=np.linspace(0,0,len(pdf_norm))
    if use_clmm==True:
        sigma_c=compute_critical_surface_density(cosmo, z_lens, 
                                                 z_source=None, use_pdz=True, 
                                                 pzbins=pzbins, pzpdf=pdf_norm)
        return sigma_c
    else:
        sigmacrit_1 = cosmo.eval_sigma_crit(z_lens, pzbins[0,:])**(-1.)
        sigmacrit_1_integrand = (pdf_norm*sigmacrit_1.T)
        return simps(sigmacrit_1_integrand, pzbins[0,:], axis=1)**(-1.)
    
def compute_p_background(z_lens, pdf, pzbins, use_clmm=False):
    r"""
    Attributes:
    -----------
    z_lens: float
        lens redshift
    pdf: array
        photoz distrib
    pzbins: array
        z_bin centers
    Returns:
    --------
    p: array
        probability background
    """
    if pdf.shape!=(len(pdf), len(pzbins)):
        pdf_new=np.zeros([len(pdf), len(pzbins[0])])
        for i, pdf_ in enumerate(pdf):
            pdf_new[i,:] = pdf_
        pdf = pdf_new
    norm=simps(pdf, pzbins, axis=1)
    pdf_norm=(pdf.T*(1./norm)).T
    pdf_norm=pdf_norm[:,pzbins[0,:]>=z_lens]
    return simps(pdf_norm, pzbins[0,:][pzbins[0,:]>=z_lens], axis=1)

def compute_photoz_quantities(z_lens, pdf, pzbins, n_samples_per_pdf=3, 
                              cosmo=None, use_clmm=False):
    r"""
    Attributes:
    z_lens: float
        lens redshift
    pdf: array
        photoz distrib
    pzbins: array
        z_bin centers
    n_samples_per_pdf: int
        number of samples from the pdf
    cosmo: Cosmology CLMM
        CLMM cosmology
    use_clmm: Bool
        use_clmm or not
    Returns:
    data: Table
        photoz_quantities for WL
    """
    
    name = ['sigma_c_photoz', 'p_background']
    name = name + ['sigma_c_photoz_estimate_' + str(k) for k in range(n_samples_per_pdf)]
    name = name + ['z_estimate_' + str(k) for k in range(n_samples_per_pdf)]
    
    #sigma_c full pdf
    sigma_c = compute_photoz_sigmac(z_lens, pdf, pzbins, cosmo=cosmo, use_clmm=use_clmm)
    
    #p_background
    p_background = compute_p_background(z_lens, pdf, pzbins, use_clmm=False)
    
    #sigma_c point estimate
    z_samples = draw_z_from_pdf(pdf, pzbins, n_samples_per_pdf, use_clmm=use_clmm)
    sigma_c_estimate = np.zeros([len(pdf), n_samples_per_pdf])
    for i in range(n_samples_per_pdf):
        sigma_c_estimate[:,i] = cosmo.eval_sigma_crit(z_lens, z_samples[:,i])
        
    data = np.zeros([len(pdf), len(name)])
    data[:,0] = sigma_c
    data[:,1] = p_background
    
    for i in range(n_samples_per_pdf): 
        data[:,1 + i + 1] = sigma_c_estimate[:,i]
        data[:,1 + n_samples_per_pdf + i + 1] = z_samples[:,i]
    res = Table(data, names=name)
    return res
    