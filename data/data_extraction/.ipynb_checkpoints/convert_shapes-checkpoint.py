import numpy as np

def e12(chi1, chi2):
    r""" epsilon 12 from chi12 """
    chi = np.sqrt(chi1**2 + chi2**2)
    zero = 1. + (1. - chi**2)**(1./2.)
    return chi1/zero, chi2/zero
    
def chi12(e1, e2):
    r""" chi12 from epsilon12 """
    e = np.sqrt(e1**2+e2**2)
    zero = 1. + e**2
    return 2*e1/zero, 2*e2/zero
    
def e_sigma(chi1, chi2, chi_sigma):
    r""" error on epsilon from error on chi"""
    chi = np.sqrt(chi1**2 + chi2**2)
    zero = 1. + (1. - chi**2)**(1/2)
    first = 1./zero
    second = chi**2/((1 - chi**2)**0.5)
    return first*(1 + second*first)*chi_sigma

def chi_sigma(e1, e2, e_sigma_):
    r""" error on chi from error on epsilon """
    e = np.sqrt(e1**2+e2**2)
    chi1, chi2 = chi12(e1, e2)
    chi = np.sqrt(chi1**2 +chi2**2)
    return e_sigma_*chi*(1./e - chi)

