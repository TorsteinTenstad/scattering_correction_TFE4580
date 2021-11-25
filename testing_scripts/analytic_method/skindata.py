import numpy as np
from dataclasses import dataclass


# function ip = skindata(lambda_)
#
# Set up data for simulating source photon distribution with source.mexl.
# ip describes a 2-layer slab, surrounded by an ambient medium.
# The data is based on the optical properties of skin.
#
# The call to skindata only allows for adjusting the absorption and
# scattering properties to the wavelength. All other data needs to be 
# edited directly in the file 'skindata.m'.
# These are:
# Number of photons in Monte Carlo simulation
# Number of scattering events before photon is considered a 'source photon'
# Resolution of coordinate system
# Refractive index of all layers


@dataclass
class Layer_info:
    d : float = None
    mua : float = None
    mus : float = None
    g : float = None


def skindata(lambda_):

    ip = {i : Layer_info() for i in [2, 3, 4]}

    # thickness of tissue layers. Thickness of second layer (index 3) should
    # be chosen such that the total thickness of the tissue slab is
    # somewhat thinner that the range of the coordinate system in z-direction
    ip[2].d = 100e-6
    ip[3].d = 200e-6


    ### absorption coefficients

    # read blood absorption. Data is in [m-1]! 
    # muabo holds data for oxygenated, muabd for deoxygenated blood
    muabo = np.genfromtxt('analytic_method\muabo.csv', delimiter=',')
    muabd = np.genfromtxt('analytic_method\muabd.csv', delimiter=',')

    oxy1 = 0.5   # oxygenation
    oxy2 = 0.5
    H = 0.55    # hematocrit, 39 weeks gestational age
    H0 = 0.45   # hematocrit of measured blood samples
    Be = 0.002  # blood fraction, epidermis
    # Bd = 0.01   # blood fraction, dermis
    Bd1 = 0.01   # blood fraction, dermis
    Bd2 = 0.01

    muam694 = 225   # melanin absorption coeff. at 694nm, in [m-1]
    # muam694 = 250   # melanin absorption coeff. at 694nm, in [m-1]
    # mua_other = 25  # non-blood, non-melanin absorption
    mua_other = 0  # non-blood, non-melanin absorption

    index = np.where(muabo[:,0]==lambda_)
    muab_blood1 = (muabo[index,1]*oxy1+muabd[index,1]*(1-oxy1))*H/H0
    muab_blood2 = (muabo[index,1]*oxy2+muabd[index,1]*(1-oxy2))*H/H0
    muab_melanin = muam694*np.power(694/lambda_, 3.46)
    ip[2].mua = (muab_melanin + muab_blood1*Be + mua_other*(1-Be))
    ip[3].mua = (muab_blood1*Bd1 + mua_other*(1-Bd1))
    ip[4].mua = (muab_blood2*Bd2 + mua_other*(1-Bd2))


    ### Scattering coefficient and av. cos. of scatt. angle
    muse, musd1, musd2, g = ScattCoeff(lambda_,H,Be,Bd1,Bd2)
    ip[2].mus = muse # ->[m^-1]
    ip[3].mus = musd1
    ip[4].mus = musd2
    ip[2].g = g
    ip[3].g = g
    ip[4].g = g

    return ip






###########################################################################
# Calculate scattering coefficients and av. cosine of the scattering angle 
# in epidermis and dermis. The value for the av. cosine of the scattering
# angle is based on tissue scattering, disregarding the scattering from
# erythrocytes.
# Some parameters depend on the gestational age chosen here: 39 weeks.
# All values are in [cm-1].
# Input parameters:  lambda_   wavelength
#                    H        hematocrit
#                    Be       blood fraction in epidermis
#                    Bd       blood fraction in dermis
# Output parameters: musre    red. scatt. coeff. epidermis
#                    musrd    red. scatt. coeff. dermis
def ScattCoeff(lambda_,H,Be,Bd1, Bd2):

    # av. cosine of scattering angle in tissue:
    g = 0.62 + lambda_*29e-5

    ## red. scatt. coeff. of Tissue:
    # Mie scattering:
    c_mie = 68     # depends on gestational age!
    musmr = c_mie*(1 - 1.745e-3*lambda_ + 9.843e-7*np.power(lambda_,2))
    # Rayleigh scattering:
    c_ray = 9.5e11
    musrr = c_ray*np.power(1.0*lambda_,-4)
    # total red. scattering coeff. in tissue:
    mustr = (musmr + musrr)*100
    # total scattering coeff. in tissue:
    must = mustr/(1-g)

    ## Erythrocytes:
    # scatt. coeff. at 685nm:
    musb685 = 55.09e-12
    # Haematocrit:
    H = 0.55
    # Erythrocyte volume:
    ve = 1.25e-16
    # scatt. coeff.:
    musb = musb685*H*(1-H)*(1.4-H)/ve*np.power((685/lambda_),0.37)

    ## total scattering coefficient
    muse = must*(1-Be)# + musb*Be
    musd1 = must*(1-Bd1)# + musb*Bd1
    musd2 = must*(1-Bd2)# + musb*Bd2

    return muse, musd1, musd2, g





