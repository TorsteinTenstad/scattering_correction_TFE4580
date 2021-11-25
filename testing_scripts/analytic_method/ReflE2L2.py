import numpy as np

def ReflE2L2_analytic(ip):

    #epidermis
    g1 = ip[2].g
    mua1 = ip[2].mua
    mus1 = ip[2].mus
    d1 = ip[2].d

    #dermis1
    g2 = ip[3].g
    mua2 = ip[3].mua
    mus2 = ip[3].mus
    d2 = ip[3].d

    #dermis2
    g3 = ip[4].g
    mua3 = ip[4].mua
    mus3 = ip[4].mus

    #boundary cond coefficient
    A = 0.14386

    #reduced transport coefficient
    mutrr1 = mus1*(1-np.power(g1,2))+mua1
    mutrr2 = mus2*(1-np.power(g2,2))+mua2
    mutrr3 = mus3*(1-np.power(g3,2))+mua3

    mutr1 = mus1*(1-g1)+mua1
    mutr2 = mus2*(1-g2)+mua2
    mutr3 = mus3*(1-g3)+mua3

    #diffuse transport coefficients
    D1 = 1/(3*(mus1*(1-g1)+mua1))
    D2 = 1/(3*(mus2*(1-g2)+mua2))
    D3 = 1/(3*(mus3*(1-g3)+mua3))

    #optical penetration depths
    del1 = np.sqrt(D1/mua1) 
    del2 = np.sqrt(D2/mua2) 
    del3 = np.sqrt(D3/mua3) 

    #factors in the source functions
    #spottløsning
    K1 = mus1*(1-np.power(g1,2))*(1/D1+3*mutrr1*g1/(1+g1))*np.power(del1,2)/(1-np.power(mutrr1,2)*np.power(del1,2))
    K2 = mus2*(1-np.power(g2,2))*(1/D2+3*mutrr2*g2/(1+g2))*np.power(del2,2)/(1-np.power(mutrr2,2)*np.power(del2,2))*np.exp(-mutrr1*d1)
    K3 = mus3*(1-np.power(g3,2))*(1/D3+3*mutrr3*g3/(1+g3))*np.power(del3,2)/(1-np.power(mutrr3,2)*np.power(del3,2))*np.exp(-mutrr1*d1)*np.exp(-mutrr2*(d2-d1))

    #diffuse reflectance
    #tolags
    #gamma = -A * (((D1 * K1 * mutrr1 * mutr1 - mus1 * (-1 + g1) * g1) * del1 + mutr1 * D1 * K1) * mutr2 * (D1 * del2 - D2 * del1) * np.exp(-(d1 / del1)) + mutr2 * ((D1 * K1 * mutrr1 * mutr1 - mus1 * (-1 + g1) * g1) * del1 - mutr1 * D1 * K1) * (D1 * del2 + D2 * del1) * np.exp((d1 / del1)) - 0.2e1 * del1 * D1 * (mutr2 * (D1 * mutr1 * del2 * K1 * mutrr1 - g1 * mus1 * (-1 + g1) * del2 - D2 * mutr1 * K1) * np.exp(-(mutrr1 * d1)) - mutr1 * (-g2 * mus2 * del2 * (-1 + g2) * np.exp(-(mutrr2 * d1)) + K2 * D2 * mutr2 * (-1 + mutrr2 * del2)))) / mutr1 / mutr2 / ((D1 * del2 - D2 * del1) * (-D1 + A * del1) * np.exp(-(d1 / del1)) + (D1 * del2 + D2 * del1) * np.exp((d1 / del1)) * (D1 + A * del1))


    #trelags
    gamma = -((np.exp(d2 / del2) * (D2 * del3 + D3 * del2) * (-D2 * del1 + D1 * del2) * ((D1 * mutr1 * K1 * mutrr1 - (mus1 * (-1 + g1) * g1)) * del1 + D1 * mutr1 * K1) * mutr2 * mutr3 * np.exp(-d1 / del1) - 0.4e1 * D1 * D2 * np.exp(d1 / del2) * del2 * mutr1 * del1 * mutr2 * mutr3 * K2 * (D2 * del3 * mutrr2 - D3) * np.exp(mutrr2 * (-d2 + d1)) + 0.4e1 * D2 * (g2 * del3 * mus2 * mutr3 * (-1 + g2) * np.exp(-mutrr2 * d2) + (-g3 * del3 * mus3 * (-1 + g3) * np.exp(-mutrr3 * d2) + D3 * K3 * mutr3 * (del3 * mutrr3 - 0.1e1)) * mutr2) * D1 * del1 * del2 * mutr1 * np.exp(d1 / del2) + np.exp(d2 / del2) * (D2 * del3 + D3 * del2) * ((D1 * del2 + D2 * del1) * ((D1 * mutr1 * K1 * mutrr1 - (mus1 * (-1 + g1) * g1)) * del1 - D1 * mutr1 * K1) * mutr2 * np.exp(d1 / del1) - 0.2e1 * ((D1 * mutr1 * del2 * K1 * mutrr1 - g1 * mus1 * (-1 + g1) * del2 - K1 * mutr1 * D2) * mutr2 * np.exp(-mutrr1 * d1) - (-g2 * mus2 * del2 * (-1 + g2) * np.exp(-mutrr2 * d1) + D2 * K2 * mutr2 * (mutrr2 * del2 - 0.1e1)) * mutr1) * D1 * del1) * mutr3) * np.exp(-d1 / del2) + np.exp(-d2 / del2) * ((D1 * del2 + D2 * del1) * ((D1 * mutr1 * K1 * mutrr1 - (mus1 * (-1 + g1) * g1)) * del1 + D1 * mutr1 * K1) * mutr2 * np.exp(-d1 / del1) + ((D1 * mutr1 * K1 * mutrr1 - (mus1 * (-1 + g1) * g1)) * del1 - D1 * mutr1 * K1) * (-D2 * del1 + D1 * del2) * mutr2 * np.exp(d1 / del1) - 0.2e1 * D1 * del1 * ((D1 * mutr1 * del2 * K1 * mutrr1 - g1 * mus1 * (-1 + g1) * del2 + K1 * mutr1 * D2) * mutr2 * np.exp(-mutrr1 * d1) - (-g2 * mus2 * del2 * (-1 + g2) * np.exp(-mutrr2 * d1) + D2 * K2 * mutr2 * (mutrr2 * del2 + 0.1e1)) * mutr1)) * np.exp(d1 / del2) * mutr3 * (D2 * del3 - D3 * del2)) * A / (((-D2 * del1 + D1 * del2) * (-D1 + A * del1) * np.exp(-d1 / del1) + (D1 * del2 + D2 * del1) * np.exp(d1 / del1) * (A * del1 + D1)) * np.exp(d2 / del2) * (D2 * del3 + D3 * del2) * np.exp(-d1 / del2) + np.exp(-d2 / del2) * ((D1 * del2 + D2 * del1) * (-D1 + A * del1) * np.exp(-d1 / del1) + (-D2 * del1 + D1 * del2) * np.exp(d1 / del1) * (A * del1 + D1)) * np.exp(d1 / del2) * (D2 * del3 - D3 * del2)) / mutr2 / mutr1 / mutr3

    return gamma
