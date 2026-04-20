import numpy as np
# we provide here a python code to estimate the Fourier power spectrum of a time series that has missing data or irregular time step. This code is used for the study 
#  "Scaling Analysis of the China France Oceanography SATellite Along-Track Wind and Wave Data" published in Journal of Geophysical Research: Oceans by Gao et al., 2021.

import numpy as np
import pandas as pd

import os



def power_law_fit(ff, fsp, fstart, fend): # power-law fit
    """
    power-law fit
    """
    if type(ff).__module__ == 'xarray.core.dataarray':
        ff=ff.to_numpy()
    if type(fsp).__module__ == 'xarray.core.dataarray':
        fsp=fsp.to_numpy()     
    if type(ff).__module__=='pandas.core.series'        :
        ff=ff.to_numpy()
    if type(fsp).__module__ == 'pandas.core.series':
        fsp=fsp.to_numpy()     
        
    Ind = np.where((ff >= fstart) & (ff <= fend))[0]
    bb = ff[Ind].copy()
    dd = fsp[Ind].copy()
    bb = bb[dd > 0]
    dd = dd[dd > 0]
    z, v = np.polyfit(np.log10(bb), np.log10(dd), 1, cov=True)
    zz = np.diagonal(v)**0.5
    return z, zz

# We estimate the Fourier power spectrum by leveraging the Wiener–Khinchin theorem (WKT). 
# First, the WKT is utilized to compute the biased autocorrelation function. 
# We then implement a correction for the bias introduced by missing data effects.
#  Finally, the corrected Fourier spectrum is indirectly derived through the application of the WKT.



def est_correlation_nan(x, fs=1, norm=False):  
    # check input data
    if (type(x).__module__
            == 'pandas.core.series') or (type(x).__module__
                                         == 'xarray.core.dataarray'):
        x_new = x.to_numpy().copy()  #if the data is from pandas or xarray, we convert it to the numpy array.
    else:
        x_new = x.copy()
    x_new = np.float64(x_new) # 
    x_new = x_new - np.nanmean(x_new) # remove the mean value
    Ind = np.where(np.isnan(x_new) == 0)[0]  #find the missing data
    x_new = x_new[Ind[0]:Ind[-1]]  # if the end missing data
    Flag = 0 # the flag for the missing data (NaN)
    Ind = np.where(np.isnan(x_new) == 1)[0]  #search for the NaN
    if len(Ind) > 0:# yes we have missing data
        Flag = 1
        y = np.ones_like(x_new)  #prepare a counter 
        y[Ind] = 0  # replace NaN by 0
        x_new[Ind] = 0  
    Nx = np.shape(x_new)[0]  #get the length of the data
    L = np.int64(2**np.ceil(np.log2(2 * Nx - 1)))  #get the Next 2 power for FFT and WKT
    xf = np.fft.rfft(x_new, n=L)  #FFT of the data with zero-padding to satisfy the requirement of the WKT 
    c1 = np.fft.irfft(xf * np.conj(xf)) # estimate the biased autocorrelation function via the WKT. 
    
    c1 = np.real(c1)  #keep only the real part.
    #     return x_new
    if Flag == 1:  # yes, we have missing data, we should do the same for the counter
        xf = np.fft.rfft(y, n=L)  
        c2 = np.fft.irfft(xf * np.conj(xf))
        c2 = np.real(c2 + 0.1).astype(np.int64) # get the sample size of each separation scale
        c2[c2 == 0] = 1  #replace 0 by 1 to aviod dividing by zero
        c3 = np.zeros_like(c1)
        c3[0:Nx] = np.arange(Nx, 0, -1)
        c3[-1:-Nx:-1] = c3[1:Nx]  #sample size without missing data
        c1 = c1 / c2 * c3  # correct the missing data effect
    Tau = np.arange(-L // 2, L // 2) / fs # separation scales
    c1 = np.fft.fftshift(c1)
    if norm == True: # do the normalizataion to force c1[0]=1
        c1 = c1 / np.max(c1)
    return c1, Tau



def est_fsp(c, fs=1, En=1): 
    # estimate the Fourier power spectrum via the WKT

    if (type(En).__module__
            == 'pandas.core.series') or (type(En).__module__
                                         == 'xarray.core.dataarray'):
        En = En.to_numpy()  
    
    if not np.argmax(c) == 0: # if the c[0]!=max(c)
        c = np.fft.fftshift(c)
    xf = np.fft.fft(c)
    xf = np.abs(np.real(xf)) # we take the absoluate value in case we have negative value due to the above correction. 
    Nx = np.size(xf)
    M = Nx // 2 + 1  #due to the symmetry of the Fourier power spectrum, keep only half of it
    f = np.arange(2, M+1 ) 
    f = f / M / 2
    xf = xf[1:M]
    f1 = f * fs  #real frequency
    xf = xf / np.sum(xf) / [f1[1] - f1[0]] * En  #normalize the Fourier power spectrum via the Parseval theorem.
    binq = 10**np.arange(np.log10(f[0]), np.log10(f[-1]),
                         0.05)  #bin average in logarithmic scale for power-law test
    Nbin = np.size(binq)
    fsp = np.zeros_like(binq) * np.nan
    ff = np.zeros_like(binq) * np.nan
    for i in range(Nbin - 1):
        Ind = np.where((f >= binq[i]) & (f <= binq[i + 1]))
        Ind = Ind[0]
        if np.size(Ind) == 0:
            fsp[i] = np.nan
            ff[i] = np.nan
        else:
            fsp[i] = np.mean(xf[Ind])
            ff[i] = np.mean(f1[Ind])
    return np.abs(xf), f1, np.abs(fsp), ff

  
