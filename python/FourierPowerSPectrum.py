import numpy as np
# we provide here a python code to estimate the Fourier power spectrum of a time series that has missing data or irregular time step. This code is used for the study 
#  "Scaling Analysis of the China France Oceanography SATellite Along-Track Wind and Wave Data" published in Journal of Geophysical Research: Oceans by Gao et al., 2021.

def est_correlation_nan(x):
  
