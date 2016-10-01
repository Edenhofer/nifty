import pyximport; pyximport.install()
import extended

import numpy as np


print "///////////////////////////////////////First thing ////////////////////////"

n=8

ksq=np.sqrt(np.arange(n))
kk=np.arange(n)
power = np.ones(n**3).reshape((n,n,n))
# power[0][4][4]=1000
# power[1][4][4]=1000
# power[2][4][4]=1000
# power[3][4][4]=1000
power[n/2][n/2][n/2]=10000
# power[5][4][4]=1000
# power[6][4][4]=1000
# power[7][4][4]=1000
k = kk
sigma=k[1]-k[0]
mirrorsize=7
startindex=mirrorsize/2
endindex=n-mirrorsize/2
print power, k, power.shape
smooth = extended.smooth_something(datablock=power, axis=(2),
                                   startindex=startindex, endindex=endindex,
                                   kernelfunction=extended.GaussianKernel, k=k,
                                   sigma=sigma)

print "Smoooooth", smooth

doublesmooth = extended.smooth_something(datablock=smooth, axis=(1),
                                   startindex=startindex, endindex=endindex,
                                         kernelfunction=extended.GaussianKernel,
                                         k=k, sigma=sigma)

print "DoubleSmooth", doublesmooth

tripplesmooth = extended.smooth_something(datablock=doublesmooth, axis=(0),
                                   startindex=startindex, endindex=endindex,
                                         kernelfunction=extended.GaussianKernel,
                                         k=k, sigma=sigma)

print "TrippleSmooth", tripplesmooth

print "///////////////////////////////////////Final thing ////////////////////////"
print "smooth.len == power.len" , smooth.shape, power.shape, power.shape==smooth.shape