import numpy as np

# just to make sure they actually all have the same size now:

glurextents = np.load("/Users/carsten/Downloads/Numpy/glurextents-int.npy")

print glurextents.shape
print np.min(glurextents)
print np.max(glurextents)

gluntlbnds = np.load("/Users/carsten/Downloads/Numpy/gluntlbnds-clipped-int.npy")

print gluntlbnds.shape
print np.min(gluntlbnds)
print np.max(gluntlbnds)

glup00ag = np.load("/Users/carsten/Downloads/Numpy/glup00ag-clipped.npy")

print glup00ag.shape
print np.min(glup00ag)
print np.max(glup00ag)
