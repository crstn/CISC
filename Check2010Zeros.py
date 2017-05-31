import os
import csv
import numpy as np
import PopFunctions as pop
import pync

pync.Notifier.notify('Reading population raster', title='Python')
p = pop.openTIFFasNParray(os.path.expanduser('~') + '/Dropbox/CISC Data/Population 2010 Raster/Pop_2010_clipped.tiff')
pync.Notifier.notify('Reading urban/rural raster', title='Python')
u = pop.openTIFFasNParray(os.path.expanduser('~') + '/Dropbox/CISC Data/GLUR Raster/GRUMP_UrbanRural.tiff')

# -1 = NAN, set to 0:
p[p<0] = 0

# select cells with 1 person or less:
nopop = p <= 1
onland = u >= 1


print "{:,}".format(p.size)
print "{:,}".format(u.size)

print "# Cells without population"
print "{:,}".format(p[nopop].size)

print "# Cells ON LAND without population"

print "{:,}".format(p[np.all((nopop, onland), axis=0)].size)


noPop = np.zeros(p.shape, dtype = np.int)
noPop[np.all((nopop, onland), axis=0)] = 1

pop.array_to_raster(noPop, os.path.expanduser('~') + '/Desktop/noPop-1.tiff', os.path.expanduser('~') + '/Dropbox/CISC Data/Population 2010 Raster/Pop_2010_clipped.tiff')

pync.Notifier.notify('Done', title='Python')
