import os
import csv
import numpy as np
import PopFunctions as pop
import pync

pync.Notifier.notify('Reading population raster', title='Python')
p = pop.openTIFFasNParray(os.path.expanduser('~') + '/Dropbox/CISC Data/Population 2010 Raster/Pop_2010_clipped.tiff')
pync.Notifier.notify('Reading urban/rural raster', title='Python')
u = pop.openTIFFasNParray(os.path.expanduser('~') + '/Dropbox/CISC Data/GLUR Raster/GRUMP_UrbanRural.tiff')
pync.Notifier.notify('Reading cities raster', title='Python')
c = pop.openTIFFasNParray(os.path.expanduser('~') + '/Dropbox/CISC Data/SDEI-Global-UHI/sdei-global-uhi-2013.tiff')

# -1 = NAN, set to 0:
p[p<0] = 0

# select cells with 1 person or less:
nopop = p <= 1
# select cells that are on land
onland = u >= 1
# select urban cells
urban = u == 2
# select non-urban cells
nonurban = u <= 1
# select cells that are in an SDEI city
insidecities = c > 0

# sanity check:
print "{:,}".format(p.size)
print "{:,}".format(u.size)
print "{:,}".format(c.size)

print

print "# Cells without population (1 person or less per cell)"
print "{:,}".format(p[nopop].size)

print "# Cells ON LAND without population"
print "{:,}".format(p[np.all((nopop, onland), axis=0)].size)

noPop = np.zeros(p.shape, dtype = np.int)
noPop[np.all((nopop, onland), axis=0)] = 1

pop.array_to_raster(noPop, os.path.expanduser('~') + '/Desktop/noPop-1.tiff', os.path.expanduser('~') + '/Dropbox/CISC Data/Population 2010 Raster/Pop_2010_clipped.tiff')

print "# Cells ON LAND that are WITHIN an SDEI city, but without population"
print "{:,}".format(p[np.all((nopop, onland, insidecities), axis=0)].size)

print "# Cells in GRUMP urban extents, but without population"
print "{:,}".format(p[np.all((nopop, urban), axis=0)].size)

print "# that are WITHIN an SDEI city, but not within an GRUMP urban area"
print "{:,}".format(p[np.all((insidecities, nonurban), axis=0)].size)

noPopInCities = np.zeros(p.shape, dtype = np.int)
noPopInCities[np.all((nopop, onland, insidecities), axis=0)] = 1

pop.array_to_raster(noPopInCities, os.path.expanduser('~') + '/Desktop/noPopInCities.tiff', os.path.expanduser('~') + '/Dropbox/CISC Data/Population 2010 Raster/Pop_2010_clipped.tiff')

pync.Notifier.notify('Done', title='Python')
