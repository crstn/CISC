# coding: utf-8
#!/usr/bin/env python

# calcuates the global total from the individual countries .npy files

import os, sys
import time
import numpy as np
import tif2num as tn
import PopFunctions as pop
from PIL import Image
from osgeo import gdal

# if this script is called with arguments, use them as the countries.
if len(sys.argv) != 5:

    print """
This script checks the total and urban population in a given scenario and year.
Call this script with 4 arguments:
1. The ID of the country
2. The SSP (SSP1, ..., SSP5)
3. The urban/rural scenario (GRUMP or GlobCover)
4. The year

Example (Sudan): python CheckOneCountry.py 508 SSP1 GRUMP 2100

"""

    sys.exit();


country = sys.argv[1]
ssp     = sys.argv[2]
urb      = sys.argv[3]
year    = sys.argv[4]

os.chdir(os.path.expanduser('~') + '/Dropbox/CISC Data')

print "Loading countries grid"
countries = pop.openTIFFasNParray('Nations Raster/ne_10m_admin_0_countries_updated_nibbled.tiff')
print countries.size

print "Loading population grid"
population = pop.openTIFFasNParray('OneRun/'+urb+'/'+ssp+'/popmean-'+year+'.tiff-no-nan.tiff')
print population.size

print "Loading urban/rural grid"
ur = pop.openTIFFasNParray('OneRun/'+urb+'/'+ssp+'/urbanization-'+year+'.tiff')
print ur.size


incountry = countries == int(country)
urban     = ur        == 2
both      = np.all((incountry, urban), axis=0)

print "Total pop: " + str(np.nansum(population[incountry]))
print "Urban pop: " + str(np.nansum(population[both]))
