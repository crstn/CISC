from PIL import Image
from osgeo import gdal
import os, numpy as np
import tif2num as tn
import PopFunctions as pop
import sys
import pync

print "loading reference tiff"
# load reference tiff
reffile = gdal.Open(os.path.expanduser('~') + '/Dropbox/CISCdata/NationsRaster/ne_10m_admin_0_countries_updated_nibbled.tiff')
geotransform = reffile.GetGeoTransform()
rasterXSize = reffile.RasterXSize
rasterYSize = reffile.RasterYSize
projection = reffile.GetProjection()


print "loading countries"
# load countries TIFF and convert to NumPy array
countries = pop.openTIFFasNParray(os.path.expanduser('~') + '/Dropbox/CISCdata/NationsRaster/ne_10m_admin_0_countries_updated_nibbled.tiff')

models = ['GRUMP', 'GlobCover']

for m in models:
    for i in range(1,6):
        for y in range(2010,2101,10):

            print "Running SSP"+str(i)+" / " +m+ " / " +str(y)

            # print "loading urban/rural"
            # load urban rural TIFF and convert to NumPy array
            ur = pop.openTIFFasNParray(os.path.expanduser('~') + '/Dropbox/CISCdata/OneRun/'+m+'/SSP'+str(i)+'/urbanization-'+str(y)+'.tiff')

            # print "replace urban cells with country ID"
            ur[ur == 2] = countries[ur == 2]

            # print "saving tiff"
            pop.array_to_raster_noref(ur, os.path.expanduser('~') + '/Dropbox/CISCdata/OneRun/'+m+'/SSP'+str(i)+'/urbanization-'+str(y)+'-countryID.tiff', geotransform, rasterXSize, rasterYSize, projection)

pync.Notifier.notify('Done', title='Python')
