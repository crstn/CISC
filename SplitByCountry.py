import os, numpy as np, tif2num as tn
from PIL import Image
from osgeo import gdal

# load countries TIFF and convert to NumPy array
f = os.path.expanduser('~') + '/Dropbox/CISC Data/Nations Raster/ne_10m_admin_0_countries_updated_nibbled.tiff'
src = gdal.Open(f, gdal.GA_Update)
band = src.GetRasterBand(1)
countryBoundaries = np.array(band.ReadAsArray())

# load urban rural TIFFs (GRUMP and GlobCover) and convert to NumPy arrays
f = os.path.expanduser('~') + '/Dropbox/CISC Data/GLUR Raster/GRUMP_UrbanRural.tiff'
src = gdal.Open(f, gdal.GA_Update)
band = src.GetRasterBand(1)
urbanRural_GRUMP = np.array(band.ReadAsArray())

f = os.path.expanduser('~') + '/Dropbox/CISC Data/GLUR Raster/GlobCover_UrbanRural.tiff'
src = gdal.Open(f, gdal.GA_Update)
band = src.GetRasterBand(1)
urbanRural_GlobCover = np.array(band.ReadAsArray())


# # load population TIFFs and convert to NumPy array
f = os.path.expanduser('~') + '/Dropbox/CISC Data/Population 2000 Raster/Pop_2000_clipped.tiff'
src = gdal.Open(f, gdal.GA_Update)
band = src.GetRasterBand(1)
population2000 = np.array(band.ReadAsArray())

f = os.path.expanduser('~') + '/Dropbox/CISC Data/Population 2010 Raster/Pop_2010_clipped.tiff'
src = gdal.Open(f, gdal.GA_Update)
band = src.GetRasterBand(1)
population2010 = np.array(band.ReadAsArray())

# just to check:
print urbanRural_GRUMP.shape
print urbanRural_GlobCover.shape
print countryBoundaries.shape
print population2000.shape
print population2010.shape

countries = np.unique(countryBoundaries)
i = 1

# Iterate through countries:
for country in countries:
    if country > 0: # skip NaN cells
        # find min and max X and Y indices for this country
        x, y = np.where(countryBoundaries==country)
        minx = np.min(x)
        maxx = np.max(x)+1 # include the last column where this country appears!
        miny = np.min(y)
        maxy = np.max(y)+1 # include the last row where this country appears!

        # cut out this rectangular block from the country Boundaries
        justCountry = countryBoundaries[minx:maxx, miny:maxy]
        np.save(os.path.expanduser('~') + '/Dropbox/CISC Data/IndividualCountries/'+str(country)+'-boundary.npy', justCountry)

        # use this to test:
        # img = Image.fromarray(justCountry)
        # img.save(str(country)+'out.tiff')

        # cut out this rectangular block from the urban/rural data
        justCountry = urbanRural_GRUMP[minx:maxx, miny:maxy]
        np.save(os.path.expanduser('~') + '/Dropbox/CISC Data/IndividualCountries/'+str(country)+'-UrbanRural-GRUMP.npy', justCountry)

        justCountry = urbanRural_GlobCover[minx:maxx, miny:maxy]
        np.save(os.path.expanduser('~') + '/Dropbox/CISC Data/IndividualCountries/'+str(country)+'-UrbanRural-GlobCover.npy', justCountry)

        # cut out this rectangular block from the 2000 and 2010 pop data
        justCountry = population2000[minx:maxx, miny:maxy]
        np.save(os.path.expanduser('~') + '/Dropbox/CISC Data/IndividualCountries/'+str(country)+'-pop2000.npy', justCountry)

        justCountry = population2010[minx:maxx, miny:maxy]
        np.save(os.path.expanduser('~') + '/Dropbox/CISC Data/IndividualCountries/'+str(country)+'-pop2010.npy', justCountry)

    print "Processed " + str(i) + " out of " + str(len(countries)) + " countries"
    i = i + 1
