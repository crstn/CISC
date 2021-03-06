from PIL import Image
from osgeo import gdal
import os, numpy as np
import tif2num as tn
import PopFunctions as pop
import sys
import pync

base = os.path.expanduser('~') + '/Dropbox/CISCdata/'

# load countries TIFF and convert to NumPy array
f = base + 'NationsRaster/ne_10m_admin_0_countries_updated_nibbled.tiff'
countryBoundaries = pop.openTIFFasNParray(f)

# save the oringal 2D size in case we need it later
xyshape = countryBoundaries.shape
# then change to 1D
countryBoundaries = countryBoundaries.ravel()

# load urban rural TIFFs (GRUMP and GlobCover) and convert to NumPy arrays
f = base + 'GLURRaster/GRUMP_UrbanRural.tiff'
urbanRural_GRUMP = pop.openTIFFasNParray(f).ravel()

f = base + 'GLURRaster/GlobCover_UrbanRural.tiff'
urbanRural_GlobCover = pop.openTIFFasNParray(f).ravel()

# load population TIFFs and convert to NumPy array
f = base + 'Population2000Raster/Pop_2000_clipped.tiff'
population2000 = pop.openTIFFasNParray(f).ravel()

f = base + 'Population2010Raster/Pop_2010_clipped.tiff'
population2010 = pop.openTIFFasNParray(f).ravel()

# load SDEI cities TIFF and convert to NumPy array
f = base + 'SDEI-Global-UHI/sdei-global-uhi-2013.tiff'
cities = pop.openTIFFasNParray(f).ravel()

# load area grid
f = base + 'AreaGrid/area-grid.tif'
areas = pop.openTIFFasNParray(f).ravel()


# load the index arrays:
rows = np.load(base + 'IndexGrids/rows.npy').ravel()
cols = np.load(base + 'IndexGrids/cols.npy').ravel()


# just to check:
print urbanRural_GRUMP.shape
print urbanRural_GlobCover.shape
print countryBoundaries.shape
print population2000.shape
print population2010.shape
print cities.shape
print areas.shape
print rows.shape
print cols.shape


countries = np.unique(countryBoundaries)

# find GRUMP/GlobCover urban cells
urbanGRUMP = urbanRural_GRUMP == 2
urbanGlobCover = urbanRural_GlobCover == 2

# find cells in SDEI cities
incities = cities > 0

# find cells that have more than one person in 2010:
withpop = population2010 > 1

# just to keep track of how many contries we have completed
i = 1

# Iterate through countries:
for country in countries:
    if country > 0: # skip NaN cells

        # find cells belonging to the country
        incountry = countryBoundaries == country

        # bring it all together: find cells that are in the country
        # AND one or more of the following:
        # - have more than 1 person in it
        # - are GRUMP urban
        # - are GlobCover urban
        # - are in an SDEI city

        matches = np.where(np.all((incountry, np.any((withpop, urbanGRUMP, urbanGlobCover, incities), axis=0)), axis=0))

        # take out matching cells from the population layers & save as np array:
        np.save(base+'IndividualCountries/'+str(country)+'-pop2000.npy', population2000[matches])
        np.save(base+'IndividualCountries/'+str(country)+'-pop2010.npy', population2010[matches])

        # repeat for the urban/rural layers from GRUMP and GlobCover
        np.save(base + 'IndividualCountries/'+str(country)+'-UrbanRural-GRUMP.npy', urbanRural_GRUMP[matches])
        np.save(base + 'IndividualCountries/'+str(country)+'-UrbanRural-GlobCover.npy', urbanRural_GlobCover[matches])

        # and for the index layers (rows and columns)
        np.save(base + 'IndividualCountries/'+str(country)+'-rows.npy', rows[matches])
        np.save(base + 'IndividualCountries/'+str(country)+'-cols.npy', cols[matches])

        # and for the area layer
        np.save(base + 'IndividualCountries/'+str(country)+'-areas.npy', areas[matches])


    print "Processed " + str(i) + " out of " + str(len(countries)) + " countries"
    i = i + 1

pync.Notifier.notify('Done', title='Python')
