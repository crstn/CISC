from PIL import Image
from osgeo import gdal
import numpy as np
import os
import xml.etree.ElementTree as ET

os.chdir(os.path.expanduser('~') + '/Dropbox/CISC - Global Population/Asia')

files = ["GLUR/GLUR_Asia.tif", "Nations/Nations_Asia.tif", "Population 2000/Population_2000_Asia.tif", "Population 2010/Population_2010_Asia.tif"]

# replaces feature IDs with UN country codes from XML attribute table
def replaceCountryCodes(countries, xmlfile):

    tree = ET.parse(xmlfile)
    countriesReplaced = np.copy(countries)
    for g in tree.iter('GDALRasterAttributeTable'):
        for r in g.iter('Row'):
            row = list(r.iter())
            # replace all cells that contain feature ID with country code in output:
            # logging.info("Replacing "+row[2].text+" with "+row[8].text)
            countriesReplaced[countries == int(row[2].text) ] = int(row[8].text)

    return countriesReplaced

for f in files:
    src = gdal.Open(f, gdal.GA_Update)
    band = src.GetRasterBand(1)
    imarray = np.array(band.ReadAsArray())

    # replace values in Nations raster (feature IDs) with country codes:
    if f == "Nations/Nations_Asia.tif":
        print "Replacing feature IDs with country codes"
        attTable = os.path.expanduser('~') + '/Dropbox/CISC - Global Population/Asia/CountriesAttributes.xml'
        imarray = replaceCountryCodes(imarray, attTable)
    # Make sure the numpy array all have the same dimensions:
    print f
    print imarray.shape

    np.save(f, imarray)
