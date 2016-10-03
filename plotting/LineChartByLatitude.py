import matplotlib, os, numpy as np
from matplotlib import pyplot
from matplotlib.backends.backend_pdf import PdfPages
from osgeo import gdal, osr

blocksize = 120 # 120 = 1 degree
maxLat = 84.0 # northern boundary of the TIFFs
resolution = 0.00833333 # size of a cell in the TIFF, in degrees

os.chdir(os.path.expanduser('~') + '/Dropbox/CISC Data/IndividualCountries/Projections/Global');

def openTIFFasNParray(file):
    src = gdal.Open(file, gdal.GA_Update)
    band = src.GetRasterBand(1)
    return np.array(band.ReadAsArray())

def rowToLat(row):
    global maxLat, resolution
    return maxLat - (row * resolution)

ys = [2050, 2040, 2030, 2020, 2010, 2000]
years = {}
for y in ys:
    years[y] = openTIFFasNParray('pop-'+str(y)+'.tiff')


matplotlib.style.use('fivethirtyeight')

for y in  ys:
    year = years[y]

    # replace NAN with 0
    year[year < 0 ] = 0

    rows = year.shape[0]
    cols = year.shape[1]

    sumsPerBlock = []
    blocks = range(0, rows, blocksize)
    for row in blocks:
        sm = np.sum(year[row:row+blocksize,])
        sumsPerBlock.append(sm)

    # now calculate the latitutde for every row that we have a number for and use those on the y axis:
    latblocks = []
    for row in blocks:
        latblocks.append(rowToLat(row))

    pyplot.plot(sumsPerBlock, latblocks, label=str(y), linewidth = 1.0)


pyplot.legend(loc='upper right')
pyplot.show()
