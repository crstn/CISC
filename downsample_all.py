import os

# This script will loop through our population files and
# downsample them to the same resolution as the SEDAC population
# projection grids; see https://github.com/crstn/CISC/wiki/Downsampling-our-projection-rasters-for-comparison-with-SEDAC-data for a detailed explanation.


datadir = os.path.expanduser('~') + '/Dropbox/CISCdata/OneRun/'
outputdir = os.path.expanduser('~') + '/Dropbox/CISCdata/OneRun/downsampled'

# creates a directoy if it doesn't exist yet
def makeSafe(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

makeSafe(outputdir)

# loop through all population and urbanization layers and downsample them
for model in ['GRUMP', 'GlobCover']:

    makeSafe(outputdir+'/'+model)

    for ssp in range(1,6): # !!!

        makeSafe(outputdir+'/'+model+'/SSP' + str(ssp))

        for year in range(2010,2101,10): # !!!

            # downsample:
            infile = datadir + model + '/SSP' + str(ssp) + '/popmean-' + str(year) +'.tiff'
            interfile = outputdir+'/'+model+'/SSP' + str(ssp) +'/inter-'+ str(year) +'.tiff'
            outfile = outputdir+'/'+model+'/SSP' + str(ssp) +'/pop-'+ str(year) +'.tiff'


            # we can't do SUM in gdalwarp, so we'll use the average and then multiply with the number of input cells in a separate step
            os.system("gdalwarp -te -180.0000000 -55.8750000 180.0000000 83.7500000 -ts 2880 1117 -wo NUM_THREADS=ALL_CPUS -co NUM_THREADS=ALL_CPUS -co COMPRESS=LZW -r average -srcnodata -2147483648 "+infile+" "+interfile)

            # multiply
            os.system('gdal_calc.py -A '+interfile+' --co NUM_THREADS=ALL_CPUS --co COMPRESS=LZW --outfile='+outfile+' --calc="A*225"')

            # delete intermedia file
            os.system('rm '+interfile)
