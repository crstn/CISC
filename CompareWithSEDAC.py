import os

# This script will loop through our population files compare the output for each year with the SEDAC population data at 1KM resolution. See https://github.com/crstn/CISC/wiki/Comparison-of-population-projections for a detailed explanation.

# Takes ca. 20 min for *one* SSP on my laptop.

SSPs = ['SSP4','SSP5'] #['SSP1', 'SSP2', 'SSP3', 'SSP4', 'SSP5']
models = ['GlobCover', 'GRUMP']

sedac_data = os.path.expanduser('~') + '/Downloads/'
our_data   = os.path.expanduser('~') + '/Dropbox/CISCdata/Projection Runs/Run 2017-07/'
outputdir = os.path.expanduser('~') + '/Dropbox/CISCdata/Comparison with SEDAC'

# creates a directoy if it doesn't exist yet
def makeSafe(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

makeSafe(outputdir)

for model in models:

    makeSafe(outputdir+'/'+model)

    for ssp in SSPs:

        makeSafe(outputdir+'/'+model+'/' + ssp)

        for year in range(2010,2101,10):

            print('Running', model, '-', ssp, '-', year)

            # clip:
            infile  = '"' + our_data + model + '/' + ssp + '/popmean-' + str(year) +'.tiff"'

            clipped = '"' + our_data + model + '/' + ssp + '/popmean-' + str(year) +'_clipped.tiff"'

            os.system('gdalwarp -te -180.0000000 -55.7750000 180.0000000 83.6416667 -wo NUM_THREADS=ALL_CPUS -co NUM_THREADS=ALL_CPUS -co COMPRESS=LZW -srcnodata -2147483648 '+infile+' '+clipped)

            # calculate difference
            sedacfile = '"' + sedac_data+ssp+'_1km/'+ssp.lower()+'_total_'+str(year)+'.tif"'

            outfile = '"'+outputdir+'/'+model+'/' + ssp +'/diff-pop-'+ str(year) +'.tiff"'

            # IMPORTANT: Numbers > 0 mean WE have more people in a cell
            #            Numbers < 0 mean SEDAC has more people in a cell
            os.system('gdal_calc.py -A '+clipped+' -B '+sedacfile+' --calc="A-B" --outfile='+outfile+' --type="Int32" --co NUM_THREADS=ALL_CPUS --co COMPRESS=LZW')

            # delete intermedia clipped file
            os.system('rm '+clipped)
