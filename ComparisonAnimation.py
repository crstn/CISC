import os

# CAREFUL! This script makes a GLOBAL gif, i.e. you won't be able to see much (and it takes a long time to generate). Better use ComparisonAnimationLocal.py to look at specific locations

# This script takes the output from CompareWithSEDAC.py and turns the generated series of GeoTIFFs into a colorized aninmated GIF using a bivariate color scale (centered around 0) from ColorBrewer: http://colorbrewer2.org/#type=diverging&scheme=PiYG&n=11

# The different steps are explained here: https://github.com/crstn/CISC/wiki/Turning-population-grids-into-colored-animated-GIFs

SSPs = ['SSP1'] #['SSP1', 'SSP2', 'SSP3', 'SSP4', 'SSP5']
models = ['GlobCover', 'GRUMP']

datadir = os.path.expanduser('~') + '/Dropbox/CISCdata/Comparison with SEDAC/'


# before we start, we'll make a color scale file for gdaldem:

for model in models:

# differences are an order of magnitude larger under GlobCover, so we'll use different value ranges:

    if model == 'GlobCover':

# TODO: instead of having fixed values here, this could be done dynamically by using the values from the 2100 data (which should in principle have the biggest difference between the two spatializations)

        os.system("""touch color.txt
        echo "nv 173 240 255
50000 142 1 82
40000 197 27 125
30000 222 119 174
20000 241 182 218
10000 253 224 239
0 247 247 247
-10000 230 245 208
-20000 184 225 134
-30000 127 188 65
-40000 77 146 33
-50000 39 100 25
        " >> color.txt""")
    else:
        os.system("""touch color.txt
        echo "nv 173 240 255
5000 142 1 82
4000 197 27 125
3000 222 119 174
2000 241 182 218
1000 253 224 239
0 247 247 247
-1000 230 245 208
-2000 184 225 134
-3000 127 188 65
-4000 77 146 33
-5000 39 100 25
        " >> color.txt""")

    for ssp in SSPs:

        # First, we'll colorize all individual images and "stamp" them with the year and a legend

        for year in range(2010,2101,10):

            print('Running', model, '-', ssp, '-', year)


            infile = '"' + datadir + model + '/' + ssp + '/diff-pop-' + str(year) + '.tiff"'

            colorfile = '"' + datadir + model + '/' + ssp + '/diff-pop-' + str(year) + '-color.tiff"'

            labelfile =  '"' + datadir + model + '/' + ssp + '/diff-pop-' + str(year) + '-label.tiff"'

            # colorize
            os.system('gdaldem color-relief '+infile+' color.txt '+colorfile)

            # label with year and delete the colorfile
            os.system('convert '+colorfile+' -font Helvetica-Neue -pointsize 2500 -fill white -gravity southwest -annotate +200+200 '+str(year)+' '+labelfile+'; rm '+colorfile)

        print("Done coloring, making a GIF")

        # All files have been colorized and labeled, let's make a GIF:
        folder = '"' + datadir + model + '/' + ssp +'"'
        os.system('cd '+folder+'; convert -delay 35 -loop 0 *label.tiff '+model+'-'+ssp+'-animation.gif')

        # clean up:
        os.system('cd '+folder+'; rm *label.tiff')



    # remove the color scale file again
    os.system("rm color.txt")

print('Done.')
