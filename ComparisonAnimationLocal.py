import os
from geopy.geocoders import Nominatim
import numpy as np
import gdal

# This script takes the output from CompareWithSEDAC.py and turns the generated series of GeoTIFFs into a colorized aninmated GIF using a bivariate color scale (centered around 0) from ColorBrewer: http://colorbrewer2.org/#type=diverging&scheme=PiYG&n=11

# The different steps are explained here: https://github.com/crstn/CISC/wiki/Turning-population-grids-into-colored-animated-GIFs

# ---------------------------
# CONFIGURATION:
# ---------------------------
# Put in a city name below, for which we'll look up the location
# using an online geocoder
# Pick the SSP(s) and urbanization model(s) you want the GIF for
# ---------------------------

places = ['Manila', 'Tokyo', 'Bejing', 'Cairo', 'Paris', 'Qatar']
size = 0.75 # in degrees, i.e. the GIF will cover an area of size x size center on the place chosen above
SSPs = ['SSP1', 'SSP2', 'SSP3', 'SSP4', 'SSP5']
models = ['GlobCover', 'GRUMP']

datadir = os.path.expanduser('~') + '/Dropbox/CISCdata/Comparison with SEDAC/'

for place in places:

    # find the place:
    geolocator = Nominatim(user_agent="CISC App")
    location = geolocator.geocode(place)
    print(location.address.encode('utf-8'))

    # extent to clip to
    size = size/2.0
    extent = str(location.longitude-size) + ' ' + str(location.latitude-size) + ' ' + str(location.longitude+size) + ' ' + str(location.latitude+size)

    # creates a directoy if it doesn't exist yet
    def makeSafe(dir):
        if not os.path.exists(dir):
            os.makedirs(dir)

    # does what it says...
    def openTIFFasNParray(file):
        src = gdal.Open(file, gdal.GA_Update)
        band = src.GetRasterBand(1)
        a = np.array(band.ReadAsArray())
        # replace nan cells with 0 - fine for the purpose of this script
        notanumber = -2147483648
        a[a==notanumber]=0
        return a

    # First, we'll clip all years to the extent around our place

    for model in models:
        for ssp in SSPs:
            for year in range(2010,2101,10):

                infile = '"' + datadir + model + '/' + ssp + '/diff-pop-' + str(year) + '.tiff"'

                clipped = '"' + datadir + model + '/' + ssp + '/diff-pop-' + str(year) + '-clipped.tiff"'

                os.system('gdalwarp -te '+ extent +' -wo NUM_THREADS=ALL_CPUS -co NUM_THREADS=ALL_CPUS -co COMPRESS=LZW -srcnodata -2147483648 '+infile+' '+clipped)

    # Now we'll go through again to figure out what the maximum value in any
    # of the rasters is, so that we can apply the the same color scale to all
    maxDiff = 0

    for model in models:
        for ssp in SSPs:
            # First, we'll clip all years to the extent around our place
            for year in range(2010,2101,10):

                clipped = datadir + model + '/' + ssp + '/diff-pop-' + str(year) + '-clipped.tiff'

                a = openTIFFasNParray(clipped)

                if(np.max(np.absolute(a)) > maxDiff):
                    maxDiff = np.max(np.absolute(a))


    # Now that we know what the maximum difference across all models, SSPs and
    # years is, we can make a divergent color centered around 0
    os.system("""touch color.txt; echo "nv 173 240 255
    """+str(maxDiff)+""" 142 1 82
    """+str(maxDiff*0.8)+""" 197 27 125
    """+str(maxDiff*0.6)+""" 222 119 174
    """+str(maxDiff*0.4)+""" 241 182 218
    """+str(maxDiff*0.2)+""" 253 224 239
    0 247 247 247
    -"""+str(maxDiff*0.2)+""" 230 245 208
    -"""+str(maxDiff*0.4)+""" 184 225 134
    -"""+str(maxDiff*0.6)+""" 127 188 65
    -"""+str(maxDiff*0.8)+""" 77 146 33
    -"""+str(maxDiff)+""" 39 100 25
                    " >> color.txt""")




    # Go through again and colorize them using the color scale above
    for model in models:
        for ssp in SSPs:
            for year in range(2010,2101,10):

                infile = '"' + datadir + model + '/' + ssp + '/diff-pop-' + str(year) + '-clipped.tiff"'

                colorfile = '"' + datadir + model + '/' + ssp + '/diff-pop-' + str(year) + '-color.tiff"'

                labelfile =  '"' + datadir + model + '/' + ssp + '/diff-pop-' + str(year) + '-label.tiff"'

                # colorize
                os.system('gdaldem color-relief '+infile+' color.txt '+colorfile)

                # label with year and delete the colorfile
                os.system('convert '+colorfile+' -font Helvetica-Neue -pointsize 10 -fill black -gravity southwest -annotate +10+10 '+str(year)+' '+labelfile+'; rm '+colorfile)

            print("Done coloring, making a GIF")

            # All files have been colorized and labeled, let's make a GIF:
            makeSafe(datadir+'gifs') # put the GIFs in a subfolder

            folder = '"' + datadir + model + '/' + ssp +'"'
            os.system('cd '+folder+'; convert -delay 40 -loop 0 *label.tiff "'+datadir+'gifs/'+model+'-'+ssp+'-'+place+'.gif"')

            # clean up:
            os.system('cd '+folder+'; rm *label.tiff; rm *clipped.tiff')



    # remove the color scale file again
    os.system("rm color.txt")

    # make a website that shows them side by side
    cmd = """touch '"""+datadir+place+""".html'
    echo "<html>

    <head>
      <style>
        body       { font-family: Helvetica, sans-serif }
        img        { width: 250px; height: 250px }
        p.vertical { writing-mode: vertical-rl }
        td         { border: 0; margin: 0}
        td.l       { color: white; text-align: center; width:30px; height: 30px; padding: 0}
        td#c1      { background: #8e0152 }
        td#c2      { background: #c51b7d }
        td#c3      { background: #de77ae }
        td#c4      { background: #f1b6da }
        td#c5      { background: #fde0ef }
        td#c6      { background: #f7f7f7 }
        td#c7      { background: #e6f5d0 }
        td#c8      { background: #b8e186 }
        td#c9      { background: #7fbc41 }
        td#c10     { background: #4d9221 }
        td#c11     { background: #276419 }

      </style>
    </head>

    <body>
      <table>
        <tr>
          <td></td>
          <td>
            <h1>"""+place+"""</h1>
            </td>
            </tr>
            <tr>
            <td></td>
            <td>SSP1</td>
            <td>SSP2</td>
            <td>SSP3</td>
            <td>SSP4</td>
            <td>SSP5</td>
            </tr>
            <tr>
            <td>
            <p class="vertical">GlobCover
            </td>
            <td><img src="gifs/GlobCover-SSP1-"""+place+""".gif" /></td>
            <td><img src="gifs/GlobCover-SSP2-"""+place+""".gif" /></td>
            <td><img src="gifs/GlobCover-SSP3-"""+place+""".gif" /></td>
            <td><img src="gifs/GlobCover-SSP4-"""+place+""".gif" /></td>
            <td><img src="gifs/GlobCover-SSP5-"""+place+""".gif" /></td>
            </tr>
            <tr>
            <td>
            <p class="vertical">GRUMP
            </td>
            <td><img src="gifs/GRUMP-SSP1-"""+place+""".gif" /></td>
            <td><img src="gifs/GRUMP-SSP2-"""+place+""".gif" /></td>
            <td><img src="gifs/GRUMP-SSP3-"""+place+""".gif" /></td>
            <td><img src="gifs/GRUMP-SSP4-"""+place+""".gif" /></td>
            <td><img src="gifs/GRUMP-SSP5-"""+place+""".gif" /></td>
            </tr>
            <tr>
            <td></td>
            <td colspan="5">
            <table>
              <tr>
                <td colspan="5" style='text-align: right'>&larr; More in CISC</td>
                <td></td>
                <td colspan="5">
                  <p style='text-align: left'>More in SEDAC &rarr; <p>
                </td>
              </tr>
              <tr>
                <td class="l" id="c1"></td>
                <td class="l" id="c2"></td>
                <td class="l" id="c3"></td>
                <td class="l" id="c4"></td>
                <td class="l" id="c5"></td>
                <td class="l" id="c6"></td>
                <td class="l" id="c7"></td>
                <td class="l" id="c8"></td>
                <td class="l" id="c9"></td>
                <td class="l" id="c10"></td>
                <td class="l" id="c11"></td>
              </tr>
              <tr>
                <td colspan="5" style='text-align: left'>"""+str(maxDiff)+"""</td>
                <td></td>
                <td colspan="5" style='text-align: right'>"""+str(maxDiff)+"""</td>
              </tr>
            </table>
            </td>
            </tr>
            </table>
            </body>

            </html>
    " >> '"""+datadir+place+""".html'; open '"""+datadir+place+""".html'"""


print('Done.')
