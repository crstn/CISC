import os, sys, numpy as np
import moviepy.editor as mpy

def arrays2GIF(arrays, gif):
    mx = 0

    for a in arrays:
        if np.nanmax(a) > mx:
            mx = np.nanmax(a)

    # normalize all arrays to the same range within [0,1] to be able to assign a color scale:
    for a in arrays:
        a = a / mx

    clip = mpy.ImageSequenceClip(arrays, fps=1)
    clip.write_gif(gif)

    print "Output written to "+gif
    # os.system('open ' + gif)

def projections2GIF(country, endyear):
    arrays = []
    arrays.append(np.load(os.path.expanduser('~') + '/Dropbox/CISC Data/IndividualCountries/'+str(country)+'.0-pop2000.npy'))

    arrays.append(np.load(os.path.expanduser('~') + '/Dropbox/CISC Data/IndividualCountries/'+str(country)+'.0-pop2010.npy'))

    year = 2020
    while year <= endyear:
        arrays.append(np.load(os.path.expanduser('~') + '/Dropbox/CISC Data/IndividualCountries/Projections/'+str(country)+'-'+str(year)+'-pop.npy'))
        year = year + 10

    arrays2GIF(arrays, os.path.expanduser('~') + '/Desktop/'+str(country)+'.gif')

if __name__ == '__main__':

    if len(sys.argv) != 3:
        print "This script expects two parameters, the country code and the last year of the projections, e.g."
        print "python MakeGIF.py 156 2050"
        print "to make a GIF from the projections for China. Check the WUP/WTP csv files for the IDs."
        sys.exit()

    projections2GIF(int(sys.argv[1]), int(sys.argv[2]))
