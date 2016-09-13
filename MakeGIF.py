import os, numpy as np
import moviepy.editor as mpy

country = 156
endyear = 2050

arrays = []

year = 2020

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

arrays.append(np.load(os.path.expanduser('~') + '/Dropbox/CISC Data/IndividualCountries/'+str(country)+'.0-pop2000.npy'))

arrays.append(np.load(os.path.expanduser('~') + '/Dropbox/CISC Data/IndividualCountries/'+str(country)+'.0-pop2010.npy'))

while year <= endyear:
    arrays.append(np.load(os.path.expanduser('~') + '/Dropbox/CISC Data/IndividualCountries/Projections/'+str(country)+'-'+str(year)+'-pop.npy'))
    year = year + 10

arrays2GIF(arrays, os.path.expanduser('~') + '/Desktop/'+str(country)+'.gif')
