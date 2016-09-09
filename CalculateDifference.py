import os, numpy as np, tif2num as tn
from PIL import Image

# Array of numpy arrays saved to disk. We'll calculate
# the difference between each consecutive pair of layers and
# save the difference as a tiff
queue = ['392-2020-pop.npy', '392-2030-pop.npy', '392-2040-pop.npy', '392-2050-pop.npy']

# they should all be in the same directory:
os.chdir(os.path.expanduser('~') + '/Dropbox/CISC - Global Population/IndividualCountries/Projections')

i = 0;
while i < len(queue)-1:
    new = np.load(queue[i+1])
    old = np.load(queue[i])
    diff = new - old

    img = Image.fromarray(diff)
    img.save(os.path.expanduser('~') + '/Dropbox/CISC - Global Population/IndividualCountries/Projections/diff-'+queue[i+1]+'-'+queue[i]+'.tiff')

    i = i + 1
