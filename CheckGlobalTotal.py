import os
import numpy as np
import PopFunctions as pop
import pync

os.chdir(os.path.expanduser('~') + '/Dropbox/CISC Data/IndividualCountries/Projections/Global');

for filename in os.listdir('.'):
    if filename.endswith(".tiff") and filename.startswith("pop-") :
        data = pop.openTIFFasNParray(filename);
        # -1 = NAN, set to 0:
        data[data<0] = 0
        print filename + ": " + "{:,}".format(np.nansum(data))

pync.Notifier.notify('Global total and urban population checked', title='Python')
