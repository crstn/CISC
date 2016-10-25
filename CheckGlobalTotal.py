import os
import numpy as np
import PopFunctions as pop

os.chdir(os.path.expanduser('~') + '/Dropbox/CISC Data/IndividualCountries/Projections/Global');

for filename in os.listdir('.'):
    if filename.endswith(".tiff") and filename.startswith("pop-") :
        data = pop.openTIFFasNParray(filename);
        # small numbers = NAN, set to 0:
        data[data < 0] = 0

        print filename + ": " + "{:,}".format(np.sum(data))
