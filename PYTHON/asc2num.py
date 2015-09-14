import numpy as np
import os

#@profile
def read_asc():

    os.chdir('/Users/carsten/Downloads/Numpy')

    gluntlbnds = np.genfromtxt('/Users/carsten/Downloads/gl_grumpv1_ntlbndid_ascii_30/gluntlbnds.asc', skip_header=6, missing_values=-9999)

    print gluntlbnds.shape
    print np.min(gluntlbnds)
    print np.max(gluntlbnds)

    np.save("gluntlbnds", gluntlbnds)



    glurextents = np.genfromtxt('/Users/carsten/Downloads/gl_grumpv1_urextent_ascii_30/glurextents.asc', skip_header=6, missing_values=-9999)

    print glurextents.shape
    print np.min(glurextents)
    print np.max(glurextents)

    np.save("glurextents", glurextents)




    glup00g = np.genfromtxt('/Users/carsten/Downloads/gl_grumpv1_pcount_00_ascii_30/glup00g.asc', skip_header=6)

    print glup00g.shape
    print np.min(glup00g)
    print np.max(glup00g)

    np.save("glup00g", glup00g)

if __name__ == '__main__':
    read_asc()
