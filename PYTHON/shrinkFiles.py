import numpy as np
import os

os.chdir('/Users/carsten/Downloads/Numpy')
# shrinks the file size of the glurextents and gluntlbnds
# by casting to more space-saving data types
glurextents = np.load("/Users/carsten/Downloads/Numpy/glurextents.npy")
glurextents.clip(0, 10)
np.save('glurextents-int', glurextents.astype('int8'))
del(glurextents)

glurextents = np.load("/Users/carsten/Downloads/Numpy/gluntlbnds-clipped.npy")
glurextents.clip(0, 10)
np.save('gluntlbnds-clipped-int', glurextents.astype('uint8'))
