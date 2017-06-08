import numpy as np
import os

"""
Creates two grids that contain the row index and column index,
respectively, and save those to disc as numpy arrays.
"""

# size in pixels of our GeoTIFFs
rowcount = 16920
colcount = 43200

def makeGrids(rows, cols):
    rowgrid = np.empty([rows, cols])
    colgrid = np.empty([rows, cols])

    for row in range(0, rows):
        for col in range(0, cols):
            rowgrid[row, col] = row
            colgrid[row, col] = col

    return rowgrid, colgrid


if __name__ == "__main__":
    rowgrid, colgrid = makeGrids(rowcount, colcount)

    np.save(os.path.expanduser('~') +
            '/Dropbox/CISC Data/Index Grids/rows.npy', rowgrid)
    np.save(os.path.expanduser('~') +
            '/Dropbox/CISC Data/Index Grids/cols.npy', colgrid)
