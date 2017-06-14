import numpy as np
import gdal
import os
import PopFunctions as pop

resolution=0.008333333333333 # in decimal degrees!
dst = os.path.expanduser('~') + "/Dropbox/CISC Data/Area Grid/"

"""
Creates a global grid of a particular resolution, where the value of
each grid cell is the area of that grid cell.

Borrowed/adapted from https://gis.stackexchange.com/questions/232813/easiest-way-to-create-an-area-raster
"""

def do_grid (resolution):
    """Calculate the area of each grid cell for a user-provided
    grid cell resolution. Area is in square meters, but resolution
    is given in decimal degrees."""
    # Calculations needs to be in radians
    lats = np.deg2rad(np.arange(-57,84, resolution))
    r_sq = 6371000**2
    n_lats = int(360./resolution)
    area = r_sq*np.ones(n_lats)[:, None]*np.deg2rad(resolution)*(
                np.sin(lats[1:]) - np.sin(lats[:-1]))
    return area.T

if __name__ == "__main__":

    areagrid = do_grid(resolution)

    # Get the GeoTIFF driver
    drv = gdal.GetDriverByName("GTiff")
    # Compressed GeoTIFF file
    dst_ds = drv.Create(dst+"area-grid.tif", int(360./resolution),
                int(141./resolution),
                1, gdal.GDT_Float32,
                options = [ 'COMPRESS=DEFLATE'] )
    # Projection using EPSG:4326
    wgs84='GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563,AUTHORITY["EPSG","7030"]],AUTHORITY["EPSG","6326"]],PRIMEM["Greenwich",0,AUTHORITY["EPSG","8901"]],UNIT["degree",0.01745329251994328,AUTHORITY["EPSG","9122"]],AUTHORITY["EPSG","4326"]]'
    dst_ds.SetProjection(wgs84)
    geotransform = (-180.,resolution,0,84.,0,-resolution)
    dst_ds.SetGeoTransform(geotransform)
    dst_ds.GetRasterBand(1).WriteArray(areagrid)
    dst_ds = None
