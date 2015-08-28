import arcpy, os
import numpy as np
from arcpy import env
env.workspace = "J:/Data/CISC.gdb"

filepath = "J:/LIVE/"
rasterList = arcpy.ListRasters("G*")
codeList = []

for raster in rasterList:
    codeList.append(raster[5:])
                    
del rasterList

for code in codeList:
    if int(code) > 840:
        print code
        inputGLUR = arcpy.Raster("J:/Data/CISC.gdb/GLUR_"+code)
        lowerLeft = arcpy.Point(inputGLUR.extent.XMin, inputGLUR.extent.YMin)
        cellSize = inputGLUR.meanCellWidth
        print "Converting GLUR_"+code, "to NumPy array"
        arr = arcpy.RasterToNumPyArray(inputGLUR, nodata_to_value=0)
        os.chdir("J:/LIVE/NumPy_GLUR")
        np.save("GLUR_"+code+".npy", arr)
        print "GLUR Converted"
        del inputGLUR
        del arr

        inputPop = arcpy.Raster("J:/Data/CISC.gdb/Pop00_"+code)
        lowerLeft = arcpy.Point(inputPop.extent.XMin, inputPop.extent.YMin)
        cellSize = inputPop.meanCellWidth
        print "Converting Pop00_"+code, "to NumPy array"
        arr = arcpy.RasterToNumPyArray(inputPop, nodata_to_value=0)
        os.chdir("J:/LIVE/NumPy_Pop")
        np.save("Pop00_"+code, arr)
        print "Population array converted"
        del inputPop
        del arr
    
