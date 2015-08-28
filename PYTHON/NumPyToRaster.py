def main(countryCode, filepath, geodatabase):
    import arcpy, os
    import numpy as np

    inputRaster = arcpy.Raster(filepath+"/"+geodatabase+"/GLUR_"+str(countryCode))
    print "Input raster loaded"
    lowerLeft = arcpy.Point(inputRaster.extent.XMin, inputRaster.extent.YMin)
    cellSize = inputRaster.meanCellWidth
    print "Input raster parameters saved"

    NumPyLoc = filepath +"/Output_"+str(countryCode)
    saveLoc = filepath+"/"+geodatabase
    total_list = os.listdir(NumPyLoc)
    
    for array in total_list:
        print "Converting", array, "to raster"
        arr = np.load(NumPyLoc+"/"+array)
        newRaster = arcpy.NumPyArrayToRaster(arr, lowerLeft, cellSize, value_to_nodata=0)
        name = array.strip(".npy")
        newRaster.save(saveLoc+"/"+name)

        print name, "created"
        
    print "done"

if __name__ == '__main__':
    main(180, "V:/Original/LIVE", "CISC.gdb")
