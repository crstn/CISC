# ----------------------------------------------------------------------------
# DESCRIPTION:  This script runs through all the heat index 
#               rasters and re-samples them from (0.05 dd) to a finer resolution
#               (0.0083333333) comparable with our other data. The script runs
#               off IDLE.  
#   
# DEVELOPER:    Peter J. Marcotullio
# DATE:         June 2017
# NOTES:        Uses python (os, datetime, and Script_1_Heat_index_Functions)  
# ----------------------------------------------------------------------------

# Import modudles
import os, datetime
##import arcpy
##from arcpy import env
##arcpy.CheckOutExtension("Spatial")
##from arcpy.sa import *
from Script_1_Heat_index_Functions import *

# Annonce the start of the script
print "Start script at: ", datetime.datetime.now().strftime("%A, %d %B %Y %I:%M.%S %p")
print

# SET IMPORTANT VARIABLES

# cell size for re-sampling
cell_size = 0.0083333333

# Get current path and file name
cwd = os.getcwd()

# rcp names list 
rcp_levels = ["RCP2p6", "RCP4p5", "RCP6p0", "RCP8p5"]


# GET HEAT WAVE RASTERS 

# loop through rcp names
for level in rcp_levels:

    # return all heat wave rasters in that rcp 
    file_paths = heat_index_rcp_files(cwd, level)

    for f in file_paths:

        zero_name = f.split("\\")[-1]
        first_name = zero_name[:-4]
        second_name = first_name + "_RS.tif"
        outpath = f.replace(zero_name, second_name)
        re_sample(f, outpath, cell_size)
        print "Re-sampled: ", zero_name
    print "Finished: ", level
    print
    
print
print "Finished: " + datetime.datetime.now().strftime("%I:%M.%S %p")
print
    

