# -------------------------------------------------------------------------
# DESCRIPTION:  This script creates several fields in the 1 degree graticule
#               file.  See also "autoincrement_play.py" file for creating
#               the other columns 
#
# DEVELOPER:    Peter J. Marcotullio
# DATE:         March 2018
# NOTES:        Uses python and arcpy.  
# ----------------------------------------------------------------------
# Import modules
import inspect, os, datetime
import arcpy
from arcpy import env
arcpy.CheckOutExtension("Spatial")
from arcpy.sa import *

print "Start script at: ", datetime.datetime.now().strftime("%A, %d %B %Y %I:%M.%S %p")

# Get current path and file name
cwd = os.getcwd()

print "Located in: " + cwd
print "Named: " + os.path.basename(__file__)
print

# FIND THE GRATICULE FOLDER 

# Create empty list 
path=[]

# list comprehension for finding directories and putting them into a  list 
all_dirs = [d for d in os.listdir('.') if os.path.isdir(d)]

# loop through directories list 
for dirs in all_dirs:

    # Find "Graticules" folders, make a path to it and put it in a list
    if dirs.startswith("Graticules"):
        
        # Create a path to the folder  
        path_to_graticule_folder=os.path.join(cwd, dirs, "new")

    # Find the folder that ends with vector
    #if dirs.endswith("vector"):

        # Create a path to the folder
        #path_to_storage_folder = os.path.join(cwd,dirs)


# Urban vector files at "E:\July_2017\Population\Data\SSPs\GlobCover\SSP_1\urbanNationID_vectors" for SSP 1

# Change directories to the SSP_3 folder
os.chdir(path_to_graticule_folder)

# Move arcpy into the working directory
arcpy.env.workspace = path_to_graticule_folder

# Make a list of all the feature classes in Graticule subdirectory called "new"
featureclass_list = arcpy.ListFeatureClasses()

# Find the ID file "2100" feature class in the list
for fc in featureclass_list:

    # find the file for SSP_3_2100
    if fc.endswith("polygons.shp"):

        # create a path to this file 
        graticule_file = os.path.join(path_to_graticule_folder, fc)

        # ADD FIELD

        # Announce the processes
        print "Add fields and field calculations"

        # Provide field name and data
        fieldName = "lat_code"
        fieldPrecision = 10
        fieldScale = 10

        # Add fields
        arcpy.AddField_management(graticule_file, fieldName, "SHORT", fieldPrecision, fieldScale)

        # FIELD CALCULATIONS

        # Provide local variables
        fieldName = "lat_code"
        exp = "autoIncrement()"
        codeblock ="""interval=0\nrec=1\ndef autoIncrement():\n  global rec\n  global interval\n  pStart = 89\n  pInterval = 1\n  if (interval == 0):\n    rec = pStart\n    interval += 1\n  else:\n    rec -= pInterval\n  return rec"""

       # Execute CalculateField 
        arcpy.CalculateField_management(in_table = graticule_file, field = fieldName, expression = exp, expression_type = "PYTHON_9.3", code_block = codeblock)

        # Announce process
        print "Finished: " + datetime.datetime.now().strftime("%I:%M.%S %p")
        print

        # Announce process
        print "Save new file"

        # Save file
        arcpy.CopyFeatures_management(graticule_file, path_to_graticule_folder + "\\" + "One_degree_latitude_numbered.shp")
        print "Finished: " + datetime.datetime.now().strftime("%I:%M.%S %p")
        print

        # DELETE FIELD

        # Announce process
        print "Delete field in the original file"  

        # Delete field in original file
        #arcpy.DeleteField_management(id_file, fieldName)
        print "Finished: " + datetime.datetime.now().strftime("%I:%M.%S %p")
        print

# Announce process
print "Script ended: ", datetime.datetime.now().strftime("%A, %d %B %Y %I:%M.%S %p")
print "DONE"                    
