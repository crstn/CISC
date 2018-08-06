# -------------------------------------------------------------------------
# DESCRIPTION:  This script creates two new sets of files: rasters with 
#               extracted urban extents (with country codes) and vectors
#               with these country codes, individual urban extent codes,
#               areas and lat and long.  The files are stored in new
#               "urbanNationalID_vectors" folders in the "tables" folder
#               for each SSP.
#
# DEVELOPER:    Peter J. Marcotullio
# DATE:         November 2017
# NOTES:        Uses python and arcpy.  
# --------------------------------------------------------------------------

# Import modules
import os, datetime, time
import arcpy
from arcpy import env
arcpy.CheckOutExtension("Spatial")
from arcpy.sa import *
from Script_1_Population_Functions import *


# Announce the start of the script
print "Start script at: \t", datetime.datetime.now().strftime("%A, %d %B %Y %I:%M.%S %p")
print "File Location: \t\t" + cwd
print "File Name: \t\t" + os.path.basename(__file__)
print

cwd = os.getcwd()

# Start timer
start = time.time()

# SET IMPORTANT VARIABLES

number_file = 0
cwd = os.getcwd()

# SET THE LAND USE MODEL

land_use = ["GRUMP", "GlobCover"]
# land_use = ["GlobCover"]
# land_use = ["GRUMP"]

# START THE PROCESS BY FIRST ITERATING THROUGH THE URBAN LAND USE TYPES

for land in land_use: # Control the land use models here

    # FIND THE SSP AND TABLE FOLDERS FOR THE URBAN LAND USE
    
    urban_folder = population_landuse_folder(land)
    ssp_folder_paths = ssp_population_paths(urban_folder)
    table_folder_paths = table_folders(ssp_folder_paths)

    # FIND THE POPULATION FILES AND THE URBAN ID FILES FOR THE URBAN LAND USE

    # pop_files = population_files(ssp_folder_paths)
    ID_files = urbanID_files(ssp_folder_paths)

    
    # ITERATE THROUGH THE SSPS FOR THE URBAN LAND USE 

    for table in table_folder_paths: # control the ssp number here

        # Get ssp number from the path and create geodatabase name
        ssp = table.split("\\")[-2]

        # Announcement
        print "We are now working in", land, ssp.split("_")[0], ssp.split("_")[1]
        print

        # Select only the current ssp urbanID files from the larger list
        ssp_ID_files = []
        for files in ID_files:
            if ssp in files:
                ssp_ID_files.append(files)

        # Select only the current ssp paths from the larger list
        ssp_paths = []
        for ssp_folders in ssp_folder_paths:
            if ssp in ssp_folders:
                ssp_paths.append(ssp_folders)


        # Create a new directory and geodatabase and paths to each
        for ssps in ssp_paths:
            
            # create new directories and new geodatabase in the new directories
            new_directory_path = create_directory(ssps, "urbanNationID_vectors")
            new_gdb_path = create_gdb(new_directory_path, "Extracted_rasters.gdb")


        # Iterate through the list of files for this ssp
        for ssp_ID in ssp_ID_files: # Control the years here
                        
            # Create raster name with ssp number, year and "EX" and make a path to the gdb
            file_name = ssp_ID.split("\\")[-1]
            year = file_name.split("-")[-2]
            name_raster = ssp + "_" + year + "_EX"
            extracted_raster_path = os.path.join(new_gdb_path, name_raster)
    
            # Announce the process 
            print "Extracting urban cells from new file: ", file_name
            
            # Extract only the Values > 1 (national iso codes) from the urbanID raster
            rasExtract = ExtractByAttributes(ssp_ID, "Value > 1")     
            rasExtract.save(extracted_raster_path)

            # Create a vector name
            name_vector = ssp + "_" + year + "_urbanID.shp"
            vector_path = os.path.join(new_directory_path, name_vector)

            # Announce the process 
            print "Creating vector file: ", name_vector
            
            # raster to polygon conversion (make the vector files)
            vectr = arcpy.RasterToPolygon_conversion(in_raster=rasExtract, out_polygon_features=vector_path, simplify="NO_SIMPLIFY", raster_field="Value")

            # Announce the processes
            print "Add fields, urban and nation codes, areas and geometries"
            print

            # Provide field name and data
            fieldName = "urban_ID"
            fieldName0 = "nationID"
            fieldPrecision = 10
            fieldScale = 10

            # Add fields to the ID file
            arcpy.AddField_management(vectr, fieldName, "LONG", fieldPrecision, fieldScale)
            arcpy.AddField_management(vectr, fieldName0, "LONG", fieldPrecision, fieldScale)

            # FIELD CALCULATIONS

            # Provide local variables
            exp = "autoIncrement()"
            codeblock ="""rec=0\ndef autoIncrement():\n  global rec\n  pStart = 1\n  pInterval = 1\n  if (rec == 0):\n    rec = pStart\n  else:\n    rec += pInterval\n  return rec"""

            # Execute CalculateField to create urban_ID and nationID field values
            arcpy.CalculateField_management(in_table = vectr, field = fieldName, expression = exp, expression_type = "PYTHON_9.3", code_block = codeblock)
            arcpy.CalculateField_management(in_table=vectr, field=fieldName0, expression="!gridcode!", expression_type="PYTHON_9.3", code_block="")

            # CALCULATE GEOMETRIES

            inFeatures = vectr
            fieldName1 = "LONG"
            fieldName2 = "LAT"
            fieldPrecision = 10
            fieldScale = 10

            # Add fields
            arcpy.AddField_management(inFeatures, fieldName1, "DOUBLE", fieldPrecision, fieldScale)
            arcpy.AddField_management(inFeatures, fieldName2, "DOUBLE", fieldPrecision, fieldScale)
            arcpy.AddField_management(inFeatures, "Area_SQKM", "DOUBLE")
             
            # Calculate centroid
            arcpy.CalculateField_management(inFeatures, fieldName1,"!SHAPE.CENTROID.X!", "PYTHON_9.3")
            arcpy.CalculateField_management(inFeatures, fieldName2,"!SHAPE.CENTROID.Y!", "PYTHON_9.3")
            # Calculate area
            exp = "!SHAPE.AREA@SQUAREKILOMETERS!"
            arcpy.CalculateField_management(vectr, "Area_SQKM", exp, "PYTHON_9.3")

            # Delete the "gridcode" and "id" fields 
            arcpy.DeleteField_management(vectr, ["id", "gridcode"])

            number_file += 1

    print "Finished", land

# end timer and get results 
end = time.time()
time_split = ((end - start)/(60*60))

# Announce the ending 
print "Finished processing: ", number_file
print "That took: ", round(time_split, 2), "hours"
print "The data and time are now", datetime.datetime.now().strftime("%A, %d %B %Y %I:%M.%S %p")
