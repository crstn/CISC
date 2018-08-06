# -------------------------------------------------------------------------------
# DESCRIPTION:  This script performs zonal statistics for both "GRUMP" and
#               "GlobCover" land use models urban population data. 
#
# DEVELOPER:    Peter J. Marcotullio
# DATE:         March 2018
# NOTES:        Uses python (os, datetime, time), arcpy and the script functions.  
# -------------------------------------------------------------------------------

# Import modules
import os, datetime, time
import arcpy
from arcpy import env
arcpy.CheckOutExtension("Spatial")
from arcpy.sa import *
from Script_1_Population_Functions import *
# arcpy.env.overwriteOutput = True

# Announce the start of the script
print "Start script at: \t", datetime.datetime.now().strftime("%A, %d %B %Y %I:%M.%S %p")
print "File Location: \t\t" + cwd
print "File Name: \t\t" + os.path.basename(__file__)
print

# Start timer
start = time.time()

# SET IMPORTANT INITIAL VARIABLES

count = 0
cwd = os.getcwd()

initial_directory = "Pop_&_Temp"

# Create initial directory and path to it
path_to_initial_directory = create_directory(cwd, initial_directory)

# Get the path to the graticules folder using list comprehension for finding directories and putting them into a  list 
all_dirs = [d for d in os.listdir('.') if os.path.isdir(d)]

# Create empty list
path_to_graticule_folder = []

# loop through directories list 
for dirs in all_dirs:

    # Find "Graticules" folders, make a path to it and put it in a list
    if dirs.startswith("Graticules"):
        
        # Create a path to the folder  
        path_to_graticule_folder=os.path.join(cwd, dirs, "new")

# Move arcpy into the graticule directory
arcpy.env.workspace = path_to_graticule_folder

# Make a list of all the feature classes in Graticule subdirectory called "new"
featureclass_list = arcpy.ListFeatureClasses()

# Find the ID file "2100" feature class in the list
for fc in featureclass_list:

    # find the file for SSP_3_2100
    if fc.endswith("numbered.shp"):

        graticule_file_name = fc

        # create a path to this file 
        graticule_file_path = os.path.join(path_to_graticule_folder, fc)

# SET IMPORTANT LISTS

land_use = ["GRUMP", "GlobCover"]

# Loop through the land use files
for land in land_use: # Control land use here (0-1)

    # return to the working initial working directory
    os.chdir(cwd)

    # Get paths and create new directories to store data
    HI_folder = land + "_HI"
    lat_folder = land + "_Latitude"
    path_to_lat_directory = os.path.join(path_to_initial_directory, HI_folder, lat_folder)
    path_to_urban_pop_directory = create_directory(path_to_lat_directory, "Urban_pop")
    path_to_dbf_folder = create_directory(path_to_urban_pop_directory, "dbf_files")

    # Announcement
    print "We are working with %s data"% land
    print
       
    # GET PATHS TO THE URBAN POPULATION RASTER DATA 

    # Puts us in the correct land use folder for population
    urban_folder = population_landuse_folder(land) 

    # From the land use folder collects the paths to each SSP folder
    ssp_folder_list = ssp_population_paths(urban_folder)

    # From the ssp list get the paths to all the urban population files
    urban_pop_files = urban_population_files(ssp_folder_list)

    # ITERATE THROUGH THE LIST OF RASTER FILES AND PERFORM ZONAL STATISTICS

    for urban_pop in urban_pop_files:

        # Get data for naming
        split_path = urban_pop.split("\\")
        ssp = split_path[-4]
        year = split_path[-1].split("_")[-1]

        urban_pop_file_name = split_path[-1]

        # Make final path 
        final_path = os.path.join(path_to_dbf_folder, ssp + "_" + year + "_urban_population_latitude.dbf")

        # Announce
        print "Zonal statistics with %s and %s"%(graticule_file_name, urban_pop_file_name)

        # DO ZONAL STATISTICS
        ZonalStatisticsAsTable(graticule_file_path, "Display", urban_pop,  final_path, "DATA", "SUM")

        print "Stored in: \t%s"% path_to_dbf_folder
        print

        count += 1

# end timer and get results 
end = time.time()
final_time_hours = round(((end - start)/(60)), 2)                      

print "Finished processing %d files in %d minutes"% (count, final_time_hours)
print "Finished: " + datetime.datetime.now().strftime("%A, %d %B %Y %I:%M.%S %p")
print      


    
    
                                        


