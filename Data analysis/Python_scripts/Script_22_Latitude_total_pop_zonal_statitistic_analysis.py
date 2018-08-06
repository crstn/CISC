# -------------------------------------------------------------------------------
# DESCRIPTION:  This script performs zonal statistics for both "GRUMP" and
#               "GlobCover" land use models total population data. 
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
# print "File Name: \t\t" + os.path.basename(__file__)
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
ssp_names = ["SSP_1", "SSP_2", "SSP_3", "SSP_4", "SSP_5"]


# Loop through the land use files
for land in land_use: # Control land use here (0-1)

    # return to the working initial working directory
    os.chdir(cwd)

    # create name for new directory
    secondary_directory = land + "_HI"
    name_new_folder = land + "_Latitude"
    path_to_new_directory = os.path.join(path_to_initial_directory, secondary_directory)

    # Create new directory 
    out_folder = create_directory(path_to_new_directory, name_new_folder)

    # Announcement
    print "We are working with %s data"% land
    print
       
    # GET PATHS TO THE POPULATION DATA FOR THIS URBAN LAND USE

    # Puts us in the correct land use folder for population
    urban_folder = population_landuse_folder(land)
 
    
    # CREATE LIST OF SSPs IN THIS URBAN LAND USE MODEL

    # From the land use folder collects the paths to each SSP folder
    ssp_population_folder_list = ssp_population_paths(urban_folder)

    pop_rasters = population_files(ssp_population_folder_list)

    # Iterate through the ssps
    for ssp in ssp_names: # Control the ssps here

        # Choose only the ssps that we need
        ssp_pop_files = []

        # Iterate through the pop rasters list 
        for pop in pop_rasters: 

            # identify the current working ssp    
            if ssp in pop:

                count += 1

               # Get year and ssp for naming file and some folders  
                path_split = pop.split("\\")
                pop_file_name = path_split[-1]
                year = pop_file_name.split("-")[1][0:4]
                ssp_name = ssp

                # MAKE NAME FOR THE FINAL FILE
                final_file_name = ssp_name + "_" + year + "_total_population_latitude.dbf"

                # Names of new final directory 
                dbf_directory_name = year + "_dbf_tables"
                csv_directory_name = year + "_csv_tables"                               

                # Create dbf output directory
                year_dbf_directory = create_directory(out_folder, dbf_directory_name)

                # Create companion csv output directory to store csv files
                year_csv_directory = create_directory(out_folder, csv_directory_name)

                # Create a final directory for dbf files and csv files  
                ssp_dbf_directory = create_directory(year_dbf_directory, ssp_name)
                ssp_csv_directory = create_directory(year_csv_directory, ssp_name)

                 
                # Announce
                print "Zonal statistics with %s and %s"%(graticule_file_name, pop_file_name)

                # DO ZONAL STATISTICS
                ZonalStatisticsAsTable(graticule_file_path, "Display", pop,  ssp_dbf_directory + "\\" + final_file_name, "DATA", "SUM")

                print "Stored in: \t%s"% ssp_dbf_directory
                print


        print                

# end timer and get results 
end = time.time()
final_time_hours = round(((end - start)/(60*60)), 2)                      

print "Finished processing %d files in %d hours"% (count, final_time_hours)
print "Finished: " + datetime.datetime.now().strftime("%A, %d %B %Y %I:%M.%S %p")
print
                
