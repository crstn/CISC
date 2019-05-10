# ----------------------------------------------------------------------------
# DESCRIPTION:  This file calls the functions in Script 1, which list the dbf
#               files and converts them into csv files.  It places them in 
#               specific folders created in a previous script.  Folders are 
#               distinguished by RCP, year, ssp and data type ("*csv*").    
#
# DEVELOPER:    Peter J. Marcotullio
# DATE:         March 2018
# NOTES:        Uses python (os, time, datetime) and the functions script.  
# ----------------------------------------------------------------------------

# IMPORT MODULES AND ANNOUNCE START

import os, datetime, time
from Script_1_Heat_index_Functions import *

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

# Create initial directory and/or path to it
path_to_initial_directory = create_directory(cwd, initial_directory)

# SET IMPORTANT LISTS

# list of land uses 
land_use_models = ["GRUMP_HI", "GlobCover_HI"]

# START THE PROCESS

# iterate through the land use models
for land in land_use_models:

    # Get path to the Heat Index land use model latitude folder  
    path_to_land_use_folder = lc_model_path(path_to_initial_directory, land)

    # Get paths to the dbf folders in the latitude folder
    paths_dbf_folders = find_dbf_urban_latitude_dir(path_to_land_use_folder)
 
    print
    print "Finished finding %s folder paths for urban population latitude data: "% land + datetime.datetime.now().strftime("%I:%M.%S %p")
    print

    # Get the path to the urban folder and create csv folder with path
    for paths in paths_dbf_folders:
        
        # splite the path and remove the last element
        split_urban_paths = paths.split("\\")[:-1]

        # Get the first element of the split paths  
        split_first = split_urban_paths[0]
        split_first = split_first + "\\"
        
        # replace first element from the entire list 
        split_urban_paths[0] = split_first
        
        # use '*' (splat) to join with list 
        urban_folder_path=os.path.join(*split_urban_paths)

    # make new path to the csv folder
    path_to_csv_folder = os.path.join(urban_folder_path, "csv_files")
    
    # Get a list of the paths to all dbf files in the above folders
    latitude_dbf_files = find_dbf_urban_latitude_files(paths_dbf_folders)
    
    print "Finished finding %s dbf file paths: "% land + datetime.datetime.now().strftime("%I:%M.%S %p")
    print

    count = len(latitude_dbf_files)
    print "Finished counting %s dbf files: "% land + datetime.datetime.now().strftime("%I:%M.%S %p")
    print

    # change the dbf to csv and store them in the correct folder
    change_dbf_to_csv_urban_lat(latitude_dbf_files, path_to_csv_folder)
        
    # end timer and get results 
    end = time.time()
    final_time_hours = round(((end - start)), 2)                      

    print "Finished processing %d %s files in %d seconds"% (count, land, final_time_hours)


# END SCRIPT

print "Finished: " + datetime.datetime.now().strftime("%A, %d %B %Y %I:%M.%S %p")
print
                
