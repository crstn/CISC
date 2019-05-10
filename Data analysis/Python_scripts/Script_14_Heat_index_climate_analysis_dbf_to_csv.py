# ----------------------------------------------------------------------------
# DESCRIPTION:  This file calls the functions in Script 1, which list the dbf
#               files and converts them into csv files.  It places them in 
#               specific folders created in a previous script.  Folders are 
#               distinguished by RCP, year, ssp and data type ("*csv*").    
#
# DEVELOPER:    Peter J. Marcotullio
# DATE:         January 2018
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

    paths_dbf_folders = find_dbf_tables_latitude_dir(path_to_land_use_folder)
    for paths in paths_dbf_folders:
        print paths
    print

    print
    print "Finished finding %s folder paths for climate data: "% land + datetime.datetime.now().strftime("%I:%M.%S %p")
    print


    # Get a list of the paths to all dbf files in the above folders
    latitude_dbf_files = find_dbf_latitude_files(paths_dbf_folders)
    print "Finished finding %s dbf file paths: "% land + datetime.datetime.now().strftime("%I:%M.%S %p")
    print

    count = len(latitude_dbf_files)
    print "Finished counting %s dbf files: "% land + datetime.datetime.now().strftime("%I:%M.%S %p")
    print

    for lat in latitude_dbf_files:
        print lat
    print

    # change the dbf to csv and store them in the correct folder
    #change_dbf_to_csv_climate(latitude_dbf_files)
        
    # end timer and get results 
    end = time.time()
    final_time_hours = round(((end - start)/(60*60)), 2)                      

    print "Finished processing %d %s files in %d hours"% (count, land, final_time_hours)


# END SCRIPT

print "Finished: " + datetime.datetime.now().strftime("%A, %d %B %Y %I:%M.%S %p")
print
                
