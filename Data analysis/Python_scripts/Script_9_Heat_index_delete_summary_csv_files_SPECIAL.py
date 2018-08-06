# ------------------------------------------------------------------------------------
# DESCRIPTION:  This script deletes all the csv files  ending in "r_ready.csv".  
#               It was written to address a mistake that created these files
#               without adequate file names (time: 5 or 15 day distinctions). 
#   
# DEVELOPER:    Peter J. Marcotullio
# DATE:         November 2017 (up-dated, but not run)
# NOTES:        Uses python (os, csv, datetime) and the functions script
# -------------------------------------------------------------------------------------

import datetime, os, csv, time
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
# Create initial directory and path to it
path_to_initial_directory = create_directory(cwd, initial_directory)

# SET IMPORTANT LISTS

land_use = ["GRUMP", "GlobCover"]

# list of directories with dbf files 
rcp_models = ["RCP2p6_HI_tables", "RCP4p5_HI_tables", "RCP6p0_HI_tables", "RCP8p5_HI_tables"]

# START THE PROCESS

# Get the folders with the HI tables
hi_table_folder_paths = find_HI_tables_dir(path_to_initial_directory, rcp_models)
print "Finished finding folder paths: " + datetime.datetime.now().strftime("%I:%M.%S %p")
print

# Get a list of the paths to all dbf files in the above folders
hi_csv_files = find_HI_csv_files(hi_table_folder_paths)
print "Finished finding csv file paths: " + datetime.datetime.now().strftime("%I:%M.%S %p")
print

# Loop through the list of files
for hi_csv in hi_csv_files:

    # find those ending with "_r_ready.csv"
    if hi_csv.endswith("_r_ready.csv"):

        # delete them
        os.remove(hi_csv)
        print "deleteed: ", hi_csv
    print

print "Script ended: ", datetime.datetime.now().strftime("%A, %d %B %Y %I:%M.%S %p")
print "DONE" 
