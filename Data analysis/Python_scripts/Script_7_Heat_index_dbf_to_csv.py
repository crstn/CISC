# ----------------------------------------------------------------------------
# DESCRIPTION:  This file calls the functions in Script 1, which list the dbf
#               files and converts them into csv files.  It places them in 
#               specific folders created in a previous script.  Folders are 
#               distinguished by RCP, year, ssp and data type ("*csv*").    
#
# DEVELOPER:    Peter J. Marcotullio
# DATE:         November 2017
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
# Create initial directory and path to it
path_to_initial_directory = create_directory(cwd, initial_directory)

# SET IMPORTANT LISTS

# list of directories with dbf files 
rcp_models = ["RCP2p6_HI_tables", "RCP4p5_HI_tables", "RCP6p0_HI_tables", "RCP8p5_HI_tables"]

# START THE PROCESS

# Get the folders with the HI tables
hi_table_folder_paths = find_HI_tables_dir(path_to_initial_directory, rcp_models)
print "Finished finding folder paths: " + datetime.datetime.now().strftime("%I:%M.%S %p")
print

print "THESE ARE THE FOLDERS"
for hi_table in hi_table_folder_paths:
    print hi_table
print

# Get a list of the paths to all dbf files in the above folders
hi_dbf_files = find_HI_dbf_files(hi_table_folder_paths)
print "Finished finding dbf file paths: " + datetime.datetime.now().strftime("%I:%M.%S %p")
print

for hi_dbf in hi_dbf_files:
    print hi_dbf
print


count = len(hi_dbf_files)
print "Finished counting dbf files: " + datetime.datetime.now().strftime("%I:%M.%S %p")
print

change_dbf_to_csv(hi_dbf_files)
    
# END SCRIPT

# end timer and get results 
end = time.time()
final_time_hours = round(((end - start)/(60*60)), 2)                      

print "Finished processing %d files in %d hours"% (count, final_time_hours)
print "Finished: " + datetime.datetime.now().strftime("%A, %d %B %Y %I:%M.%S %p")
print
                
