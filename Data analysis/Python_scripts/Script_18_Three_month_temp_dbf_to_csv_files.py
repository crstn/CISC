# ----------------------------------------------------------------------------
# DESCRIPTION:  This file calls the functions in Script 1 to change dbf files
#               from the zonal statistics output to csv files with new columns
#               and column headers.  Data are stored in folders created by the 
#               previous script.  Folders are distinguished by land use 
#               model, RCP, SSP model and titled "*csv*".
#               The script runs off IDLE.  
#
# DEVELOPER:    Peter J. Marcotullio
# DATE:         November 2017
# NOTES:        Uses python (os, datetime, time) and arcpy.
# ----------------------------------------------------------------------------

# IMPORT MODULES AND ANNOUNCE START

import os, datetime, time
from Script_1_Three_month_temp_Functions import *

# Announce the start of the script
print
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
land_use_models = ["GlobCover_3M", "GRUMP_3M"]

# CALL THE FUNCTIONS

for land in land_use_models:
        
    ft = find_tables_dir(path_to_initial_directory, land) #CHANGE THIS (CHOICES ARE "GlobCover_tables" or "GRUMP_tables")
    print "Found %s folder "% land + datetime.datetime.now().strftime("%I:%M.%S %p")  # CHANGE THIS (CHOIDES ARE "GlobCover" or "GRUMP")
    print

    # print ft

    dbf_folders = find_dbf_folders(ft)
    print "Found dbf folders " + datetime.datetime.now().strftime("%I:%M.%S %p")
    print

    dbf_files = get_dbf_files(dbf_folders)
    print "Listed dbf files " + datetime.datetime.now().strftime("%I:%M.%S %p")

    for dbf_f in dbf_files:
        count += 1

    print "Changing dbf to csv"
    print
    number_file = change_dbf_to_csv(dbf_files)
    print "Finished coverting dbf files to csv files" + datetime.datetime.now().strftime("%I:%M.%S %p")
    print

# end timer and get results 
end = time.time()
time_split = ((end - start)/(60 *60))

print "Finished processing: ", count
print "That took: ", round(time_split, 2), " hours"
