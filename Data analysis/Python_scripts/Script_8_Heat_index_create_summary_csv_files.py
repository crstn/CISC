# ------------------------------------------------------------------------------------
# DESCRIPTION:  This script takes all the csv files created for the heat index and 
#               re-formats the data file with appropriate headers and columns to 
#               continue the analysis is R-Studio.  It places the new data into the  
#               same folders.  
#   
# DEVELOPER:    Peter J. Marcotullio
# DATE:         November 2017
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

##for hi_table in hi_table_folder_paths:
##    print hi_table

# Get a list of the paths to all csv files in the above folders
hi_csv_files = find_HI_csv_files(hi_table_folder_paths)
print "Finished finding csv file paths: " + datetime.datetime.now().strftime("%I:%M.%S %p")
print

##for hi_csv in hi_csv_files:
##    print hi_csv
##print
##print
##print

for hi_csv in hi_csv_files:

    # Get names, ssp, year, model name
    file_name = os.path.basename(hi_csv)
    folder = os.path.dirname(hi_csv)
    ssp = file_name[:5]
    model_name_first = file_name[6:]
    year = model_name_first[:4]
    time_1 = model_name_first[-12:-10]
    model_name = model_name_first[5:-13]

##    print "This is the filename:", file_name
##    print "This is the folder:", folder
##    print "This is the spp:", ssp
##    print "This is the model name:", model_name
##    print "This is the year:", year
##    print "This is the heatwave length:", time_1
##    print
##    print

    # CHANGE THE STRING TO A NUMERIC FOR HEAT WAVE LENGHT 

    if time_1 == "05":
        time_2 = 5
    elif time_1 == "15":
        time_2 = 15
    elif time_1 == "01":
        time_2 = 1
    else:
        print "This file is a problem", hi_csv
    
    

    data_1 = []

    # open the csv file in python
    f = open(hi_csv, "rb")

    # create reader object to examine inside the file
    reader = csv.reader(f, delimiter = ",")

    # get the table column names
    header = reader.next()

    # PREPARE A NEW HEADER

    # Make counts for the columns  
    ind = 0
    col = 0

    # loop through the header names to find the "MEAN" column
    for head in header:
        if head == "MEAN":
            col = ind
        else:
            ind += 1

    # Create names for the new header
    col_1 = "urban_code"
    col_2 = "GCM_Model" 
    col_3 = "SSP"
    col_4 = "Year"
    col_5 = "Heat_Wave_length"
    col_6 = "Heat_Index"

    # put new names in a header list
    new_header = [col_1, col_2, col_3, col_4, col_5, col_6]

    # loop through the reader object
    for row in reader:

        # read the rows and put the values into a new list
        data_row =[row[0], model_name, ssp, year, time_2, row[col]]

        # append to make a list of lists
        data_1.append(data_row)

    # Open a new csv file
    with open(folder + "\\" + ssp + "_" + year + "_" + model_name + "_" + time_1 + "_r_ready.csv", "wb") as csvFile:

        # Create a csv writer object
        writer = csv.writer(csvFile)

        # write the header
        writer.writerow(new_header)

        # write the rest of the rows
        writer.writerows(data_1)
      
        print "Finished writing: ", ssp, model_name, year, time_1
    count += 1

# END SCRIPT

# end timer and get results 
end = time.time()
final_time_hours = round(((end - start)/(60*60)), 2)
final_time_seconds = round((end-start),2)
print
print "Finished processing %d files in %d hours"% (count, final_time_hours)
# print "So it will take %d hours to finish "% ((final_time_seconds*240)/(60*60))
print "Finished: " + datetime.datetime.now().strftime("%A, %d %B %Y %I:%M.%S %p")
print "DONE"
print
                
