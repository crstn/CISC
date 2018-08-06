# -------------------------------------------------------------------------
# DESCRIPTION:  This script moves the urban ID csv data tables from the   
#               population folders to new folder created in the "Pop_&_Temp"
#               folder area.  It also reduces the variables to "urban_code"
#               "nation_ID"
#
# DEVELOPER:    Peter J. Marcotullio
# DATE:         December 2017
# NOTES:        Uses python and arcpy.  
# --------------------------------------------------------------------------
# IMPORT MODULES AND ANNOUNCE START

import os, datetime, time, glob, csv
from Script_1_Population_Functions import *

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
land_use_temp_models = ["GlobCover_3M", "GlobCover_HI", "GRUMP_3M", "GRUMP_HI"]
land_use_models = ["GlobCover", "GRUMP"]


ssp_number_list = ["SSP_1", "SSP_2", "SSP_3", "SSP_4", "SSP_5"]

for land in land_use_models:

    # Get the path to the population files in this land use
    land_use_model_path = population_landuse_folder(land)

    # Get the list of SSPs in this land use
    ssp_list = ssp_population_paths(land_use_model_path)

    # get the list of paths to the urbanID files
    urbanID_files = get_urbanID_csv(ssp_list)

    # iterate through the land use temp models
    for land_use in land_use_temp_models:

        # select files only if they are in the correct land use model
        if land in land_use:

            # create a path to the Pop_&_Temp folders for the land use temp model
            land_use_temp_model_path = os.path.join(path_to_initial_directory, land_use)
            
            # create a new directory in the land use temp model folder by adding "_ID_tables" as an ending
            land_use_temp_model_ID_path = create_directory(land_use_temp_model_path, land_use + "_ID_tables")

            # Iterate through the urbanID file paths
            for urbanID in urbanID_files:

                # get the file name
                file_name = urbanID.split("\\")[-1]
                
                # Get the ssp of the urbanID file
                ssp = urbanID.split("\\")[-4]

                # use the ssp to create a new folder to store the data 
                output_folder_path = create_directory(land_use_temp_model_ID_path, ssp)
                
                # make an empty list to hold all our row data
                data_1 = []
                
                # open each file
                csv_file = open(urbanID, "rb")

                # create a reader file object
                reader = csv.reader(csv_file, delimiter = ",")

                # get the header names
                header = reader.next()

                # create new names for the header and put them in a list
                col_1 = "urban_code"
                col_2 = "nation_ID"
                new_header = [col_1, col_2]

                # Prepare for row identification
                ind = 0
                col = 0

                # loop through the header file to find the "SUM" column
                for head in header:
                    if head == "NATIONID":
                        col = ind
                    else:
                        ind += 1

                # loop through the other rows
                for row in reader:

                    # read the row data adn put the values into a new list
                    data_row = [row[0], row[col]]
                    
                    # append the data_row data to the data_1 list 
                    data_1.append(data_row)
                

                with open(output_folder_path + "\\" + file_name, "wb") as csvFile:

                    # Create a csv writer object
                    writer =csv.writer(csvFile)

                    # Write the header
                    writer.writerow(new_header)

                    # Write the rest of the rows
                    writer.writerows(data_1)

                    print "Finished writing: ", land, "in", land_use + ":", file_name
                    count += 1
        print
            
# end timer and get results 
end = time.time()
time_split = ((end - start)/(60 *60))

print "Finished processing: ", count
print "That took: ", round(time_split, 2), " hours"
print "Script ended: ", datetime.datetime.now().strftime("%A, %d %B %Y %I:%M.%S %p")
print "DONE"
