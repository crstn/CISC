# -------------------------------------------------------------------------
# DESCRIPTION:  This script creates population csv data tables from shape  
#               file dbf files and stores them in the "tables" folder within
#               each population SSP section in a folder called "urbanID"
#
# DEVELOPER:    Peter J. Marcotullio
# DATE:         December 2017
# NOTES:        Uses python and arcpy.  
# --------------------------------------------------------------------------
# IMPORT MODULES AND ANNOUNCE START

import os, datetime, time, glob
from dbfpy import dbf
from Script_1_Three_month_temp_Functions import *
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
land_use_models = ["GlobCover_3M", "GlobCover_HI", "GRUMP_3M", "GRUMP_HI"]
ssp_number_list = ["SSP_1", "SSP_2", "SSP_3", "SSP_4", "SSP_5"]

# START PROCESS

for land in land_use_models:  # CONTROL THE LAND USE MODELS AND TEMPERATURE FOLDERS HERE

    # Get land use model name with ending
    land_model_name = land[:-3]

    # Get the land use directory path
    path_land_use = lc_model_path(path_to_initial_directory, land)

    # Create directory to hold the population data
    path_to_output_landuse_folder = create_directory(path_land_use, land + "_ID_tables")

    # list the table paths to the SSPs
    paths_to_population = lc_model_path(cwd, land_model_name)
    b_paths = ssp_paths(paths_to_population)

    # WE ONLY NEED TO DO THIS ONCE FOR EACH LAND USE MODEL

    if "3M" in land:
        
        # Get the paths to the folders with the IDs
        path_to_urbanID_vectors = urbanID_vector_folders(b_paths)
 
        # Get paths to the tables folders
        paths_to_pop_table_folders = table_folders(b_paths)

        # simultaneously iterate through the two lists
        for table, id_vector in zip(paths_to_pop_table_folders, path_to_urbanID_vectors): # CONTROL THE SSPS HERE

            # get ssp number
            ssp_number = table.split("\\")[-2]

            # create folders in the output folder ("Pop_&_Temp") to hold the final data
            path_to_output_ssp_folder = create_directory(path_to_output_landuse_folder, ssp_number)

            # Create folder in the table directory to hold the original csv files
            path_to_csv_output = create_directory(table, "urbanID")
            
            # Get all the dbf files from the vector folder
            for name in glob.glob(id_vector + "\\*.dbf"):

                # Get the name of the file 
                file_name = name.split("\\")[-1][:-4]
 
                # open a new csv file with new name in the "urbanNationalID" folder)
                with open(path_to_csv_output + "\\" + file_name + ".csv",'wb') as csvfile:

                    # Get the dbf file
                    in_db = dbf.Dbf(name)

                    # create a csv writer object to put the lines into 
                    out_csv = csv.writer(csvfile)

                    # create an empty list for header names 
                    names = []

                    # fill the list with the dbf header names 
                    for field in in_db.header.fields:
                        
                        # Append the list to the names list
                        names.append(field.name)

                    # put the header names in the writer object
                    out_csv.writerow(names)

                    # iterate through the dbf file and put the lines into the new csv writer object 
                    for rec in in_db:
                        out_csv.writerow(rec.fieldData)
                        
                    print "Finished writing: ", land, file_name + ".csv"
            print
  
# end timer and get results
end = time.time()
time_split = ((end - start)/(60 *60))

print "Finished processing: ", count
print "That took: ", round(time_split, 2), " hours"
print "Script ended: ", datetime.datetime.now().strftime("%A, %d %B %Y %I:%M.%S %p")
print "DONE"
        
