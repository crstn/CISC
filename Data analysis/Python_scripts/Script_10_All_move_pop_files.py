# -------------------------------------------------------------------------
# DESCRIPTION:  This script changes the population dbf files into csv files
#               and moves the population data tables from the 
#               population folders to the "Pop_&_Temp" folders for each land
#               use model (GlobCover and GRUMP) and each temperature type
#               (3 Months and Heat waves).  The tables are moved for R access.
#
# DEVELOPER:    Peter J. Marcotullio
# DATE:         December 2017
# NOTES:        Uses python and arcpy.  
# --------------------------------------------------------------------------
# IMPORT MODULES AND ANNOUNCE START

import os, datetime, time, glob
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
land_use_models = ["GlobCover_3M", "GlobCover_HI", "GRUMP_3M", "GRUMP_HI"]
ssp_number_list = ["SSP_1", "SSP_2", "SSP_3", "SSP_4", "SSP_5"]

# START PROCESS

for land in land_use_models:

    # Get land use model name with ending
    land_model_name = land[:-3]

    # Get the land use directory path
    path_land_use = lc_model_path(path_to_initial_directory, land)

    # Create directory to hold the population data
    path_to_output_landuse_folder = create_directory(path_land_use, land + "_pop_tables")

    # list the table paths to the SSPs
    paths_to_population = lc_model_path(cwd, land_model_name)

    # Get paths to the ssp folders for the population data
    b_paths = ssp_paths(paths_to_population)

    # Iterate through the ssp number list
    for ssp_number in ssp_number_list:

        print "Getting .dbf files for %s, %s and changing them to .csv files:"%(land, ssp_number)
        print 

        # create new folder to hold the data
        path_to_output_ssp_folder = create_directory(path_to_output_landuse_folder, ssp_number)

        # create a new folder to hold the dbf files
        csv_output_path = create_directory(path_to_output_ssp_folder, "CSV_original")
        
        # get the urban csv file paths
        f_paths = pop_dbf_urban_files(b_paths, ssp_number)


        # go through file paths, change the headers and store the data in the new folder 
        for f in f_paths:

            # get basename of folder
            dbf_folder_path1 = os.path.dirname(f)

            dbf_name = os.path.basename(f)
            csv_name = dbf_name.replace(".dbf", ".csv")

            # CHANGE FILE FROM DBF INTO CSV 
            
            # open a new csv file with new name
            with open(csv_output_path + "\\" + csv_name,'wb') as csvfile:

                # Get the dbf file
                in_db = dbf.Dbf(f)

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

        for name_path in glob.glob(csv_output_path + "\\*.csv"):         

            # Create a list to hold the csv data
            data_1 = []

            # name of file 
            file_name = name_path.split("\\")[-1]

            # year
            year = file_name.split("_")[2]

            # open the csv file in python
            file_x = open(name_path, "rb")

            # create a reader object to examine the data in the csv
            reader = csv.reader(file_x, delimiter = ",")

            # get the header data
            header = reader.next()

            # Prepare a new header
            ind = 0
            col = 0

            # loop through the header file to find the "SUM" column
            for head in header:
                if head == "POPULATION":
                    col = ind
                else:
                    ind += 1

            # Create names for the new header
            col_1 = "urban_code"
            col_2 = "Land_model"
            col_3 = "SSP"
            col_4 = "Year"
            col_5 = "Population"

            # put the new names in a header list
            new_header = [col_1, col_2, col_3, col_4, col_5]

            # loop through the reader object
            for row in reader:

                # read the rows and put the values into a new list
                data_row = [row[0], land_model_name, ssp_number, year, row[col]]

                # append the data to a list of lists
                data_1.append(data_row)

            # Open a new csv file
            with open(path_to_output_ssp_folder + "\\" + file_name, "wb") as csvFile:

                # Create a csv writer object
                writer = csv.writer(csvFile)

                # write the header
                writer.writerow(new_header)

                # write the rest of teh rows
                writer.writerows(data_1)

                print "Finished writing: ", ssp_number, file_name
                count += 1
        print
         
# end timer and get results 
end = time.time()
time_split = ((end - start)/(60 *60))

print "Finished processing: ", count
print "That took: ", round(time_split, 2), " hours"
print "Script ended: ", datetime.datetime.now().strftime("%A, %d %B %Y %I:%M.%S %p")
print "DONE"
        
