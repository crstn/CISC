# ------------------------------------------------------------------------------------
# DESCRIPTION:  This script takes all the csv files created for the latitude for total
#               population and reformat the data file with appropriate headers and columns
#               to facilitate the R-Studio analysis.  It places the new data into the  
#               same folders.  One of the columns is changed to include values of 1 degree
#   
# DEVELOPER:    Peter J. Marcotullio
# DATE:         March 2018
# NOTES:        Uses python (os, csv, datetime) and the functions script
# -------------------------------------------------------------------------------------

# IMPORT MODULES AND ANNOUNCE START

import os, datetime, time
from Script_1_Heat_index_Functions import *

# Start timer
start = time.time()

# SET IMPORTANT INITIAL VARIABLES

count = 0
cwd = os.getcwd()
initial_directory = "Pop_&_Temp"
# Create initial directory and path to it
path_to_initial_directory = create_directory(cwd, initial_directory)

# SET IMPORTANT LISTS

# list of land uses 
land_use_models = ["GRUMP_HI", "GlobCover_HI"]


# MAKE INITIAL ANNOUNCEMENTS

# Announce the start of the script
print "Start script at: \t", datetime.datetime.now().strftime("%A, %d %B %Y %I:%M.%S %p")
print "File Location: \t\t" + cwd
#print "File Name: \t\t" + os.path.basename(__file__)
print

# START THE PROCESS

# iterate through the land use models
for land in land_use_models: # Control the land use models here

    # Get path to the Heat Index land use model latitude folder  
    path_to_land_use_folder = lc_model_path(path_to_initial_directory, land)

    # MAKE NEW FUNCTION TO GET THE CSV FOLDER
    # REPLACE THE NEXT FEW LINES 

    # Get paths to the dbf folders in the latitude folder
    paths_csv_folders = find_csv_urban_latitude_dir(path_to_land_use_folder)
 
    print
    print "Finished finding %s folder paths for urban population latitude data: "% land + datetime.datetime.now().strftime("%I:%M.%S %p")
    print

    # Get paths to the csv urban latitude files
    csv_files = find_csv_urban_latitude_files(paths_csv_folders)

    print
    print "Finished finding %s file paths for urban population latitude data: "% land + datetime.datetime.now().strftime("%I:%M.%S %p")
    print


    # Iterate through the files list
    for csv_f in csv_files:

        # split the path and get the name of the file 
        csv_file_name = csv_f.split("\\")[-1]
        #new_file_name = csv_file_name.replace(".csv", "_ready.csv")
        folder = os.path.dirname(csv_f)
    
        # Get names, ssp, year, model name
        ssp1 = csv_file_name.split("_")[0]
        ssp2 = csv_file_name.split("_")[1]
        ssp = ssp1 + "_" + str(ssp2)
        year = csv_file_name.split("_")[2] 
        land_use = land[:-3]

##        print "This is the filename:", csv_file_name
##        print "This is the folder:", folder
##        print "This is the spp:", ssp
##        print "This is the land use model:", land_use
##        print "This is the year:", year
##        print
##        print

        # START THE CSV TRANSFORMATION
        
        # Create empty list
        data_1 = []

        # open the csv file in python
        f = open(csv_f, "rb")
        
        # create reader object to examine inside the file
        reader = csv.reader(f, delimiter = ",")

        # get the table column names
        header = reader.next()

        # PREPARE A NEW HEADER

        # MAKE NEW VARIABLE FOR "CITIES" 

        # Make counts for the columns  
        ind = 0
        col_pop = 0
        col_cities=0
        col_display_num = 0

        # loop through the header names to find the "MEAN" column
        for head in header:
            if head == "SUM":
                col_pop = ind
                ind += 1
            elif head == "DISPLAY":
                col_display_num = ind
                ind += 1
            elif head == "COUNT":
                col_cities = ind
                ind += 1
            else:
                ind += 1

        # Create names for the new header
        col_1 = "Display"
        col_2 = "Land_use"
        col_3 = "SSP"
        col_4 = "Year"
        col_5 = "Urb_Population"

        # put new names in a header list
        new_header = [col_1, col_2, col_3, col_4, col_5]

        # loop through the reader object
        for row in reader:

            # Change the values in the display column to include 1 degree 
            if "S" in row[col_display_num]:

                value = row[col_display_num].split(" ")
                num_value_plus = int(value[0]) + 1
                new_display = value[0] + "-" + str(num_value_plus) + " " + value[-1]

            elif row[col_display_num] == "0":

                new_display = "0-1 S"

            elif "N" in row[col_display_num]:

                value = row[col_display_num].split(" ")
                num_value_minus = int(value[0]) - 1
                new_display = str(num_value_minus) + "-" + value[0] + " " + value[-1]
            else:
                print "There is a problem with:, ", row[col_display_num]

            
            # read the rows and put the values into a new list
            data_row =[new_display, land_use, ssp, year, row[col_pop]]

            # append to make a list of lists
            data_1.append(data_row)

        # WHERE TO PUT THE FINAL FILES???

        # Open a new csv file
        with open(folder + "\\" + ssp + "_" + year + "_" + land_use + "_urban_pop_latitude_r_ready.csv", "wb") as csvFile:

            # Create a csv writer object
            writer = csv.writer(csvFile)

            # write the header
            writer.writerow(new_header)

            # write the rest of the rows
            writer.writerows(data_1)
          
            print "Finished writing: ", ssp, land_use, year
        count += 1

# END SCRIPT

# end timer and get results 
end = time.time()
final_time_hours = round(((end - start)/(60*60)), 2)
final_time_seconds = round((end-start),2)
print
print "Finished processing %d files in %d seconds"% (count, final_time_seconds)
print "Finished: " + datetime.datetime.now().strftime("%A, %d %B %Y %I:%M.%S %p")
print "DONE"
print
                
