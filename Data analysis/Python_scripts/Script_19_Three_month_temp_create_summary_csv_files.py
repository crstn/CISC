# ------------------------------------------------------------------------------------
# DESCRIPTION:  This script creates summary csv files that are ready for r scripting
#               Data are stored in the same cvs folders as "*r_ready.csv" files
#   
# DEVELOPER:    Peter J. Marcotullio
# DATE:         May 2017
# NOTES:        Uses python (os, csv, datetime) and  functions script               
# -------------------------------------------------------------------------------------

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

# START PROCESS

for land in land_use_models:
    
    # Get the folders in the tables
    lum_tables_folder = find_tables_dir(path_to_initial_directory, land)  

    # Get the csv folders in the land use model 
    csv_folders = find_csv_folders(lum_tables_folder)

    # Create empty list to hold the various CSV files
    rcp2p6_csv_folders = []
    rcp4p5_csv_folders = []
    rcp6p0_csv_folders = []
    rcp8p5_csv_folders = []

    # loop through the full list and put the csv paths in the appropriate folder
    # This snipit of code is not necesary.  
    for csv_x in csv_folders:

        if "RCP4.5" in csv_x:

            rcp4p5_csv_folders.append(csv_x)

        elif "RCP2.6" in csv_x:

            rcp2p6_csv_folders.append(csv_x)
        
        elif "RCP6.0" in csv_x:

            rcp6p0_csv_folders.append(csv_x)

        elif "RCP8.5" in csv_x:

            rcp8p5_csv_folders.append(csv_x)
        else:
            print "This folder is incorrect", csv_x
    print

    all_rcp_folders = []

    all_rcp_folders.append(rcp2p6_csv_folders)
    all_rcp_folders.append(rcp4p5_csv_folders)
    all_rcp_folders.append(rcp6p0_csv_folders)
    all_rcp_folders.append(rcp8p5_csv_folders)


    all_rcp_folders.sort()

    for rcp_folders in all_rcp_folders:

        csv_files = get_csv_files(rcp_folders)

        dup_csv_folders = []

        for csv_f in csv_files:

            rcp = csv_f.split("\\")[4][:6]

            # Create a list to hold the data
            data_1=[]

            # get folder
            folder = os.path.dirname(csv_f)

            # Append to list
            dup_csv_folders.append(folder)
            
            # Get file name
            file_name = os.path.basename(csv_f)
            model_name = file_name[6:-4]

            # get year
            year_folder = csv_f.split("\\")[-3]
            year = year_folder[:4]

            # get ssp
            ssp = csv_f.split("\\")[-2]

##            print csv_f
##            print land
##            print rcp
##            print year
##            print ssp
##            print model_name
##            print
            
            # open the csv file in python
            f = open(csv_f, "rb")

            # create reader object to examine inside the file
            reader = csv.reader(f, delimiter = ",")

            # get the table column names
            header = reader.next()

            # Prepare a new header 
            ind = 0
            col = 0

            # loop through the header to find the "MEAN" column
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
            col_5 = "Temperature"

            # put new names in a header list
            new_header = [col_1, col_2, col_3, col_4, col_5]

            # loop through the reader object
            for row in reader:

                # read the rows and put the values into a new list
                data_row =[row[0], model_name, ssp, year, row[col]]

                # append to make a list of lists
                data_1.append(data_row)

            # Open a new csv file
            with open(folder + "\\" + ssp + "_" + year + "_" + model_name + "_r_ready.csv", "wb") as csvFile:

                # Create a csv writer object
                writer = csv.writer(csvFile)

                # write the header
                writer.writerow(new_header)

                # write the rest of the rows
                writer.writerows(data_1)
              
                print "Finished writing: ", land, rcp, ssp, model_name, year

            count += 1

        print    
        print "Finished entire ", rcp, datetime.datetime.now().strftime("%A, %d %B %Y %I:%M.%S %p")
        print

# end timer and get results 
end = time.time()
time_split = ((end - start)/(60 *60))

print "Finished processing: ", count
print "That took: ", round(time_split, 2), " hours"
print "Script ended: ", datetime.datetime.now().strftime("%A, %d %B %Y %I:%M.%S %p")
print "DONE"


