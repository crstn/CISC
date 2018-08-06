# ----------------------------------------------------------------------------
# DESCRIPTION:  Sixth in a series. This file calls the functions in Script 1
#   This script should not be run before "R_Script_1_Create_Temp_&_pop_data.R"
#   is run.  In that file, names of GCM models for 2010 are created by RCP. This 
#   script uses the output from that analysis.  The script creates a 2010 set
#   of 3 month temperature data that include information for all urban areas
#   for all GCM models.  
#
# DEVELOPER:    Peter J. Marcotullio
# DATE:         May 2017
# NOTES:        Uses python and arcpy.  
# ----------------------------------------------------------------------------

# IMPORT MODULES

import os, sys, datetime, csv, time, glob
from os import rename
from Script_1_Three_month_temp_Functions import *

# INITIAL ANNOUNCEMENTS AND PRE-SCRIPT INITIATIONS

# Annonce the start of the script
print "Start script at: ", datetime.datetime.now().strftime("%A, %d %B %Y %I:%M.%S %p")
print

# Start timer
start = time.time()

# get current directory
cwd = os.getcwd() # = E:\

print "Script is located in: " + cwd
print "Named: " + os.path.basename(__file__)
print

# SET LOCAL VARIABLES

land_cover_table_name = "GRUMP_3M_tables" #CHANGE THIS (CHOICES ARE "GlobCover_3M_tables" or "GRUMP_3M_tables")
land_use_table_model_name = land_cover_table_name[:-10]

# GET LIST OF GCM MODEL NAMES FILES 

# Get the path to land cover table folder
path = find_tables_dir(cwd, land_cover_table_name)

# change to that directory and get path
os.chdir(path)

# change one directory up and create path and then change directories to that path
# os.chdir("..")
direct = os.getcwd()
name_files_directory = os.path.join(direct, "GRUMP_3M_Final")
os.chdir(name_files_directory)
##print(name_files_directory)

# list all the files that end with "names.csv"
names_files_first = glob.glob("*names.csv")

##for names_files in names_files_first:
##    print names_files

# Create special lists
rcps=["RCP2p6", "RCP4p5", "RCP6p0", "RCP8p5"]
ssps =["SSP_1", "SSP_2", "SSP_3", "SSP_4", "SSP_5"]

# Loop through rcp list
for rcp in rcps:

    # Loop through the ssp list
    for ssp in ssps:

        # put python the the working directory
        os.chdir(name_files_directory)

        # create empty list to fill
        names_files=[]

        # loop through the names list
        for names_f in names_files_first:

            # find files with rcp 
            if rcp in names_f:

                # within those files find files with ssp number
                if ssp in names_f:

                    # add file name to empty lst
                    names_files.append(names_f)

        # Announcement
        print "We are working in: ", rcp, ssp
        print
        
        # loop through the list open each file and append all the files of the gcm to the list 
        for names in names_files:

##            print names
##            print
                  
            not_models=[]
            all_gcm_model_names = []
            with open(names, "rb") as csvfile:
                for line in csvfile.readlines():
                    columns = line.split(",")
                    for c in columns:
                        if "X" not in c:
                            not_models.append(c)

                not_models.sort()

                for m in not_models[2:]:
                    all_gcm_model_names.append(m)
            
        # create a final list with only the unique gcm model names 
        unique_gcm_names = list(set(all_gcm_model_names))

        # sort list
        unique_gcm_names.sort()

##        for unique in unique_gcm_names:
##            print unique
    
        # GET TO THE TEMPERATURE CSV DATAFILES FOR 2010

        os.chdir(cwd)

        # Within the land cover table folder, get the folder paths that hold all the csv files
        csv_folders = find_csv_folders(path)

        # Create an empty list to use later
        csv_2010_folders = []

        # Loop through the list of folder paths 
        for csv_x in csv_folders:

            # find the paths that are for 2010
            if "2010_csv" in csv_x:

                # Append those to a list
                csv_2010_folders.append(csv_x)

##        for csv_x in csv_2010_folders:
##            print csv_x
##        print

        rcp_num=rcp.replace("p", ".")
        
        rcp_csv_2010_folders=[]
        for csv_2010 in csv_2010_folders:
            if rcp_num in csv_2010:
                rcp_csv_2010_folders.append(csv_2010)

        
        # Get the paths to the csv files that are in this list of folders
        all_2010_csv_files = get_csv_files(rcp_csv_2010_folders)


##        for all_2010 in all_2010_csv_files:
##            print all_2010
##        print
        
        # Create an empty list to use later 
        r_not_ready_files = []

        # loop through all the csv paths to only take the "r_ready" files
        for all_2010 in all_2010_csv_files:

            # rename files with "not_ready" or just include the "r_ready" file in the new list
            if "r_ready" in all_2010:

                # rename the file to "not_ready"
                # os.rename(all_2010, all_2010.replace("r_ready", "r_not_ready"))

                # append these paths to th empty list 
                r_not_ready_files.append(all_2010)

        # sort this list 
        r_not_ready_files.sort()

##        for r_not in r_not_ready_files:
##            print r_not
        
        # create an empty list to use later
        working_file_paths=[]

        # loop through the list of csv paths to only get the path to the current ssp file
        for r_not_ready in r_not_ready_files:

            # find those that have RCP and SSP number in the paths 
            if rcp_num in r_not_ready and ssp in r_not_ready:

                # append the files to a working path list
                working_file_paths.append(r_not_ready)

        for working in working_file_paths:
            print "This is the working file: ", working.split("\\")[-1]

        # OPEN THE TEMPERATURE FILES AND ADD THE UNIQUE GCM MODEL NAMES FOR EACH URBAN EXTENT 
             
        # loop through the new list 
        for working_file in working_file_paths:

            path_to_output = os.path.dirname(working_file)
            print "This is the path to output the table:", path_to_output

            f_temp_data = open(working_file, "rb")
            reader_temp = csv.reader(f_temp_data, delimiter = ",")

            header_temp = reader_temp.next()

            temp_data = []
            for row_temp in reader_temp:

                temp_row_data = [row_temp[0], row_temp[2], row_temp[3], row_temp[4]]
                temp_data.append(temp_row_data)

            data_1 = []

            for unique in unique_gcm_names:

                gcm_name = unique
                
                for temp in temp_data:

                    data_row = [temp[0], gcm_name, temp[1], temp[2], temp[3]]

                    data_1.append(data_row)

            # save file as "r_ready" in correct location
            with open(path_to_output + "/" + ssp + "_2010_r_ready.csv", "wb") as csvFile:

                writer = csv.writer(csvFile)

                writer.writerow(header_temp)

                writer.writerows(data_1)
            print  "Finished: "
            print

            f_temp_data.close()
    
# end timer and get results 
end = time.time()
time_split = ((end - start)/60)

print "Finished processing: " 
print "That took: ", round(time_split, 2), " miunutes"



