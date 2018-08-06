# -------------------------------------------------------------------------------
# DESCRIPTION:  This script performs zonal statistics for both "GRUMP" and
#               "GlobCover" land use models.  Each SSP includes 10 files by
#               decade which are matched with 3 rasters from 5 ISIMIP model
#               outputs (1, 5 and 15 day heat waves) for 4 time periods (
#               2010, 2030, 2070, 2100).  The data are stored in new folders
#               created by the script. 
#
# DEVELOPER:    Peter J. Marcotullio
# DATE:         November 2017
# NOTES:        Uses python (os, datetime, time), arcpy and the script functions.  
# -------------------------------------------------------------------------------

# Import modudles
import os, datetime, time
import arcpy
from arcpy import env
arcpy.CheckOutExtension("Spatial")
from arcpy.sa import *
from Script_1_Heat_index_Functions import *
arcpy.env.overwriteOutput = True

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
model_years = ["2010", "2030", "2070",  "2100"]
rcp_models =  ["RCP2p6", "RCP4p5","RCP6p0", "RCP8p5"]
ssp_names = ["SSP_1", "SSP_2", "SSP_3", "SSP_4", "SSP_5"]


# START THE PROCESS BY LOOPING THROUGH THE DIFFERENT URBAN LAND USE MODELS

for land in land_use[1:]: ## Control the land use models here

    # Announcement
    print "We are working with %s data"% land
    print

    # GET PATHS TO THE POPULATION DATA FOR THIS URBAN LAND USE

    # Puts us in the correct land use folder for population
    urban_folder = lc_model_path(cwd, land)

    # CREATE NEW SECONDARY FOLDERS BASED UPON THE LAND USE MODEL
    
    # create secondary directory name based upon land use model
    secondary_directory = land + "_HI"
    path_to_secondary_directory = create_directory(path_to_initial_directory, secondary_directory)
 
    # CREATE LIST OF SSPs IN THIS URBAN LAND USE MODEL

    # From the land use folder collects the paths to each SSP
    urban_ssp_folder_list = ssp_paths(urban_folder) # should include 50 files (5 SSPs and 10 years for each)

    # LOOP THROUGH HEAT WAVE RCP DATA 
    for rcp in rcp_models[:1]: ## control the rcp numbers here 

        # CREATE NEW DIRECTORIES BASED UPON THE RCP
        
        # Create a new folder with the RCP number, if it isn't there already
        path_to_tables_output = create_directory(path_to_secondary_directory, rcp + "_HI_tables")

        # MAKE A LIST OF ALL THE HEAT WAVE DATA FILES FOR THE CURRENT RCP

        # List all the files for the 5 GCMs for the current rcp
        hi_rcp_file_paths = heat_index_rcp_files(cwd, rcp) # should include 120 paths (5 models with 24 files each)

        # Choose only the heat wave files that have been re-sampled (end with "RS.tif") - reduces list to 60 files (5 models with 12 files each)
        working_rcp_file_paths = []
        for hi_rcp in hi_rcp_file_paths:
            if "RS.tif" in hi_rcp:
                working_rcp_file_paths.append(hi_rcp)
               
        # Iterate through the years of the analyses 
        for year in model_years[1:]: ## control the years here 

            # Announce
            print "We are working with %s data from %s"% (rcp, year)
            print

            # ORGANIZE THE HEAT WAVE DATA FOR ANALYSIS

            # Select the heat wave data by year and place them into a "working_files" list, reducing the list to 15 (5 models with 3 files each)
            working_files =[]
            for working_f in working_rcp_file_paths:
                
                # If the year is 2010, then pick find the files with 2009 in their name and use them
                if year == "2010":
                    if "2009" in working_f:
                        working_files.append(working_f)

                # OR if the year is 2030, then pick the files with "2030" in their name and use them 
                elif year == "2030":
                    if "2039" in working_f:
                        working_files.append(working_f)

                # OR if the year is 2070, then pick the files with "2069" in their name and use them
                elif year == "2070":
                    if "2069" in working_f:
                        working_files.append(working_f)

                # OR if the year is 2100, then pick the files with "2100" in their name and use them
                elif year == "2100":
                    if "2099" in working_f:
                        working_files.append(working_f)
                            
            # Select the urban files for the current year
            urban_files = ssp_urban_files_year_new(urban_ssp_folder_list, year) # reduces the SSP list to 5 (5 SSPs with 1 model year each)

            # Iterate through the urban  and ssp files simultaneously, create folders and paths and do the analysis for all SSPs 
            for urban, ssp in zip(urban_files, ssp_names): ## Control the number of ssps here 

                # CREATE FILE NAME

                urban_file_name = urban.split("\\")[-1]
                
                # CREATE FOLDERS AND PATHS TO FOLDERS TO HOLD THE DATA

                new_dbf_directory_name = year + "_dbf_tables"
                new_csv_directory_name = year + "_csv_tables"

                # Make two new directories
                path_to_dbf_directory = create_directory(path_to_tables_output, new_dbf_directory_name) 
                path_to_csv_directory = create_directory(path_to_tables_output, new_csv_directory_name)

                # Create path to output folder
                path_to_dbf_output = create_directory(path_to_dbf_directory, ssp)

                # make different folders to hold the csv data for later
                create_directory(path_to_csv_directory, ssp)
 
                # ITERATE THROUGH THE LIST OF HEAT WAVE FILES AND START ZONAL STATISTICS PROCESS
                
                for working in working_files: ## Control number of heat wave models for analysis here
                    
                    # CREATE NAMES 
                    
                    # Get the heat wave days from the input file for the output file name ending
                    if "RMean05" in working:
                        ending = "05_dayHW"
                    elif "RMean15" in working:
                        ending = "15_dayHW"
                    else:
                        ending = "01_dayHW"

                    # Get the model name of the input file for the output file name
                    model_name_first = working.split("\\")[4]
                    model_name = model_name_first[:-3]
                    model_name = model_name.replace("-", "_")
                    # Create a name to use in an announcement
                    announce_model_name = model_name + "_" + ending
                    # Create a final output file name
                    final_model_name = ssp + "_" + year + "_" + model_name + "_" + ending + ".dbf" 

                    # Announcement
                    print "Zonal statistics with %s and %s"% (urban_file_name, announce_model_name), datetime.datetime.now().strftime("%I:%M.%S %p")

                    # Zonal statistics and store the table in "dbf" format in the tables folder
                    ZonalStatisticsAsTable(urban, "urban_ID", working,  path_to_dbf_output + "\\" + final_model_name, "DATA", "MEAN")
                    count += 1
                print
                print "Finished: ", ssp, datetime.datetime.now().strftime("%I:%M.%S %p")
                print
        print "Finished: ", rcp, datetime.datetime.now().strftime("%I:%M.%S %p")
        print
               
# end timer and get results 
end = time.time()
final_time_hours = round(((end - start)/(60*60)), 2)                      

print "Finished processing %d files in %d hours"% (count, final_time_hours)
print "Finished: " + datetime.datetime.now().strftime("%A, %d %B %Y %I:%M.%S %p")
print
                

                
        
    



