# ----------------------------------------------------------------------------
# DESCRIPTION:  This file calls the functions in Script 1 and performs zonal
#               statistics using the different urban areas (by decade) on the 
#               different temperature outputs.  DBF files are stored in newly  
#               created folders titled by RCP, year, dbf or csv and SSP model.
#               The script runs off IDLE.  
#
# DEVELOPER:    Peter J. Marcotullio
# DATE:         November 2017
# NOTES:        Uses python and arcpy.
# ----------------------------------------------------------------------------

# Import modudles
import os, datetime, time
import arcpy
from arcpy import env
arcpy.CheckOutExtension("Spatial")
from arcpy.sa import *
from Script_1_Three_month_temp_Functions import *

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
model_years = ["2010", "2030", "2050", "2070",  "2080"]
rcp_levels = ["RCP2.6", "RCP4.5", "RCP6.0","RCP8.5"]  
ssp_names = ["SSP_1", "SSP_2", "SSP_3", "SSP_4", "SSP_5"]

# Set counter and start timer
count = 0
start = time.time()

# START THE PROCESS BY LOOPING THROUGH THE DIFFERENT URBAN LAND USE MODELS

for land in land_use[1:]: # control the land use model here

    # GET PATHS TO THE POPULATION DATA FOR THIS URBAN LAND USE

    # Get the land use model data    
    land_use_model_path = lc_model_path(cwd, land)

    # Lists the SSPs of the land use model data
    land_use_model_ssp_list = ssp_paths(land_use_model_path)


    # CREATE NEW SECONDARY FOLDERS BASED UPON THE LAND USE MODEL
    
    # create secondary directory name based upon land use model
    secondary_directory = land + "_3M"
    path_to_secondary_directory = create_directory(path_to_initial_directory, secondary_directory)
      

    for rcp_level in rcp_levels[1:]: # control the rcp numbers here

        print rcp_level

        # CREATE NEW DIRECTORIES BASED UPON THE RCP
        
        # Create a new folder with the RCP number, if it isn't there already
        path_to_tables_output = create_directory(path_to_secondary_directory, rcp_level + "_3M_tables")


        # loop through the years
        for year in model_years:

            print year

            # ORGANIZE THE RCP DATA FOR ANALYSIS

            # Lists the paths to the files for year (all SSPs) for land use model data
            urban_files = ssp_urban_files_year_new(land_use_model_ssp_list, year) 

            # create year name 
            year_name = year + "_Mean"

            # Lists all the temperature folders by RCP and YEAR    
            rcp_paths = get_rcp_paths(cwd, year_name) 

            # List the temperature data files
            rcp_max_model_files = temp_file_paths(rcp_paths, rcp_level) 

            
            # CREATE FOLDERS AND PATHS TO FOLDERS TO HOLD THE DATA

            # Create names for folders
            new_dbf_directory_name = year + "_dbf_tables"
            new_csv_directory_name = year + "_csv_tables"

            # Make two new directories
            path_to_dbf_directory = create_directory(path_to_tables_output, new_dbf_directory_name) 
            path_to_csv_directory = create_directory(path_to_tables_output, new_csv_directory_name)            

            for ssp in ssp_names: # Control the SSPs level here

                path_to_ssp_dbf_folder = create_directory(path_to_dbf_directory, ssp)
                path_to_ssp_csv_folder = create_directory(path_to_csv_directory, ssp)

                # PERFORM ZONAL STATISTICS

                # Announce process
                print "Performing zonal statistics"
            
                # Simultaneously loop through two lists (one with globcover data and the other with folder names
                for urban in urban_files:
                    if ssp in urban:
                        working_pop_file = urban

                # Create a name for the ssp model
                name_1 = os.path.basename(working_pop_file)
                ssp_name = name_1[:5]

                #Announce
                print "Spatial analysis for: ",land, rcp_level, ssp_name, year
                print

                # get rcp count and start rcp time
                rcp_count = 0
                rcp_time_start = time.time()
                
                # loop through the GCM outputs
                for rcp_model in rcp_max_model_files: # Control number of GCM outputs here [:1]:
                   
                        # Create a name for the GCM output
                        name = os.path.basename(rcp_model)
                        temp_model_name = name[:-17]

                        # Announce the process and count activity
                        print "GCM name: %s and: %s"%(temp_model_name, name_1), datetime.datetime.now().strftime("%I:%M.%S %p")
                        count += 1
                        rcp_count += 1
        
                        # Do the zonal statistics and store the table in "dbf" format in the tables folder 
                        ZonalStatisticsAsTable(working_pop_file, "urban_ID", rcp_model,  path_to_ssp_dbf_folder + "\\" + ssp_name + "_" + temp_model_name + ".dbf", "DATA", "MEAN") # arcpy.gp.ZonalStatisticsAsTable_sa

                # Get rcp time end and get results
                rcp_time_end = time.time()
                rcp_time_minutes = round((rcp_time_end - rcp_time_start)/60, 2)
                
                print "Finished: " + datetime.datetime.now().strftime("%I:%M.%S %p")
                print "That took %d minutes to process %d files"%(rcp_time_minutes, rcp_count)
                print
                
# end timer and get results 
end = time.time()
final_time_hours = round(((end - start)/(60*60)), 2)                      

print "Finished processing %d files in %d hours"% (count, final_time_hours)
print "Finished: " + datetime.datetime.now().strftime("%A, %d %B %Y %I:%M.%S %p")
print
       
