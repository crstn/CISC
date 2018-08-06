# -------------------------------------------------------------------------------
# DESCRIPTION:  This script performs zonal statistics for both "GRUMP" and
#               "GlobCover" land use models, but only for 2010 climate.  Each SSP
#               includes 4 files by decade (2010, 2030, 2070 and 2010) which are matched 
#               1 raster output for each model for 1, 5 and 15 day heat waves for 1 time
#               periods (2010).  The data are stored in new folders in
#               "XX_HI/Climate_2010" folders created by the script. 
#
# DEVELOPER:    Peter J. Marcotullio
# DATE:         January 2018
# NOTES:        Uses python (os, datetime, time), arcpy and the script functions.  
# -------------------------------------------------------------------------------

# Import modules
import os, datetime, time
import arcpy
from arcpy import env
arcpy.CheckOutExtension("Spatial")
from arcpy.sa import *
from Script_1_Heat_index_Functions import *
# arcpy.env.overwriteOutput = True

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


# Loop through the land use files
for land in land_use[1:]: # Control land use here (0-1)

    # return to the working initial working directory
    setwd(cwd)
    
    # Announcement
    print "We are working with %s data"% land
    print

    secondary_directory = land + "_HI"
    teriary_directory = "Climate_2010"
    out_folder = os.path.join(path_to_initial_directory, secondary_directory, teriary_directory)
       
    # GET PATHS TO THE POPULATION DATA FOR THIS URBAN LAND USE

    # Puts us in the correct land use folder for population
    urban_folder = lc_model_path(cwd, land)
    
    # CREATE LIST OF SSPs IN THIS URBAN LAND USE MODEL

    # From the land use folder collects the paths to each SSP folder
    urban_ssp_folder_list = ssp_paths(urban_folder)
        

    # loop through the years list
    for ssp_n in ssp_names: # Control ssp here (0-4)

        # Get the paths to the UrbanID vector file for the year (all SSPs)
        ssp_paths = ssp_urban_files_number_new(urban_ssp_folder_list, ssp_n)

        # Choose only those years that we need
        paths = []
        for years in model_years:
            for ssp in ssp_paths:
                if years in ssp:
                    paths.append(ssp)

        # loop through the rcp list
        for rcp in rcp_models: # Control RCP here (0-3)

            # Get all climate files for the rcp
            climate_files = heat_index_rcp_files(cwd, rcp)
            
            # Choose only the heat wave files that have been re-sampled (end with "RS.tif") - reduces list to 60 files (5 models with 12 files each)
            working_rcp_file_paths = []
            for hi_rcp in climate_files:
                if "RS.tif" in hi_rcp and "2009" in hi_rcp:
                    working_rcp_file_paths.append(hi_rcp)

            
            for path in paths: 
            
               # Split string to get the population file name 
                pop_working_path = path.split("\\")
                pop_file_name = pop_working_path[-1]
                ssp_name = pop_working_path[-3]
                year = pop_file_name.split("_")[-2]
                               
                # Names of new final directory 
                dbf_directory_name = year + "_dbf_tables"
                csv_directory_name = year + "_csv_tables"


                # Create dbf output directory
                year_dbf_directory = create_directory(out_folder, dbf_directory_name)

                # Create companion csv output directory to store csv files
                year_csv_directory = create_directory(out_folder, csv_directory_name)

                # Create a final directory for dbf files and csv files  
                ssp_dbf_directory = create_directory(year_dbf_directory, ssp_name)
                ssp_csv_directory = create_directory(year_csv_directory, ssp_name)

                # loop through the climate files
                for working in working_rcp_file_paths: # control heat waves here (0-2)

                    # split working rcp file path to get model name 
                    climate_file_name=working.split("\\")[-1]

                    # Create final file name 
                    name_1 = climate_file_name.split("_")[2]
                    name_2 = name_1.replace("+", "_")
                    cli_name = name_2.replace("-RMean", "_")
                    if "-RMean" not in name_2:
                        cli_name = name_2 + "_01"
                    cl_name = cli_name.replace("-", "_")
                    final_file_name = ssp_name + "_" + year + "_" + cl_name + ".dbf"
                    
                    print "Zonal statistics with %s and %s"%(cl_name, pop_file_name)

                    # DO ZONAL STATISTICS
                    ZonalStatisticsAsTable(path, "urban_ID", working,  ssp_dbf_directory + "\\" + final_file_name, "DATA", "MEAN")

                    print "Stored in: \t%s"% ssp_dbf_directory
                    print
                    
                    count += 1
                   

# end timer and get results 
end = time.time()
final_time_hours = round(((end - start)/(60*60)), 2)                      

print "Finished processing %d files in %d hours"% (count, final_time_hours)
print "Finished: " + datetime.datetime.now().strftime("%A, %d %B %Y %I:%M.%S %p")
print
                
