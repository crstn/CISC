# -------------------------------------------------------------------------------
# DESCRIPTION:  This script performs extract by mask and creates urban population
#               rasters that can then be analyszed by zonal statistics. 
#
# DEVELOPER:    Peter J. Marcotullio
# DATE:         March 2018
# NOTES:        Uses python (os, datetime, time), arcpy and the script functions.  
# -------------------------------------------------------------------------------

# Import modules
import os, datetime, time
import arcpy
from arcpy import env
arcpy.CheckOutExtension("Spatial")
from arcpy.sa import *
from Script_1_Population_Functions import *
# arcpy.env.overwriteOutput = True

# Announce the start of the script
print "Start script at: \t", datetime.datetime.now().strftime("%A, %d %B %Y %I:%M.%S %p")
print "File Location: \t\t" + cwd
# print "File Name: \t\t" + os.path.basename(__file__)
print

# Start timer
start = time.time()

# SET IMPORTANT INITIAL VARIABLES

count = 0
cwd = os.getcwd()

# Create initial directory and path to it
initial_directory = "Pop_&_Temp"
path_to_initial_directory = create_directory(cwd, initial_directory)

# SET IMPORTANT LISTS

land_use = ["GRUMP", "GlobCover"]
ssp_names = ["SSP_1", "SSP_2", "SSP_3", "SSP_4", "SSP_5"]


# Loop through the land use files
for land in land_use: # Control land use here (0-1)

    # return to the working initial working directory
    os.chdir(cwd)

    # create name for new directory
    secondary_directory = land + "_HI"
    name_new_folder = land + "_Latitude"
    path_to_new_directory = os.path.join(path_to_initial_directory, secondary_directory)

    # Create new directory 
    out_folder = create_directory(path_to_new_directory, name_new_folder)

    # Announcement
    print "We are working with %s data"% land
    print
       
    # GET PATHS TO THE POPULATION DATA FOR THIS URBAN LAND USE

    # Puts us in the correct land use folder for population
    urban_folder = population_landuse_folder(land) 
    
    # CREATE LIST OF SSPs IN THIS URBAN LAND USE MODEL

    # From the land use folder collects the paths to each SSP folder
    ssp_population_folder_list = ssp_population_paths(urban_folder)

    # list population rasters 
    pop_rasters = population_files(ssp_population_folder_list)

    # list urban vectors
    urban_vectors = urbanID_vectors(ssp_population_folder_list)

    for pop, urban in zip(pop_rasters, urban_vectors):


        # Get data for announcement
        urban_file_name = urban.split("\\")[-1]

        # Get data for the name
        path_split = pop.split("\\")
        pop_file_name = path_split[-1]
        year = pop_file_name.split("-")[1][0:4]
        ssp_name = path_split[-3]

        # Create file name
        urban_raster_name = ssp_name + "_" + year 

        # Create path to final folder
        population_folder = os.path.dirname(pop)

        # Create full final path 
        final_path = os.path.join(population_folder, "urban_rasters", urban_raster_name)

        # Extract by mask
        outExtractByMask = ExtractByMask(pop, urban)

        # Save raster
        outExtractByMask.save(final_path)

        # Add to counter
        count += 1

        print "Finished processing %s and %s"%(pop_file_name, urban_file_name)
    print
        
# end timer and get results 
end = time.time()
final_time_hours = round(((end - start)/(60*60)), 2)                      

print "Finished processing %d files in %d hours"% (count, final_time_hours)
print "Finished: " + datetime.datetime.now().strftime("%A, %d %B %Y %I:%M.%S %p")
print
        




# Replace a layer/table view name with a path to a dataset (which can be a layer file) or create the layer/table view within the script
# The following inputs are layers or table views: "popmean-2010.tiff-no-nan.tiff", "SSP_1_2010_urbanID"
#arcpy.gp.ExtractByMask_sa("popmean-2010.tiff-no-nan.tiff","SSP_1_2010_urbanID","F:/July_2017/Test/Population/GRUMP_newest/test2")














##
##    # Iterate through the ssps
##    for ssp in ssp_names: # Control the ssps here
##
##        # Choose only the ssps that we need
##        ssp_pop_files = []
##
##        # Iterate through the pop rasters list 
##        for pop in pop_rasters: 
##
##            # identify the current working ssp    
##            if ssp in pop:
##
##                count += 1
##
##               # Get year and ssp for naming file and some folders  
##                path_split = pop.split("\\")
##                pop_file_name = path_split[-1]
##                year = pop_file_name.split("-")[1][0:4]
##                ssp_name = ssp
##
##                # MAKE NAME FOR THE FINAL FILE
##                final_file_name = ssp_name + "_" + year + "_total_population_latitude.dbf"
##
##                # Names of new final directory 
##                dbf_directory_name = year + "_dbf_tables"
##                csv_directory_name = year + "_csv_tables"                               
##
##                # Create dbf output directory
##                year_dbf_directory = create_directory(out_folder, dbf_directory_name)
##
##                # Create companion csv output directory to store csv files
##                year_csv_directory = create_directory(out_folder, csv_directory_name)
##
##                # Create a final directory for dbf files and csv files  
##                ssp_dbf_directory = create_directory(year_dbf_directory, ssp_name)
##                ssp_csv_directory = create_directory(year_csv_directory, ssp_name)
##
##                 
##                # WORK ON THESE PART OF CODE TOMORROW XX
##                print "Zonal statistics with %s and %s"%(graticule_file_name, pop_file_name)
##
##                # DO ZONAL STATISTICS
##                ZonalStatisticsAsTable(graticule_file_path, "Display", pop,  ssp_dbf_directory + "\\" + final_file_name, "DATA", "SUM")
##
##                print "Stored in: \t%s"% ssp_dbf_directory
##                print
##
##
##        print                
##
### end timer and get results 
##end = time.time()
##final_time_hours = round(((end - start)/(60*60)), 2)                      
##
##print "Finished processing %d files in %d hours"% (count, final_time_hours)
print "Finished: " + datetime.datetime.now().strftime("%A, %d %B %Y %I:%M.%S %p")
print
                
