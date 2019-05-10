# -------------------------------------------------------------------------
# DESCRIPTION:  This script produces zonal statistic dbf data files by decade
#               by SSP for the IDed urban extents.  The data include population
#               by urban extent.  The files are first stored in a geodatabase
#               located in the "tables/urban/DBF" folder so they can be altered.
#               The files are then transfered outside the geodatabase as dbf
#               files.  These files in combination with the tables from the
#               associated ".shp" files provide spatially articulated national
#               urban populations.  
#
# DEVELOPER:    Peter J. Marcotullio
# DATE:         November 2017
# NOTES:        Uses python and arcpy.  
# --------------------------------------------------------------------------

# Import modules
import os, datetime, time, glob
import arcpy
from arcpy import env
arcpy.CheckOutExtension("Spatial")
from arcpy.sa import *
from Script_1_Population_Functions import *

# Announce the start of the script
print "Start script at: \t", datetime.datetime.now().strftime("%A, %d %B %Y %I:%M.%S %p")
print "File Location: \t\t" + cwd
print "File Name: \t\t" + os.path.basename(__file__)
print


# Start timer
start = time.time()

# SET IMPORTANT VARIABLES

number_file = 0
cwd = os.getcwd()
land_use = ["GRUMP", "GlobCover"]

# START THE PROCESS BY FIRST ITERATING THROUGH THE URBAN LAND USE TYPES

for land in land_use: # Control the land use models here

    # FIND THE SSP AND TABLE FOLDERS FOR THE URBAN LAND USE
    
    urban_folder = population_landuse_folder(land)
    ssp_folder_paths = ssp_population_paths(urban_folder)
    table_folder_paths = table_folders(ssp_folder_paths)

    # get the population raster data files
    population_data_paths = population_files(ssp_folder_paths)

    # get the urban ID vector folder paths
    urban_ID_vector_folder_paths = urbanID_vector_folders(ssp_folder_paths)

    # get the urban ID vector files 
    urban_ID_vector_paths = urbanID_vectors(ssp_folder_paths)

    for urban_ID in  urban_ID_vector_paths:
        print urban_ID
    print

    for table in table_folder_paths: # control the ssp number here
    
        ssp = table.split("\\")[-2]

        # Announcement
        print "We are now working in", land, ssp.split("_")[0], ssp.split("_")[1]
        print

        # Select only the urban vector folders in this ssp
        ssp_vector_folders = []
        for urban_vector_folder in urban_ID_vector_folder_paths:
            if ssp in urban_vector_folder:
                ssp_vector_folders.append(urban_vector_folder)

        for ssp_folder in ssp_vector_folders:
            print ssp_folder
        print
        
        # Select only the urban vectors in this ssp
        ssp_vector_files = []
        for urban_vector in urban_ID_vector_paths:
            if ssp in urban_vector:
                ssp_vector_files.append(urban_vector)

        for ssp_file in ssp_vector_files:
            print ssp_file
        print
        
        # Select only the population data for this ssp
        ssp_pop_files = []
        for pop_data in population_data_paths:
            if ssp in pop_data:
                ssp_pop_files.append(pop_data)

        for ssp_pop in ssp_pop_files:
            print ssp_pop
        print
        
        # Create directory and file geodatabase
        urban_table_path = create_directory(table, "urban")
        dbf_urban_table_path = create_directory(urban_table_path, "DBF")
        dbf_gdb = create_gdb(dbf_urban_table_path, "UrbanPop_" + ssp + ".gdb") 
        
        # Iterate through the vector list and the population list together 
        for vector, pop in zip(ssp_vector_files, ssp_pop_files):

            # Get the names of the files
            vector_file_name = vector.split("\\")[-1]
            population_file_name = pop.split("\\")[-1]

            # Create name and path for final file
            gdb_name = vector_file_name[:-6]
            gdb_path = os.path.join(dbf_gdb, gdb_name)

            # Announce the process
            print "Zonal statistics for %s & %s"%(vector_file_name, population_file_name)

            # Zonal statistics and add one to the number of files processes
            # ZonalStatisticsAsTable (vector, "urban_ID", pop, gdb_path, "DATA", "SUM")

            # add number to file 
            number_file += 1

        print
        print "Finished zonal statistics"
        print

        # LAST SECTION JOIN TABLES TO CHANGE THE NAME OF THE "SUM" VARIABLE, JOIN TO SHAPE FILE TABLE AND SAVE AS DBF 

        # This first part if funky, I make list of paths to the files using ListTables and glob.glob
        # put arcpy into gdb
        arcpy.env.workspace = dbf_gdb
        
        # Get the list of tables in the gdb
        tlist = arcpy.ListTables()

        # Get list of paths to these tables
        #t_paths = []
        fieldname = "Population"
        for t in tlist:
            dbf_name = t + "_pop.dbf"
            print "Saving table", dbf_name
            arcpy.AlterField_management(t, "SUM", fieldname)
            arcpy.DeleteField_management(t, ["COUNT", "AREA"])
            arcpy.TableToTable_conversion(t, dbf_urban_table_path, dbf_name)

    print

# end timer and get results 
end = time.time()
time_split = ((end - start)/(60*60))

# Announce the ending 
print "Finished processing: ", number_file
print "That took: ", round(time_split, 2), "hours"
print "The data and time are now", datetime.datetime.now().strftime("%A, %d %B %Y %I:%M.%S %p")
