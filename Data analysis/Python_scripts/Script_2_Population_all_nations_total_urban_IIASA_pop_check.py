# ---------------------------------------------------------------------------------   
# DESCRIPTION:  This scripts creates dbf population data files for each SSP.
#               Data include both the total population and urban population
#               by country for the decades 2010-2100.  Outputs are for both types of
#               urban land use models (GRUMP and GlobCover). The script was written to
#               check our spatial output national urban and total numbers with those
#               of IIASA.  
#                 
# DEVELOPER:    Peter J. Marcotullio
# DATE:         November 2017
# NOTES:        Uses python (os, datetime, time) and arcpy.
# ---------------------------------------------------------------------------------   

# IMPORT MODULES

import os, datetime, time
from Script_1_Population_Functions import *

# Annonce the start of the script
print "Start script at: ", datetime.datetime.now().strftime("%A, %d %B %Y %I:%M.%S %p")
print "File Location: " + cwd
print "File Name: " + os.path.basename(__file__)
print

cwd = os.getcwd()

# Start timer
start = time.time()

# SET IMPORTANT VARIABLES

number_file = 0
cwd = os.getcwd()
land_use = ["GRUMP", "GlobCover"]

# get national boundary map
nations_map = nation_borders_file()

# START THE PROCESS BY FIRST ITERATING THROUGH THE URBAN LAND USE TYPES

for land in land_use:

    # FIND THE SSP AND TABLE FOLDERS FOR THE URBAN LAND USE
    
    urban_folder = population_landuse_folder(land)
    ssp_folder_paths = ssp_population_paths(urban_folder)
    table_folder_paths = table_folders(ssp_folder_paths)

    # FIND THE POPULATION FILES AND THE URBAN ID FILES FOR THE URBAN LAND USE

    pop_files = population_files(ssp_folder_paths)
    ID_files = urbanID_files(ssp_folder_paths)

    # ITERATE THROUGH THE SSPS FOR THE URBAN LAND USE 

    for table in table_folder_paths: # control the ssp number here

        # Get ssp number from the path and create geodatabase name
        ssp = table.split("\\")[-2]
        gdb_name = "Nation_" + ssp + ".gdb"
        print "Start analysis of: ", land, ssp.split("_")[0], ssp.split("_")[1]
        print

        # MAKE NEW FOLDERS IN THE TABLES FOLDER AND PUT IN A GEODATABASE WITH THE APPROPRIATE NAME (IF NECESSARY)

        # THIS CAN BE CHANGED TO USE DEFINITIONS (SEE SCRIPT_3_POPULATION)
        # create new directory path
        new_directory = os.path.join(table, "nation")
        # create a new directory, if it already exists ignore
        if not os.path.exists(new_directory):
            os.makedirs(new_directory)
        # create a new geodatabase path
        gdb_path = os.path.join(new_directory, gdb_name)
        # Create a new geodatabase, if already exists ignore
        if not arcpy.Exists(gdb_path):
            # print "%s Doesn't exist"% gdb_name
            arcpy.CreateFileGDB_management(new_directory, gdb_name)
        #else:
            # print "%s exists here"% gdb_name  

        # CREATE NEW LISTS TO HOLD FILES FOR EACH SSP (POPULATION AND URBAN ID)

        ssp_pop_files = []
        for pop in pop_files:
            if ssp in pop:
                ssp_pop_files.append(pop)

        ssp_ID_files = []
        for ID_x in ID_files:
            if ssp in ID_x:
                ssp_ID_files.append(ID_x)

        # sort each list 
        ssp_pop_files.sort()   
        ssp_ID_files.sort()
        
        # ITERATE THROUGH THE LISTS OF POPLATION AND URBAN ID TOGETHER SIMULATIOUSLY BY YEAR
        
        for ssp_pop, ssp_ID in zip(ssp_pop_files, ssp_ID_files): # control the years here 
            #print ssp_pop, "\t", ssp_ID
            
            # DO THE ZONAL STATISTICS

            # Get the year from the path
            urban_id_file = ssp_ID.split("\\")[-1]
            year = urban_id_file.split("-")[-2]

            # Create names for the urban and total tables
            urban_table_name = "nation_urban_" + str(year)
            zonal_stat_path_urban = os.path.join(gdb_path, urban_table_name)
            nation_table_name = "nation_total_" + str(year)
            zonal_stat_path_total = os.path.join(gdb_path, nation_table_name)

            # Create names for field
            urban_field = "urbpop" + str(year)
            total_field = "totpop" + str(year)

            # Announce the process
            print "Zonal statistics for %s & %s"%(urban_table_name, nation_table_name)

            ZonalStatisticsAsTable (ssp_ID, "Value", ssp_pop, zonal_stat_path_urban, "DATA", "SUM")
            number_file += 1
            ZonalStatisticsAsTable (nations_map, "ISO_N3", ssp_pop, zonal_stat_path_total, "DATA", "SUM")
            number_file += 1

            print "Changing field names and deleting fields for urban pop table"
            arcpy.AlterField_management(zonal_stat_path_urban, "SUM", urban_field)
            arcpy.AlterField_management(zonal_stat_path_urban, "Value", "nation_id")
            final_urban = arcpy.DeleteField_management(zonal_stat_path_urban, drop_field="COUNT;AREA")

            print "Changing field names and deleting fields for total pop table" 
            arcpy.AlterField_management(zonal_stat_path_total, "SUM", total_field)
            arcpy.AddField_management(zonal_stat_path_total, "nation_id", "LONG")
            # Recalculate string to integer data 
            arcpy.CalculateField_management(zonal_stat_path_total, "nation_id", expression="myFunction( !ISO_N3!)", expression_type="PYTHON_9.3", code_block="def myFunction(value):\n    input = int(value)\n    return input")
            final_total = arcpy.DeleteField_management(zonal_stat_path_total, drop_field="ISO_N3;COUNT;ZONE_CODE;AREA")
            print 

        #  JOIN ALL THE TABLES IN THE GEODATABASE TOGETHER AND OUTPUT THE FINAL AS .DBF TABLE
        
        # Announce the next process     
        print "Join all tables for: ", land, ssp.split("_")[0], ssp.split("_")[1]
        print

        # put arcpy in gdb and list all the tables
        arcpy.env.workspace = gdb_path
        tlist = arcpy.ListTables()

        # SEPARATE TABLES (URBAN AND TOTAL), FIRST JOIN URBAN TOGETHER AND THEN TOTAL TO THEM, REMOVING THE EXTRA COLUMNS EACH TIME

        # Remove one table from the list to start the join process
        for t in tlist:
            if t.endswith('urban_2010'):
                final_table = t
                tlist.remove(t)

        # Separate the list into urban pop tables and total pop tables
        urban_tables = []
        total_tables = []
        for t in tlist:
            if "urban" in t:
                urban_tables.append(t)
            elif "total" in t:
                total_tables.append(t)
            else:
                print "Problem with the code"

        # reverse the order of the urban pop table list and join them all together (so that they end up (mostly) in order), delete the extra columns  
        urban_tables.reverse()
        for urban in urban_tables:
            final_table = arcpy.JoinField_management(in_data=urban, in_field="nation_id", join_table= final_table, join_field="nation_id")
            arcpy.DeleteField_management(final_table, drop_field="nation_id_1")

        # reverse the order of the total pop table list and join them all together with the urban tables,  (so that they end up (mostly) in order), delete the extra columns
        total_tables.reverse()
        for total in total_tables:
            final_table = arcpy.JoinField_management(in_data=total, in_field="nation_id", join_table= final_table, join_field="nation_id")
            arcpy.DeleteField_management(final_table, drop_field="nation_id_1")
                    
        # make name for final table to be put outside of gdb
        final_file_name = ssp + "_national_urban_and_total_pop_all_years.dbf"

        # move final table to outside geodatabase as ".dbf" file 
        arcpy.TableToTable_conversion(final_table, new_directory, final_file_name)

# We could convert the dbf to csv and change the final order of the columns, but not necessary as we do not use these data in any detailed analysis (they're just to check)

# end timer and get results 
end = time.time()
time_split = ((end - start)/60)

# Announce the ending 
print "Finished processing: ", number_file
print "That took: ", round(time_split, 2), " minutes"

