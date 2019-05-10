# -------------------------------------------------------------------------
# DESCRIPTION:  This file holds functions for the analysis of population and    
#               heat index data.  The script runs off IDLE.  
#
# DEVELOPER:        Peter J. Marcotullio
# DATE:             July 2017
# LAST AMENDMENT:   March 2018
# NOTES:            Uses python (os, sys, datetime, glob, csv, dbfpy) and arcpy.
#                   Note that on some computers the dbfpy is not installed!  
# --------------------------------------------------------------------------
# Import modules
import os, sys, datetime, glob, csv
from csv import DictReader
# from dbfpy import dbf
import arcpy
from arcpy import env
arcpy.CheckOutExtension("Spatial")
from arcpy.sa import *

# Get current path and file name
cwd = os.getcwd()


# HEAT INDEX FUNCTIONS

#returns all the paths to folders with output models for a specific rcp
def heat_index_rcp_folders(current_direct, rcp_number):

    list_rcp_paths = []
    list_dirs = os.walk(current_direct)

    for root, dirs, files in list_dirs:

        for d in dirs:

            if d.endswith(rcp_number):

                folder_path = os.path.join(root, rcp_number)

                list_rcp_paths.append(folder_path)
                
    return list_rcp_paths


#returns all the paths to file output models for a specific rcp
def heat_index_rcp_files(current_direct, rcp_number):

    list_rcp_file_paths = []
    list_dirs = os.walk(current_direct)

    for root, dirs, files in list_dirs:

        for d in dirs:

            if d.endswith(rcp_number):

                folder_path = os.path.join(root, rcp_number)

                arcpy.env.workspace = folder_path

                rasters = arcpy.ListRasters()

                for raster in rasters:

                    file_path = os.path.join(folder_path, raster)

                    list_rcp_file_paths.append(file_path)
                

    return list_rcp_file_paths

# returns the paths for a specific Heat Index model
def heat_index_model_folders(current_direct, rcp_model):

    rcp_folders = ["RCP2p6", "RCP4p5", "RCP6p0", "RCP8p5"]
    list_model_rcp_paths=[]
    list_dirs = os.walk(current_direct)

    for root, dirs, files in list_dirs:

        for d in dirs:

            if d.startswith(rcp_model):

                model_folder_path = os.path.join(root, d)

                for rcp in rcp_folders:

                    model_rcp_path = os.path.join(model_folder_path, rcp)

                    list_model_rcp_paths.append(model_rcp_path)

    return list_model_rcp_paths

# Returns a re-sampled raster file 
def re_sample(in_file, out_file, size):

    new_file = arcpy.Resample_management(in_file, out_file, size, "CUBIC")

    return new_file

# returns a list of rcp folder paths that hold HI data 
def find_HI_tables_dir(cwd, list_of_rcp_folders):

    paths_to_rcp_HI_tables = []
    
    for rcp in list_of_rcp_folders:

        list_dirs = os.walk(cwd)

        for root, dirs, files in list_dirs:

            for d in dirs:

                if d == rcp:

                    path = os.path.join(root, d)

                    paths_to_rcp_HI_tables.append(path)

    return paths_to_rcp_HI_tables




  
# FINDING LAND USE MODEL FUNCTIONS




# To find path to Population folder of either "GRUMP" or "GlobCover"
def lc_model_path(rootDir, model_name):

    # Create a list of everything that walk gives
    list_dirs = os.walk(rootDir)

    # Divide the entries into root, dirs and files
    for root, dirs, files in list_dirs:

        # Loop through the directories 
        for d in dirs:

            # Find those that end with the model name
            if d == model_name:

                # Create a path
                path = os.path.join(root, d)

    # Return the path
    return path



# FINDING SSP FOLDERS AND SSP POPULATION FILES



# To create a list of paths to the various ssp in either the GRUMP or GlobCover Population data
def ssp_paths(model_path):

    # Create an empty list 
    paths = []

    # Create a list of everything that walk gives
    list_dirs = os.walk(model_path)

    # Loop through the entries root, dirs and files
    for root, dirs, files in list_dirs:

        # Loop through the directories
        for d in dirs:

            # Find those that start with "SSP"
            if d.startswith("SSP"):

                # Create a path to the folder
                path = os.path.join(root, d)

                # Append the path to the list
                paths.append(path)

    # Return the final list of all paths to "SSPs"
    return paths

# Returns the paths to the urban vector files in the SSPs by year
def ssp_urban_files_year(ssp_paths, year):

    urban_file_paths = []

    # Loop through the ssp_paths
    for ssp in ssp_paths:

        # Make a list of the directories in this folder
        list_dirs = os.walk(ssp)

        # identify all the files & folders
        for root, dirs, files in list_dirs:


            # Loop through the folders                
            for d in dirs:

                # Find the folder that ends with "Rural"
                if d.endswith("Rural"):

                        # Make a path to that folder
                        urbanRural_path = os.path.join(root, d)

                        # Put python in that folder
                        os.chdir(urbanRural_path)

                        # Put arcpy in that folder
                        arcpy.env.workspace = urbanRural_path

                        # Make a list of the feature classes
                        urban_vectors = arcpy.ListFeatureClasses()

                        # loop through the list
                        for urban in urban_vectors:

                                # Select the files with the specific year
                                if year in urban:

                                    # get the path to the vector
                                    vector_path = os.path.join(urbanRural_path, urban)

                                    # append the path to the list of the SSP
                                    urban_file_paths.append(vector_path)    

    return urban_file_paths

# Returns the paths to the urban vector files in the SSPs by year
def ssp_urban_files_year_new(ssp_paths, year):

    urban_file_paths = []

    # Loop through the ssp_paths
    for ssp in ssp_paths:

        # Make a list of the directories in this folder
        list_dirs = os.walk(ssp)

        # identify all the files & folders
        for root, dirs, files in list_dirs:


            # Loop through the folders                
            for d in dirs:

                # Find the folder that ends with "Rural"
                if d == "urbanNationID_vectors":

                        # Make a path to that folder
                        urbanID_path = os.path.join(root, d)

                        # Put python in that folder
                        os.chdir(urbanID_path)

                        # Put arcpy in that folder
                        arcpy.env.workspace = urbanID_path

                        # Make a list of the feature classes
                        urban_vectors = arcpy.ListFeatureClasses()

                        # loop through the list
                        for urban in urban_vectors:

                                # Select the files with the specific year
                                if year in urban:

                                    # get the path to the vector
                                    vector_path = os.path.join(urbanID_path, urban)

                                    # append the path to the list of the SSP
                                    urban_file_paths.append(vector_path)    

    return urban_file_paths

# Returns the urban files in the Population SSPs files by SSP number
def ssp_urban_files_number(ssp_paths, ssp_number):

    urban_file_paths= []

    # Loop through the ssp_paths
    for ssp in ssp_paths:

        # Find the specific ssp model
        if ssp.endswith(ssp_number):

            # Make a list of the directories in this folder
            list_dirs = os.walk(ssp)

            # identify all the files & folders
            for root, dirs, files in list_dirs:

                # Loop through the folders                
                for d in dirs:

                    # Find the folder that ends with "Rural"
                    if d.endswith("Rural"):

                        # Make a path to that folder
                        urbanRural_path = os.path.join(root, d)

                        # Put python in that folder
                        os.chdir(urbanRural_path)

                        # Put arcpy in that folder
                        arcpy.env.workspace = urbanRural_path

                        # Make a list of the feature classes
                        urban_vectors = arcpy.ListFeatureClasses()

                        # loop through the list
                        for urban in urban_vectors:

                            # get the path to the vector
                            vector_path = os.path.join(urbanRural_path, urban)

                            # append the path to the list of the SSP
                            urban_file_paths.append(vector_path)                        

    return urban_file_paths

# Returns the urban files in the Population SSPs files by SSP number
def ssp_urban_files_number_new(ssp_paths, ssp_number):

    urban_file_paths= []

    # Loop through the ssp_paths
    for ssp in ssp_paths:

        # Find the specific ssp model
        if ssp_number in ssp:

            # Make a list of the directories in this folder
            list_dirs = os.walk(ssp)

            # identify all the files & folders
            for root, dirs, files in list_dirs:

                # Loop through the folders                
                for d in dirs:

                    # Find the folder that ends with "Rural"
                    if d == "urbanNationID_vectors":

                        # Make a path to that folder
                        urban_vectors_path = os.path.join(root, d)

                        # Put python in that folder
                        os.chdir(urban_vectors_path)

                        # Put arcpy in that folder
                        arcpy.env.workspace = urban_vectors_path

                        # Make a list of the feature classes
                        urban_vectors = arcpy.ListFeatureClasses()

                        # loop through the list
                        for urban in urban_vectors:

                            # get the path to the vector
                            vector_path = os.path.join(urban_vectors_path, urban)

                            # append the path to the list of the SSP
                            urban_file_paths.append(vector_path)                        

    return urban_file_paths



# FUNCTIONS FOR DBF AND CSV FOLDERS AND FILES (OUTPUTS FROM ZONAL STATISTICS)



# FIND DBF FOLDERS AND FILES 



# Returns all the paths to four hundrd dbf files 
def find_HI_dbf_files(paths_to_HI_folders):

    ssps = ["SSP_1", "SSP_2", "SSP_3", "SSP_4", "SSP_5"]

    dbf_table_files = []

    for path in paths_to_HI_folders:

        list_dirs = os.walk(path)

        for root, dirs, files in list_dirs:

            for d in dirs:

                if "dbf_tables" in d:

                    path = os.path.join(root, d)

                    os.chdir(path)

                    for ssp in ssps:

                        new_path = os.path.join(path, ssp)

                        arcpy.env.workspace = new_path

                        list_dbf_tables = arcpy.ListTables("*.dbf")

                        for list_dbf in list_dbf_tables:

                            dbf_file_path = os.path.join(new_path, list_dbf)

                            dbf_table_files.append(dbf_file_path)

    return dbf_table_files

# returns a list of dbf folder paths that hold data for the climate analysis data 
def find_dbf_tables_climate_dir(heat_index_folder):

    # Create empty list
    paths_to_climate_table_folders = []

    # List the directories in the heat index folder
    list_dirs = os.walk(heat_index_folder)

    # Iterate through the list of directories to find the "climate_2010" folder
    for root, dirs, files in list_dirs:

        for d in dirs:

            if d == "Climate_2010":

                # Get path to the folder
                path_to_climate = os.path.join(root, d)

                # Create a list of directories in this folder 
                list_dirs_climate = os.walk(path_to_climate)

                # Iterate through these directories and get only those with "dbf" in the name 
                for root, dirs, files in list_dirs_climate:

                    for d in dirs:
                        
                        if "dbf" in d:

                            # create a path to the dbf folders and append the path to the list of climate table folders 
                            path = os.path.join(root, d)
                            
                            paths_to_climate_table_folders.append(path)

    return paths_to_climate_table_folders

# returns a list of dbf folder paths that hold data for the climate analysis data 
def find_dbf_tables_latitude_dir(heat_index_folder):

    # Create empty list
    paths_to_latitude_table_folders = []

    # List the directories in the heat index folder
    list_dirs = os.walk(heat_index_folder)

    # Iterate through the list of directories to find the "climate_2010" folder
    for root, dirs, files in list_dirs:

        for d in dirs:

            if d.endswith("Latitude"):

                # Get path to the folder
                path_to_latitude = os.path.join(root, d)

                # Create a list of directories in this folder 
                list_dirs_climate = os.walk(path_to_latitude)

                # Iterate through these directories and get only those with "dbf" in the name 
                for root, dirs, files in list_dirs_climate:

                    for d in dirs:
                        
                        if "dbf" in d:

                            # create a path to the dbf folders and append the path to the list of climate table folders 
                            path = os.path.join(root, d)
                            
                            paths_to_latitude_table_folders.append(path)

    return paths_to_latitude_table_folders

# Returns all the paths to 1200 dbf files for each land use model
def find_dbf_climate_files(paths_dbf_climate_folders):

    ssps = ["SSP_1", "SSP_2", "SSP_3", "SSP_4", "SSP_5"]

    dbf_table_files = []

    for path in paths_dbf_climate_folders:

        list_dirs = os.walk(path)

        for root, dirs, files in list_dirs:

            for d in dirs:

               for ssp in ssps:

                    if ssp in d:
                        
                        path = os.path.join(root, ssp)

                        arcpy.env.workspace = path

                        list_dbf_tables = arcpy.ListTables("*.dbf")

                        for list_dbf in list_dbf_tables:

                            dbf_file_path = os.path.join(path, list_dbf)

                            dbf_table_files.append(dbf_file_path)

    return dbf_table_files

# Returns all the paths to 50 dbf files for each land use model
def find_dbf_latitude_files(paths_dbf_latitude_folders):

    ssps = ["SSP_1", "SSP_2", "SSP_3", "SSP_4", "SSP_5"]

    dbf_table_files = []

    for path in paths_dbf_latitude_folders:

        list_dirs = os.walk(path)

        for root, dirs, files in list_dirs:

            for d in dirs:

               for ssp in ssps:

                    if ssp in d:
                        
                        path = os.path.join(root, ssp)

                        arcpy.env.workspace = path

                        list_dbf_tables = arcpy.ListTables("*.dbf")

                        for list_dbf in list_dbf_tables:

                            dbf_file_path = os.path.join(path, list_dbf)

                            dbf_table_files.append(dbf_file_path)

    return dbf_table_files

# returns a list of dbf folder paths that hold data for the urban latitude analysis 
def find_dbf_urban_latitude_dir(heat_index_folder):

    # Create empty list
    paths_to_urban_latitude_folders = []

    # List the directories in the heat index folder
    list_dirs = os.walk(heat_index_folder)

    # Iterate through the list of directories to find the "climate_2010" folder
    for root, dirs, files in list_dirs:

        for d in dirs:

            if d.endswith("Latitude"):

                # Get path to the folder
                path_to_latitude = os.path.join(root, d)

                # Create a list of directories in this folder 
                list_dirs_climate = os.walk(path_to_latitude)

                # Iterate through these directories and get only those with "dbf" in the name 
                for root, dirs, files in list_dirs_climate:

                    for d in dirs:
                        
                        if d == "dbf_files" in d:

                            # create a path to the dbf folders and append the path to the list of climate table folders 
                            path = os.path.join(root, d)
                            
                            paths_to_urban_latitude_folders.append(path)

    return paths_to_urban_latitude_folders

# Returns all the paths to 50 urban population latitude dbf files for each land use model
def find_dbf_urban_latitude_files(paths_dbf_latitude_folders):

    dbf_urban_files = []

    for path in paths_dbf_latitude_folders:

        arcpy.env.workspace = path

        list_dbf_tables = arcpy.ListTables("*.dbf")

        for list_dbf in list_dbf_tables:

            dbf_file_path = os.path.join(path, list_dbf)

            dbf_urban_files.append(dbf_file_path)

    return dbf_urban_files



# CHANGE DBF TO CSV FILES 



# Changes the dbf files to csv files 
def change_dbf_to_csv(dbf_file_list):

    # count the files
    file_number = 0
    
    # loop through the list
    for table in dbf_file_list:

        # get name for csv file replace the "dbf" string from path
        full_dbf_path = table.split("\\")
        dbf_fn = table.split("\\")[-1]
        csv_fn = table.split("\\")[-1].replace(".dbf", ".csv")

        # get the name ofthe csv folder, replace the "dbf" string from path
        dbf_fn_folder = table.split("\\")[-3]
        dbf_fn_folder = table.split("\\")[-3]
        csv_fn_folder = dbf_fn_folder.replace("dbf", "csv")

        # Create final output paths for csv files
        final_dbf_path = table.replace(dbf_fn, csv_fn)
        final_path = final_dbf_path.replace(dbf_fn_folder, csv_fn_folder)

        # Announce the process
        # print "%s to %s"%(dbf_fn, csv_fn) 
        
        # open a new csv file with new name
        with open(final_path,'wb') as csvfile:

            # Get the dbf file
            in_db = dbf.Dbf(table)

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

            file_number += 1

    # return file number
    return file_number


# Changes the dbf files to csv files 
def change_dbf_to_csv_climate(dbf_file_list):

    # count the files
    file_number = 0
    
    # loop through the list
    for table in dbf_file_list:

        # get name for csv file replace the "dbf" string from path
        full_dbf_path = table.split("\\")
        dbf_fn = table.split("\\")[-1]
        csv_fn = table.split("\\")[-1].replace(".dbf", ".csv")

        # get the name of the csv folder, replace the "dbf" string from path
        dbf_fn_folder = table.split("\\")[-3]
        csv_fn_folder = dbf_fn_folder.replace("dbf", "csv")

        # Create final output paths for csv files
        final_dbf_path = table.replace(dbf_fn, csv_fn)
        final_path = final_dbf_path.replace(dbf_fn_folder, csv_fn_folder)

        # Announce the process
        print "Final path is %s"%(final_path) 
        
        # open a new csv file with new name
        with open(final_path,'wb') as csvfile:

            # Get the dbf file
            in_db = dbf.Dbf(table)

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

            file_number += 1

    # return file number
    return file_number

# Changes the dbf files to csv files 
def change_dbf_to_csv_urban_lat(dbf_file_list, final_folder):

    # count the files
    file_number = 0
    
    # loop through the list
    for table in dbf_file_list:

        # get name for csv file replace the "dbf" string from path
        full_dbf_path = table.split("\\")
        dbf_fn = table.split("\\")[-1]
        csv_fn = dbf_fn.replace(".dbf", ".csv")

        # Create final output paths for csv files
        final_path = os.path.join(final_folder, csv_fn)
 
        # Announce the process
        print "Final path is %s"%(final_path) 
        
        # open a new csv file with new name
        with open(final_path,'wb') as csvfile:

            # Get the dbf file
            in_db = dbf.Dbf(table)

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

            file_number += 1

    # return file number
    return file_number



# FIND CSV FILES



# Returns the paths to the csv files for a specific SSP for all years
def pop_csv_urban_files(ssp_paths, ssp_number):

    # make that variable an empty list
    csv_urban_file_paths = []

    # sort the paths
    ssp_paths.sort()
    
    # loop through paths
    for ssp in ssp_paths:

        # find the ssp of interest
        if ssp.endswith(ssp_number):
            
            # Get name of ssp in case we need it?
            ssp_name = ssp.split("\\")[-1]
            
            # make a list of the directories in the folder
            dirs_list = os.walk(ssp)

            # identify all the files & folders
            for root, dirs, files in dirs_list:

                # Loop through the directories 
                for d in dirs:
                    
                    # Find those that end with the model name
                    if d.endswith("tables"):

                        # Get the directory path
                        tables_path = os.path.join(ssp, d)
                  
                        # put python in the directory
                        os.chdir(tables_path)

                        # walk through directories and files in the directory
                        for (root, dirs, filenames) in os.walk(tables_path):

                            # loop through the directories 
                            for d in dirs:

                                # find directories 
                                if d.endswith("urban"):

                                    # Get teh path to the directory
                                    csv_urban_path = os.path.join(tables_path, d)
                                    
                                    # put python in the directory 
                                    os.chdir(csv_urban_path)

                                    # walk through the directories, files 
                                    for (root,dirs,filenames) in os.walk(csv_urban_path):

                                        # look only for files
                                        for f in filenames:

                                            # find the csv files
                                            if f.endswith("urban.csv"):

                                                # create the path to the file
                                                csv_urban_file_path = os.path.join(csv_urban_path, f)
                                                
                                                # append the path to the original list
                                                csv_urban_file_paths.append(csv_urban_file_path)

    return csv_urban_file_paths


# Returns all the paths to four hundrd csv files 
def find_HI_csv_files(paths_to_HI_folders):

    ssps = ["SSP_1", "SSP_2", "SSP_3", "SSP_4", "SSP_5"]

    csv_table_files = []

    for path in paths_to_HI_folders:

        list_dirs = os.walk(path)

        for root, dirs, files in list_dirs:

            for d in dirs:

                if "csv_tables" in d:

                    path = os.path.join(root, d)

                    os.chdir(path)

                    for ssp in ssps:

                        new_path = os.path.join(path, ssp)

                        arcpy.env.workspace = new_path

                        list_csv_tables = arcpy.ListTables("*.csv")

                        for list_csv in list_csv_tables:

                            csv_file_path = os.path.join(new_path, list_csv)

                            csv_table_files.append(csv_file_path)

    return csv_table_files


# Returns a list of csv folder paths that hold data for the climate analysis data 
def find_csv_tables_climate_dir(heat_index_folder):

    # Create empty list
    paths_to_climate_table_folders = []

    # List the directories in the heat index folder
    list_dirs = os.walk(heat_index_folder)

    # Iterate through the list of directories to find the "climate_2010" folder
    for root, dirs, files in list_dirs:

        for d in dirs:

            if d == "Climate_2010":

                # Get path to the folder
                path_to_climate = os.path.join(root, d)

                # Create a list of directories in this folder 
                list_dirs_climate = os.walk(path_to_climate)

                # Iterate through these directories and get only those with "dbf" in the name 
                for root, dirs, files in list_dirs_climate:

                    for d in dirs:
                        
                        if "csv" in d:

                            # create a path to the dbf folders and append the path to the list of climate table folders 
                            path = os.path.join(root, d)
                            
                            paths_to_climate_table_folders.append(path)

    return paths_to_climate_table_folders



# Returns all the paths to 1200 csv files for each land use model
def find_csv_climate_files(paths_csv_climate_folders):

    ssps = ["SSP_1", "SSP_2", "SSP_3", "SSP_4", "SSP_5"]

    csv_table_files = []

    for path in paths_csv_climate_folders:

        list_dirs = os.walk(path)

        for root, dirs, files in list_dirs:

            for d in dirs:

               for ssp in ssps:

                    if ssp in d:
                        
                        path = os.path.join(root, ssp)

                        arcpy.env.workspace = path

                        list_csv_tables = arcpy.ListTables("*.csv")

                        for list_csv in list_csv_tables:

                            csv_file_path = os.path.join(path, list_csv)

                            csv_table_files.append(csv_file_path)

    return csv_table_files


# Returns a list of csv folder paths that hold data for the latitude analysis data 
def find_csv_tables_latitude_dir(heat_index_folder):

    # Create empty list
    paths_to_latitude_table_folders = []

    # List the directories in the heat index folder
    list_dirs = os.walk(heat_index_folder)

    # Iterate through the list of directories to find the "climate_2010" folder
    for root, dirs, files in list_dirs:

        for d in dirs:

            if d.endswith("Latitude"):

                # Get path to the folder
                path_to_latitude = os.path.join(root, d)

                # Create a list of directories in this folder 
                list_dirs_latitude = os.walk(path_to_latitude)

                # Iterate through these directories and get only those with "dbf" in the name 
                for root, dirs, files in list_dirs_latitude:

                    for d in dirs:
                        
                        if "csv" in d:

                            # create a path to the dbf folders and append the path to the list of climate table folders 
                            path = os.path.join(root, d)
                            
                            paths_to_latitude_table_folders.append(path)

    return paths_to_latitude_table_folders

# Returns all the paths to 100 csv files for each land use model
def find_csv_latitude_files(paths_csv_latitude_folders):

    ssps = ["SSP_1", "SSP_2", "SSP_3", "SSP_4", "SSP_5"]

    csv_table_files = []

    for path in paths_csv_latitude_folders:

        list_dirs = os.walk(path)

        for root, dirs, files in list_dirs:

            for d in dirs:

               for ssp in ssps:

                    if ssp in d:
                        
                        path = os.path.join(root, ssp)

                        arcpy.env.workspace = path

                        list_csv_tables = arcpy.ListTables("*.csv")

                        for list_csv in list_csv_tables:

                            csv_file_path = os.path.join(path, list_csv)

                            csv_table_files.append(csv_file_path)

    return csv_table_files




# returns a list of dbf folder paths that hold data for the urban latitude analysis 
def find_csv_urban_latitude_dir(heat_index_folder):

    # Create empty list
    paths_to_urban_latitude_folders = []

    # List the directories in the heat index folder
    list_dirs = os.walk(heat_index_folder)

    # Iterate through the list of directories to find the "climate_2010" folder
    for root, dirs, files in list_dirs:

        for d in dirs:

            if d.endswith("Latitude"):

                # Get path to the folder
                path_to_latitude = os.path.join(root, d)

                # Create a list of directories in this folder 
                list_dirs_climate = os.walk(path_to_latitude)

                # Iterate through these directories and get only those with "dbf" in the name 
                for root, dirs, files in list_dirs_climate:

                    for d in dirs:
                        
                        if d == "csv_files" in d:

                            # create a path to the dbf folders and append the path to the list of climate table folders 
                            path = os.path.join(root, d)
                            
                            paths_to_urban_latitude_folders.append(path)

    return paths_to_urban_latitude_folders

# Returns all the paths to 50 urban population latitude csv files for each land use model
def find_csv_urban_latitude_files(paths_csv_latitude_folders):

    csv_urban_files = []

    for path in paths_csv_latitude_folders:

        arcpy.env.workspace = path

        list_csv_tables = arcpy.ListTables("*.csv")

        for list_csv in list_csv_tables:

            csv_file_path = os.path.join(path, list_csv)

            csv_urban_files.append(csv_file_path)

    return csv_urban_files





# CREATE DIRECTORY




# creates a new directory if one is not already there, returns the path to the new directory
def create_directory(path, directory_name):

    # create new directory path
    new_directory_path = os.path.join(path, directory_name)

    # create a new directory, if it already exists ignore
    if not os.path.exists(new_directory_path):
        os.makedirs(new_directory_path)

    return new_directory_path                      

                        

                        

                    

                    

                    
    

        


 
