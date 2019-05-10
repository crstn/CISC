# -------------------------------------------------------------------------
# DESCRIPTION:  First in a series. This file holds functions for the 3 month
#   temperature analysis.  The script runs off IDLE.  
#   
# DEVELOPER:    Peter J. Marcotullio
# DATE:         May 2017
# NOTES:        Uses python (os, sys, datetime, csv, glob, dbfpy) and arcpy.
#               Some computers do not have dbfpy installed!  
# --------------------------------------------------------------------------
# Import modules
import os, sys, datetime, glob, csv
from csv import DictReader
from dbfpy import dbf
import arcpy
from arcpy import env
arcpy.CheckOutExtension("Spatial")
from arcpy.sa import *


# Get current path and file name
cwd = os.getcwd()


# POPULATION FUNCTIONS

# Returns the path to Population folder of either "GRUMP" or "GlobCover" land use models
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

# Returns a list of paths to the various ssp in either the GRUMP or GlobCover land use Population data
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

# Returns the paths to the urban vector files in the Population SSPs folders by selected year
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

# Returns the paths to the urban vector files in the Population SSPs folders by selected year given new folder names
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



# Returns the paths to the urban vector files in the Population SSPs folders by selected SSP number
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



# Returns the paths to the new dbf files (December 2017)for the SSP for all years
def pop_dbf_urban_files(ssp_paths, ssp_number):

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
                    if d.endswith("DBF"):

                        # Get the directory path
                        tables_path = os.path.join(root, d)
                  
                        # put python in the directory
                        os.chdir(tables_path)

                        # walk through directories and files in the directory
                        for (root, dirs, filenames) in os.walk(tables_path):

                            # loop through the directories 
                            for f in filenames:

                                # find the csv files
                                if f.endswith(".dbf"):

                                    # create the path to the file
                                    csv_urban_file_path = os.path.join(root, f)
                                    
                                    # append the path to the original list
                                    csv_urban_file_paths.append(csv_urban_file_path)

    return csv_urban_file_paths





# TEMPERATURE FUNCTIONS

# Returns a list of the paths to the rcps given a year
def get_rcp_paths(rootDir, year_mean):

    rcp_paths = []

    # Create a list of everything that walk gives
    list_dirs = os.walk(rootDir)

    # Divide the entries into root, dirs and files
    for root, dirs, files in list_dirs:

        # Loop through the directories 
        for d in dirs:

            # Find those that end with the model name
            if d.endswith(year_mean):

                # Create a path
                path = os.path.join(root, d)

                # Append paths to list
                rcp_paths.append(path)

    # Sort list
    rcp_paths.sort()
    
    # Return the path
    return rcp_paths


def temp_file_paths(rcp_paths, rcp_model):

    # Create an empty list 
    paths_to_files = []

    for rcp in rcp_paths:

        if rcp_model in rcp:

            # put python in the directory
            os.chdir(rcp)

            # Create a list of everything that walk gives
            list_dirs = os.walk(rcp)

            # Loop through the entries root, dirs and files
            for root, dirs, files in list_dirs:

                # Loop through the directories
                for d in dirs:

                    # Find those that start with "SSP"
                    if d.endswith("MAX"):

                        # Create a path to the folder
                        max_path = os.path.join(root, d)

                        # Put python in the directory
                        os.chdir(max_path)

                        # put arcpy in the directory
                        arcpy.env.workspace = max_path

                        # List rasters
                        max_file = arcpy.ListRasters("*RS.tif")

                        # Loop through list to get files
                        for max_f in max_file:

                            # create path to the file
                            max_files_path = os.path.join(max_path, max_f)

                            # Append the path to the list
                            paths_to_files.append(max_files_path)

    # Return the final list of all paths to "SSPs"
    return paths_to_files


# TABLES FUNCITONS

# Returns the path to the temperature data folders with csv and dbf folders 
def find_tables_dir(rootDir, land_use_model_tables): # (i.e., cwd, "GRUMP_tables")
        
    # Create a list of everything that walk gives
    list_dirs = os.walk(rootDir)

    # Divide the entries into root, dirs and files
    for root, dirs, files in list_dirs:

        # Loop through the directories 
        for d in dirs:

            # Find those that end with the model name
            if d.endswith(land_use_model_tables):

                # Create a path
                path = os.path.join(root, d)

    # Return the path
    return path

# Returns the path to the population data folders with csv files by ssp folders 
def find_pop_tables_dir(rootDir, land_use_model_pop_tables):

    # Create a list of everything that walk gives
    list_dirs = os.walk(rootDir)

    # Divide the entries into root, dirs and files
    for root, dirs, files in list_dirs:

        # Loop through the directories 
        for d in dirs:

            # Find those that end with the model name
            if d.endswith(land_use_model_pop_tables):

                # Create a path
                path = os.path.join(root, d)

    # Return the path
    return path
   

def find_dbf_folders(tables_dir):

    # Create empty directory to hold the paths
    dbf_tables_dirs = []

    # find subfolders in tables directories
    list_subdirs = os.walk(tables_dir)

    # Loop through root, sub-folders & files
    for root, dirs, files in list_subdirs:

        # loop trhough sub-folders
        for di in dirs:

            # Find the RCP folder
            if di.startswith("RCP"):

                # make a path to the RCP folder
                rcp_path = os.path.join(tables_dir, di)

                # Move python into the folder
                os.chdir(rcp_path)

                # List the files and folders in this sub-directory
                dbf_tables = os.walk(rcp_path)

                # Loop through root, sub-folders & files                
                for root, dirs, files in dbf_tables:

                    # Loop through the sub-folders
                    for dd in dirs:

                        # find the folders that hold the dbf files
                        if dd.endswith("dbf_tables"):

                            # Create a path to these folders
                            dbf_tables_path = os.path.join(rcp_path, dd)

                            # append the folder paths to the folder path list 
                            dbf_tables_dirs.append(dbf_tables_path)

    return dbf_tables_dirs


def find_csv_folders(tables_dir):

    # Create empty directory to hold the paths
    csv_tables_dirs = []

    # find subfolders in tables directories
    list_subdirs = os.walk(tables_dir)

    # Loop through root, sub-folders & files
    for root, dirs, files in list_subdirs:

        # loop trhough sub-folders
        for di in dirs:

            # Find the RCP folder
            if di.startswith("RCP"):

                # make a path to the RCP folder
                rcp_path = os.path.join(tables_dir, di)

                # Move python into the folder
                os.chdir(rcp_path)

                # List the files and folders in this sub-directory
                csv_tables = os.walk(rcp_path)

                # Loop through root, sub-folders & files                
                for root, dirs, files in csv_tables:

                    # Loop through the sub-folders
                    for dd in dirs:

                        # find the folders that hold the dbf files
                        if dd.endswith("csv_tables"):

                            # Create a path to these folders
                            csv_tables_path = os.path.join(rcp_path, dd)

                            # append the folder paths to the folder path list 
                            csv_tables_dirs.append(csv_tables_path)

    return csv_tables_dirs

def get_dbf_files(dbf_table_directories):

    # Create a list to hold the paths to the files
    dbf_files =[]

    # loop through the list that holds the folders for dbf files
    for dbf_table in dbf_table_directories:

        # make a list of the files, folders and root in the folder
        list_dbf_files = os.walk(dbf_table)

        # loop through the root, files and directories
        for root, dirs, files in list_dbf_files:

            # loop through the directories
            for ddd in dirs:

                # make a path to the folder
                ssp_folder = os.path.join(dbf_table, ddd)

                # put arcpy in the folder
                arcpy.env.workspace = ssp_folder

                # list the dbf tables in the folder
                list_dbf_tables = arcpy.ListTables("*.dbf")

                # Loop through the list of dbf tables
                for list_dbf in list_dbf_tables:

                    # Create a path to the dbf table
                    dbf_file_path = os.path.join(ssp_folder, list_dbf)

                    # Append the path to the path list
                    dbf_files.append(dbf_file_path)

    # Return the full path list
    return dbf_files

def get_csv_files(csv_table_directories):

    # Create a list to hold the paths to the files
    csv_files =[]

    # loop through the list that holds the folders for dbf files
    for csv_table in csv_table_directories:

        # make a list of the files, folders and root in the folder
        list_csv_files = os.walk(csv_table)

        # loop through the root, files and directories
        for root, dirs, files in list_csv_files:

            # loop through the directories
            for ddd in dirs:

                # make a path to the folder
                ssp_folder = os.path.join(csv_table, ddd)

                # put arcpy in the folder
                arcpy.env.workspace = ssp_folder

                # list the dbf tables in the folder
                list_csv_tables = arcpy.ListTables("*.csv")

                # Loop through the list of dbf tables
                for list_csv in list_csv_tables:

                    # Create a path to the dbf table
                    csv_file_path = os.path.join(ssp_folder, list_csv)

                    # Append the path to the path list
                    csv_files.append(csv_file_path)

    # Return the full path list
    return csv_files


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

        # Get rcp from path
        rcp_number = table.split("\\")[3]

        # Get the year from path
        year_folder = table.split("\\")[4]
        year = year_folder[:4]

        # Create final output paths for csv files
        final_dbf_path = table.replace(dbf_fn, csv_fn)
        final_path = final_dbf_path.replace(dbf_fn_folder, csv_fn_folder)

        # Announce the process
        print "%s to %s: %s, %s"%(dbf_fn, csv_fn, rcp_number, year)
        
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
        

def change_dbf_to_csv_new(dbf_file_list, output_folder):

    # count the files
    file_number = 0
    
    # loop through the list
    for table in dbf_file_list:

        # get name for csv file replace the "dbf" string from path
        full_dbf_path = table.split("\\")
        dbf_fn = table.split("\\")[-1]
        csv_fn = table.split("\\")[-1].replace(".dbf", ".csv")

        # Create final output paths for csv files
        final_path = os.path.join(output_folder, csv_fn)

        # Announce the process
        print "%s to %s"%(dbf_fn, csv_fn)
        
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
     
# CREATE DIRECTORY

# creates a new directory if one is not already there, returns the path to the new directory
def create_directory(path, directory_name):

    # create new directory path
    new_directory_path = os.path.join(path, directory_name)

    # create a new directory, if it already exists ignore
    if not os.path.exists(new_directory_path):
        os.makedirs(new_directory_path)

    return new_directory_path                      

                        
