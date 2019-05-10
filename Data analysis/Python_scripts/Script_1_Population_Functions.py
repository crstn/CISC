# -------------------------------------------------------------------------
# DESCRIPTION:  First in a series. Contains the functions that are used
#               often in working with population data.  
#
# DEVELOPER:    Peter J. Marcotullio
# DATE:         Novmeber 2017
# NOTES:        Uses python and arcpy.  
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


# FIND GlobCover OR GRUMP FOLDERS  

#returns the path to Globcover or GRUMP population folders 
def population_landuse_folder(land_use_model):

    list_dirs = os.walk(cwd)

    for root, dirs, files in list_dirs:

        for d in dirs:

            if d == land_use_model:

                folder_path = os.path.join(root, land_use_model)
              
    return folder_path
# CHECK
##urban_folder = population_landuse_folder("GRUMP")
##print urban_folder
##urban_folder = population_landuse_folder("GlobCover")
##print urban_folder


# FIND THE NATIONAL BOUNDARY MAP

# Returns the path to the nations border map
def nation_borders_file():

    list_dirs = os.walk(cwd)

    for root, dirs, files in list_dirs:

        for name in files:

            if name == "ne_10m_admin_0_countries_USE.shp":

                folder_path = os.path.abspath(os.path.join(root, name))

    return folder_path
# CHECK
##nations_map = nation_borders_file()
##print nations_map



# FIND SSP FOLDERS IN GlobCover OR GRUMP POPULATION FOLDERS

# Returns the list of SSP folder paths for GlobCover or GRUMP population data
def ssp_population_paths(model_path):

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
# CHECK
##urban_folder = population_landuse_folder("GRUMP")
##ssp_folder_paths = ssp_population_paths(urban_folder)
##for ssp_folders in ssp_folder_paths:
##    print ssp_folders 
##print    
##urban_folder = population_landuse_folder("GlobCover")
##ssp_folder_paths = ssp_population_paths(urban_folder)
##for ssp_folders in ssp_folder_paths:
##    print ssp_folders 
##print    


# FIND TABLES FOLDER IN THE SSP FOLDERS

# Returns a list of paths to the tables folders for each SSP folder

def table_folders(ssp_folders_list):

    tables_folders = []

    # walk through the ssp folders one at a time
    for ssp_folder in ssp_folders_list:

        # Create list of the directors in the folder
        list_dirs = os.walk(ssp_folder)

        # find the directories names "population"
        for root, dirs, files in list_dirs:
            for d in dirs:
                if d == "tables":

                    # create a path to the "population" folder
                    tables_folder_path = os.path.join(root, d)
                    # append this to the list
                    tables_folders.append(tables_folder_path)

    return tables_folders 
### CHECK
##urban_folder = population_landuse_folder("GRUMP")
##ssp_folder_paths = ssp_population_paths(urban_folder)
##table_folder_paths = table_folders(ssp_folder_paths)
##table_folder_paths.sort()
##for table in table_folder_paths:
##    print table                   
##print
##urban_folder = population_landuse_folder("GlobCover")
##ssp_folder_paths = ssp_population_paths(urban_folder)
##table_folder_paths = table_folders(ssp_folder_paths)
##table_folder_paths.sort()
##for table in table_folder_paths:
##    print table                   
##print

# RETURNS A LIST OF PATHS TO "population" FOLDERS 

def population_folders(ssp_folders_list):

    population_folders = []

    # walk through the ssp folders one at a time
    for ssp_folder in ssp_folders_list:

        # Create list of the directors in the folder
        list_dirs = os.walk(ssp_folder)

        # find the directories names "population"
        for root, dirs, files in list_dirs:
            for d in dirs:
                if d == "population":

                    # create a path to the "population" folder
                    population_folder_path = os.path.join(root, d)
                    # append this to the list
                    population_folders.append(population_folder_path)

    return population_folders 


# RETURNS THE LIST OF PATHS TO 'urban_rasters" FOLDERS

def urban_raster_folders(population_folders_list):

    urban_rasters_folders = []

    # walk through the ssp folders one at a time
    for pop_folder in population_folders_list:

        # Create list of the directors in the folder
        list_dirs = os.walk(pop_folder)

        # find the directories names "population"
        for root, dirs, files in list_dirs:
            for d in dirs:
                if d == "urban_rasters":

                    # create a path to the "population" folder
                    urban_raster_folder_path = os.path.join(root, d)
                    # append this to the list
                    urban_rasters_folders.append(urban_raster_folder_path)

    return urban_rasters_folders 

# LIST ALL URBAN RASTER FILES PER SSP IN GlobCover OR GRUMP POPULATION FOLDERS

# Returns a list of population files 
def urban_population_files(ssp_folders_list):
    # create an empty list
    urban_population_files = []

    # walk through the ssp folders one at a time
    for ssp_folder in ssp_folders_list:

        # Create list of the directors in the folder
        list_dirs = os.walk(ssp_folder)

        # find the directories names "population"
        for root, dirs, files in list_dirs:
            for d in dirs:
                if d == "urban_rasters":

                    # create a path to the "population" folder
                    urban_population_folder_path = os.path.join(root, d)

                    # Put arcpy in that folder
                    arcpy.env.workspace = urban_population_folder_path

                    # list all the rasters in that folder
                    urban_population_rasters = arcpy.ListRasters()

                    # add each of these raster paths to the population file list
                    for raster in urban_population_rasters:
                        # Get the path to the file and append it to the population file list
                        raster_path = os.path.join(urban_population_folder_path, raster)
                        urban_population_files.append(raster_path)

    # return the populatoin file list
    return urban_population_files



# LIST ALL POPULATION RASTER FILES PER SSP IN GlobCover OR GRUMP POPULATION FOLDERS

# Returns a list of population files 
def population_files(ssp_folders_list):
    # create an empty list
    population_files = []

    # walk through the ssp folders one at a time
    for ssp_folder in ssp_folders_list:

        # Create list of the directors in the folder
        list_dirs = os.walk(ssp_folder)

        # find the directories names "population"
        for root, dirs, files in list_dirs:
            for d in dirs:
                if d == "population":

                    # create a path to the "population" folder
                    population_folder_path = os.path.join(root, d)

                    # Put arcpy in that folder
                    arcpy.env.workspace = population_folder_path

                    # list all the rasters in that folder
                    population_rasters = arcpy.ListRasters()

                    # add each of these raster paths to the population file list
                    for raster in population_rasters:
                        # Get the path to the file and append it to the population file list
                        raster_path = os.path.join(population_folder_path, raster)
                        population_files.append(raster_path)

    # return the populatoin file list
    return population_files
### CHECK
##urban_folder = population_landuse_folder("GRUMP")
##ssp_folder_paths = ssp_population_paths(urban_folder)
##pop_files = population_files(ssp_folder_paths)
##pop_files.sort()
##for pop in pop_files:
##    print pop
##print    
##urban_folder = population_landuse_folder("GlobCover")
##ssp_folder_paths = ssp_population_paths(urban_folder)
##pop_files = population_files(ssp_folder_paths)
##pop_files.sort()
##for pop in pop_files:
##    print pop
##print    



# LIST ALL URBAN ID RASTER FILES PER SSP IN GlobCover OR GRUMP POPULATION FOLDERS

# Returns a list of rasters with national IDs 
def urbanID_files(ssp_folders_list):

    # create an empty list for urban national ID files
    urbanID_files = []

    # walk through the ssp folders one at a time
    for ssp_folder in ssp_folders_list:

        # Create list of the directors in the folder
        list_dirs = os.walk(ssp_folder)

        # find the directories names "urbanNationID"
        for root, dirs, files in list_dirs:
            for d in dirs:
                if d == "urbanNationID":

                    # create a path to the "urbanNationID" folder
                    urbanID_folder_path = os.path.join(root, d)

                    # Put arcpy in that folder
                    arcpy.env.workspace = urbanID_folder_path

                    # list all the rasters in that folder
                    urbanID_rasters = arcpy.ListRasters()

                    # add each of these raster paths to the urban national ID file list
                    for raster in urbanID_rasters:
                        # Get the path to the file and append it to the urban national ID file list
                        raster_path = os.path.join(urbanID_folder_path, raster)
                        urbanID_files.append(raster_path)

    # return the urban national ID file list
    return urbanID_files
# CHECK
##urban_folder = population_landuse_folder("GRUMP")
##ssp_folder_paths = ssp_population_paths(urban_folder)
##pop_files = population_files(ssp_folder_paths)
##ID_files = urbanID_files(ssp_folder_paths)
##for i, (pop, ID) in enumerate(zip(pop_files, ID_files)):
##    print i, pop, ID
##print
# CHECK
##urban_folder = population_landuse_folder("GlobCover")
##ssp_folder_paths = ssp_population_paths(urban_folder)
##pop_files = population_files(ssp_folder_paths)
##ID_files = urbanID_files(ssp_folder_paths)
##for i, (pop, ID) in enumerate(zip(pop_files, ID_files)):
##    print i, pop, ID


# LIST ALL URBAN ID VECTOR FOLDERS SSPs IN GlobCover OR GRUMP POPULATION FOLDERS

# Returns a list of urban vector folders 
def urbanID_vector_folders(ssp_folders_list):

    # create an empty list for urban national ID vector folders
    urbanID_folders = []

    # walk through the ssp folders one at a time
    for ssp_folder in ssp_folders_list:

        # Create list of the directors in the folder
        list_dirs = os.walk(ssp_folder)

        # find the directories names "urbanNationID_vectors"
        for root, dirs, files in list_dirs:
            for d in dirs:
                if d == "urbanNationID_vectors":

                    # create a path to the "urbanNationID_vectors" folder
                    urbanID_folder_path = os.path.join(root, d)

                    # Append this path to the path list 
                    urbanID_folders.append(urbanID_folder_path)

    # return the urban national ID vector file list
    return urbanID_folders




# LIST ALL URBAN ID VECTOR FILES PER SSP IN GlobCover OR GRUMP POPULATION FOLDERS

# Returns a list of urban vector files 
def urbanID_vectors(ssp_folders_list):

    # create an empty list for urban national ID vector files
    urbanID_files = []

    # walk through the ssp folders one at a time
    for ssp_folder in ssp_folders_list:

        # Create list of the directors in the folder
        list_dirs = os.walk(ssp_folder)

        # find the directories names "urbanNationID_vectors"
        for root, dirs, files in list_dirs:
            for d in dirs:
                if d == "urbanNationID_vectors":

                    # create a path to the "urbanNationID_vectors" folder
                    urbanID_folder_path = os.path.join(root, d)

                    # Put arcpy in that folder
                    arcpy.env.workspace = urbanID_folder_path

                    # list all the shape files in that folder
                    urbanID_vectors = arcpy.ListFeatureClasses()

                    # add each of these vector paths to the urban national ID vectpr file list
                    for vector in urbanID_vectors:
                        
                        # Get the path to the file and append it to the urban national ID vector file list
                        vector_path = os.path.join(urbanID_folder_path, vector)
                        urbanID_files.append(vector_path)

    # return the urban national ID vector file list
    return urbanID_files


# CREATE DIRECTORY

# creates a new directory if one is not already there, returns the path to the new directory
def create_directory(path, directory_name):

    # create new directory path
    new_directory_path = os.path.join(path, directory_name)

    # create a new directory, if it already exists ignore
    if not os.path.exists(new_directory_path):
        os.makedirs(new_directory_path)

    return new_directory_path

# CREATE FILE GEODATABASE

# creates a new file geodatabase if one is not already there, returns the path to the new gdb
def create_gdb(path, gdb_name):

    # create a new geodatabase path
    gdb_path = os.path.join(path, gdb_name)

    # Create a new geodatabase, if already exists ignore
    if not arcpy.Exists(gdb_path):
        arcpy.CreateFileGDB_management(path, gdb_name)

    return gdb_path

def get_urbanID_csv(ssp_folders_list):

   # create an empty list for urbanID csv files
    urbanID_files = []

    # walk through the ssp folders one at a time
    for ssp_folder in ssp_folders_list:

        # Create list of the directors in the folder
        list_dirs = os.walk(ssp_folder)

        # find the directories names "urbanID"
        for root, dirs, files in list_dirs:
            for d in dirs:
                if d == "urbanID":

                    # create a path to the "urbanID" folder
                    urbanID_folder_path = os.path.join(root, d)

                    # In the folder get the .csv files
                    for csv_file in glob.glob(urbanID_folder_path + "\\*.csv"): 
                        
                        # Put the path to each in the urbanID csv file list
                        csv_file_path = os.path.join(urbanID_folder_path, csv_file)
                        urbanID_files.append(csv_file_path)

    return urbanID_files
    
