# 
# ==============================================================================================
# THIS SCRIPT OPENS URBAN DESCRIPTIVES DATA AND MERGES THESE WITH URBAN POPULATION DATA FOR ALL 
# ALL YEARS 2010-2100.  IT STORES THE RESULTS IN THE "*_LATITUDE/URBAN_POP_LATITUDE_FINAL" 
# FOLDERS.  THE SCRIPT CAN BE RUN ONCE FOR BOTH GRUMP AND GLOBCOVER DATA. THE PROBLEM WITH THE 
# RESULTS, HOWEVER, IS THAT THE LATITUDES ARE FOR THE CENTROIDS OF CITIES, WHICH RESULTS IN A 
# SHIFT IN THE POPULATION FIGURES.  THEY DO NOT MATCH THE ANALYSIS OF THE TOTAL POPULATION 
# DISTRIBUTIONS! 
#
# Peter J. Marcotullio, MARCH 2018
# ==============================================================================================

# GET LIBRARIES 

# get libraries
library(dplyr)
library(stringr)
library(reshape2)

# MAKE FUNCTIONS

line.space = function(x){
  cat(x, sep="\n")
}

# CREATE VARIABLES

# Create path templates to the different folders for files 
folder_x<-"F:/July_2017/Population/Data/SSPs/XX/SSP_X/tables/urbanID"
folder_y<-"F:/July_2017/Pop_&_Temp/XX_HI/XX_Latitude/Urban_pop_latitude_final"
folder_z<-"F:/July_2017/Pop_&_Temp/XX_HI/XX_HI_pop_tables"

# CREATE LISTS AND VECTORS
years<-c("2010", "2020", "2030", "2040", "2050", "2060", "2070", "2080", "2090", "2100")
ssps<-c("SSP_1", "SSP_2", "SSP_3", "SSP_4", "SSP_5")
land_use_names<-c("GRUMP", "GlobCover")
# CREATE LISTS AND VECTORS

breaks_lat<-c(Inf,
              80.0, 79.0, 78.0, 77.0, 76.0,
              75.0, 74.0, 73.0, 72.0, 71.0,
              70.0, 69.0, 68.0, 67.0, 66.0,
              65.0, 64.0, 63.0, 62.0, 61.0,
              60.0, 59.0, 58.0, 57.0, 56.0,
              55.0, 54.0, 53.0, 52.0, 51.0,
              50.0, 49.0, 48.0, 47.0, 46.0,
              45.0, 44.0, 43.0, 42.0, 41.0,
              40.0, 39.0, 38.0, 37.0, 36.0,
              35.0, 34.0, 33.0, 32.0, 31.0,
              30.0, 29.0, 28.0, 27.0, 26.0,
              25.0, 24.0, 23.0, 22.0, 21.0,
              20.0, 19.0, 18.0, 17.0, 16.0,
              15.0, 14.0, 13.0, 12.0, 11.0,
              10.0, 9.0, 8.0, 7.0, 6.0,
              5.0, 4.0, 3.0, 2.0, 1.0,
              0.0, -1.0, -2.0, -3.0, -4.0, 
              -5.0, -6.0, -7.0, -8.0, -9.0,
              -10.0, -11.0, -12.0, -13.0, -14.0,
              -15.0, -16.0, -17.0, -18.0, -19.0,
              -20.0, -21.0, -22.0, -23.0, -24.0,
              -25.0, -26.0, -27.0, -28.0, -29.0,
              -30.0, -31.0, -32.0, -33.0, -34.0,
              -35.0, -36.0, -37.0, -38.0, -39.0,
              -40.0, -41.0, -42.0, -43.0, -44.0,
              -45.0, -46.0, -47.0, -48.0, -49.0,
              -50.0, -51.0, -52.0, -53.0, -54.0,  
              -55.0, -56.0,-Inf)

labels_lat<- c("80-81 N", "79-80 N", "78-79 N", "77-78 N", "76-77 N", 
               "75-76 N", "74-75 N", "73-74 N", "72-73 N", "71-72 N",
               "70-71 N", "69-70 N", "68-69 N", "67-68 N", "66-67 N", 
               "65-66 N", "64-65 N", "63-64 N", "62-63 N", "61-62 N", 
               "60-61 N", "59-60 N", "58-59 N", "57-58 N", "56-57 N", 
               "55-56 N", "54-55 N", "53-54 N", "52-53 N", "51-52 N", 
               "50-51 N", "49-50 N", "48-49 N", "47-48 N", "46-47 N", 
               "45-46 N", "44-45 N", "43-44 N", "42-43 N", "41-42 N", 
               "40-41 N", "39-40 N", "38-39 N", "37-38 N", "36-37 N", 
               "35-36 N", "34-35 N", "33-34 N", "32-33 N", "31-32 N", 
               "30-31 N", "29-30 N", "28-29 N", "27-28 N", "26-27 N",
               "25-26 N", "24-25 N", "23-24 N", "22-23 N", "21-22 N", 
               "20-21 N", "19-20 N", "18-19 N", "17-18 N", "16-17 N", 
               "15-16 N", "14-15 N", "13-14 N", "12-13 N", "11-12 N", 
               "10-11 N", "9-10 N",  "8-9 N",   "7-8 N",   "6-7 N",
               "5-6 N",   "4-5 N",   "3-4 N",   "2-3 N",   "1-2 N", 
               "0-1 N",   "0-1 S",   "1-2 S",   "2-3 S",   "3-4 S", 
               "4-5 S",   "5-6 S",   "6-7 S",   "7-8 S",   "8-9 S", 
               "9-10 S",  "10-11 S", "11-12 S", "12-13 S", "13-14 S",
               "14-15 S", "15-16 S", "16-17 S", "17-18 S", "18-19 S", 
               "19-20 S", "20-21 S", "21-22 S", "22-23 S", "23-24 S", 
               "24-25 S", "25-26 S", "26-27 S", "27-28 S", "28-29 S", 
               "29-30 S", "30-31 S", "31-32 S", "32-33 S", "33-34 S", 
               "34-35 S", "35-36 S", "36-37 S", "37-38 S", "38-39 S", 
               "39-40 S", "40-41 S", "41-42 S", "42-43 S", "43-44 S", 
               "44-45 S", "45-46 S", "46-47 S", "47-48 S", "48-49 S", 
               "49-50 S", "50-51 S", "51-52 S", "52-53 S", "53-54 S",
               "54-55 S", "55-56 S", "56-57 S")

# # Make sure one more break than label!
# length_labels<-length(labels_lat)
# length_breaks<-length(breaks_lat)
# print(length_labels)
# print(length_breaks)

# For some reason we need to reverse the order of the labels, otherwise, the are matched incorrectly
labels_lat<-rev(labels_lat)

# Loop through the land use names list (do both GRUMP and GLobCover)
for(f in 1:1){ #length(land_use_names)){ # Change for land use models here 
  
  # Create path to final folder
  final_folder<-gsub("XX", land_use_names[f], folder_y)
  
  # Create path to the population tables
  pop_data_folder<-gsub("XX", land_use_names[f], folder_z)
  
  # Start creating Area_ID file folder 
  desc_folder<-gsub("XX", land_use_names[f], folder_x)
  
  # CHECK
  #print(working_folder_1)

  # Start the loop through the ssps
  for(i in 1:1){ #length(ssps)){ # change for SSPs here 
    
    # COLLECT DESCRIPTIVE FILES WITH AREA DATA
    
    # Finishing creating Area_ID folder
    descriptive_folder<-gsub("SSP_X", ssps[i], desc_folder)
   
    # Create a pattern to identify id files in working directory and list them. 
    ps<-"new"
    desc_files<-list.files(path = descriptive_folder, pattern = ps)
    
    # COLLECT POPULATION FILES 
    
    # population folder
    pop_folder<-file.path(pop_data_folder, ssps[i])
    
    # set this population folder as the directory
    setwd(pop_folder)
    
    # make a list of population files
    pop_files<-list.files(pop_folder)
    
    # loop through the Area ID files list (which includes only 1 file)
    for(g in 1:1){ #length(years)){
      
      # Select the descriptive file for the correct year
      working_desc_file<-desc_files[grepl(years[g], desc_files)]
      
      # Select the population file for the correct year
      working_pop_file<-pop_files[grepl(years[g], pop_files)]
      
      # Check
      # print(working_desc_file)
      # print(working_pop_file)
      # line.space(c("", ""))
   
      # Set working directory to read in the descriptive folder
      setwd(descriptive_folder)
      
      # Read in the descriptive file
      desc_file<-read.csv(working_desc_file)
      
      # Change column name to match population file 
      colnames(desc_file)[1]<-"urban_code"
      
      # Set working directory to read in the population file 
      setwd(pop_folder)
      
      # Read in the population file
      pop_file<-read.csv(working_pop_file)
      
      # merge files 
      final_file<-merge(desc_file, pop_file, by= c("urban_code", "Year", "SSP"))
      
      # Tranform the current dataframe creating a variable called "Latitude_cat" of 1 degree and give it labels
      final_file <-
        transform(final_file, Latitude_cat = cut(
          LAT,
          breaks = breaks_lat,
          labels = labels_lat))
      
      # Summarize the data by latitude
      final_file<-final_file%>%
        group_by(Latitude_cat)%>%
        summarize(Model = land_use_names[f],
                  SSP = ssps[i],
                  Year = as.numeric(years)[g],
                  number_urban_areas = n(),
                  Lat_urban_area=sum(AREA_SQKM),
                  Lat_density_p_km2 = sum(as.numeric(Population)) / sum(AREA_SQKM), 
                  Lat_density_m2_p = (sum(AREA_SQKM) / sum(as.numeric(Population)))*10^6,
                  Lat_population = sum(as.numeric(Population)))%>%
        as.data.frame()
        
      # Create a file name
      file_name<-paste(ssps[i], years[g], land_use_names[f],"urban_pop_latitude.csv", sep = "_")
      final_file_path<-file.path(final_folder, file_name)
      
      # save file
      write.csv(final_file, final_file_path, row.names=FALSE)
      
    }
  }
}

