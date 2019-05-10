# 
# ==============================================================================================
# THIS SCRIPT OPENS URBAN DESCRIPTIVES DATA AND MERGES THESE WITH URBAN POPULATION DATA FOR ALL 
# YEARS 2010, 2030, 2070 AND 2100.  IT STORES THE RESULTS IN THE "*_LATITUDE/URBAN_POP_LATITUDE_FINAL" 
# FOLDERS.  THE SCRIPT CAN BE RUN ONCE FOR BOTH GRUMP AND GLOBCOVER DATA. THE PROBLEM WITH THE 
# RESULTS, HOWEVER, IS THAT THE LATITUDES ARE FOR THE CENTROIDS OF CITIES, WHICH RESULTS IN A 
# SHIFT IN THE POPULATION FIGURES.  THEY DO NOT MATCH THE ANALYSIS OF THE TOTAL POPULATION 
# DISTRIBUTIONS! 
#
# Peter J. Marcotullio, MARCH 2018
# ==============================================================================================

# INSTALL PACKAGES 
# install.packages("dplyr")
# install.packages("stringr")

# GET LIBRARIES 

library(dplyr)
library(stringr)
library(reshape2)

# CREATE GROUPINGS NAMES TO SUMMARIZE BY

# For each set of categories, create groups and put these into a list, called dots with numbers
group_names_5<-list("urban_code", "Year")
dots5<-lapply(group_names_5, as.symbol)

group_names_6<-list("Latitude_cat", "Year")
dots6<-lapply(group_names_6, as.symbol)

# CREATE VARIABLES
#  Go to the folders for each of the SSPs
folder_x<-"E:/July_2017/Population/Data/SSPs/XX/SSP_X/tables/urbanID"

# Get the files labeled "All_SSP_1_Area.csv" (only file that starts with "All") one per folder

# go to "E:\July_2017\Pop_&_Temp\GRUMP_HI\GRUMP_HI_pop_tables\SSP_1" for all ssps 
# Pick up all the 10 urban files merge them with the descriptives files by year by urban_code to pick up population

# merge these files then perform the binning and summarization

# output files into the latitude folders 


start_directory<-"E:/July_2017"

#print(start_directory)

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

# Make sure one more break than label!
length_labels<-length(labels_lat)
length_breaks<-length(breaks_lat)
print(length_labels)
print(length_breaks)

# For some reason we need to reverse the order of the labels, otherwise, the are matched incorrectly
labels_lat<-rev(labels_lat)

# List of SSPs 
ssps<-c("SSP_1", "SSP_2", "SSP_3", "SSP_4", "SSP_5")

# List of land use model names 
land_use_names<-c("GRUMP", "GlobCover")

# MAKE FUNCTIONS 

# Creates a empty line in a printout
print_space<-function(x){
  cat(x, sep = "\n")
}

# Generate 2 tables at the national scale (one for with UHI and one for with UHI) by summarizing the data twice.    
latitude.tables<-function(file_x, landuse, name_file_x){
  
  # GROUP THE DATA BY (LATITUDE CATEGORY, YEAR) -> GET POP(SUM), AREA(SUM), NUMBER(SUM), DENSITITES (MEAN), OVERALL_DENSITY 
  
  # Summarize urban extents by urban code and year -> GET POP(MEAN), AREA(MEAN), DENSITY, LATITUDE(MEAN)
  l_file_5<-file_x%>%
    group_by_(.dots=dots5)%>%
    summarize(Population = mean(Population),
              Area = mean(AREA_SQKM),
              Density_p_km2 = Population / Area, 
              Density_rev_m2_p = (Area / Population)*10^6,
              latitude = mean(LAT)) %>%
    as.data.frame()
  # Only get the complete cases  
  l_file_5<-l_file_5[complete.cases(l_file_5),]
  
  # Tranform the current dataframe creating a variable called "Latitude_cat" of 1 degree and give it labels
  l_file_5 <-
    transform(l_file_5, Latitude_cat = cut(
      latitude,
      breaks = breaks_lat,
      labels = labels_lat))
  

  # MAKE NAME FOR TABLE 1
  # Make names for final file
  parts_file_name<-strsplit(name_file_x, "_")
  first<-paste("Summary_", landuse, "_latitude_cities", sep = "")
  very_last_parts<-parts_file_name[[1]][5]
  last_parts<-parts_file_name[[1]][4]
  last <- substr(last_parts, 1, 1)
  file_name<- paste(first, parts_file_name [[1]][2], parts_file_name[[1]][3], last_parts, very_last_parts, sep="_")
  final_path<-file.path(final_directory, file_name)
  final_path_file <- paste(final_path, ".csv", sep="")

  # WRITE TABLE 1 AND MAKE ANNOUNCEMENT
  # write file an announce process
  write.csv(l_file_5, file = final_path_file, row.names=FALSE)
  announcement<-paste("Writing: ", file_name)
  print(announcement)
  
  # CREATE TABLE 2 TO PRINT
  # Summarize the summary data by ISO, Year, and Heat_Index_cat
  l_file_6<-l_file_5%>%
    group_by_(.dots=dots6)%>%
    summarize(Lat_population = sum(as.numeric(Population)),
              number_urban_areas = n(),
              Lat_urban_area=sum(Area),
              Lat_density_p_km2 = sum(as.numeric(Population)) / sum(Area), 
              Lat_density_m2_p = (sum(Area) / sum(as.numeric(Population)))*10^6,
              Lat_mean_density = mean(Density_p_km2))%>%
    as.data.frame()
  # Only get the complete cases  
  l_file_6<-l_file_6[complete.cases(l_file_6),]  

  # MAKE NAME FOR TABLE 2   
  # Make names for final file
  parts_file_name<-strsplit(name_file_x, "_")
  first<-paste("Summary_", landuse, "_Latitude", sep = "")
  very_last_parts<-parts_file_name[[1]][5]
  last_parts<-parts_file_name[[1]][4]
  last <- substr(last_parts, 1, 1)
  file_name<- paste(first, parts_file_name [[1]][2], parts_file_name[[1]][3], last_parts, very_last_parts, sep="_")
  final_path<-file.path(final_directory, file_name)
  final_path_file <- paste(final_path, ".csv", sep="")
  
  # WRITE TABLE AND MAKE ANNOUNCEMENT 
  # write file an announce process
  write.csv(l_file_6, file = final_path_file, row.names=FALSE) 
  announcement<-paste("Writing: ", file_name)
  print(announcement)
  
}

# Start the loop between land use models 
for(q in 1:length(land_use_names)){
  
  # Get the correct land use folder
  land_use_folder<-gsub("XX", land_use_names[q], folder_x)
  
  # CHECK
  # print(land_use_folder)
  


# Get initial directory



# LOOP THROUGH THE WORKING DIRECTORIES AND CREATE FINAL DIRECTORY PATHS
for(i in 1:1){ #length(paths)){
  
  # set paths 
  setwd(paths[i])
  data_directory<-getwd()
  #print(data_directory)
  
  # get land use name 
  new_name_list<-strsplit(paths[i], "/")
  get_name<-new_name_list[[1]][4]
  get_x<-nchar(get_name)
  landuse_name<-substr(get_name, 1, get_x-3)
  landuse_lat<- paste(landuse_name, "_Latitude", sep = "")
  
  #print(landuse_name)
  
  final_directory<-file.path(start_directory, "Pop_&_Temp", get_name, landuse_lat, "Urban_pop_latitude_final")
  #print (final_directory)

  # COLLECT NECESSARY DATA FILES 
  
  # Create a pattern to identify all necessary files and list them. 
  patter<-"^All_RCP.*desc_Area.csv"
  summ_files<-list.files(path = data_directory, pattern = patter)
  
  # LOOP THROUGH THE FILES READ THEM INTO R AS .CSVS AND MAKE THE SUMMARY TABLES 
  for(j in 1:1){ #length(summ_files)){
    #print(summ_files[j])
    
    # read in the file (note: use basic R)
    new_file<-read.csv(summ_files[j])
    
    # Function to create two tables based upon nation
    latitude.tables(new_file, landuse_name, summ_files[j])
    
  }
  
  # Announcement
  announcement<-paste("FINISHED", start_directory, sep = " ")
  print_space(c(announcement, ""))
}

print("FINISHED")





# Location of the area nad latitude urban files by Urban_code, ssp and year: see  "F:\July_2017\Population\Data\SSPs\GRUMP\SSP_1\tables\urbanID"
# Location of the Urban pop files by "urban_code, SSP and year" F:\July_2017\Pop_&_Temp\GRUMP_HI\GRUMP_HI_pop_tables\SSP_1




