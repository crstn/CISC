# 
# ==================================================================================================== 
# THIS SCRIPT MERGES the POPULATION-ID-HEATWAVE FILES WITH A DESCRIPTIVES FILE THAT ALLOWS FOR  
# DISTINCTIONS BETWEEN COUNTRY, REGION, SUB-REGION, ETC.  IT ALSO CREATES HEAT WAVE CATEGORICAL 
# VARAIBLES.  IT PICKS UP ALL OF THE "ALL_*" FILES AND MERGES EACH WITH A DESCRIPTIVE FILE AND 
# THEN CALCULATES A UHI ADDITION TO THE HEAT INDEX.  IT THEN CREATES THE HEAT WAVE AND HEAT WAVE WITH UHI 
# CATEORICAL VARIABLES.  THE FINAL OUTPUT FILES HAVE THE SIMILAR NAMES AS THE "ALL_*" FILES, BUT THEY
# END WITH "DESC.CSV" AND ARE STORED IN THE SAME FOLDER AS THE "ALL_*" FILES.  THE SCRIPT CAN BE RUN
# ONCE FOR THE GRUMP AND GLOBCOVER DATA FILES.  IT PRODUCES 60 FILES FOR EACH LAND USE MODEL.  
#
# DEVELOPER:  PETER J. MARCOTULLIO
# DATE:       DECEMBER 2017
# ====================================================================================================

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

folder_x<-"E:/July_2017/Population/Data/SSPs/XX/SSP_X/tables/urbanID"

# CREATE LISTS AND VECTORS

ssps<-c("SSP_1", "SSP_2", "SSP_3", "SSP_4", "SSP_5")
land_use_names<-c("GRUMP", "GlobCover")

# Start the loop between land use models 
for(q in 1:length(land_use_names)){
  
  # Get the correct land use folder
  land_use_folder<-gsub("XX", land_use_names[q], folder_x)
  
  # CHECK
  # print(land_use_folder)

  # Start the loop for the ssp folders
  for(i in 1:length(ssps)){
    
    # Get the correct SSP folder folder
    working_folder<-gsub("SSP_X", ssps[i], land_use_folder)
    
    setwd(working_folder)
    
    # Make announcement
    text<-"We are now working in:"
    announce<-paste(text, working_folder, sep = " ")
    line.space(c(announce, ""))

    # Create a pattern to identify id files in working directory and list them. 
    ps<-"^SSP.*urbanID"
    id_files<-list.files(path = working_folder, pattern = ps)
    
    # CHECK
    # for(z in 1:length(id_files)){
    #   print(id_files[z])
    # }
    # line.space(" ")

    # Loop through the id_files
    for(j in 1:length(id_files)){
      
      # Get year of file data
      year<-strsplit(id_files[j], "_")[[1]][3]
      
      path_to_file<-file.path(working_folder, id_files[j])
      # print(path_to_file)
      
      # Read in file
      working_file<-read.csv(path_to_file)
      
      # Create two new varables "Year" and "SSP" and fill the columns with values 
      working_file$Year<-year
      working_file$SSP<-ssps[i]
      
      # Create name of new file
      temp_name<-paste(ssps[i], "_", year, "_new", ".csv", sep = "")
      full_file_path<-file.path(working_folder, temp_name)
      
      # CHECK
      print(full_file_path)
      line.space(" ")
      
      # Write the individual files with new name by SSP and year
      write.csv(working_file, file = full_file_path, row.names = FALSE)
    }

    # CHECK
    #print(working_folder)
    
    # Create the pattern to search 
    p = "*_new"
    
    # collect all the files just created and put them in a file list
    temp_fs<-list.files(path = working_folder, pattern = p)
    
    # # CHECK
    # for(z in 1:length(temp_fs)){
    #   print(temp_fs[z])
    #   
    # }
    
    #Read in all the csv files for this ssp
    file_list<-lapply(temp_fs, read.csv)
    
    # Bind the files 
    test<-do.call(rbind.data.frame, file_list) 
    
    # Need the change the some names before we merge with the descriptive files
    colnames(test)[colnames(test)=="NATIONID"] <- "nation_ID"
    colnames(test)[colnames(test)=="URBAN_ID"] <- "urban_code"
    
    # CHECK
    # print(colnames(test))
    
    # Create name for final file
    file_name<- paste("All", ssps[i], "Area.csv", sep = "_")
    full_file_path<-file.path(working_folder, file_name)
    
    # print(full_file_path)
    
    # Save file
    write.csv(test, file = full_file_path, row.names = FALSE)
  }
}
print("END")