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

# SET LOCAL VARIABLES 

# get the descriptives files, attach and get names
descr<-read.csv("E:/July_2017/zz_Descriptives_short.csv")
attach(descr)
names(descr)

# set the working directories
# Create a vector with the working directories 
path_a<-"E:/July_2017/Pop_&_Temp/GRUMP_HI/GRUMP_HI_Final"
path_b<-"E:/July_2017/Pop_&_Temp/GlobCover_HI/GlobCover_HI_Final"
paths<-c(path_a, path_b)

# MAKE FUNCTIONs

# Creates a empty line in a printout
print_space<-function(x){
  cat(x, sep = "\n")
}

# START THE PROCESS

# LOOP THROUGH THE WORKING DIRECTORIES AND CREATE FINAL DIRECTORY PATHS
for(p in 1:length(paths)){

# LIST ALL THE DATAFILES 

  # set the working directory
  setwd(paths[p]) 
  start_directory<-getwd()
  #print(start_directory)
  
  # make a pattern
  pat <- "All_"
  #print (p)

  # Create a list of all files 
  all_files<-list.files(path = start_directory, pattern  = pat)
  
  # loop through the list of all the files 
  for(i in 1:length(all_files)){
    
    # MAKE NAME FOR NEW FILE 
    
    # Get name of file 
    old_name<-all_files[i]
    # Cut out the ".csv" (substring from characters 1-19)
    mid_name<-substr(old_name, 1, 19)
    # Add "_desc.csv" to the end
    new_name<-paste(mid_name, "_desc.csv", sep = "")
    #print(new_name)
    
    # READ IN THE TEMP AND POP FILE AND PERFORM ADDITIONAL ANALYSES
    
    # Read in the file as csv
    temp_pop_data<-read.csv(all_files[i])
    
    # Create new variable for UHI addition
    temp_pop_data$HI_w_UHI<-0
    
    # Give variable new values based upon population size (adding temp by pop categories)
    temp_pop_data$HI_w_UHI<-ifelse(temp_pop_data$Population <= 2500, temp_pop_data$Heat_Index + 1.38, temp_pop_data$HI_w_UHI)
    temp_pop_data$HI_w_UHI<-ifelse(temp_pop_data$Population > 2500 & temp_pop_data$Population <=50000, temp_pop_data$Heat_Index + 1.496, temp_pop_data$HI_w_UHI)
    temp_pop_data$HI_w_UHI<-ifelse(temp_pop_data$Population > 50000 & temp_pop_data$Population <=100000, temp_pop_data$Heat_Index + 1.609, temp_pop_data$HI_w_UHI)
    temp_pop_data$HI_w_UHI<-ifelse(temp_pop_data$Population > 100000 & temp_pop_data$Population <=1000000, temp_pop_data$Heat_Index + 1.759, temp_pop_data$HI_w_UHI)
    temp_pop_data$HI_w_UHI<-ifelse(temp_pop_data$Population > 1000000 & temp_pop_data$Population <=10000000, temp_pop_data$Heat_Index + 2.169, temp_pop_data$HI_w_UHI)
    temp_pop_data$HI_w_UHI<-ifelse(temp_pop_data$Population > 10000000, temp_pop_data$Heat_Index + 2.988, temp_pop_data$HI_w_UHI)
    
    # Tranform the current dataframe creating a variable called "Heat_index_cat" and give it labels
    temp_pop_data<-transform(temp_pop_data, Heat_Index_cat = cut(Heat_Index, breaks = c(0, 28.0, 34.0, 42.0, 55.0, Inf),
                                                                 labels =c("<=28", ">28 & <=34", ">34 & <=42", ">42 & <=55", ">55")))
    
    # Tranform the current dataframe creating a variable called "Heat_index_cat" and give it labels
    temp_pop_data<-transform(temp_pop_data, Heat_Index_cat_UHI = cut(HI_w_UHI, breaks = c(0, 28.0, 34.0, 42.0, 55.0, Inf),
                                                                     labels =c("<=28", ">28 & <=34", ">34 & <=42", ">42 & <=55", ">55")))
    # MERGE POP FILE WITH ALL DESCRIPTIVES
    
    # merge the descriptives data with the temp and pop data  
    merge.data<-merge(temp_pop_data, descr, by ='nation_ID')
    
    # Make announcement
    announcement = paste("We are now writing:", new_name, sep = " ")
    print_space(c(announcement, ""))
    
    # SAVE FILE
    
    # write file 
    write.csv(merge.data, file = new_name, row.names = FALSE)
  }
}

print ("FINISHED")

rm(temp_pop_data)
rm(merge.data)
rm(descr)
gc()

