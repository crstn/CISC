# ======================================================================================================= 
# DESCRIPTION:  This script collects all the 15 day heat wave national population files and selects only
#               nations in Asia and the population experiencing very warm heat indices (>42C).  It outputs
#               these data by nation, SSP, region and sub-region.  It then takes these individual tables 
#               and combines them into 5 final output (Summary) files by SSP.
#               
# DATE:         21 July 2018
# Developer:    Peter J. Marcotullio
# ======================================================================================================= 

# Get libraries
library(dplyr)
library(stringr)
library(reshape2)

# Make lists
years <-c(2010, 2030, 2070, 2100)
ssps<-c("SSP1", "SSP2", "SSP3", "SSP4", "SSP5")

# Put R in appropriate folder
start_folder<-"E:/July_2017/Pop_&_Temp/GRUMP_HI/GRUMP_HI_Final/Summary_tables/Population/15_day"
setwd(start_folder)

# make path to final folder (created this folder previously)
final_path<-"E:/July_2017/Pop_&_Temp/GRUMP_HI/GRUMP_HI_Final/Summary_tables/Population/15_day/Asia"

# Get all the files we want 
pat<-"Nation_w_UHI"
all_files<-list.files(start_folder, pattern = pat)

# Loop through the files
for(p in 1:length(all_files)){
  
  # Get the RCP and the SSP from the name 
  file_name_list<-strsplit(all_files[p], "_")
  rcp<-file_name_list[[1]][7]
  ssp_n<-file_name_list[[1]][9]
  ssp<-paste("SSP", ssp_n, sep="")
  
  # Read in the data
  df<-read.csv(all_files[p])
  
  # select only Asian countries
  df_asia<-df[ which(df$region == 142), ]
  # select only hot populations
  df_asia_1<-df_asia[ which(df_asia$Heat_Index_cat_UHI == ">42 & <=55" | df_asia$Heat_Index_cat_UHI == ">55"),]

  # Create new datasets by looping through years
  for(i in 1:length(years)){
    
    # Select only for the current year
    df_asia_y<-df_asia_1[which(df_asia_1$Year == years[i]), ]
    
    # Create new variables
    df_asia_y$SSP<-ssp
    df_asia_y$RCP<-rcp
    
    # Create a new variable for heat index categories to help with the summary
    df_asia_y$HI_cat<-">42"

    # Get year in characters and make new name
    Asia_year<-as.character(years[i])
    label_y<-paste("Nation", "hotPop", rcp, ssp, Asia_year, sep = "_")
    
    # Summarize data by country iso and new heat index category 
    df_1<-df_asia_y%>%
      group_by(ISO, HI_cat, SSP, RCP, Year)%>%
      summarize(Region = mean(region),
                Sub_region = mean(sub_region),
                Population = sum(as.numeric(Population)))
    
    # Create final name and path for the file
    final_file_name<-paste(label_y, ".csv", sep = "")
    final_file_path<-file.path(final_path, final_file_name)
    
    # save the file
    write.csv(df_1, final_file_path, row.names =FALSE)
    
  }
}

# re-set the working folder
setwd(final_path)

# Loop through the new csv files by SSP
for(q in 1:length(ssps)){
  
  # Collect all the new files into a list
  list.data<-list()
  
  # create a pattern of SSP
  patt<-ssps[q]
  
  # Select all files by ssp
  all_new_files<-list.files(final_path, pattern = patt)
  
  # loop through the files and put them into a new list of dataframes 
  for(m in 1:length(all_new_files)){
    
    list.data[[m]]<-read.csv(all_new_files[m])
    
  }
  
  # bind the files from the dataframe list
  big_file<-do.call(rbind, list.data)
  
  # Create a Summary_file_name and path
  new_file_name<-paste("AC_Summary_", ssps[q], ".csv", sep="")
  new_final_file_path<-file.path(final_path, new_file_name)
  
  # save file
  write.csv(big_file, new_final_file_path, row.names = FALSE)  
}

# DONE! 