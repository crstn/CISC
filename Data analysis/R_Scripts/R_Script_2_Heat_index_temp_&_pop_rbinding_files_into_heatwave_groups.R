# 
# ==============================================================================================
# THIS SCRIPT ITERATES THROUGH THE FILES IN THE "*_HI_FINAL" FOLDERS AND BINDS THEM TOGETHER BY 
# RCP-SSP-HEAT WAVE LENGTH.  OUTPUTS ARE PLACED IN THE SAME FOLDER WITH THE TITLE "ALL_*".  
# THE SCRIPT ONLY NEEDS TO BE RUN ONCE FOR GRUMP AND GLOBCOVER DATA AND PRODUCES 60 FILES 
# FOR EACH LAND USE MODEL (4 X 3 X 5). 
#
# Peter J. Marcotullio, DECEMBER 2017
# ==============================================================================================

# GET LIBRARIES

library(dplyr)
library(stringr)

# SET INITIAL CONDITIONS

# set the working directories
# Create a vector with the working directories 
path_a<-"E:/July_2017/Pop_&_Temp/GRUMP_HI/GRUMP_HI_Final"
path_b<-"E:/July_2017/Pop_&_Temp/GlobCover_HI/GlobCover_HI_Final"
paths<-c(path_a, path_b)
#paths_b<-list(path_b)

#  CREATE LISTS 

# Create an SSPs list
ssps<-list("SSP_1", "SSP_2", "SSP_3", "SSP_4", "SSP_5")

# Create a rcp list
rcps<-list("RCP2p6", "RCP4p5", "RCP6p0", "RCP8p5")

# Create a heat wave days list
days = list("_01", "_05", "_15")

# MAKE FUNCTIONs

# Creates a empty line in a printout
print_space<-function(x){
  cat(x, sep = "\n")
}

# START THE PROCESS

# LOOP THROUGH THE WORKING DIRECTORIES AND CREATE FINAL DIRECTORY PATHS
for(p in 1:length(paths)){
  
  # set the working directory
  setwd(paths[p]) 
  start_directory<-getwd()

  # Iterate through the rcp list 
  for(i in 1:length(rcps)){
    
    # Get the rcp name 
    rcp_name<-rcps[i]
    #print(start_y)
    
    # Iterate through the ssp list
    for(j in 1:length(ssps)){
      
      # get the ssp name fo
      ssp_name<-ssps[j]
      # get the rcp and ssp
      patte <- paste(rcp_name, "_", ssp_name, "_", "*", sep = "")
      #print (p)
      
      # Create a list of all files 
      rcp_ssp_files<-list.files(path = start_directory, pattern  = patte)
 
       
      # create sub-lists of all the files based upon heat wave day length
      rcp_ssp_files_01<-rcp_ssp_files[!grepl("_05|_15", rcp_ssp_files)]
      rcp_ssp_files_05<-rcp_ssp_files[!grepl("_01|_15", rcp_ssp_files)]
      rcp_ssp_files_15<-rcp_ssp_files[!grepl("_01|_05", rcp_ssp_files)]

      # Make a list of lists
      rcp_ssp_days_files<-list( rcp_ssp_files_01,  rcp_ssp_files_05,  rcp_ssp_files_15)

      # Iterate through the heat wave days list
      for(m in 1:length(days)){

        #get the days
        heatwave_days<-days[m]

        #Read in all the csv files for 1 day heat wave
        file_list<-lapply(rcp_ssp_days_files[[m]], read.csv)

        # bind them
        tf<-do.call(rbind.data.frame, file_list)

        #create file name
        end<-paste(ssp_name, heatwave_days, ".csv", sep = "")
        final_name<-paste("All", rcp_name, end, sep="_")

        # Make announcement
        announcement = paste("We are now writing:", final_name, sep = " ")
        print(announcement)
        print_space("")

        # write file
        write.csv(tf, file = final_name, row.names = FALSE)

      }
      print_space("")
    }
  }
}

print("FINISHED")

rm(tf)
gc()