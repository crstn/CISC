# 
# ==================================================================================================
# THIS SCRIPT OPENS ALL THE CLIMATE HEAT INDEX AND THE POPULATION DATA CSV FILES AND NATION ID FILES. 
# IT MERGES THE POP AND ID FILES BY SSP AND YEAR AND THEN MERGES THIS FILE WITH THE 2010 CLIMATE 
# HEATWAVE DATA BY HEAT WAVE LENGTH (1, 5 AND 15 DAYS).  THUS THREE FILES ARE CREATED FOR EACH 
# RCP-SSP COMBINATION.  IN TOTAL 240 (4 X 5 X 4 x 3) FILES ARE CREATED AND STORED IN THE 
# "*_HI_FINAL/CLIMATE_2010/CLIMATE_FINAL" FOLDER. THE SCRIPT CAN BE RUN ONCE FOR BOTH GRUMP AND 
# GLOBCOVER DATA.  
#
# Peter J. Marcotullio, JANUARY 2018
# ==================================================================================================

# INSTALL PACKAGES

# install.packages("dplyr")
# install.packages("stringr")

# GET LIBRARIES  

# library(readr)
library(dplyr)
library(stringr)
library(reshape2)

# LOCAL VARIABLES 

# set the working directories
# Create a vector with the working directories 
path_a<-"E:/July_2017/Pop_&_Temp/GRUMP_HI"
path_b<-"E:/July_2017/Pop_&_Temp/GlobCover_HI"
paths<-c(path_a, path_b)

# MAKE LISTS

# Create lists of the SSPs and years to examine
ssps<-list("SSP_1", "SSP_2", "SSP_3", "SSP_4", "SSP_5")

# Create a list of years 
years<-list(2010, 2030, 2070, 2100)

# Create a list of heat wave lengths 
times <-list("01", "05", "15")

rcps<-list("RCP2p6", "RCP4p5", "RCP6p0", "RCP8p5")

# LOOP THROUGH THE WORKING DIRECTORIES AND CREATE FINAL DIRECTORY PATHS
for(p in 1:length(paths)){
  
  setwd(paths[p]) # MIGHT NEED TO CHANGE THIS! 
  start_directory<-getwd()
  # print(start_directory)
  
  # GET ALL DIRECTORIES IN THE CURRENT FOLDER
  
  # create a list of all directories in the initial directory
  all_dirs<-list.dirs(path = start_directory, recursive=FALSE)
  
  # Go through the list of all directories and 1) put all the RCP directories in the new rcp_list
  # 2) finds the "Final" directory; 3) finds the "pop" directory and 4) finds the "id" directory
  for(q in 1:length(all_dirs)){
    if(grepl("Climate",all_dirs[q])){
      climate_folder<-all_dirs[q] 
    } else if (grepl("pop", all_dirs[q])){
      pop_directory<-all_dirs[q]
    } else if (grepl("ID", all_dirs[q])){
      id_directory<-all_dirs[q]
    }
  }

  # create an empty list to put paths into for the temperature data 
  csv_tables_path<-list()
  
  final_directory<-file.path(climate_folder, "Climate_Final")
  
  #print(climate_folder)
  #print(final_directory)

  # create path name from current directory to the folders that hold the data files for a specific RCP
  # create parts of the path first and then a list of all paths to all SSPs in that RCP
  for(i in 1:length(years)){
    for(j in 1:length(ssps)){
      path_start<-paste(climate_folder, "/", sep="")
      path_second<-paste(years[i], "csv_tables/", sep="_")
      path_third<-ssps[j]
      path_name_list=c(path_start, path_second, path_third)
      path_final = paste(path_name_list, sep='', collapse= '')
      csv_tables_path[length(csv_tables_path)+1]<-file.path(path_final)
      }
    }

  # # Start the iteration through the paths
  for(i in 1:length(csv_tables_path)){
    
    # get the path to the files
    path_s<-unlist(csv_tables_path[i])
    # print(path_s)
    
    # set the working directory at the current path
    setwd(path_s)
  
    # break up the path to get parts 
    y <- strsplit(path_s, "/")
    #print(y)

    #get ssp
    ssp<-y[[1]][7]
    #print(ssp)
    
    # get year
    t<-y[[1]][6]
    year<- substr(t, 1, 4)
    #print(year)
    
    # COLLECT ALL THE HEAT WAVE DATA IN THIS DIRECTORY AND CREATE ONE LARGE FILE
    
    # collect all the files there that end in "r_ready.csv" and put them in a file list
    temp_fs<-list.files(path = path_s, pattern = "*r_ready")
    
    # Create three different lists 
    temp_fs_05<-list()
    temp_fs_15<-list()
    temp_fs_01<-list()
    
    for (j in 1:length(temp_fs)){
      if (grepl("05_r_ready.csv$", temp_fs[j])){
        temp_fs_05[length(temp_fs_05)+1]<-temp_fs[j]
      } else if (grepl("15_r_ready.csv$", temp_fs[j])){
        temp_fs_15[length(temp_fs_15)+1]<-temp_fs[j]
      } else if (grepl("01_r_ready.csv$", temp_fs[j])){
        temp_fs_01[length(temp_fs_01)+1]<-temp_fs[j]
      } else print("Mistake")
    }
    
    # Create a list of lists
    all_temp<-list(temp_fs_01, temp_fs_05, temp_fs_15)
    
    # GET POPULATION DATA
    
    # get the population table path
    pop_table_path<-paste(pop_directory, ssp, sep="/")
    
    # Get the urban id path
    urbanID_path<-paste(id_directory, ssp, sep = "/")
    
    #create name for the pop  and ID data
    pop_file_name_first<-paste(ssp, "_", sep = "")
    pop_file_name_second<-paste(year, "_urban_pop.csv", sep="")
    id_file_name_second<-paste(year, "_urbanID.csv", sep ="")
    pop_file_name_final<-paste(pop_file_name_first, pop_file_name_second, sep="")
    id_file_name_final<-paste(pop_file_name_first, id_file_name_second, sep = "")
    
    # create entire paths for both pop and id data
    pop_file_path_full<-file.path(pop_table_path, pop_file_name_final)
    id_file_path_full<-file.path(urbanID_path, id_file_name_final)
    
    #print(pop_file_path_full)
    #print(id_file_path_full)
    
    # read in the population  and id datafiles
    pop_data<-read.csv(pop_file_path_full)
    id_data<-read.csv(id_file_path_full)

    # attach data
    attach(pop_data)
    attach(id_data)

    # MERGE THE POP AND ID FILES AND THEN MERGE THAT FILE WITH THE BIG HEAT WAVE DATA FILE

    # Merge the pop data and the urban ID data together
    urban_pop_id_data<-merge(pop_data, id_data, by = "urban_code", all.x=TRUE)
    
    # Loop through the list of lists and bind the files by RCP, SSP, heat wave length and year
    for(k in 1:length(all_temp)){
      
      # Get the length of the heat wave for use in the final file name from the heat wave times list
      heat_wave_length<-times[k]

      print(heat_wave_length)
      
      #print(all_temp[k])
      working_files<-unlist(all_temp[k])
      
      for(r in 1:length(rcps)){
        
        # Create a RCP variable
        rcp<-unlist(rcps[r])
        #print(rcp)
        
        # create empty list to hold files
        ssp_rcp_hw_year_files<-list()  
        
        # Iterate through all the working files list and pull out those of the Needed RCP
        for(s in 1:length(working_files)){
          
          # Iterate through working files and put the files with the correct values in the empty file list
          if(grepl(rcps[r], working_files[s])){
            ssp_rcp_hw_year_files[length(ssp_rcp_hw_year_files)+1]<-working_files[s]
          }
        } 
        
        # Go through the file list and read all these csv files
        filelist<-lapply(ssp_rcp_hw_year_files, read.csv)
            
        # combine them together into a data.frame
        all_hi_files<-do.call(rbind.data.frame, filelist)

        # Merge the temperature files with the population  and Id file 
        # by CODE, SSP and Year to create a master file for this RCP, SSP and Year! and perhaps Heat WAVE lenght? 
        heat_index_pop<-merge(urban_pop_id_data, all_hi_files, by = c("urban_code", "SSP", "Year"), all.x=TRUE)
        
        # Create final variable for the new file
        heat_index_pop$RCP<-rcp
            
        # CREATE A NAME FOR THE OUTPUT FILE AND SAVE IT
            
        #Create new file names for the final files
        new_file_name_first<-paste("Climate", rcp, ssp, year, heat_wave_length, sep="_")
        new_file_name_second<-paste(new_file_name_first, ".csv", sep="")
        new_file_path<-file.path(final_directory, new_file_name_second)
            
        announcement = paste("Writing: ", new_file_name_first, sep = " ")
        print(announcement)
            
        # save each file!!
        write.csv(heat_index_pop, file = new_file_path, row.names = FALSE)
      }
    }
  }
}
print("FINISHED")









