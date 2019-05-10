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
folder_y<-"E:/July_2017/Pop_&_Temp/XX_HI/XX_HI_Final"

# CREATE LISTS AND VECTORS

ssps<-c("SSP_1", "SSP_2", "SSP_3", "SSP_4", "SSP_5")
land_use_names<-c("GRUMP", "GlobCover")
# remove_desc<-c("Least_Developed", "Land_locked", "small_island", "WB_other")

# Loop through the land use names list (do both GRUMP and GLobCover)
for(f in 1:length(land_use_names)){ # Changee for land use models here 
  
  # Create land use folder
  land_use_folder<-gsub("XX", land_use_names[f], folder_y)
  
  # CHECK
  # print(paste("This is the current land use folder", land_use_folder, sep = " "))
  
  # Start creating Area_ID file folder 
  working_folder_1<-gsub("XX", land_use_names[f], folder_x)
  
  # CHECK
  # print(working_folder_1)
  
  # Start the loop through the ssps
  for(i in 1:length(ssps)){ # change for SSPs here 
    
    # COLLECT IDED FILES WITH AREA DATA
    
    # Finishing creating Area_ID folder
    working_folder<-gsub("SSP_X", ssps[i], working_folder_1)
    
    # set this working folder as the working directory
    setwd(working_folder)
    current_directory<-getwd()

    # CHECK
    # print(paste("This is the Area ID directory", current_directory, sep = " "))
        
    # Make announcement
    # text<-"We are now collecting IDed_Area files from:"
    # announce<-paste(text, working_folder, sep = " ")
    # line.space(c(announce, ""))
    
    # Create a pattern to identify id files in working directory and list them. 
    ps<-"^All.*Area"
    id_files<-list.files(path = working_folder, pattern = ps)
    
    # loop through the Area ID files list (which includes only 1 file)
    for(g in 1:length(id_files)){

      # Announce the Area ID file      
      # line.space(c(paste("We are working with Area ID file:", id_files[g], sep = " "), " " ))

      # Read in the ID file
      id_file<-read.csv(id_files[g])
      
      # Change the directory
      setwd(land_use_folder)
      new_dir<-getwd()
      
      # Make announcement
      # text<-"We are now collecting descriptive files from:"
      # announce<-paste(text, land_use_folder, sep = " ")
      # line.space(c(announce, ""))
      
      # COLLECT DESCRIPTIVE FILES 
      
      # Find all descriptive files 
      pq<-"^All_RCP.*desc.csv"
      descriptive_files<-list.files(path = land_use_folder, pattern = pq)
      
      # in descriptive files find those with the current ssp
      working_desc_files<-descriptive_files[grep(ssps[i], descriptive_files)]
      
      # Loop through the descriptive files
      for(z in 1:length(working_desc_files)){ # Change for the RCP and heat wave length here
        
        # CHECK
        # print(working_desc_files[z])
        # line.space(" ")
        
        # Read in the file as csv file
        desc_file<-read.csv(working_desc_files[z])
        
        df <- subset(desc_file, select = -c(Land_model, RCP, Country.name, Least_Developed, Land_locked, small_island, ISO, WB_other))
        
        #desc_file[, !(names(desc_file) %in% remove_desc)]
        
        # CHECK
        #line.space(c(names(df)), " ")
        
        # Announce
        line.space(paste("Working with data from", land_use_names[f], sep = " "))
        line.space(paste("ID files come from", working_folder, sep = " "))
        line.space(paste("Description files come from", land_use_folder, sep = " "))
        line.space(c(paste("We merge files", id_files[g], "with", working_desc_files[z], sep = " "), " "))
        
        names(df)
        names(id_file)
        
        # Create new name for new file
        file_name<-gsub("_desc", "_desc_Area", working_desc_files[z])
        full_file_path<-file.path(land_use_folder, file_name)
        
        # merge the descriptives data with the temp and pop data  
        merge.data<-merge(x=df, y=id_file, by=c("urban_code", 'nation_ID', "Year", "SSP"))
        
        # Save file
        write.csv(merge.data, file = full_file_path, row.names = FALSE)
      }
    }
  }
}
