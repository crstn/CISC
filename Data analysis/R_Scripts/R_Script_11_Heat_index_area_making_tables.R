# 
# ==============================================================================================
# THIS SCRIPT CREATES SUMMARY DATA (MEAN, MAX, MIN, TOTAL) FOR AREA DATA BY SSP AND URBAN LAND 
# USE MODEL AT THE GLOBAL, REGIONAL AND SUB-REGIONAL SCALES. IT STORES THE DATA IN THE 
# "URBAN_AREAS/1-DAY" FOLDERS.  THE SCRIPT CAN BE RUN ONCE FOR BOTH GRUMP AND GLOBCOVER DATA.   
#
# Peter J. Marcotullio, JANUARY 2018
# ==============================================================================================

# INSTALL PACKAGES 
# install.packages("dplyr")
# install.packages("stringr")

# GET LIBRARIES 

library(dplyr)
library(stringr)

# MAKE FUNCTIONS 

# Creates a empty line in a printout
print_space<-function(x){
  cat(x, sep = "\n")
}

# Create a vector with the working directories 
path_a<-"E:/July_2017/Pop_&_Temp/GRUMP_HI/GRUMP_HI_Final/Summary_tables/Urban_areas/1_day"
path_b<-"E:/July_2017/Pop_&_Temp/GlobCover_HI/GlobCover_HI_Final/Summary_tables/Urban_areas/1_day"
paths<-c(path_a, path_b)

# Paths to data
datafile_1<-"E:/July_2017/Pop_&_Temp/xx_HI/xx_HI_Final/Summary_tables/Urban_areas/1_day/Summary_xx_by_Urban_Extent_RCP2p6_SSP_1_01.csv"
datafile_2<-"E:/July_2017/Pop_&_Temp/xx_HI/xx_HI_Final/Summary_tables/Urban_areas/1_day/Summary_xx_by_Urban_Extent_RCP2p6_SSP_2_01.csv"
datafile_3<-"E:/July_2017/Pop_&_Temp/xx_HI/xx_HI_Final/Summary_tables/Urban_areas/1_day/Summary_xx_by_Urban_Extent_RCP2p6_SSP_3_01.csv"
datafile_4<-"E:/July_2017/Pop_&_Temp/xx_HI/xx_HI_Final/Summary_tables/Urban_areas/1_day/Summary_xx_by_Urban_Extent_RCP2p6_SSP_4_01.csv"
datafile_5<-"E:/July_2017/Pop_&_Temp/xx_HI/xx_HI_Final/Summary_tables/Urban_areas/1_day/Summary_xx_by_Urban_Extent_RCP2p6_SSP_5_01.csv"

datas = c(datafile_1, datafile_2, datafile_3, datafile_4, datafile_5)

ssps = c("SSP_1", "SSP_2", "SSP_3", "SSP_4", "SSP_5")

land_use_names<-c("GRUMP", "GlobCover")


# Loop through the land use names list 

for(i in 1:length(paths)){
  
  # Get the land use name
  land<-land_use_names[i]
  print_space(c(paste("Working in", land, sep = " "), " "))
  
  # LOOP THROUGH THE FILES READ THEM INTO R AS .CSVS AND MAKE THE SUMMARY TABLES 
  for(j in 1:length(datas)){
    
    df<-gsub("xx", land, datas[j])
    
    # read in the file (note: use basic R)
    new_file<-read.csv(df)
    
    # Remove all urban extents without population
    new_file[new_file$Population >0, ]
    
    # Get summary statistics
    sum_data<-new_file%>%
      group_by(Year)%>%
      summarize(urban_extents = n(),
                Pop_sum = sum(as.numeric(Population)),
                Pop_mean = mean(Population),
                Pop_max = max(Population),
                Area_sum = sum(as.numeric(Area)),
                Area_mean = mean(Area),
                Area_min = min(Area),
                Area_max = max(Area),
                density_mean = mean(Density_p_km2),
                density_max = max(Density_p_km2),
                density_min = min(Density_p_km2))%>%
      as.data.frame()
    
    # Create name
    file_name<-paste(land, "_Summary_Area_Global_", ssps[j], ".csv", sep="")
    final_path_file<-file.path(paths[i], file_name)
    write.csv(sum_data, file = final_path_file, row.names=FALSE) 
    
    
    # Get summary statistics
    sum_data<-new_file%>%
      group_by(Year, region)%>%
      summarize(urban_extents = n(),
                Pop_sum = sum(as.numeric(Population)),
                Pop_mean = mean(Population),
                Pop_max = max(Population),
                Area_sum = sum(as.numeric(Area)),
                Area_mean = mean(Area),
                Area_min = min(Area),
                Area_max = max(Area),
                density_mean = mean(Density_p_km2),
                density_max = max(Density_p_km2),
                density_min = min(Density_p_km2))%>%
      as.data.frame()
    
    # Create name
    file_name<-paste(land, "_Summary_Area_region_", ssps[j], ".csv", sep="")
    final_path_file<-file.path(paths[i], file_name)
    write.csv(sum_data, file = final_path_file, row.names=FALSE) 
    
    # Get summary statistics
    sum_data<-new_file%>%
      group_by(Year, sub_region)%>%
      summarize(urban_extents = n(),
                Pop_sum = sum(as.numeric(Population)),
                Pop_mean = mean(Population),
                Pop_max = max(Population),
                Area_sum = sum(as.numeric(Area)),
                Area_mean = mean(Area),
                Area_min = min(Area),
                Area_max = max(Area),
                density_mean = mean(Density_p_km2),
                density_max = max(Density_p_km2),
                density_min = min(Density_p_km2))%>%
      as.data.frame()
    
    # Create name
    file_name<-paste(land, "_Summary_Area_Sub_region_", ssps[j], ".csv", sep="")
    final_path_file<-file.path(paths[i], file_name)
    write.csv(sum_data, file = final_path_file, row.names=FALSE) 
    
  }
}

print("FINISHED")





