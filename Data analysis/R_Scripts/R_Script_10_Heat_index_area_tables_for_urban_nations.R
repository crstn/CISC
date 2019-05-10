# 
# ==============================================================================================
# THIS SCRIPT OPENS ALL THE COMBINED HEAT INDEX FILES WITH UHI DATA AND DESCRIPTIVES.  IT 
# SUMMARIZES THE INFORMATION INTO URBAN EXTENT AND NATIONAL LEVEL UNITS WITH TAGS FOR ALLOWING 
# REGIONAL, SUB-REGIONAL, WORLD BANK INCOME STATUS AND OTHER AGGREGATIONS.  THE DATA ARE STORED 
# AS SUMMARY FILES IN "*_HI_FINAL/SUMMARY_TABLES" FOLDERS.  THE SCRIPT CAN BE RUN ONCE FOR BOTH 
# GRUMP AND GLOBCOVER DATA.   
#
# Peter J. Marcotullio, DECEMBER 2017
# ==============================================================================================

# INSTALL PACKAGES 
# install.packages("dplyr")
# install.packages("stringr")

# GET LIBRARIES 

library(dplyr)
library(stringr)

# CREATE GROUPINGS NAMES FOR HEAT INDEX ANALYSES (TO SUMMARIZE BY)

# For each set of categories, create groups and put these into a list, called dots with numbers (1-16) 
# converted into names (or symbols) for use in the "summarize" method 
group_names_5<-list("urban_code", "nation_ID", "Year")
dots5<-lapply(group_names_5, as.symbol)

group_names_6<-list("nation_ID", "Year", "Heat_Index_cat")
dots6<-lapply(group_names_6, as.symbol)

group_names_7<-list("nation_ID", "Year", "Heat_Index_cat_UHI")
dots7<-lapply(group_names_7, as.symbol)

# MAKE FUNCTIONS 

# Creates a empty line in a printout
print_space<-function(x){
  cat(x, sep = "\n")
}

# Generate 2 tables at the national scale (one for with UHI and one for with UHI) by summarizing the data twice.    
nation.tables<-function(file_x, landuse, name_file_x){
  
  # Summarize urban extents by urban code, ISO and year 
  l_file_5<-file_x%>%
    group_by_(.dots=dots5)%>%
    summarize(Population = mean(Population),
              Mean_HI = mean(Heat_Index),
              Mean_HI_UHI = mean(HI_w_UHI),
              Area = mean(AREA_SQKM),
              Density_p_km2 = Population / Area, 
              Density_rev_m2_p = (Area / Population)*10^6,
              longitude = mean(LONG),
              latitude = mean(LAT),
              region = mean(Region),
              sub_region = mean(Sub.region),
              dev_status = mean(Develop_status),
              income_grp = mean(WB_income)) %>%
    as.data.frame()
  # Only get the complete cases  
  l_file_5<-l_file_5[complete.cases(l_file_5),]
  
  # Tranform the current dataframe creating a variable called "Heat_index_cat" and give it labels
  l_file_5<-transform(l_file_5, Heat_Index_cat = cut(Mean_HI, breaks = c(0, 28.0, 34.0, 42.0, 55.0, Inf),
                                                               labels =c("<=28", ">28 & <=34", ">34 & <=42", ">42 & <=55", ">55")))
  
  # Tranform the current dataframe creating a variable called "Heat_index_cat_UHI" and give it labels
  l_file_5<-transform(l_file_5, Heat_Index_cat_UHI = cut(Mean_HI_UHI, breaks = c(0, 28.0, 34.0, 42.0, 55.0, Inf),
                                                                   labels =c("<=28", ">28 & <=34", ">34 & <=42", ">42 & <=55", ">55")))
  
  # MAKE NAME FOR TABLE 1   
  # Make names for final file
  parts_file_name<-strsplit(name_file_x, "_")
  first<-paste("Summary_", landuse, "_by_Urban_Extent", sep = "")
  very_last_parts<-parts_file_name[[1]][5]
  last_parts<-parts_file_name[[1]][4]
  last <- substr(last_parts, 1, 1)
  file_name<- paste(first, parts_file_name [[1]][2], parts_file_name[[1]][3], last_parts, very_last_parts, sep="_")
  final_path<-file.path(final_directory, file_name)
  final_path_file <- paste(final_path, ".csv", sep="")
  
  # WRITE TABLE 1 AND MAKE ANNOUNCEMENT 
  # write file an announce process
##  write.csv(l_file_5, file = final_path_file, row.names=FALSE) 
  announcement<-paste("Writing: ", file_name)
  print(announcement)
  
  # CREATE SECOND TABLE TO PRINT
  # Summarize the summary data by ISO, Year, and Heat_Index_cat
  l_file_6<-l_file_5%>%
    group_by_(.dots=dots6)%>%
    summarize(Nation_population = sum(as.numeric(Population)),
              number_urban_areas = n(),
              nation_urban_area=sum(Area),
              Density_nation_p_km2 = sum(as.numeric(Population)) / sum(Area), 
              Density_Nation_rev_m2_p = (sum(Area) / sum(as.numeric(Population)))*10^6,
              region = mean(region),
              sub_region = mean(sub_region),
              dev_status = mean(dev_status),
              income_grp = mean(income_grp)) %>%
    as.data.frame()
  # Only get the complete cases  
  l_file_6<-l_file_6[complete.cases(l_file_6),]  

  # MAKE NAME FOR TABLE 2   
  # Make names for final file
  parts_file_name<-strsplit(name_file_x, "_")
  first<-paste("Summary_", landuse, "_by_Nation", sep = "")
  very_last_parts<-parts_file_name[[1]][5]
  last_parts<-parts_file_name[[1]][4]
  last <- substr(last_parts, 1, 1)
  file_name<- paste(first, parts_file_name [[1]][2], parts_file_name[[1]][3], last_parts, very_last_parts, sep="_")
  final_path<-file.path(final_directory, file_name)
  final_path_file <- paste(final_path, ".csv", sep="")
  
  # WRITE TABLE 2 AND MAKE ANNOUNCEMENT 
  # write file an announce process
  write.csv(l_file_6, file = final_path_file, row.names=FALSE) 
  announcement<-paste("Writing: ", file_name)
  print(announcement)
  
  # CREATE A THIRD TABLE FOR PRINTING 
  # Summarizes the summary data by ISO, Year, and Heat_Index_cat_UHI
  l_file_7<-l_file_5%>%
    group_by_(.dots=dots7)%>%
    summarize(Population = sum(as.numeric(Population)),
              number_urban_areas = n(),
              nation_urban_area=sum(Area),
              Density_nation_p_km2 = sum(as.numeric(Population)) / sum(Area), 
              Density_nation_rev_m2_p = (sum(Area) / sum(as.numeric(Population)))*10^6,
              region = mean(region),
              sub_region = mean(sub_region),
              dev_status = mean(dev_status),
              income_grp = mean(income_grp)) %>%
    as.data.frame()
  # Only get the complete cases  
  l_file_7<-l_file_7[complete.cases(l_file_7),]  
  
  # MAKE NAME FOR TABLE 3  
  # Make names for final file
  parts_file_name<-strsplit(name_file_x, "_")
  first<-paste("Summary_", landuse, "_by_Nation_w_UHI", sep = "")
  very_last_parts<-parts_file_name[[1]][5]
  last_parts<-parts_file_name[[1]][4]
  last <- substr(last_parts, 1, 1)
  file_name<- paste(first, parts_file_name [[1]][2], parts_file_name[[1]][3], last_parts, very_last_parts, sep="_")
  final_path<-file.path(final_directory, file_name)
  final_path_file <- paste(final_path, ".csv", sep="")
  
  # WRITE TABLE 3 AND MAKE ANNOUNCEMENT 
  # write file an announce process
  write.csv(l_file_7, file = final_path_file, row.names=FALSE) 
  announcement<-paste("Writing: ", file_name)
  print(announcement)
}


# MAIN SCRIPT

# Create a vector with the working directories 
path_a<-"E:/July_2017/Pop_&_Temp/GRUMP_HI/GRUMP_HI_Final"
path_b<-"E:/July_2017/Pop_&_Temp/GlobCover_HI/GlobCover_HI_Final"
paths<-c(path_a, path_b)

# LOOP THROUGH THE WORKING DIRECTORIES AND CREATE FINAL DIRECTORY PATHS
for(i in 1:length(paths)){
  
  # set paths 
  setwd(paths[i])
  start_directory<-getwd()
  #print(start_directory)
  final_directory<-file.path(start_directory, "Summary_tables", "Urban_areas")
  #print (final_directory)
  
  # get land use name 
  new_name_list<-strsplit(paths[i], "/")
  get_name<-new_name_list[[1]][4]
  get_x<-nchar(get_name)
  landuse_name<-substr(get_name, 1, get_x-3)

  # COLLECT NECESSARY DATA FILES 
  
  # Create a pattern to identify all necessary files and list them. 
  patter<-"^All_RCP.*desc_Area.csv"
  summ_files<-list.files(path = start_directory, pattern = patter)
  
  # LOOP THROUGH THE FILES READ THEM INTO R AS .CSVS AND MAKE THE SUMMARY TABLES 
  for(j in 1:length(summ_files)){
    #print(summ_files[j])
    
    # read in the file (note: use basic R)
    new_file<-read.csv(summ_files[j])
    
    # Function to create two tables based upon nation
    nation.tables(new_file, landuse_name, summ_files[j])
    
  }
  
  # Announcement
  announcement<-paste("FINISHED", start_directory, sep = " ")
  print_space(c(announcement, ""))
}

print("FINISHED")





