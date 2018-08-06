# ======================================================================================================= 
# DESCRIPTION:  This script collects all the 15 day heat wave summary files created from the previous 
#               script and merges these files with a GDP per capita file by SSP.  It calcualte the median 
#               and lower income thresholds at the global level and identifies the populations in these
#               categories that are also experiencing very warm heat indices.  
#               
# DATE:         21 July 2018
# Developer:    Peter J. Marcotullio
# ======================================================================================================= 

# GET LIBRARIES AND PRE-MANAGE THE DATA

# Get libraries
library(dplyr)
library(stringr)
library(reshape2)

# Make lists
ssps<-c("SSP1", "SSP2", "SSP3", "SSP4", "SSP5")
years<-c(2010,2030,2070,2100)
low_add<-data.frame("NONE", "SSP", 1999, ">42", "RCP", 142, 0, 0, 0)

# Put R in appropriate folder
work_folder<-"E:/July_2017/Pop_&_Temp/GRUMP_HI/GRUMP_HI_Final/Summary_tables/Population/15_day/Asia"
setwd(work_folder)

# Get all the summary files 
pat<-"Summary"
all_files<-list.files(work_folder, pattern = pat)

# Get the gdp file and read in
gdp_dat<-"E:/July_2017/Pop_&_Temp/GRUMP_HI/GRUMP_HI_Final/Summary_tables/Population/15_day/Asia/GDP_per_capita_by_SSP.csv"
gdp<-read.csv(gdp_dat)

# FUNCTIONS

line.space<-function(x){
  cat(x, sep = "\n\n")
}

# START THE ANALYSIS

# create empty income threshold values list
income_threshold_list<-list()

# loop through the list of ssps
for(i in 1:length(ssps)){
  
  # GET SSP NUMBER FROM THE FILE NAME
  
  # Get the RCP and the SSP from the name 
  file_name_list<-strsplit(all_files[i], "_")
  ssp_name<-file_name_list[[1]][3]
  ssp_split<-strsplit(ssp_name, "[.]")
  ssp<-ssp_split[[1]][1]
  
  # CREATE A TABLE OF THE MEDIAN AND LOWER INCOME THRESHOLDS FOR THE SSP BY YEAR
  
  # Select for the correct ssp in the gdp file
  gdp_ssp<-gdp[which(gdp$SSP == ssp),]
  
  # Create an empty list of fill and loop through year list  
  median_list<-list()
  for(q in 1:length(years)){
    
    # subset gdp-ssp df for only the specific year
    gdp_ssp_year<-gdp_ssp[which(gdp_ssp$Year == years[q]),]
    # remove nations with 0 GDP
    gdp_ssp_year_no_0<-gdp_ssp_year[ which(gdp_ssp_year$GDP_capita > 0),]
    
    # summarize the data by ssp, year, low middle income and low income thresholds
    gdp_median<-gdp_ssp_year_no_0%>%
      group_by(SSP, Year)%>%
      summarize(mean_gdp = mean(GDP_capita),
                median = median(GDP_capita),
                max = max(GDP_capita),
                min = min(GDP_capita),
                low_middle = median(GDP_capita)/2 +1000,
                low_income = (low_middle)/2,
                count_low= length(ISO[GDP_capita<=low_middle]),
                count_total = length(ISO))%>%
      as.data.frame()
    
    # put the median value tables into a list
    median_list[[q]]<-gdp_median
  }
  
  # Create one table with all the median value tables created 
  median_df<-do.call(rbind, median_list)
  
  # Put values in a bind later
  income_threshold_list[[i]]<-median_df

  # read in Asia summary file 
  asia_df<-read.csv(all_files[i])
  
  # merge the datafiles
  merge.file<-merge(asia_df, gdp_ssp, by = c("ISO","SSP", "Year"), all.x = TRUE)
  
  # replace NAs with 0
  merge.file[is.na(merge.file)]<-0
  
  adapt_capacity_list<-list()
  # loop through the data by year again 
  for(m in 1:length(years)) {
    
    # Get middle income and low income thresholds 
    median_year<-subset(median_df, Year == years[m])
    median<-round(median_year$low_middle, 0)
    low<-round(median_year$low_income, 0) 
    
    # announce values
    announce<-paste("This is the ", years[m], ", ", ssp, " lower middle class income threshold: ", median, " and this is the lower income threshold: ", low, sep = "")
    print(announce)
    
    # select the data from the correct year
    merge.file.year<-merge.file[ which(merge.file$Year == years[m]), ]
    
    # Select all Middle income nations
    asia_middle<-merge.file.year[ which(merge.file.year$GDP_capita <= median & merge.file.year$GDP_capita > low),]
    
    # Select all Low income nations with GDP > 0
    asia_low<-merge.file.year[which(merge.file.year$GDP_capita <= low & merge.file.year$GDP_capita > 0),]
    
    if(nrow(asia_low)<=1){
      names(low_add)<-names(asia_low)
      asia_low<-rbind(low_add, asia_low)
    }
    
    # summarize to get total population by sub-region
    mid_low_Adap_cap<-asia_middle%>%
      group_by(SSP, Year, RCP, Sub_region)%>%
      summarize(pop = sum(as.numeric(Population)))
    
    low_income_Adap_cap<-asia_low%>%
      group_by(SSP, Year, RCP, Sub_region)%>%
      summarize(pop = sum(as.numeric(Population)))
    
    mid_low_Adap_cap$income_cat<-"Lower_middle"
    low_income_Adap_cap$income_cat<-"Low"
    
    final_table<-rbind(mid_low_Adap_cap, low_income_Adap_cap)
    
    adapt_capacity_list[[m]]<-final_table
  } 

  # Create one table with all the median value tables created 
  low_adapt<-do.call(rbind, adapt_capacity_list)
  
  # SAVE THESE FILES
  
  # file name
  file_name<-paste("Adapt_capacity_low", ssp, "low.csv", sep="_")
  final_file_path<- file.path(work_folder, file_name)
  
  #save file
  write.csv(low_adapt, final_file_path, row.names =FALSE)
  
  line.space(" ")
}


# Create one table with all the income values
income_thresholds<-do.call(rbind, income_threshold_list)

# Create name and path for saving
income_file_name<-"Adapt_capacity_Incomes.csv"
final_income_file_path<-file.path(work_folder, income_file_name)

# save file
write.csv(income_thresholds, final_income_file_path, row.names=FALSE)