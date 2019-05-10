# 
# ==============================================================================================
# THIS SCRIPT UP THE SUMMARY FILES FOR GRUMP AND GLOBCOVER THAT WERE CREATED FROM THE 1, 3 AND 15 
# DAY ANALYSES.  IT FURTHER SUMMERIZES THESE DATA BY WORLD, REGION AND SUB-REGION (3 TABLES) AND 
# PRODUCES A SERIES OF 5 FACET CHARTS FOR POPULATION IN HOT ( > 42 c HEAT WAVES) BY SSP AND RCP.     
# THE SCRIPT CAN BE RUN ONCE FOR BOTH GRUMP AND GLOBCOVER DATA.  
#
# Peter J. Marcotullio, JANUARY 2018
# ==============================================================================================

# INSTALL PACKAGES

# install.packages("dplyr")
# install.packages("stringr")

# GET LIBRARIES  

library(ggplot2)
library(dplyr)
library(reshape2)

# MAKE LISTS AND vectors

# Create a vector with the working directories 
path_table_a<-"E:/July_2017/Pop_&_Temp/GRUMP_HI/GRUMP_HI_Final/Summary_tables"
path_table_b<-"E:/July_2017/Pop_&_Temp/GlobCover_HI/GlobCover_HI_Final/Summary_tables"
paths_table<-c(path_table_a, path_table_b)

# Create a vector with the working directories 
path_chart_a<-"E:/July_2017/Pop_&_Temp/GRUMP_HI/GRUMP_HI_Final/Summary_charts"
path_chart_b<-"E:/July_2017/Pop_&_Temp/GlobCover_HI/GlobCover_HI_Final/Summary_charts"
paths_chart<-c(path_chart_a, path_chart_b)

# Create other lists to use later
heat_wave_length<-c(1,5,15)
Region_list<-c("Africa", "Asia")
file_land_use_names = c("GRUMP", "GlobCover")

# LOOP THROUGH THE WORKING DIRECTORIES AND CREATE FINAL DIRECTORY PATHS

for(m in 1:length(paths_table)){

  # Create a name for the data files to retrieve
  dat<-"XX_heat_wave_data_summary.csv"
  dat<-gsub("XX", file_land_use_names[m], dat)

  # Create paths to the tables and charts folders
  table_path<-paths_table[m]
  chart_path<-paths_chart[m]

  # Create path to the working data
  working_data_path<-paste(table_path, dat, sep = "/")

  # Read in the working data and attach file
  working_dat<-read.csv(working_data_path, header = TRUE, sep=",")
  attach(working_dat)

  # Select for hot temperature only
  new_working_dat<-working_dat[ which(working_dat$HI_cat == ">42 & <=55" | working_dat$HI_cat == ">55"), ]

  # Change year from 2030 to 2040 for better charts
  new_working_dat$Year[new_working_dat$Year == 2030]<-2040

  # MAKE THE TABLES 

  # Create two tables for UHI AND NO UHI, to merge (creating two different variables (from long to wide)
  regions_sen_UHI<-new_working_dat[which(new_working_dat$UHI_cat == "With UHI"),]
  regions_sen_no_UHI<-new_working_dat[which(new_working_dat$UHI_cat == "Without UHI"),]


  # Combine/merge tables 
  merged_regions<-merge(regions_sen_UHI, regions_sen_no_UHI, by = c("Region", "Region_name", "sub_region",
                                                                 "Sub_region_name", "SSP", "RCP", "Year",
                                                                 "Heat_wave_days", "HI_cat"))

  # Create new variables by substracting the total and sensitive "with UHI" population from the "without UHI"   
  merged_regions$Added_pop<-merged_regions$Population.x - merged_regions$Population.y
  merged_regions$Added_pop_sen<-merged_regions$Sensitive_pop.x - merged_regions$Sensitive_pop.y
  
  # Summarize table by sub-regions NOTE: combined populations for HI_cat so we combine population for two categories
  summary.merged.sub_regions<-merged_regions%>%
    group_by(sub_region, Sub_region_name, Region_name, SSP, RCP, Year, Heat_wave_days)%>%
    summarize(pop_UHI = sum(as.numeric(Population.x)),
              pop_no_UHI = sum(as.numeric(Population.y)),
              sen_pop_UHI = sum(as.numeric(Sensitive_pop.x)),
              sen_pop_no_UHI = sum(as.numeric(Sensitive_pop.y)),
              UHI_pop_added = sum(as.numeric(Added_pop)),
              UHI_pop_sen_added = sum(as.numeric(Added_pop_sen)),
              Share_sen_pop_UHI = sen_pop_UHI / pop_UHI,
              Share_sen_pop_no_UHI = sen_pop_no_UHI / pop_no_UHI)%>%
    as.data.frame()
  
  # Create name for file
  file_name<-paste (file_land_use_names[m], "sub_regional_heat_wave_data_summary.csv", sep = "_")
  full_file_path<-paste(table_path, file_name, sep = "/")
  
  # Save dataframe
  write.csv(summary.merged.sub_regions, file = full_file_path, row.names = FALSE)
  
  # Summarize table by regions 
  summary.merged.regions<-merged_regions%>%
    group_by(Region_name, SSP, RCP, Year, Heat_wave_days)%>%
    summarize(pop_UHI = sum(as.numeric(Population.x)),
              pop_no_UHI = sum(as.numeric(Population.y)),
              sen_pop_UHI = sum(as.numeric(Sensitive_pop.x)),
              sen_pop_no_UHI = sum(as.numeric(Sensitive_pop.y)),
              UHI_pop_added = sum(as.numeric(Added_pop)),
              UHI_pop_sen_added = sum(as.numeric(Added_pop_sen)),
              Share_sen_pop_UHI = sen_pop_UHI / pop_UHI,
              Share_sen_pop_no_UHI = sen_pop_no_UHI / pop_no_UHI)%>%
    as.data.frame()

  # Create name for file
  file_name<-paste (file_land_use_names[m], "regional_heat_wave_data_summary.csv", sep = "_")
  full_file_path<-paste(table_path, file_name, sep = "/")
  
  # Save dataframe
  write.csv(summary.merged.regions, file = full_file_path, row.names = FALSE)

  # Summarize table for the world
  summary.merged.world<-merged_regions%>%
    group_by(SSP, RCP, Year, Heat_wave_days)%>%
    summarize(pop_UHI = sum(as.numeric(Population.x)),
              pop_no_UHI = sum(as.numeric(Population.y)),
              sen_pop_UHI = sum(as.numeric(Sensitive_pop.x)),
              sen_pop_no_UHI = sum(as.numeric(Sensitive_pop.y)),
              UHI_pop_added = sum(as.numeric(Added_pop)),
              UHI_pop_sen_added = sum(as.numeric(Added_pop_sen)),
              Share_sen_pop_UHI = sen_pop_UHI / pop_UHI,
              Share_sen_pop_no_UHI = sen_pop_no_UHI / pop_no_UHI)%>%
    as.data.frame()

  # Create name for file
  file_name<-paste (file_land_use_names[m], "global_heat_wave_data_summary.csv", sep = "_")
  full_file_path<-paste(table_path, file_name, sep = "/")
    
  # Save dataframe
  write.csv(summary.merged.world, file = full_file_path, row.names = FALSE)

  
  # MAKE CHARTS 
  
  # Create two lists of variables we want later for graphs of world
  UHI_pop_data<-c("SSP", "RCP", "Year", "Heat_wave_days", "pop_UHI", "sen_pop_UHI")
  pop_added_data<-c("SSP", "RCP", "Year", "Heat_wave_days", "UHI_pop_added", "UHI_pop_sen_added")
  
  # Select for the list of variables for two different files 
  summary.merged.world.UHI<-summary.merged.world[UHI_pop_data]
  summary.merged.world.added<-summary.merged.world[pop_added_data]
  
  # make long file for world different graphs by melting
  summary.merged.world.UHI.long<-melt(summary.merged.world.UHI, id = c("SSP", "RCP", "Year", "Heat_wave_days"))
  summary.merged.world.added.long<-melt(summary.merged.world.added, id = c("SSP", "RCP", "Year", "Heat_wave_days"))
  
  # CREATE GRAPHS FOR THE REGIONS
  
  # create POPULATION graphs for each heat wave length 
  for(i in 1:length(heat_wave_length)){
    
    # Select regional data for the correct heat wave 
    regions_hw_len<-summary.merged.regions[which(summary.merged.regions$Heat_wave_days == heat_wave_length[i]),]
    
    # Select world data for correct heat wave
    world_hw_len_UHI<-summary.merged.world.UHI.long[which(summary.merged.world.UHI.long$Heat_wave_days == heat_wave_length[i]),]
    
    # Select world data for correct heat wave
    world_hw_len_added<-summary.merged.world.added.long[which(summary.merged.world.added.long$Heat_wave_days == heat_wave_length[i]),]
    
    
    # Correct for plurals for title
    if(heat_wave_length[i]==1){
      days = "day"
    } else {
      days = "days"
    }
    
    # Make chart titles 
    title_q<-paste("Urban population by region exposed to hot temperatures in heat wave of", heat_wave_length[i], days, sep = " ")
    title_w<-paste("World urban population exposed to hot temperatures in heat wave of", heat_wave_length[i], days, sep = " ")
    title_a<-paste("World urban population added to hot temp category due to UHI in heat wave of", heat_wave_length[i], days, sep = " ")
    title_b<-paste("Regional share of sensitive urban population exposed to hot temperatures in heat wave of", heat_wave_length[i], days, sep = " ")
    
    
    # CHART 1
    # FACET PLOT OF REGIONS BY SSP BY RCP TOTAL POPULATION EXPOSED TO HOT HEAT INDEX (>42 C)
    
    # Use ggplot to create a set of line graphs (geom_smooth no confidence intervals) with specific y units (billions), facet the graph and plot it
    gg_pop_region <- ggplot(regions_hw_len, aes(x = Year, y = pop_UHI, color = Region_name)) + geom_smooth(se=FALSE) +
      scale_y_continuous(limits = c(0, 4000000000), labels = function(x) format(x/1000000000)) 
    
    gg_pop_region<- gg_pop_region + facet_grid(RCP ~ SSP)
    
    gg_pop_region<-gg_pop_region + labs(title = title_q, color = "Region", caption = "(Hot temp: > 42 C)") + ylab("Population (billions)")+
      theme(axis.line = element_line(color="black", size = 0.5), plot.title = element_text(size = 14, face = "bold"), panel.grid.major = element_blank(), panel.grid.minor = element_blank())  + theme(panel.background = element_blank())
    
    plot(gg_pop_region)
    
    # make final file name
    file_name_identifier<-paste(heat_wave_length[i], days, sep = "_" )
    file_name_end<-paste(file_name_identifier, ".pdf", sep="")
    file_name<-paste(file_land_use_names[m],"Regional_pop_with_UHI", file_name_end, sep="_" )
    full_path<-paste(chart_path, file_name, sep = "/")
    
    # Save chart file 
    ggsave(filename = full_path, plot = gg_pop_region, width = 10, height = 7, units = "in",  dpi = 300)
    
    
    # CHART 2
    # WORLD CHARTS OF URBAN POPULATION EXPOSED TO HOT TEMPERATURES WITH UHI

    # Use ggplot to create a set of line graphs (geom_smooth no confidence intervals) with specific y units (billions), facet the graph and plot it    
    gg_pop_world<-ggplot(world_hw_len_UHI, aes(x = Year, y = value, color = variable)) + geom_smooth(se=FALSE) +
      scale_y_continuous(limits = c(0, 8000000000), labels = function(x) format(x/1000000000)) 
    
    gg_pop_world<- gg_pop_world + facet_grid(RCP ~ SSP)
    
    gg_pop_world<-gg_pop_world + labs(title = title_w, caption = "(Hot temp: > 42 C)") + ylab("Population (billions)")+
      theme(axis.line = element_line(color="black", size = 0.5), plot.title = element_text(size = 14, face = "bold"), panel.grid.major = element_blank(), panel.grid.minor = element_blank())  + theme(panel.background = element_blank())
    
    gg_pop_world<-gg_pop_world + scale_color_discrete(name = "Population type", breaks = c("pop_UHI", "sen_pop_UHI"),
                                                      labels = c("Total Pop", "Sensitive Pop"))
    plot(gg_pop_world)  
    
    # make final file name
    file_name_identifier<-paste(heat_wave_length[i], days, sep = "_" )
    file_name_end<-paste(file_name_identifier, ".pdf", sep="")
    file_name<-paste(file_land_use_names[m],"World_pop_with_UHI", file_name_end, sep="_" )
    full_path<-paste(chart_path, file_name, sep = "/")
    
    # Save chart file 
    ggsave(filename = full_path, plot = gg_pop_world, width = 10, height = 7, units = "in",  dpi = 300)
    

    # CHART 3
    # WORLD CHARTS OF URBAN POPULATIONS ADDED TO HOT TEMPERATURE CATEGORIES DUE TO UHI

    # Use ggplot to create a set of line graphs (geom_smooth no confidence intervals) with specific y units (billions), facet the graph and plot it       
    gg_pop_world<-ggplot(world_hw_len_added, aes(x = Year, y = value, color = variable)) + geom_smooth(se=FALSE) +
      scale_y_continuous(limits = c(0, 2000000000), labels = function(x) format(x/1000000000)) 
    
    gg_pop_world_a<- gg_pop_world + facet_grid(RCP ~ SSP)
    
    gg_pop_world_a<-gg_pop_world_a + labs(title = title_a, caption = "(Hot temp: > 42 C)") + ylab("Population (billions)")+
      theme(axis.line = element_line(color="black", size = 0.5), plot.title = element_text(size = 14, face = "bold"), panel.grid.major = element_blank(), panel.grid.minor = element_blank())  + theme(panel.background = element_blank())
    
    gg_pop_world_a<-gg_pop_world_a + scale_color_discrete(name = "Population type", breaks = c("UHI_pop_added", "UHI_pop_sen_added"),
                                                          labels = c("Total Pop added", "Sensitive Pop added"))
    plot(gg_pop_world_a)  
    
    # make final file name
    file_name_identifier<-paste(heat_wave_length[i], days, sep = "_" )
    file_name_end<-paste(file_name_identifier, ".pdf", sep="")
    file_name<-paste(file_land_use_names[m], "World_pop_added_with_UHI", file_name_end, sep="_" )
    full_path<-paste(chart_path, file_name, sep = "/")
    
    # Save chart file 
    ggsave(filename = full_path, plot = gg_pop_world_a, width = 10, height = 7, units = "in",  dpi = 300)  

        
    # CHART 4
    # FACET PLOT OF REGIONS BY SSP BY RCP SHARE OF SENSITIVE POPULATION EXPOSED TO HOT HEAT INDEX

    # Use ggplot to create a set of line graphs (geom_smooth no confidence intervals) with share as y-axis units (0-1), facet graph and plot it           
    gg_pop_region_share <- ggplot(regions_hw_len, aes(x = Year, y = Share_sen_pop_UHI, color = Region_name)) + geom_smooth(se=FALSE) 
    
    gg_pop_region_share<- gg_pop_region_share + facet_grid(RCP ~ SSP)
    
    gg_pop_region_share<-gg_pop_region_share + labs(title = title_b, color = "Region", caption = "(Hot temp: > 42 C, Senstive pop: <5 yrs & >65 yrs)") + ylab("Share Sensitive Population")+
      theme(axis.line = element_line(color="black", size = 0.5), plot.title = element_text(size = 14, face = "bold"), panel.grid.major = element_blank(), panel.grid.minor = element_blank())  + theme(panel.background = element_blank())
    
    plot(gg_pop_region_share)
    
    # make final file name
    file_name_identifier<-paste(heat_wave_length[i], days, sep = "_" )
    file_name_end<-paste(file_name_identifier, ".pdf", sep="")
    file_name<-paste(file_land_use_names[m], "Regional_share_sensitive_with_UHI", file_name_end, sep="_" )
    full_path<-paste(chart_path, file_name, sep = "/")
    
    # Save chart file 
    ggsave(filename = full_path, plot = gg_pop_region_share, width = 10, height = 7, units = "in",  dpi = 300)    
 
       
    # CHARTS 5 & 6
    # FACET PLOTS OF POPULATIONS EXPERIENCING HOT HEAT WAVES BY SUB-REGION WITH SPECIAL REGIONS (AFRICA AND ASIA)
    
    # Select for the region from the region list
    for(q in 1:length(Region_list)){
      
      # Get the region name
      region_name<-Region_list[q]
      print(region_name)
      
      # Set the limits for the y axis units
      if(Region_list[q]=="Africa"){
        limit = c(0, 1500000000)
      }else {
        limit = c(0, 2500000000)
      }
      
      # Select only data for that region
      sub_regions_sen_current<-summary.merged.sub_regions[which(summary.merged.sub_regions$Region_name == Region_list[q]),]
      
      # Select for data for the correct heat wave and with UHI
      sub_regions_hw_len<-sub_regions_sen_current[which(sub_regions_sen_current$Heat_wave_days == heat_wave_length[i]),]
      
      # Make a chart title 
      title_sp<-paste(region_name, "sub-regional urban population exposed to hot temperatures in heat wave of", heat_wave_length[i], days, sep = " ")
      
      # FACET PLOT OF REGIONS BY SSP BY RCP TOTAL POPULATION EXPOSED TO HOT HEAT INDEX

      # Use ggplot to create a set of line graphs (geom_smooth no confidence intervals) with specific y-axis units as specified by the conditional snipet, facet graph and plot it                 
      gg_pop_special <- ggplot(sub_regions_hw_len, aes(x = Year, y = pop_UHI, color = Sub_region_name)) + geom_smooth(se=FALSE) +
        scale_y_continuous(limits = limit, labels = function(x) format(x/1000000000)) 
      
      gg_pop_special<- gg_pop_special + facet_grid(RCP ~ SSP)
      
      gg_pop_special<-gg_pop_special + labs(title = title_sp, color = "Sub-region", caption = "(Hot temp: > 42 C)") + ylab("Population (billions)")+
        theme(axis.line = element_line(color="black", size = 0.5), plot.title = element_text(size = 14, face = "bold"), panel.grid.major = element_blank(), panel.grid.minor = element_blank())  + theme(panel.background = element_blank())
      
      plot(gg_pop_special)
      
      # make final file name
      file_name_identifier<-paste(region_name, heat_wave_length[i], days, sep = "_" )
      file_name_end<-paste(file_name_identifier, ".pdf", sep="")
      file_name<-paste(file_land_use_names[m], "Sub_Regional_share_sensitive_with_UHI", file_name_end, sep="_" )
      full_path<-paste(chart_path, file_name, sep = "/")
      
      # Save chart file 
      ggsave(filename = full_path, plot = gg_pop_special, width = 10, height = 7, units = "in",  dpi = 300)            
      
    }
    
  }
  
}  
  
# STOP HERE!!












