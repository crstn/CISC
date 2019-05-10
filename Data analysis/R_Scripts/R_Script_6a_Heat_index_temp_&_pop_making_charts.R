# 
# ==============================================================================================
# THIS SCRIPT OPENS ALL THE FINAL HEAT INDEX CSV FILES WITH DESCRIPTIVES IN BASE R READ.CSV. IT 
# CREATES FIVE CHARTS AT THE GLOBAL, REGIONAL AND SUBREGIONAL SCALES FOR POPULATIONS AND 
# TEMPERATURES .  IT ALSO PRODUCES FOUR TABLES THAT SUMMARIZE RESULTS BY RCP, SSP AND HEAT WAVE 
# LENGTH.  IT RUNS FOR BOTH GRUMP AND GLOBCOVER DATA
#
# Peter J. Marcotullio, JANUARY 2018
# ==============================================================================================
#
# INSTALL PACKAGES
#
# install.packages("dplyr")
# install.packages("stringr")
# install.packages("ggplot2")

# GET LIBRARIES 

library(dplyr)
library(stringr)
library(ggplot2)

# MAKE FUNCTIONS

line.space = function(x){
  cat(x, sep="\n")
}

# MAKE LISTS AND VECTORS
file_land_use_names = c("GRUMP", "GlobCover")

# CREATE COLOR RAMP

# Make a color ramp white to red through yellow
colfunc<-colorRampPalette(c("white", "yellow", "orange", "red"))
# prints out the color numbers in 12 steps
colfunc(12)
# print out the color ramp using 12 steps
plot(rep(1,12),col=colfunc(12),pch=19,cex=3)

# CREATE GROUPINGS NAMES FOR HEAT INDEX ANALYSES 

# Create group names for first analysis
group_names_1<-list("Year", "GCM_Model")
# # convert the character vector to list of symbols
dots1<-lapply(group_names_1, as.symbol)

# Create group names for second analysis 
group_names_2<-list("Year", "GCM_Model", "New_HI_cat")
# # convert the character vector to list of symbols
dots2<-lapply(group_names_2, as.symbol)

# Create group names for third analysis 
group_names_3<-list("Year", "New_HI_cat")
# # convert the character vector to list of symbols
dots3<-lapply(group_names_3, as.symbol)

# Create group names for fourth analysis 
group_names_4<-list("Year", "GCM_Model", "ISO")
# # convert the character vector to list of symbols
dots4<-lapply(group_names_4, as.symbol)

# Create group names for fourth analysis 
group_names_5<-list("Year","GCM_Model", "Sub.region")
# # convert the character vector to list of symbols
dots5<-lapply(group_names_5, as.symbol)

# Create group names for fourth analysis 
group_names_6<-list("Year","GCM_Model", "Region")
# # convert the character vector to list of symbols
dots6<-lapply(group_names_6, as.symbol)

# set the working  and final directories
path_a<-"E:/July_2017/Pop_&_Temp/GRUMP_HI/GRUMP_HI_Final"
path_b<-"E:/July_2017/Pop_&_Temp/GlobCover_HI/GlobCover_HI_Final" # MIGHT NEED TO CHANGE THIS!
paths<-c(path_a, path_b)


# LOOP THROUGH BOTH LAND USE SETS OF DATA
for(j in 1:length(paths)){
  
  # SET DIRECTORIES AND NAMES
  
  # Set current directory and call is "start directory"
  setwd(paths[j])
  start_directory<-getwd()
  #print(start_directory)
  
  # Set final directory for charts 
  final_directory<-file.path(start_directory, "Summary_charts")
  #print (final_directory)
  
  # Set final directory for most tables 
  final_table_directory<-file.path(start_directory, "Summary_tables")
  
  # set final directory heat index temps
  final_HI_table_directory<-file.path(start_directory, "Summary_tables", "Heat_Index_temp")
  
  # Get land use name for file naming
  land_use_name<-file_land_use_names[j]

  # Create a pattern to identify all necessary files and list them. 
  p<-"^All_RCP.*desc.csv"
  sum_files<-list.files(path = start_directory, pattern = p)

  # ITERATE THROUGH ALL THE FILES AND PERFORM ANALYSIS TO PRODUCE TABLES AND CHARTS

  # loop through all the files
  for(i in 1:length(sum_files)){
    
    # Make parts of name for ssp number 
    parts_file_name<-strsplit(sum_files[i], "_")
    last_parts<-parts_file_name[[1]][4]
    ssp <- unlist(strsplit(last_parts, split = ".", fixed = TRUE))[1]
    ssp_number<-paste(parts_file_name[[1]][3], ssp, sep = " ")
    ssp_number_for_file<- paste(parts_file_name[[1]][3], ssp, sep="_")
    rcp_number<-parts_file_name[[1]][2]
    heat_wave_number<-parts_file_name[[1]][5]
    #print(parts_file_name)
    #print(ssp_number)
    #print(rcp_number)
    #print(heat_wave_number)
    #line.space(c(ssp_number_for_file, " "))

    # Get RCP number for the chart
    if(rcp_number == "RCP2p6"){
      rcp_name<-"RCP 2.6"
    }else if(rcp_number == "RCP4p5"){
      rcp_name<- "RCP 4.5"
    }else if(rcp_number == "RCP6p0"){
      rcp_name<- "RCP 6.0"
    }else if(rcp_number == "RCP8p5"){
      rcp_name <- "RCP 8.5"
    } else
      rcp_name <- "Unknown"
    
    # Get heat wave number for the chart
    if(heat_wave_number == "01"){
      heat_wave_name<-1
    }else if(heat_wave_number == "05"){
      heat_wave_name<-5
    }else if(heat_wave_number == "15"){
      heat_wave_name<-15
    } else
      heat_wave_name <- "Unknown"
    
    # Correct for plurals for title
    if(heat_wave_name==1){
      days = "day"
    } else {
      days = "days"
    }
    
    # read in file
    new_file<-read.csv(sum_files[i])
    
    # Change year from 2030 to 2040 for better charts
    new_file$Year[new_file$Year == 2030]<-2040
    
    # Recode into new variable Change categories adding >55 to ">42"
    new_file$New_HI_cat<-""
    new_file$New_HI_cat[new_file$Heat_Index_cat_UHI == ">55"]<-">42"
    new_file$New_HI_cat[new_file$Heat_Index_cat_UHI == ">42 & <=55"]<-">42"
    new_file$New_HI_cat[new_file$Heat_Index_cat_UHI == ">28 & <=34"]<-">28 & <=34"
    new_file$New_HI_cat[new_file$Heat_Index_cat_UHI == ">34 & <=42"]<-">34 & <=42"
    new_file$New_HI_cat[new_file$Heat_Index_cat_UHI == "<=28"]<-"<=28"
    
    # CHART 1 
    
    # summarize data by "Year", "GCM_Model" & "HI_cat_UHI" 
    l_chart_file<-new_file%>%
      group_by_(.dots = dots2) %>%
      summarize(Urban_extents = n(),
                Population = sum(as.numeric(Population)))%>%
      as.data.frame()
    
    # Select for only complete cases (removes NAs)
    l_chart_file<-l_chart_file[complete.cases(l_chart_file),]
    
    # Graph the data by Population size under different heat index categories
    plot_1<-ggplot(data=l_chart_file, aes(x=Year, y=Population, color=factor(New_HI_cat))) + 
      geom_smooth(size = 1, level = 0.95) + 
      scale_color_manual(values = colfunc(4)) + 
      scale_y_continuous(limits = c(0, 7000000000), labels = function(x) format(x/1000000000)) 
    
    # Create title name
    title_q<-paste("Urban population by heat index category for heat wave of ", heat_wave_name, " ", days, ", ", rcp_name, ", ", ssp_number, sep = "")
    
    # Create chart
    plot_1 <- plot_1 + labs(title = title_q, color = "Heat Index Category (C)")+ 
      ylab("Population (billions)") + 
      theme(axis.line = element_line(color="black", size = 0.5), plot.title = element_text(size = 14, face = "bold"), panel.grid.major = element_blank(), panel.grid.minor = element_blank()) + 
      theme(panel.background = element_blank())
    
    # plot chart    
    print(plot_1)

    # Make file name for chart
    file_name_identifier<-paste(rcp_number, ssp_number_for_file, heat_wave_number, sep = "_" )
    file_name_end<-paste(file_name_identifier, ".jpg", sep="")
    file_name<-paste(land_use_name,"Global_pop_by_HI_cat", file_name_end, sep="_" )
    full_path<-paste(final_directory, file_name, sep = "/")

    # Announce process saving chart
    announcement_1<-paste("Graphing: ", title_q, sep = " ")
    line.space(c(announcement_1, " "))
    ggsave(filename=full_path, plot=plot_1,  width = 10, height = 7, units = "in",  dpi = 300)

    # Get heat wave number for the chart
    if(heat_wave_number == "01"){
      limit<-c(35, 50)
    }else if(heat_wave_number == "05"){
      limit<-c(33, 47)
    }else if(heat_wave_number == "15"){
      limit<-c(31, 43)
    } else
      limit<-c(32, 50)
    
    #  SUMMARY TABLES AND CHARTS FOR GLOBAL, REGIONAL, SUB-REGIONAL AND NATIONAL TEMPERATURES
    
    # CHART 2 GLOBAL TEMPERATURES BY RCP, SSP AND GCM MODELS (3 versions)
    
    # VERSION 1 GLOBAL TEMPERATURES BY RCP, SSP AND GCM MODELS
    
    # Summarize data by Year & GCM model 
    l_chart_file_1<-new_file%>%
      group_by_(.dots = dots1) %>%
      summarize(Urban_extents = n(),
                Population = sum(as.numeric(Population)),
                Mean_HI_w_UHI = mean(HI_w_UHI),
                Mean_HI_wo_UHI = mean(Heat_Index))%>%
      as.data.frame()
    
    # use "complete cases" to remove NAs
    l_chart_file_1<-l_chart_file_1[complete.cases(l_chart_file_1),]
    
    
    # Graph the data for  global urban heat index over century  
    plot_2<-ggplot(data=l_chart_file_1, aes(x=Year, y=Mean_HI_w_UHI)) + 
      geom_point(aes(color = GCM_Model), size = 2) + 
      geom_smooth() + 
      coord_cartesian(ylim = limit) 
    
    # Create title
    title_q<-paste("Mean global urban heat index for heat wave of ", heat_wave_name, " ", days, ", ", rcp_name, ", ", ssp_number, sep = "")
    
    plot_2<- plot_2 + labs(title = title_q) + ylab("Heat Index (Celcius)")
    
    plot_2<- plot_2 + scale_color_discrete(name = "GCM Model",
                                           breaks = c("GFDL_ESM2M", "HadGEM2_ES", "IPSL_CM5A_LR", "MIROC_ESM_CHEM", "NorESM1_M"),
                                           labels = c("GFDL ESM2M", "HadGEM2 ES", "IPSL CM5A LR", "MIROC ESM CHEM", "NorESM1 M")) + 
      theme(axis.line = element_line(color="black", size = 0.5), plot.title = element_text(size = 14, face = "bold"), panel.grid.major = element_blank(), panel.grid.minor = element_blank()) + 
      theme(panel.background = element_blank())
    
    # print the graph
    print(plot_2)
    
    # Make file name for chart
    file_name_identifier<-paste(rcp_number, ssp_number_for_file, heat_wave_number, sep = "_" )
    file_name_end<-paste(file_name_identifier, ".jpg", sep="")
    file_name<-paste(land_use_name,"Global_mean_Heat_Index_with_points", file_name_end, sep="_" )
    full_path<-paste(final_directory, file_name, sep = "/")
    
    # Announce process saving chart
    announcement_1<-paste("Graphing: ", title_q, "V.1", sep = " ")
    line.space(c(announcement_1, " "))
    ggsave(filename=full_path, plot=plot_2,  width = 10, height = 7, units = "in",  dpi = 300)      
    
    # CREATE NEW TABLEs FOR NATIONAL, REGIONAL AND SUB-REGIONAL URBAN TEMPERATURES 
    
    # Summarize data by Year, GCM & NATION 
    l_chart_file_1a<-new_file%>%
      group_by_(.dots = dots4) %>%
      summarize(Urban_extents = n(),
                Population = sum(as.numeric(Population)),
                Mean_HI_w_UHI = mean(HI_w_UHI),
                Mean_HI_wo_UHI = mean(Heat_Index),
                sub_region = mean(Sub.region),
                region = mean(Region))%>%
      as.data.frame()
    
    # use "complete cases" to remove NAs
    l_chart_file_1a<-l_chart_file_1a[complete.cases(l_chart_file_1a),]
    
    # Create name for table file
    file_name<-paste (land_use_name, rcp_number, ssp_number_for_file, heat_wave_number, "national_temperature.csv", sep = "_")
    full_file_path<-paste(final_HI_table_directory, file_name, sep = "/")
    
    # Change year back to 2030 for table data
    l_chart_file_1a$Year[l_chart_file_1a$Year == 2040]<-2030
    
    # Save dataframe
    write.csv(l_chart_file_1a, file = full_file_path, row.names = FALSE)
    
    # Summarize data by Year, GCM & SUB-REGION 
    l_chart_file_1b<-new_file%>%
      group_by_(.dots = dots5) %>%
      summarize(Urban_extents = n(),
                Population = sum(as.numeric(Population)),
                Mean_HI_w_UHI = mean(HI_w_UHI),
                Mean_HI_wo_UHI = mean(Heat_Index),
                region = mean(Region))%>%
      as.data.frame()
    
    # use "complete cases" to remove NAs
    l_chart_file_1b<-l_chart_file_1b[complete.cases(l_chart_file_1b),]
    
    # Create name for table file
    file_name<-paste (land_use_name, rcp_number, ssp_number_for_file, heat_wave_number, "sub_regional_temperature.csv", sep = "_")
    full_file_path<-paste(final_HI_table_directory, file_name, sep = "/")
    
    # Change year back to 2030 for table data
    l_chart_file_1b$Year[l_chart_file_1b$Year == 2040]<-2030
    
    # Save dataframe
    write.csv(l_chart_file_1b, file = full_file_path, row.names = FALSE) 
    
    # Summarize data by Year, GCM & REGION 
    l_chart_file_1c<-new_file%>%
      group_by_(.dots = dots6) %>%
      summarize(Urban_extents = n(),
                Population = sum(as.numeric(Population)),
                Mean_HI_w_UHI = mean(HI_w_UHI),
                Mean_HI_wo_UHI = mean(Heat_Index),
                region = mean(Region))%>%
      as.data.frame()
    
    # use "complete cases" to remove NAs
    l_chart_file_1c<-l_chart_file_1c[complete.cases(l_chart_file_1c),]
    
    # Create name for table file
    file_name<-paste (land_use_name, rcp_number, ssp_number_for_file, heat_wave_number, "regional_temperature.csv", sep = "_")
    full_file_path<-paste(final_HI_table_directory, file_name, sep = "/")
    
    # Change year back to 2030 for table data
    l_chart_file_1c$Year[l_chart_file_1c$Year == 2040]<-2030
    
    # Save dataframe
    write.csv(l_chart_file_1c, file = full_file_path, row.names = FALSE) 
    
    
    # VERSION 2 GLOBAL TEMPERATURES BY RCP, SSP AND GCM MODELS
    
    # get the stadard error for each group 
    l_chart_file_1_summary<-l_chart_file_1%>%
      group_by(Year) %>%
      summarise(mean_HI = mean(Mean_HI_w_UHI), # Get mean of mean HIs for each group
                sd_HI = sd(Mean_HI_w_UHI), # Get standard deviation for each group
                n_HI = n(), # Returns the sample size of each group
                SE_HI = sd(Mean_HI_w_UHI)/sqrt(n())) # cacluates the standard error of each group
    
    # Create title 
    title_q<-paste("Mean global urban heat index for heat wave of ", heat_wave_name, " ", days, ", ", rcp_name, ", ", ssp_number, sep = "")
    
    
    plot_2a<-ggplot(data=l_chart_file_1_summary, aes(x=Year, y=mean_HI)) + 
      geom_errorbar(data = l_chart_file_1_summary, mapping=aes(x= Year, ymin= mean_HI - sd_HI, ymax = mean_HI + sd_HI), color = "red", width=1.0) +
      geom_point(data=l_chart_file_1_summary, aes(x=Year, y=mean_HI), size =3, shape = 21, fill = "blue") +
      geom_smooth(data=l_chart_file_1, aes(x=Year, y=Mean_HI_w_UHI), se = FALSE) +
      coord_cartesian(ylim = limit)
    
    plot_2a <- plot_2a + labs(title = title_q) + ylab("Heat Index (Celcius)") +
      ylab("Mean Global Heat Index (C)") + 
      labs(caption = "Note: Error bars represent one standard deviation\n from mean global urban temperatures") + 
      theme(axis.line = element_line(color="black", size = 0.5), plot.title = element_text(size = 14, face = "bold"), panel.grid.major = element_blank(), panel.grid.minor = element_blank()) + 
      theme(panel.background = element_blank())
    
    print(plot_2a)
    
    # Make file name for chart
    file_name_identifier<-paste(rcp_number, ssp_number_for_file, heat_wave_number, sep = "_" )
    file_name_end<-paste(file_name_identifier, ".jpg", sep="")
    file_name<-paste(land_use_name,"Global_mean_Heat_Index_with_ErrorBars", file_name_end, sep="_" )
    full_path<-paste(final_directory, file_name, sep = "/")
    
    # Announce process saving chart
    announcement_1<-paste("Graphing: ", title_q, "V.2", sep = " ")
    line.space(c(announcement_1, " "))
    ggsave(filename=full_path, plot=plot_2a,  width = 10, height = 7, units = "in",  dpi = 300)            
    
    # VERSION 3 GLOBAL TEMPERATURES BY RCP, SSP AND GCM MODELS
    
    # Graph the data for  global urban heat index over century  
    plot_2b<-ggplot(data=l_chart_file_1, aes(x=Year, y=Mean_HI_w_UHI)) + 
      geom_smooth() + 
      coord_cartesian(ylim = limit) 
    
    # Create title
    title_q<-paste("Mean global urban heat index for heat wave of ", heat_wave_name, " ", days, ", ", rcp_name, ", ", ssp_number, sep = "")
    
    plot_2b<- plot_2b + labs(title = title_q) + ylab("Heat Index (Celcius)") +
      theme(axis.line = element_line(color="black", size = 0.5), plot.title = element_text(size = 14, face = "bold"), panel.grid.major = element_blank(), panel.grid.minor = element_blank()) + 
      theme(panel.background = element_blank())
    
    # print the graph
    print(plot_2b)
    
    # Make file name for chart
    file_name_identifier<-paste(rcp_number, ssp_number_for_file, heat_wave_number, sep = "_" )
    file_name_end<-paste(file_name_identifier, ".jpg", sep="")
    file_name<-paste(land_use_name,"Global_mean_Heat_Index", file_name_end, sep="_" )
    full_path<-paste(final_directory, file_name, sep = "/")
    
    # Announce process saving chart
    announcement_1<-paste("Graphing: ", title_q, "V.3", sep = " ")
    line.space(c(announcement_1, " "))
    ggsave(filename=full_path, plot=plot_2b,  width = 10, height = 7, units = "in",  dpi = 300)      
    
    # Chart 3
    
    # Graph the average global urban heat wave for 5 days by GCM
    plot_3<-ggplot(data=l_chart_file_1, aes(x=Year, y=Mean_HI_w_UHI, color = factor(GCM_Model))) + 
      geom_smooth(size=1.0, se = FALSE) + 
      coord_cartesian(ylim = limit)
    
    # Create title for chart
    title_q<-paste("Mean global urban Heat Index by GCM for heat wave of ", heat_wave_name, " ", days, ", ", rcp_name, ", ", ssp_number, sep = "")
    
    # Create chart
    plot_3 <- plot_3 + labs(title = title_q, color = "GCM Model") + ylab("Heat Index (Celcius)")
    
    plot_3 <- plot_3 + scale_color_discrete(name = "GCM Model",
                                            breaks = c("GFDL_ESM2M", "HadGEM2_ES", "IPSL_CM5A_LR", "MIROC_ESM_CHEM", "NorESM1_M"),
                                            labels = c("GFDL ESM2M", "HadGEM2 ES", "IPSL CM5A LR", "MIROC ESM CHEM", "NorESM1 M")) + 
      theme(axis.line = element_line(color="black", size = 0.5), plot.title = element_text(size = 14, face = "bold"), panel.grid.major = element_blank(), panel.grid.minor = element_blank()) + 
      theme(panel.background = element_blank())
    
    # Plot chart
    print(plot_3)
    
    # Make file name for chart
    file_name_identifier<-paste(rcp_number, ssp_number_for_file, heat_wave_number, sep = "_" )
    file_name_end<-paste(file_name_identifier, ".jpg", sep="")
    file_name<-paste(land_use_name,"Global_mean_Heat_Index_by_GCM", file_name_end, sep="_" )
    full_path<-paste(final_directory, file_name, sep = "/")
    
    # Announce process saving chart
    announcement_1<-paste("Graphing: ", title_q, sep = " ")
    line.space(c(announcement_1, " "))
    ggsave(filename=full_path, plot=plot_3,  width = 10, height = 7, units = "in",  dpi = 300)      
    
    # Save global temperature data, but with correct dates
    
    # Create name for table file
    file_name<-paste (land_use_name, rcp_number, ssp_number_for_file, heat_wave_number, "global_temperature.csv", sep = "_")
    full_file_path<-paste(final_HI_table_directory, file_name, sep = "/")
    
    # Change year back to 2030 for table data
    l_chart_file_1$Year[l_chart_file_1$Year == 2040]<-2030 
    
    # Save dataframe
    write.csv(l_chart_file_1, file = full_file_path, row.names = FALSE)
    
  } 
  # Announce finishing RCP and SSP number
  announcement_3<-paste("Finished", land_use_name, rcp_name, sep = " ")
  line.space(c(" ", announcement_3, " "))
}  


