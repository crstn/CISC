

# REFERENCES FOR GGPLOT
# see http://www.cookbook-r.com/Graphs/Facets_(ggplot2)/ 
# THIS IS REALLY COOL STUFF!!
# http://r-statistics.co/Top50-Ggplot2-Visualizations-MasterList-R-Code.html
# FACETING
# https://stackoverflow.com/questions/33139247/ggplot2-save-individual-facet-wrap-facets-as-separate-plot-objects


# REFERNCE FOR DPLYR
# http://www.datacarpentry.org/R-genomics/04-dplyr.html



# Get libraries 
library(ggplot2)
library(dplyr)


# CREATE COLOR RAMP

# Make a color ramp white to red through yellow
colfunc<-colorRampPalette(c("red", "orange", "yellow", "white"))
# prints out the color numbers in 12 steps
colfunc(6)


# GET DATA 

dat<-"E:/July_2017/Pop_&_Temp/GRUMP_HI/GRUMP_HI_Final/Summary_tables/GRUMP_heat_wave_data_summary.csv"
grump_dat<-read.csv(dat, header = TRUE, sep=",")


# PREPARE THE DATA 

# remove "Total" from data and select for "With UHI" and 15 day heat waves
# These could be turned into loops!! 
new_grump_dat<-grump_dat[ which(grump_dat$HI_cat != "Total" & grump_dat$UHI_cat =="With UHI" & grump_dat$Heat_wave_days == 15), ]

# change the year to make the chart look better
new_grump_dat$Year[new_grump_dat$Year == 2030]<-2040

# Reverse order of HI_cat
new_grump_dat$HI_cat<-factor(new_grump_dat$HI_cat, levels = c(">55", ">42 & <=55", ">34 & <=42", ">28 & <=34", "<=28"))





# CREATE A GRID OF BAR CHARTS RCPS BY SSPS!!

# Do a bar plot and use our colors (note "scale_fill_manual")
p<-ggplot(new_grump_dat, aes(x = Year, y = Population, fill = factor(HI_cat))) + geom_bar(stat="Identity") + 
  scale_fill_manual("15 Day heat wave temp", values = colfunc(5)) 

# facet by ssp 
p + facet_grid(RCP ~ SSP)



# CREATING THE SAME FACET BUT WITH GEOM_SMOOTH!! 

# Do a bar plot and use our colors (note "scale_fill_manual") (se=FALSE removes the confidence intervals)
p<-ggplot(new_grump_dat, aes(x = Year, y = Population, color = factor(HI_cat))) + geom_smooth(se = FALSE) + 
  scale_color_manual("15 Day heat wave temp", values = colfunc(5)) 

# facet by ssp 
p + facet_grid(RCP ~ SSP)



# CREATE THE SAME AS ABOVE BUT ONLY FOR POPULATIONS WITH HIGH TEMPERATURE VALUES - CHECK VALUES 

# remove "Total" from data and select for "With UHI, RCP 6.0 and 15 day heat waves
new_grump_dat1<-new_grump_dat[ which(new_grump_dat$HI_cat == ">55" |  new_grump_dat$HI_cat == ">42 & <=55"), ]

gg<-ggplot(new_grump_dat, aes(x = Year, y = Population, color = Region_name))+
  geom_smooth(method = "loess", se = F)

gg<- gg + facet_grid(RCP ~ SSP)

plot(gg)










### MAKING BAR CHARTS with loop (didn't work) 

rcp_list<-list("RCP 2.6", "RCP 4.5", "RCP 6.0", "RCP 8.5")

for(i in 1:length(rcp_list)){
  
  # remove "Total" from data and select for "With UHI" and "15 day heat waves" and the current rcp
  new_grump_dat<-grump_dat[ which(grump_dat$HI_cat != "Total" & grump_dat$UHI_cat =="With UHI" & grump_dat$RCP == rcp_list[i] & grump_dat$Heat_wave_days == 15), ]
  
  # Attempt to reverse order of HI_cat
  new_grump_dat$HI_cat<-factor(new_grump_dat$HI_cat, levels = c(">55", ">42 & <=55", ">34 & <=42", ">28 & <=34", "<=28"))
  
  # Do a bar plot and use our colors (note "scale_fill_manual")
  p<-ggplot(new_grump_dat, aes(x = Year, y = Population, fill = factor(HI_cat))) + geom_bar(stat="Identity") + 
    scale_fill_manual(values = colfunc(5)) 
  
  # facet by ssp 
  p<- p + facet_wrap(. ~SSP)
  print(p)
  
  
}



# MAKE BAR CHART OF SHARE IN EACH GROUP WITH HIGHEST CATEORY ON TOP 


# Do a bar plot and use our colors (note "scale_fill_manual")
p<-ggplot(new_grump_dat, aes(x = Year, y = Population, fill = factor(HI_cat))) + geom_bar(stat="Identity", position = position_fill(reverse = TRUE)) + 
  scale_fill_manual(values = colfunc(5)) 

# facet by ssp 
p + facet_grid(. ~SSP)






# =======================================================================================================================
# EXTRA - THESE CODES MIGHT BE USEFUL IN LATER SCRIPTS 

# Select for population of regions with hot temperatures USE IF WANT TO FIND OUT WHAT POP IN 42-55 & > 55 CATEGORIES
regions<-new_grump_dat%>%
  group_by(Region_name, SSP, RCP, Year, UHI_cat, Heat_wave_days, HI_cat)%>%
  summarize(Pop = sum(as.numeric(Population)),
            Sen_pop = sum(as.numeric(Sensitive_pop)))%>%
  as.data.frame()

# Make new variable (share sensitive population)
regions$Share_sen_pop<-  regions$Sen_pop/ regions$Pop

# Select for population of regions with hot temperatures USE IF WANT TO FIND OUT WHAT POP IN 42-55 & > 55 CATEGORIES
sub_regions<-new_grump_dat%>%
  group_by(Sub_region_name, Region, SSP, RCP, Year, UHI_cat, Heat_wave_days, HI_cat)%>%
  summarize(Pop = sum(as.numeric(Population)),
            Sen_pop = sum(as.numeric(Sensitive_pop)))%>%
  as.data.frame()


# CREATE GRAPHS FOR ADDITIONAL POPULATION WITH UHI

# create ADDITONAL POPULATION graphs for each heat wave length 
for(i in 1:length(heat_wave_length)){
  
  # Select for data for the correct heat wave and with UHI
  regions_hw_len<-summary.merged[which(summary.merged$Heat_wave_days == heat_wave_length[i]),]
  
  # Correct for plurals for title
  if(heat_wave_length[i]==1){
    days = "day"
  } else {
    days = "days"
  }
  
  # Make a chart title 
  title_q<-paste("Urban population added to exposed to hot temperatures by UHI in heat wave of", heat_wave_length[i], days, sep = " ")
  
  # FACET PLOT OF REGIONS BY SSP BY RCP TOTAL POPULATION EXPOSED TO HOT HEAT INDEX
  
  gg_pop <- ggplot(regions_hw_len, aes(x = Year, y = UHI_pop_added, color = Region_name)) + geom_smooth(se=FALSE) +
    scale_y_continuous(limits = c(0, 1000000000), labels = function(x) format(x/1000000000)) 
  
  gg_pop<- gg_pop + facet_grid(RCP ~ SSP)
  
  gg_pop<-gg_pop + labs(title = title_q, color = "Region", caption = "(Hot temp: > 42 C)") + ylab("Population (billions)")+
    theme(axis.line = element_line(color="black", size = 0.5), plot.title = element_text(size = 14, face = "bold"), panel.grid.major = element_blank(), panel.grid.minor = element_blank())  + theme(panel.background = element_blank())
  
  plot(gg_pop)
  
  # make final file name
  file_name_identifier<-paste(heat_wave_length[i], days, sep = "_" )
  file_name_end<-paste(file_name_identifier, ".pdf", sep="")
  file_name<-paste("E:/July_2017/Pop_&_Temp/GRUMP_HI/GRUMP_HI_Final/Summary_charts/Regional_UHI_added_pop", file_name_end, sep = "_")
  
  # Save file 
  ggsave(filename = file_name, plot = gg_pop, dpi = 300)
  
  
}







