# 
# ==============================================================================================
# THIS SCRIPT XX.  
#
#
# Peter J. Marcotullio, AUGUST 2017
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

data_file_a<-"E:/July_2017/Pop_&_Temp/GRUMP_HI/GRUMP_HI_Final/Summary_tables/Urban_areas/GRUMP_Summary_Area_Sub_region_data.csv"
data_file_b<-"E:/July_2017/Pop_&_Temp/GRUMP_HI/GRUMP_HI_Final/Summary_tables/Urban_areas/GRUMP_Summary_Area_Region_data.csv"
data_file_c<-"E:/July_2017/Pop_&_Temp/GRUMP_HI/GRUMP_HI_Final/Summary_tables/Urban_areas/Summary_Area_Region_data.csv"

df_a<-read.csv(data_file_a)
df_b<-read.csv(data_file_b)
df_c<-read.csv(data_file_c)


# Graph the data by Population size under different heat index categories
plot_1<-ggplot(data=df_b, aes(x=Year, y=Area_sum, color=factor(region))) + 
  geom_smooth(size = 1, level = 0.95) +
  scale_y_continuous(limits = c(0, 3000000), labels = function(x) format(x/1000000)) 

# Create title name
title_q<-"Change in urban land use area by region"

  # Create chart
plot_1 <- plot_1 + labs(title = title_q, color = "Region")+ 
  ylab("Area (million sq km)") + 
  theme(axis.line = element_line(color="black", size = 0.5), plot.title = element_text(size = 14, face = "bold"), panel.grid.major = element_blank(), panel.grid.minor = element_blank()) + 
  theme(panel.background = element_blank())

plot_1<- plot_1 + scale_color_discrete(name = "Region", breaks = c(2,9,21,142,150,419),
                                       labels= c("Africa", "Oceania", "North America", "Asia", "Europe", "Latin America and Caribbean"))

plot_1<- plot_1 + facet_grid(. ~ SSP)

# plot chart    
print(plot_1)

# Graph the data by Population size under different heat index categories
plot_2<-ggplot(data=df_c, aes(x=Year, y=Area_sum, color=factor(region))) + 
  geom_smooth(size = 1, level = 0.95) +
  scale_y_continuous(limits = c(0, 3500000), labels = function(x) format(x/1000000)) 

# Create title name
title_q<-"Change in urban land use area by region"

# Create chart
plot_2 <- plot_2 + labs(title = title_q, color = "Region")+ 
  ylab("Area (million sq km)") + 
  theme(axis.line = element_line(color="black", size = 0.5), plot.title = element_text(size = 14, face = "bold"), panel.grid.major = element_blank(), panel.grid.minor = element_blank()) + 
  theme(panel.background = element_blank())

plot_2<- plot_2 + scale_color_discrete(name = "Region", breaks = c(2,9,21,142,150,419),
                                       labels= c("Africa", "Oceania", "North America", "Asia", "Europe", "Latin America and Caribbean"))

plot_2<- plot_2 + facet_grid(Land_use ~ SSP)

# plot chart    
print(plot_2)

