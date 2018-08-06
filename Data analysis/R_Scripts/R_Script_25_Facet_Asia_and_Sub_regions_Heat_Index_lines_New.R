# ==========================================================================================================
# DESCRIPTION:  This script creates a single image output of two different facet graphs (RCP by SSP) of 
#               population in very warm heat waves (>42C) over time.  It starts with two summary .csv files
#               which hold data at the sub-regional level, globally.  The multi-plot function used, which
#               creates the final output from ggplot objects, was found on the internet.
#
# DATE:         27 July 2018
# DEVELOPER:    Peter J. Marcotullio
# ==========================================================================================================

# GET LIBRARIES

library(dplyr)
library(ggplot2)
library(RColorBrewer)
library(ggthemes)
library(rstudioapi)
library(ggpubr)

# MAKE FUNCTIONS AND OTHER 

# Multiple plot function: taken from
# http://www.cookbook-r.com/Graphs/Multiple_graphs_on_one_page_(ggplot2)/
#
# ggplot objects can be passed in ..., or to plotlist (as a list of ggplot objects)
# - cols:   Number of columns in layout
# - layout: A matrix specifying the layout. If present, 'cols' is ignored.
#
# If the layout is something like matrix(c(1,2,3,3), nrow=2, byrow=TRUE),
# then plot 1 will go in the upper left, 2 will go in the upper right, and
# 3 will go all the way across the bottom.
#
multiplot <- function(..., plotlist=NULL, file, cols=1, layout=NULL) {
  library(grid)
  
  # Make a list from the ... arguments and plotlist
  plots <- c(list(...), plotlist)
  
  numPlots = length(plots)
  
  # If layout is NULL, then use 'cols' to determine layout
  if (is.null(layout)) {
    # Make the panel
    # ncol: Number of columns of plots
    # nrow: Number of rows needed, calculated from # of cols
    layout <- matrix(seq(1, cols * ceiling(numPlots/cols)),
                     ncol = cols, nrow = ceiling(numPlots/cols))
  }
  
  if (numPlots==1) {
    print(plots[[1]])
    
  } else {
    # Set up the page
    grid.newpage()
    pushViewport(viewport(layout = grid.layout(nrow(layout), ncol(layout))))
    
    # Make each plot, in the correct location
    for (i in 1:numPlots) {
      # Get the i,j matrix positions of the regions that contain this subplot
      matchidx <- as.data.frame(which(layout == i, arr.ind = TRUE))
      
      print(plots[[i]], vp = viewport(layout.pos.row = matchidx$row,
                                      layout.pos.col = matchidx$col))
    }
  }
}

my_colors<-colorRampPalette(c('yellow', 'gold', 'orange', 'darkorange3', 'red2')) # 'orangered','brown1',


# SET UP ANALYSIS

# Set working directory to the folder containing this script:
# (assumes you use R studio)
work_dir<-setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

# Get data and read into dataframe
DF_GRUMP<-read.csv("Data_Asia_GRUMP_regional_temperature.csv")
DF_GlobCover<-read.csv("Data_Asia_GlobCover_regional_temperature.csv")

#Combine the files
DF<-rbind(DF_GRUMP, DF_GlobCover)

# SELECTIVE APPROPRIATE DATA

# Select for only Asia, with UHI, 15 dcay heat waves  and remove "Total" category 
DF_1<-subset(DF, Region == 142) # sub_region == 30 | sub_region == 34 | sub_region == 35 | sub_region == 143 | sub_region == 145)
DF_2<-subset(DF_1, Heat_wave == 15)

# get two datasets
DF_3<-subset(DF_2, Model == "GRUMP")
DF_4<-subset(DF_2, Model =="GlobCover")

# WORK WITH GRUMP

# Summarise for Asia
asia_GRUMP<-DF_3%>%
  group_by(SSP, RCP, Year)%>%
  summarize(Temp = mean(Mean_HI_w_UHI),
            sd = sd(Mean_HI_w_UHI))%>%
  as.data.frame()

# WORK WITH GLOBCOVER

# Summarize for Asia

# Summarise for Asia
asia_GLOBCOVER<-DF_4%>%
  group_by(SSP, RCP, Year)%>%
  summarize(Temp = mean(Mean_HI_w_UHI),
            sd = sd(Mean_HI_w_UHI))%>%
  as.data.frame()


# PLOT THE CHART

# Remove scientific notation
options(scipen = 10000)

# FOR GRUMP

# First plot asia with colors from ramp and Heat Index categories
p<-ggplot(asia_GRUMP) +
  geom_smooth(aes(x=Year, y=Temp, color = RCP), size = 1, se = FALSE) +
  scale_color_manual(values = c(my_colors(5))) +
  labs(title = "GRUMP", x = "Year", y = "Heat Index (C)")

p<-p + scale_y_continuous(breaks=c(36, 38, 40, 42, 44, 46, 48, 50), labels = c("36", "38", "40", "42", "44", "46", "48", "50"),limits=c(36, 51))

# facet by ssp and rcp
p<-p + facet_grid(~ SSP)

r<-p + theme_bw() 

r<-r + theme(legend.position = c(0.9, 0.975),
             legend.justification = c("right", "top"),
             legend.title=element_blank(),
             panel.border = element_rect(color = "black", fill=NA),
             legend.text=element_text(size = rel(0.7)))
             

# theme(axis.title.y = element_text("Heat index (C)") )

print(r)

# FOR GLOBCOVER

# First plot asia with colors from ramp and Heat Index categories
q<-ggplot(asia_GLOBCOVER) +
  geom_smooth(aes(x=Year, y=Temp, color = RCP), size = 1, se = FALSE) +
  scale_color_manual(values = c(my_colors(5))) +
  labs(title = "GlobCover", x = "Year", y = "Heat Index (C)")

q<-q + scale_y_continuous(breaks=c(36, 38, 40, 42, 44, 46, 48, 50), labels = c("36", "38", "40", "42", "44", "46", "48", "50"),limits=c(36, 51))

# facet by ssp and rcp
q<-q + facet_grid(~ SSP)

s<-q + theme_bw() 

s<-s + theme(legend.position = "none")

print(s)

png(filename = "Heat_index_Asia.png", width = 10, height = 7, units = 'in', res = 300)
z<-multiplot(r, s, cols=1)
dev.off()