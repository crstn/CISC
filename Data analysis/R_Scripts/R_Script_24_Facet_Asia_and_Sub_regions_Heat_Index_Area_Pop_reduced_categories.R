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

# MAKE FUNCTIONS

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

# SET UP ANALYSIS

# Set working directory to the folder containing this script:
# (assumes you use R studio)
work_dir<-setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

# Get data and read into dataframe
DF_GRUMP<-read.csv("Data_GRUMP_Pop.csv")
DF_GlobCover<-read.csv("Data_GlobCover_Pop.csv")

#Combine the files
DF<-rbind(DF_GRUMP, DF_GlobCover)


# SELECTIVE APPROPRIATE DATA

# Select for only Asia, with UHI, 15 dcay heat waves  and remove "Total" category 
DF_1<-subset(DF, sub_region == 30 | sub_region == 34 | sub_region == 35 | sub_region == 143 | sub_region == 145)
DF_1<-subset(DF_1, UHI_cat == "With UHI")
DF_1<-subset(DF_1, ! HI_cat == "\"Total\"")
DF_2<-subset(DF_1, Heat_wave_days == 15)

# get two datasets
DF_3<-subset(DF_2, Model == "GRUMP")
DF_4<-subset(DF_2, Model =="GlobCover")

# WORK WITH GRUMP

# Summarise for Asia
asia_GRUMP<-DF_3%>%
  group_by(SSP, RCP, Year, HI_cat)%>%
  summarize(pop = sum(as.numeric(Population)),
            sen_pop =sum(as.numeric(Sensitive_pop)))%>%
  as.data.frame()

# remove the quotes from the categories
asia_GRUMP$HI_cat <- gsub('"', '', asia_GRUMP$HI_cat)

# Smoothing and area charts don't seem to go so well together,
# so we'll need to summarize the data before we can make the area chart.
# Fist, we'll add numeric categories to make this easier:
asia_GRUMP$HI_cat_num <- asia_GRUMP$HI_cat
asia_GRUMP$HI_cat_num <- gsub('<=28', 1, asia_GRUMP$HI_cat_num)
asia_GRUMP$HI_cat_num <- gsub('>28 & <=34', 2, asia_GRUMP$HI_cat_num)
asia_GRUMP$HI_cat_num <- gsub('>34 & <=42', 3, asia_GRUMP$HI_cat_num)
asia_GRUMP$HI_cat_num <- gsub('>42 & <=55', 4, asia_GRUMP$HI_cat_num)
asia_GRUMP$HI_cat_num <- gsub('>55', 4, asia_GRUMP$HI_cat_num)

# make sure they are integers and not characters 
asia_GRUMP$HI_cat_num <- as.integer(asia_GRUMP$HI_cat_num)

asia_GRUMP$HI_cat<-gsub('>42 & <=55', '>42', asia_GRUMP$HI_cat)
asia_GRUMP$HI_cat<-gsub('>55', '>42', asia_GRUMP$HI_cat)


# Summarise again to remove the extra rows 
asia_GRUMP<-asia_GRUMP%>%
  group_by(SSP, RCP, Year, HI_cat, HI_cat_num)%>%
  summarize(pop = sum(as.numeric(pop)),
            sen_pop =sum(as.numeric(sen_pop)))%>%
  as.data.frame()

# next, we'll sum up the population per SSP/RCP/year
# up to the corresponding level (e.g. on level 3, the number
# will be the sum of people in cats 1, 2 and 3):

asia_GRUMP_1 <- group_by(asia_GRUMP, SSP, RCP, Year) %>% 
  filter(HI_cat_num == 1) %>% 
  summarise(pop_below = sum(pop), sen_pop_below = sum(sen_pop), HI_cat = "<=28")

asia_GRUMP_2 <- group_by(asia_GRUMP, SSP, RCP, Year) %>% 
  filter(HI_cat_num <= 2) %>% 
  summarise(pop_below = sum(pop), sen_pop_below = sum(sen_pop), HI_cat = "<=34")

asia_GRUMP_3 <- group_by(asia_GRUMP, SSP, RCP, Year) %>% 
  filter(HI_cat_num <= 3) %>% 
  summarise(pop_below = sum(pop), sen_pop_below = sum(sen_pop), HI_cat = "<=42")

asia_GRUMP_4 <- group_by(asia_GRUMP, SSP, RCP, Year) %>% 
  filter(HI_cat_num <= 4) %>% 
  summarise(pop_below = sum(pop), sen_pop_below = sum(sen_pop), HI_cat = ">42")

# asia_GRUMP_5 <- group_by(asia_GRUMP, SSP, RCP, Year) %>% 
#   filter(HI_cat_num <= 5) %>% 
#   summarise(pop_below = sum(pop), sen_pop_below = sum(sen_pop), HI_cat = "> 55")

# combine them into one table:
asia_GRUMP_sums <- ungroup(bind_rows(asia_GRUMP_1, asia_GRUMP_2, asia_GRUMP_3, asia_GRUMP_4))

# clean up
remove(asia_GRUMP_1, asia_GRUMP_2, asia_GRUMP_3, asia_GRUMP_4, DF, DF_2)

# last preparation step: Turn the HI_cat column into a factor
# and change the order of the levels to make sure >55 gets plotted first,
# tthen <=42, and so on:
asia_GRUMP_sums$HI_cat <- as.factor(asia_GRUMP_sums$HI_cat)
asia_GRUMP_sums$HI_cat <- factor(asia_GRUMP_sums$HI_cat, levels = levels(asia_GRUMP_sums$HI_cat)[c(4,3,2,1)])


# WORK WITH GLOBCOVER

# Summarise for Asia
asia_GlobCover<-DF_4%>%
  group_by(SSP, RCP, Year, HI_cat)%>%
  summarize(pop = sum(as.numeric(Population)),
            sen_pop =sum(as.numeric(Sensitive_pop)))%>%
  as.data.frame()

# remove the quotes from the categories
asia_GlobCover$HI_cat <- gsub('"', '', asia_GlobCover$HI_cat)

# Smoothing and area charts don't seem to go so well together,
# so we'll need to summarize the data before we can make the area chart.
# Fist, we'll add numeric categories to make this easier:
asia_GlobCover$HI_cat_num <- asia_GlobCover$HI_cat
asia_GlobCover$HI_cat_num <- gsub('<=28', 1, asia_GlobCover$HI_cat_num)
asia_GlobCover$HI_cat_num <- gsub('>28 & <=34', 2, asia_GlobCover$HI_cat_num)
asia_GlobCover$HI_cat_num <- gsub('>34 & <=42', 3, asia_GlobCover$HI_cat_num)
asia_GlobCover$HI_cat_num <- gsub('>42 & <=55', 4, asia_GlobCover$HI_cat_num)
asia_GlobCover$HI_cat_num <- gsub('>55', 4, asia_GlobCover$HI_cat_num)

# make sure they are integers and not characters 
asia_GlobCover$HI_cat_num <- as.integer(asia_GlobCover$HI_cat_num)


# Change values in HI_cat
asia_GlobCover$HI_cat<-gsub('>42 & <=55', '>42', asia_GlobCover$HI_cat)
asia_GlobCover$HI_cat<-gsub('>55', '>42', asia_GlobCover$HI_cat)

# Summarise again to remove the extra rows 
asia_GlobCover<-asia_GlobCover%>%
  group_by(SSP, RCP, Year, HI_cat, HI_cat_num)%>%
  summarize(pop = sum(as.numeric(pop)),
            sen_pop =sum(as.numeric(sen_pop)))%>%
  as.data.frame()

# next, we'll sum up the population per SSP/RCP/year
# up to the corresponding level (e.g. on level 3, the number
# will be the sum of people in cats 1, 2 and 3):

asia_GlobCover_1 <- group_by(asia_GlobCover, SSP, RCP, Year) %>% 
  filter(HI_cat_num == 1) %>% 
  summarise(pop_below = sum(pop), sen_pop_below = sum(sen_pop), HI_cat = "<=28")

asia_GlobCover_2 <- group_by(asia_GlobCover, SSP, RCP, Year) %>% 
  filter(HI_cat_num <= 2) %>% 
  summarise(pop_below = sum(pop), sen_pop_below = sum(sen_pop), HI_cat = "<=34")

asia_GlobCover_3 <- group_by(asia_GlobCover, SSP, RCP, Year) %>% 
  filter(HI_cat_num <= 3) %>% 
  summarise(pop_below = sum(pop), sen_pop_below = sum(sen_pop), HI_cat = "<=42")

asia_GlobCover_4 <- group_by(asia_GlobCover, SSP, RCP, Year) %>% 
  filter(HI_cat_num <= 4) %>% 
  summarise(pop_below = sum(pop), sen_pop_below = sum(sen_pop), HI_cat = ">42")

# asia_GlobCover_5 <- group_by(asia_GlobCover, SSP, RCP, Year) %>% 
#   filter(HI_cat_num <= 5) %>% 
#   summarise(pop_below = sum(pop), sen_pop_below = sum(sen_pop), HI_cat = ">55")

# combine them into one table:
asia_GlobCover_sums <- ungroup(bind_rows(asia_GlobCover_1, asia_GlobCover_2, asia_GlobCover_3, asia_GlobCover_4))

# clean up
remove(asia_GlobCover_1, asia_GlobCover_2, asia_GlobCover_3, asia_GlobCover_4)

# last preparation step: Turn the HI_cat column into a factor
# and change the order of the levels to make sure >55 gets plotted first,
# tthen <=42, and so on:
asia_GlobCover_sums$HI_cat <- as.factor(asia_GlobCover_sums$HI_cat)
asia_GlobCover_sums$HI_cat <- factor(asia_GlobCover_sums$HI_cat, levels = levels(asia_GlobCover_sums$HI_cat)[c(4,3,2,1)])


# MAKE PLOTs AND SAVE THE FINAL CHART

# Remove scientific notation
options(scipen = 10000)

# DO GLOBCOVER
m<-ggplot(as.data.frame(ungroup(asia_GlobCover_sums))) + 
  stat_smooth(geom = 'area', method = 'loess', formula = y ~ x, aes(x=Year, y=pop_below, fill = HI_cat)) + 
  scale_fill_manual(values = rev(brewer.pal(4, "YlOrRd"))) 

m<-m + scale_y_continuous(breaks=c(0, 1000000000, 2000000000, 3000000000, 4000000000, 5000000000, 6000000000), labels = c("0", "1", "2", "3", "4", "5", "6"),limits=c(-1, 4000000000)) 

r<- m + facet_grid(SSP ~ RCP) 

r<- r + labs(title = "Globcover", y = "Population (billions)", color = "Heat Index Category") +
  guides(fill = guide_legend(reverse=T,title = "Heat index "))

r<-r +theme(strip.background = element_rect(color="black", fill = "grey"), 
            strip.text.x = element_text(size = rel(0.9)), 
            strip.text.y = element_text(size = rel(0.9)),
            axis.text.x = element_text(size = rel(0.7)), 
            axis.text.y = element_text(size = rel(0.7)),
            axis.title.y = element_blank(),
            legend.title=element_text(size=rel(0.7)),
            legend.text = element_text(size=rel(0.5)))
print(r)

# DO GRUMP
q<-ggplot(as.data.frame(ungroup(asia_GRUMP_sums))) + 
  stat_smooth(geom = 'area', method = 'loess', formula = y ~ x, aes(x=Year, y=pop_below, fill = HI_cat)) + 
  scale_fill_manual(values = rev(brewer.pal(4, "YlOrRd"))) 

q<- q + scale_y_continuous(breaks=c(0, 1000000000, 2000000000, 3000000000, 4000000000, 5000000000, 6000000000), labels = c("0", "1", "2", "3", "4", "5", "6"),limits=c(-1, 4000000000)) 

s<- q + facet_grid(SSP ~ RCP) 

s<- s+ ggtitle("GRUMP") +
  labs(x = "Year", y = "Population (billions)") 

s<-s +theme(strip.background = element_rect(color="black", fill = "grey"), 
            strip.text.x = element_text(size = rel(0.9)), 
            strip.text.y = element_text(size = rel(0.9)),
            axis.text.x = element_text(size = rel(0.7)), 
            axis.text.y = element_text(size = rel(0.7)),
            legend.position = "none")
print(s)

# COMBINE THE PLOTS AND SAVE

png(filename = "Pop_in_heat.png", width = 10, height = 7, units = 'in', res = 300)
z<-multiplot(s, r, cols=2)
dev.off()










# PLOT AND SAVE THE CHART FROM PREVIOUS SCRIPT

# Carsten's script
# p<-ggplot(as.data.frame(ungroup(asia_GRUMP_sums))) + 
#   theme_fivethirtyeight() +
#   stat_smooth(geom = 'area', method = 'loess', formula = y ~ x, aes(x=Year, y=pop_below, fill = HI_cat)) + 
#   scale_fill_manual(values = rev(brewer.pal(5, "YlOrRd"))) +
#   scale_y_continuous(name = "Population (billions)", breaks=c(0, 1000000000, 2000000000, 3000000000, 4000000000, 5000000000, 6000000000), labels = c("0", "1", "2", "3", "4", "5", "6"),limits=c(-1, 4000000000)) +
#   facet_grid(SSP ~ RCP) +
#   labs(title = "GRUMP", color = "Heat Index Category") +
#   guides(fill = guide_legend(reverse=T,title = "Heat index "))
# 
# Save the plot
# ggsave("Asia_heat_index.pdf", width = 10, height = 7, units = c('in'), dpi = 300)
# write.csv(DF_2, "Asia_sub_regions.csv", row.names = FALSE)