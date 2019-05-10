library(dplyr)
library(ggplot2)
library(RColorBrewer)
library(ggthemes)


# Set working directory to the folder containing this script:
# (assumes you use R studio)

# @Peter: That way the script is always going to work, 
# as long as you have it in the same folder as the data, even if you move 
# it to a different folder
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

# Get data and read into dataframe
DF<-read.csv("Data_GRUMP_Pop.csv")

# Select for only Asia, with UHI, 15 dcay heat waves  and remove "Total" category 
DF_1<-subset(DF, sub_region == 30 | sub_region == 34 | sub_region == 35 | sub_region == 143 | sub_region == 145)
DF_1<-subset(DF_1, UHI_cat == "With UHI")
DF_1<-subset(DF_1, ! HI_cat == "\"Total\"")
DF_2<-subset(DF_1, Heat_wave_days == 15)

# Summarise for Asia
asia<-DF_2%>%
  group_by(SSP, RCP, Year, HI_cat)%>%
  summarize(pop = sum(as.numeric(Population)),
            sen_pop =sum(as.numeric(Sensitive_pop)))%>%
  as.data.frame()

# remove the quotes from the categories
asia$HI_cat <- gsub('"', '', asia$HI_cat)

# Smoothing and area charts don't seem to go so well together,
# so we'll need to summarize the data before we can make the area chart.
# Fist, we'll add numeric categories to make this easier:
asia$HI_cat_num <- asia$HI_cat
asia$HI_cat_num <- gsub('<=28', 1, asia$HI_cat_num)
asia$HI_cat_num <- gsub('>28 & <=34', 2, asia$HI_cat_num)
asia$HI_cat_num <- gsub('>34 & <=42', 3, asia$HI_cat_num)
asia$HI_cat_num <- gsub('>42 & <=55', 4, asia$HI_cat_num)
asia$HI_cat_num <- gsub('>55', 5, asia$HI_cat_num)

asia$HI_cat_num <- as.integer(asia$HI_cat_num)

# next, we'll sum up the population per SSP/RCP/year
# up to the corresponding level (e.g. on level 3, the number
# will be the sum of people in cats 1, 2 and 3):

asia_1 <- group_by(asia, SSP, RCP, Year) %>% 
  filter(HI_cat_num == 1) %>% 
  summarise(pop_below = sum(pop), sen_pop_below = sum(sen_pop), HI_cat = "<= 28")

asia_2 <- group_by(asia, SSP, RCP, Year) %>% 
  filter(HI_cat_num <= 2) %>% 
  summarise(pop_below = sum(pop), sen_pop_below = sum(sen_pop), HI_cat = "<= 34")

asia_3 <- group_by(asia, SSP, RCP, Year) %>% 
  filter(HI_cat_num <= 3) %>% 
  summarise(pop_below = sum(pop), sen_pop_below = sum(sen_pop), HI_cat = "<= 42")

asia_4 <- group_by(asia, SSP, RCP, Year) %>% 
  filter(HI_cat_num <= 4) %>% 
  summarise(pop_below = sum(pop), sen_pop_below = sum(sen_pop), HI_cat = "<= 55")

asia_5 <- group_by(asia, SSP, RCP, Year) %>% 
  filter(HI_cat_num <= 5) %>% 
  summarise(pop_below = sum(pop), sen_pop_below = sum(sen_pop), HI_cat = "> 55")

# combine them into one table:
asia_sums <- ungroup(bind_rows(asia_1, asia_2, asia_3, asia_4, asia_5))

# clean up
remove(asia_1, asia_2, asia_3, asia_4, asia_5, DF, DF_1)

# last preparation step: Turn the HI_cat column into a factor
# and reverse the order of the levels to make sure >55 gets plotted first,
# then <=42, and so on:
asia_sums$HI_cat <- as.factor(asia_sums$HI_cat)
asia_sums$HI_cat <- factor(asia_sums$HI_cat, levels = levels(asia_sums$HI_cat)[c(5,4,3,2,1)])

# PLOT AND SAVE THE CHART

# Remove scientific notation
options(scipen = 10000)

ggplot(as.data.frame(ungroup(asia_sums))) + 
  theme_fivethirtyeight() +
  stat_smooth(geom = 'area', method = 'loess', formula = y ~ x, aes(x=Year, y=pop_below, fill = HI_cat)) + 
  scale_fill_manual(values = rev(brewer.pal(5, "YlOrRd"))) +
  scale_y_continuous(name = "Population (billions)", breaks=c(0, 1000000000, 2000000000, 3000000000, 4000000000, 5000000000, 6000000000), labels = c("0", "1", "2", "3", "4", "5", "6"),limits=c(-1, 4000000000)) +
  facet_grid(SSP ~ RCP) +
  labs(title="Asia population by heat category", color = "Heat Index Category") +
  guides(fill = guide_legend(reverse=T,title = "Heat index "))

# Save the plot
ggsave("Asia_heat_index_total_population.pdf", width = 10, height = 7, units = c('in'), dpi = 300)



# repeat for sensitive population
ggplot(as.data.frame(ungroup(asia_sums))) + 
  theme_fivethirtyeight() +
  stat_smooth(geom = 'area', method = 'loess', formula = y ~ x, aes(x=Year, y=sen_pop_below, fill = HI_cat)) + 
  scale_fill_manual(values = rev(brewer.pal(5, "YlOrRd"))) +
  scale_y_continuous(name = "Population (billions)", breaks=c(0, 1000000000, 2000000000), labels = c("0", "1", "2"),limits=c(-1, 2000000000)) +
  facet_grid(SSP ~ RCP) +
  labs(title="Asia sensitive population by heat category", color = "Heat Index Category") +
  guides(fill = guide_legend(reverse=T,title = "Heat index "))

# Save the plot
ggsave("Asia_heat_index_sensitive_population.pdf", width = 10, height = 7, units = c('in'), dpi = 300)

write.csv(DF_2, "Asia_sub_regions.csv", row.names = FALSE)
