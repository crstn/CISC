from osgeo import gdal
import numpy as np
import csv

#load cities numpy array
cities = np.load('/Users/mikmaks/Documents/Jobs/CISC/SDEI-Global-UHI/sdei-global-uhi-2013.npy')

#load pop raster and convert to array
src = gdal.Open('/Users/mikmaks/Documents/Jobs/CISC/Pop_2010_clipped.tiff', gdal.GA_Update)
band = src.GetRasterBand(1)
pop = np.array(band.ReadAsArray())
pop[pop < 0] = np.nan #set all population values below 0 as NaN

uniqueCities = np.unique(cities) #get a SORTED list of all city ID's

with open('citiesPop.csv', 'wb') as csvfile:
    citiesCSV = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL) #open the csv file to write to
    citiesCSV.writerow(['ID', 'Total Population']) #write the header row

    #go through all the cities
    for city in uniqueCities:
        #if the city ID is not 0 (0 is not a city)
        if city != 0:
            totalPopForCity = np.nansum(pop[cities == city]) #calculate the total population for the city
            citiesCSV.writerow([city, totalPopForCity]) #write the row to the csv
