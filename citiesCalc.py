from osgeo import gdal
import numpy as np
import csv               #for reading csv
import codecs            #for utf-8
import os

print "loading population raster"
src = gdal.Open(os.path.expanduser('~') + '/Dropbox/CISC Data/Population 2010 Raster/Pop_2010_clipped.tiff', gdal.GA_Update)
band = src.GetRasterBand(1)
print "converting to array"
pop = np.array(band.ReadAsArray())
pop[pop < 0] = np.nan #set all population values below 0 as NaN

os.chdir(os.path.expanduser('~') + '/Dropbox/CISC Data/SDEI-Global-UHI')

print "loading cities numpy array"
cities = np.load('sdei-global-uhi-2013.npy')
uniqueCities = np.unique(cities) #get a SORTED list of all city ID's

#open cities CSV file for names
with open('CitiesAttributes.csv', 'r') as citiesFile:
    cityNames = csv.DictReader(citiesFile)

    #open output file
    with codecs.open('citiesPop.txt', 'w', 'utf-8') as outputFile:
        outputFile.write('ID, Name, Total Population\n') #write the header row

        #go through all the cities
        for city in uniqueCities:
            #if the city ID is not 0 (0 is not a city)
            if city != 0:
                totalPopForCity = int(np.nansum(pop[cities == city])) #calculate the total population for the city

                #find the name correspoonding to the city ID
                name = ""
                for row in cityNames:
                    if (row['URBID'] == str(city)):  #if the ID matches, get the name and break
                        name = unicode(row['NAME'], 'utf8')
                        break

                newLine = str(city) + ',"' + name + '",' + str(totalPopForCity) + '\n'
                outputFile.write(newLine) #write the row to the csv
