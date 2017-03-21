# coding: utf-8
#!/usr/bin/env python

# Takes 2 or 3 arguments:
# 1. model - (SSP1 through SSP5)
# 2. year - 2020 through 2100
# 3. (optional) one or more cities. If this parameter is skipped, the numbers for all cities will be calculated.
# Example:
#
# "citiesCalc.py SSP3 2030" for calculation of all cities or in model SSP3 for the year 2030
# "citiesCalc.py SSP3 2030 'Moscow' 'New York' etc. for calculation of those cities in model SSP3 for the year 2030

from osgeo import gdal
import numpy as np
import csv             #for reading csv
import codecs          #for utf-8
import os, sys

del sys.argv[0]  #delete the first argument (the script name)

if len(sys.argv) < 2:
    print 'Script takes 2 or 3 arguments:'
    print '1. model – (SSP1 through SSP5)'
    print '2. year – 2020 through 2100'
    print '3. (optional) one or more cities. If this parameter is skipped, the numbers for all cities will be calculated.'
    print 'Example:'
    print '"citiesCalc.py SSP3 2030" for calculation of all cities or in model SSP3 for the year 2030'
    print '"citiesCalc.py SSP3 2030 \'Moscow\' \'New York\' etc. for calculation of those cities in model SSP3 for the year 2030'
    sys.exit()

model = sys.argv[0]
year = sys.argv[1]

# also remove these two from the command line arguments, so that only the city names remain:
del sys.argv[0]
del sys.argv[0]

models = ["SSP1", "SSP2", "SSP3", "SSP4", "SSP5"]

if model not in models:
    print model + ' is not among the supported models: '
    print models
    sys.exit()

years = range(2020, 2101, 10)

if int(year) not in years:
    print year + ' is not among the supported years: '
    print years
    sys.exit()


datadir = os.path.expanduser('~') + '/Dropbox/CISC Data/IndividualCountries/Projections/Global/'+model
filename = 'citiesPop-'+year+'.csv'

# Make sure we don't overwrite an existing citiesPop.csv, this one takes
# hours to generate!
if (len(sys.argv) == 0) and (os.path.exists(datadir+'/'+filename)):
    print filename+" already exists. If you want to recalculate it, delete it first."
    sys.exit()

#open cities CSV file for names
with open(os.path.expanduser('~') + '/Dropbox/CISC Data/SDEI-Global-UHI/CitiesAttributes.csv', 'rb') as citiesFile:
    cityNames = csv.DictReader(citiesFile)

    citiesToCalculate = []
    #find cities passed in as arguments
    for argument in sys.argv:
        cityExists = False
        for row in cityNames:
            if (row['NAME'] == argument):  #if the name is found, add it to the search list
                cityExists = True
                print argument, 'found'
                citiesToCalculate.append(int(row['URBID']))
                break
        if (not cityExists):
            print argument, 'NOT found'

        citiesFile.seek(0)  #reset the cities csv to be back at the start (so search begins from the start again)

    print

    print "loading population raster"
    src = gdal.Open(datadir+'/pop-'+year+'.tiff', gdal.GA_Update)
    band = src.GetRasterBand(1)
    print "converting to array"
    pop = np.array(band.ReadAsArray())
    pop[pop < 0] = 0  #set all population values below 0 as 0

    print

    print "loading cities numpy array"
    cities = np.load(os.path.expanduser('~') + '/Dropbox/CISC Data/SDEI-Global-UHI/sdei-global-uhi-2013.npy')
    #either calculate data for all cities or just ones passed
    if (len(citiesToCalculate) == 0):
        citiesToCalculate = np.unique(cities)  #get a SORTED list of all city ID's
        print 'Outputting data for all cities'
    else:
        citiesToCalculate = sorted(citiesToCalculate)  #if cities are passed, sort them (to make name lookup efficient)
        filename = 'citiesPop - '+year+' - '.join(sys.argv)+'.csv'
    print

    print "Saving numbers to " + filename

    #open output file
    with codecs.open(datadir+"/"+filename, 'w', 'utf-8') as outputFile:
        outputFile.write('ID, Name, Total Population\n')  #write the header row

        #go through all the cities
        for city in citiesToCalculate:
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
