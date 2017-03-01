from osgeo import gdal
import numpy as np
import csv             #for reading csv
import codecs          #for utf-8
import os, sys

del sys.argv[0]  #delete the first argument (the script name)

#open cities CSV file for names
with open(os.path.expanduser('~') + '/Dropbox/CISC Data/SDEI-Global-UHI/CitiesAttributes.csv', 'rb') as citiesFile:
    cityNames = csv.DictReader(citiesFile)

    citiesToCalculate = []
    #find cities passed in as arguments
    for argument in sys.argv:
        print 'Finding', argument
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
    src = gdal.Open(os.path.expanduser('~') + '/Dropbox/CISC Data/Population 2010 Raster/Pop_2010_clipped.tiff', gdal.GA_Update)
    band = src.GetRasterBand(1)
    print "converting to array"
    pop = np.array(band.ReadAsArray())
    pop[pop < 0] = np.nan  #set all population values below 0 as NaN

    print

    os.chdir(os.path.expanduser('~') + '/Dropbox/CISC Data/SDEI-Global-UHI')

    print "loading cities numpy array"
    cities = np.load('sdei-global-uhi-2013.npy')
    #either calculate data for all cities or ust ones passed
    if (len(citiesToCalculate) == 0):
        citiesToCalculate = np.unique(cities)  #get a SORTED list of all city ID's
        print 'Outputting data for all cities'
    else:
        citiesToCalculate = sorted(citiesToCalculate)  #if cities are passed, sort them (to make name lookup efficient)

    print

    #open output file
    with codecs.open('citiesPop.txt', 'w', 'utf-8') as outputFile:
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
