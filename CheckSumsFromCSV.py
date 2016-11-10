# coding: utf-8
#!/usr/bin/env python

# calculates the global totals from the DESA csv

import sys, multiprocessing, subprocess, os, time, os.path, csv
import PopFunctions as pop

tasks = []
for filename in os.listdir(os.path.expanduser('~') + '/Dropbox/CISC Data/IndividualCountries'):
    if filename.endswith(".npy"):
        # the year is between the dashes
        end = filename.find('.0-')
        if filename[:end] not in tasks:
            tasks.append(filename[:end])

print "running the following countries:"
print tasks

WTP = pop.transposeDict(csv.DictReader(open(os.path.expanduser('~') + '/Dropbox/CISC Data/DESA/WPP2015_POP_F01_1_TOTAL_POPULATION_BOTH_SEXES.csv')), "Country code")

endyear = 2100
year = 2020
step = 10
while year <= endyear:
    summ = 0
    for country in tasks:
        try:
            summ = summ + pop.getNumberForYear(WTP, year, country, 1000)
        except Exception as e:
            pass
            # print "Error caught while looking up country " + str(country)

    print str(year) + ": " + str(summ/1000000000.0)
    year = year + step
