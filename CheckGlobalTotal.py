import os
import csv
import numpy as np
import PopFunctions as pop
import pync

# os.chdir(os.path.expanduser('~') + '/Dropbox/CISCdata/IndividualCountries/Projections/');
os.chdir('/Volumes/Solid Guy/SSPs 2017-06-16/Global/')

print "urbExt;SSP;Year;TotalTIFF;UrbanTIFF;TotalIIASA;UrbanIIASA;TotalDiff;UrbanDiff;TotalDiffPercent;UrbanDiffPercent"

def getIIASA(csvfile, year):
    with open("/Users/carsten/Dropbox/CISCdata/SSPs/"+csvfile) as fin:
        total = sum(int(r[year]) for r in csv.DictReader(fin))
        return total


for m in ["GRUMP", "GlobCover"]:
    for ssp in ["SSP1", "SSP2", "SSP3", "SSP4", "SSP5"]:
        pync.Notifier.notify('Checking ' + m + ' / ' + ssp , title='Python')
        for y in range (2010, 2101, 10):
            year = str(y)
            folder = m+"/"+ssp+"/"

            p = pop.openTIFFasNParray(folder+'pop-'+year+'.tiff')
            u = pop.openTIFFasNParray(folder+'urbanRural-'+year+'.tiff')
            # -1 = NAN, set to 0:
            p[p<0] = 0

            totalTIFF = np.nansum(p)
            urbTIFF   = np.nansum(p[u == 2])

            totalIIASA = getIIASA("pop-"+ssp+".csv", year)
            urbIIASA   = getIIASA("urbpop-"+ssp+".csv", year)

            totalDiff = totalIIASA - totalTIFF
            urbDiff   = urbIIASA - urbTIFF

            totalDiffPercent = float(totalDiff) / float(totalIIASA) * 100.0
            urbanDiffPercent = float(urbDiff) / float(urbIIASA) * 100.0

            print m + ";" + ssp + ";"+ year + ";" + "{:,}".format(totalTIFF) + ";" + "{:,}".format(urbTIFF) + ";" + "{:,}".format(totalIIASA) + ";" + "{:,}".format(urbIIASA) + ";" + "{:,}".format(totalDiff) + ";" + "{:,}".format(urbDiff) + ";" + str(totalDiffPercent) + ";" + str(urbanDiffPercent)

pync.Notifier.notify('Global total and urban population checked', title='Python')
