import os
import datetime
import sys
import math
import csv
import numpy as np
import PopFunctions as p
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

"""Iterates through a series of countries, and, for each country and each year (2010 to 2100,
   in 10 year steps) goes through all listed SSPs and urbal/rural models. For each combination
   (e.g., SSP3/GRUMP) is goes through all simulation runs and generates the following:

   1. A historgram
   2. An output array with the MEAN per cell
   3. An output array with the MEDIAN per cell
   4. An output array with the MIN per cell
   5. An output array with the MAX per cell
   6. An output array with the probability of urbanization per cell.
      This array will reflect the percentage of simulation runs
      in which the cell was urban (e.g. if a cell turns urban in 31 out
      of 50 runs, the cell value will be 0.62)."""


countries=['392','528']
runs = 50
ssps = ['SSP3']
urbanRuralVersions = ['GRUMP']
folder = os.path.expanduser('~') + '/Desktop/statstest'

urbanCell = 2

plt.style.use('fivethirtyeight')

scale_x = 1e-6 # we'll scale the x-axis to millions of people
scale_x = 1

for c in countries:

    for u in urbanRuralVersions:

        for ssp in ssps:

            for y in range(2010,2101,10):

                WTP = p.transposeDict(csv.DictReader(open(os.path.expanduser('~') + '/Dropbox/CISC Data/SSPs/pop-'+ssp+'.csv')), "Country code")

                WUP = p.transposeDict(csv.DictReader(open(os.path.expanduser('~') + '/Dropbox/CISC Data/SSPs/urbpop-'+ssp+'.csv')), "Country code")

                country = p.getCountryByID(c, WTP)

                # these will help us calculate stats across runs
                pops = []
                urbs = []

                # make a stack of all runs to calculate the sum of means per
                # cell ("vertical mean") as well as min, max etc
                # Also chance of urbanization
                popstack = np.array([])
                urbstack = np.array([])
                for r in range(runs):

                    pop = np.load(folder+'/'+str(r)+'/'+u+'/'+ssp+'/'+c+'-'+str(y)+'-pop.npy')
                    urbanRural = np.load(folder+'/'+str(r)+'/'+u+'/'+ssp+'/'+c+'-'+str(y)+'-urbanRural.npy')


                    if(len(popstack) == 0):
                        # initiate:
                        popstack = np.append(popstack, pop)
                    else:
                        popstack = np.vstack((popstack, pop))



                    # before we stack up the urban cells, set all rural cells to 0,
                    # and urban cells to 1. That way, we can just calculate the mean
                    # later, and that will give us the chance of urbanization.

                    urbanRural[urbanRural == 1] = 0
                    urbanRural[urbanRural == 2] = 1

                    if(len(urbstack) == 0):
                        # initiate:
                        urbstack = np.append(urbstack, urbanRural)
                    else:
                        urbstack = np.vstack((urbstack, urbanRural))



                    popsum = np.sum(pop)

                    print country + " " + str(y) + " (run " + str(r) + "): " + str(popsum)

                    pops.append(popsum)
                    urbs.append(np.sum(pop[urbanRural == 1])) # urban is 1 now!



                # Now we'll caculate the mean, median, min and max population number for each individual cell
                # and spit them out as numpy arrays:

                # Check if we already have the output folder:
                target = folder+'/summaries/'+u+'/'+ssp

                if not os.path.exists(target):
                    os.makedirs(target)

                vmean_array = np.mean(popstack, axis=0)
                np.save(target + '/'+c+'-'+str(y)+'-popmean.npy', vmean_array)

                vmedian_array = np.median(popstack, axis=0)
                np.save(target + '/'+c+'-'+str(y)+'-popmedian.npy', vmedian_array)

                vmin_array = np.min(popstack, axis=0)
                np.save(target + '/'+c+'-'+str(y)+'-popmin.npy', vmin_array)

                vmax_array = np.max(popstack, axis=0)
                np.save(target + '/'+c+'-'+str(y)+'-popmax.npy', vmax_array)

                # and finally calculate the chance of urbanization per cell:
                urbmean_array = np.mean(urbstack, axis=0)
                np.save(target + '/'+c+'-'+str(y)+'-urbanization.npy', urbmean_array)


                # next up: generate some stats and histograms
                # get the sum of thoe vertical means:
                vmean = np.sum(vmean_array)

                meanFromSimulations = np.mean(pops)
                sspNumber = p.getNumberForYear(WTP, y, c)

                print
                print "Mean:    " + '{0:,}'.format(meanFromSimulations)
                print "SSP:     " + '{0:,}'.format(sspNumber)
                print "V. Mean: " + '{0:,}'.format(vmean)
                print "V. Mean - SSP: "  + '{0:,}'.format(vmean-sspNumber)
                print


                # plot a histogram across runs:

                plt.hist(pops, color='#008fd5')
                # plt.vline(np.mean(pops))
                plt.title(country + ' population histogram for '+str(runs)+' runs ('+str(y)+')')
                plt.ylabel('Frequency')
                plt.xlabel('Total population')
                plt.axvline(meanFromSimulations * scale_x, linewidth=2, color='#fc4f30', label="Mean across runs (" + '{0:,}'.format(int(meanFromSimulations)) + ")")
                plt.axvline(sspNumber * scale_x, linestyle='dashed', linewidth=1, color='#e5ae38', label="Population SSP (" + '{0:,}'.format(sspNumber) + ")")
                # plt.axvline(vmean * scale_x, linestyle='dashed', linewidth=1, color='#6d904f', label='"Vertical" mean')
                plt.legend(loc='upper left')

                plt.savefig(folder+'/'+country+'-'+str(y)+'-pop.pdf', bbox_inches='tight')
                # clear this figure, start a new one:
                plt.clf()
