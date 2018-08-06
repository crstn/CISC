import os, matplotlib
import numpy as np
from matplotlib import pyplot
from matplotlib.backends.backend_pdf import PdfPages

"""Prints a histogram of the population in individual calls in a given country."""


country = "124"
limit = 2100

matplotlib.style.use('fivethirtyeight')

boundary = np.load(os.path.expanduser('~') + '/Dropbox/CISCdata/IndividualCountries/'+country+'.0-boundary.npy').ravel()

for year in range(2020, limit+1, 10):
    population = np.load(os.path.expanduser('~') + '/Dropbox/CISCdata/IndividualCountries/Projections/'+country+'-'+str(year)+'-pop.npy').ravel()
    incountry = boundary == int(country)
    p = population[incountry]

    print year
    # print np.min(p)
    # print np.nanmin(p)
    # print np.max(p)
    # print np.nanmax(p)
    print np.sum(p)

    r = [np.nanmin(p), np.nanmax(p)]
    print r

    pyplot.hist(np.log(p), normed=True, range=r)
    pyplot.savefig(os.path.expanduser('~') + '/Desktop/Histograms/'+country+'-'+str(year)+'.pdf', bbox_inches='tight')
