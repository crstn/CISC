import os, csv, numpy as np, PopFunctions as pop, matplotlib
from matplotlib import pyplot
from matplotlib.backends.backend_pdf import PdfPages


limit = 2100 # we'll look for countries that have projections up to this year

os.chdir(os.path.expanduser('~') + '/Dropbox/CISC Data/IndividualCountries/Projections')

countries = []

# collect all countries that have projections up to our limit:
for filename in os.listdir('.'):
    if filename.endswith(".npy"):
        # the year is between the dashes
        start = filename.find('-')
        end = filename.rfind('-')

        country = filename[:start]
        year = filename[start+1:end]

        if year == str(limit) and country not in countries:
            countries.append(country)


# load the population projection numbers:
# world URBAN population
WUP = pop.transposeDict(csv.DictReader(open(os.path.expanduser('~') + '/Dropbox/CISC Data/DESA/WUPto2100_Peter_MEAN.csv')), "Country Code")
# world TOTAL population
WTP = pop.transposeDict(csv.DictReader(open(os.path.expanduser('~') + '/Dropbox/CISC Data/DESA/WPP2015_POP_F01_1_TOTAL_POPULATION_BOTH_SEXES.csv')), "Country code")

matplotlib.style.use('fivethirtyeight')
# make the font smaller, so that the legends don't take up too much space
matplotlib.rcParams.update({'font.size': 8})

# for each country and each projected year, compare the numbers (total, rural, urban):
for country in countries:

    try:

        # the boundary will stay the same, so only load it once
        boundary = np.load(os.path.expanduser('~') + '/Dropbox/CISC Data/IndividualCountries/'+country+'.0-boundary.npy').ravel()

        years = []

        popcsv = []
        urbcsv = []
        rurcsv = []
        popraster = []
        urbraster = []
        rurraster = []

        globaltotal = []
        globalurban = []
        globalrural = []

        for year in range(2020, limit+1, 10):
            years.append(year)

            #load CSV numbers
            p = pop.getNumberForYear(WTP, year, country, 1000)
            popcsv.append(p)
            u = pop.getNumberForYear(WUP, year, country)
            urbcsv.append(u)
            rurcsv.append(p-u)

            # initialize the globals with 0s:
            globaltotal.append(0)
            globalurban.append(0)
            globalrural.append(0)

            #load numpy arrays for country/year
            urbanRural = np.load(os.path.expanduser('~') + '/Dropbox/CISC Data/IndividualCountries/Projections/'+country+'-'+str(year)+'-urbanRural.npy').ravel()
            population = np.load(os.path.expanduser('~') + '/Dropbox/CISC Data/IndividualCountries/Projections/'+country+'-'+str(year)+'-pop.npy').ravel()

            popraster.append(np.nansum(population[boundary == int(country)]))
            urbraster.append(np.nansum(population[
                np.logical_and(boundary == int(country),
                               urbanRural == pop.urbanCell)]))
            rurraster.append(np.nansum(population[
                np.logical_and(boundary == int(country),
                               urbanRural == pop.ruralCell)]))

        print " --- "
        print " "
        print pop.getCountryByID(country, WTP)
        print " "
        print years

        print popcsv
        print popraster

        print urbcsv
        print urbraster

        print rurcsv
        print rurraster
        print " "

        pyplot.plot(years, popcsv, label="Total CSV", linewidth = 1.0)
        pyplot.plot(years, popraster, label="Total Raster", linewidth = 1.0)
        pyplot.plot(years, urbcsv, label="Urban CSV", linewidth = 1.0)
        pyplot.plot(years, urbraster, label="Urban raster", linewidth = 1.0)
        pyplot.plot(years, rurcsv, label="Rural CSV", linewidth = 1.0)
        pyplot.plot(years, rurraster, label="Rural raster", linewidth = 1.0)

        pyplot.xlabel('Year')
        pyplot.ylabel('Population')
        pyplot.title(pop.getCountryByID(country, WTP))
        pyplot.legend(loc='lower right')
        pyplot.savefig(os.path.expanduser('~')+'/Desktop/plots/'+pop.getCountryByID(country, WTP)+'.pdf', bbox_inches='tight', dpi=300)

        # clear this figure, start a new one:
        pyplot.clf()

        # update the global numbers:
        for i in range(len(years)):
            globaltotal[i] = globaltotal[i] + popraster[i]
            globalurban[i] = globalurban[i] + urbraster[i]
            globalrural[i] = globalrural[i] + rurraster[i]

    except Exception as e:
        print e


# when we're done with all countries, spit out the global numbers from the raster:
print "Global totals:"
print years
print globaltotal
print "Global urban:"
print globalurban
print "Global rural:"
print globalrural
