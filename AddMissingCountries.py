import os
import csv
import PopFunctions as pop

# scenarios to add the missing data for:
def addMissingCountries(scenarios = ['SSP1', 'SSP2', 'SSP3', 'SSP4', 'SSP5']):

    # missing countries (key) and the country they are part of in the SSPs (value)
    missing = {"158": "156", "499": "688", "728": "729", "732": "504"}

    # country codes
    codes = {"158": "TWN", "499": "MNE", "728": "SSD", "732": "ESH",
             "156": "CHN", "688": "SRB", "729": "SDN", "504": "MAR"}

    for scenario in scenarios:

        print scenario

        DESA_pop = pop.transposeDict(csv.DictReader(open(os.path.expanduser(
            '~') + '/Dropbox/CISC Data/DESA/WPP2015_POP_F01_1_TOTAL_POPULATION_BOTH_SEXES.csv')), "Country code")
        SSP_pop = pop.transposeDict(csv.DictReader(open(os.path.expanduser(
            '~') + '/Dropbox/CISC Data/SSPs/pop-' + scenario + '.csv')), "Country code")

        DESA_urban = pop.transposeDict(csv.DictReader(open(os.path.expanduser(
            '~') + '/Dropbox/CISC Data/DESA/WUPto2100_Peter_MEAN.csv')), "Country Code")
        SSP_urban = pop.transposeDict(csv.DictReader(open(os.path.expanduser(
            '~') + '/Dropbox/CISC Data/SSPs/urbpop-' + scenario + '.csv')), "Country code")

        # we need a list of all parent countries later to remove their old entries
        # from the CSV
        parents = []

        output_missed_pop = ""
        output_parent_pop = ""
        output_missed_urban = ""
        output_parent_urban = ""

        for missed, parent in missing.iteritems():

            parents.append(parent)

            # print pop.getCountryByID(missed, DESA_pop) + " is part of " + pop.getCountryByID(parent, DESA_pop) + " in the SSPs"

            output_missed_pop = output_missed_pop + missed + "," + codes[missed] + ", ,"
            output_parent_pop = output_parent_pop + parent + "," + codes[parent] + ", ,"
            output_missed_urban = output_missed_urban + missed + "," + codes[missed] + ", ,"
            output_parent_urban = output_parent_urban + parent + "," + codes[parent] + ", ,"

            # for every year, subtract the DESA number from the missed country from
            # the SSP parent country
            for year in range(2020, 2101, 10):
                # print
                # print year

                SSP_parent_pop = pop.getNumberForYear(SSP_pop, year, parent)
                DESA_missed_pop = pop.getNumberForYear(
                    DESA_pop, year, missed, 1000) # numbers are in thousands
                updateparentpop = SSP_parent_pop - DESA_missed_pop

                # rinse and repeat for urban:
                SSP_parent_urban = pop.getNumberForYear(SSP_urban, year, parent)
                DESA_missed_urban = pop.getNumberForYear(DESA_urban, year, missed) # numbers are NOT in thousands!

                updateparenturbanpop = int(SSP_parent_urban - DESA_missed_urban)

                output_missed_pop = output_missed_pop + str(DESA_missed_pop) + ','
                # subtract this number from the parent country!
                output_parent_pop = output_parent_pop + str(updateparentpop) + ','

                # add to the output string:
                output_missed_urban = output_missed_urban + \
                    str(DESA_missed_urban) + ','
                output_parent_urban = output_parent_urban + \
                    str(updateparenturbanpop) + ','

            # add the country
            output_missed_pop = output_missed_pop + '"' + \
                pop.getCountryByID(missed, DESA_pop) + '"\n'
            output_parent_pop = output_parent_pop + '"' + \
                pop.getCountryByID(parent, DESA_pop) + '"\n'
            output_missed_urban = output_missed_urban + '"' + \
                pop.getCountryByID(missed, DESA_pop) + '"\n'
            output_parent_urban = output_parent_urban + '"' + \
                pop.getCountryByID(parent, DESA_pop) + '"\n'

        # remove the lines for the PARENT countries in the two csv files
        # then add the output lines from above

        f = open(os.path.expanduser('~') +
                 '/Dropbox/CISC Data/SSPs/pop-' + scenario + '.csv', 'r')
        lines = f.readlines()
        f.close()

        # overwrite the same file
        f = open(os.path.expanduser('~') +
                 '/Dropbox/CISC Data/SSPs/pop-' + scenario + '.csv', 'w')
        for l in lines:
            # don't copy over the old data from the parent countries, otherwise
            # we would have two entries for those!
            add = True
            for p in parents:
                if l.startswith(p):
                    add = False

            if add:
                f.write(l)

        # then add our new numbers:
        f.write('\n')
        f.write(output_missed_pop)
        f.write(output_parent_pop)

        f.close()





        # repeat for urban:

        f = open(os.path.expanduser('~') +
                 '/Dropbox/CISC Data/SSPs/urbpop-' + scenario + '.csv', 'r')
        lines = f.readlines()
        f.close()

        # overwrite the same file
        f = open(os.path.expanduser('~') +
                 '/Dropbox/CISC Data/SSPs/urbpop-' + scenario + '.csv', 'w')
        for l in lines:
            # don't copy over the old data from the parent countries, otherwise
            # we would have two entries for those!
            add = True
            for p in parents:
                if l.startswith(p):
                    add = False

            if add:
                f.write(l)

        # then add our new numbers:
        f.write('\n')
        f.write(output_missed_urban)
        f.write(output_parent_urban)

        f.close()
