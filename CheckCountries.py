#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, csv, numpy as np, PopFunctions as pop


countryBoundaries = pop.openTIFFasNParray(os.path.expanduser('~') + '/Dropbox/CISC Data/Nations Raster/ne_10m_admin_0_countries_nibbled.tiff')
countriesInRaster = np.unique(countryBoundaries)

WTP = pop.transposeDict(csv.DictReader(open(os.path.expanduser('~') + '/Dropbox/CISC Data/DESA/WTP2014.csv')), "Country Code")

print "Checking for countries that are in raster, but not in CSV. This should only spit out countries that are actually small island (carribean Netherlands etc.):"
for c in countriesInRaster:
    if str(int(c)) not in WTP:
        print str(int(c)) + " is not in CSV"

print " "
print "Checking for countries that are in CSV, but not in raster. "
for w in WTP:
    if float(w) not in countriesInRaster:
        print WTP[w]['Major area, region, country or area'] + " (" + w + ") is not in countries raster"
