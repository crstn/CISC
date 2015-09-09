
# ----------------------------------------------------------------------
# Name: has_3_months-above.py
# Author: Carsten Kessler
# Date: 8 September 2015
# Description: This script iterates through a list of 12 items, and checks
# whether the list has 3 consecutive items above a value X, where X is
# pulled in as a command line argument.
# -----------------------------------------------------------------------

import sys

x = int(sys.argv[1])
templist = [6,5,4,3,2,1,1,2,3,4,4,6] # input some list of temperatures
monthlist = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec", "Jan", "Feb"]

for i in range(0, len(templist)):
    print monthlist[i],":",templist[i]

# copy the first two months to the end of the array
# so that we can always check for 3 consecutive months 
templist.append(templist[0])
templist.append(templist[1])
monthlist.append(monthlist[0])
monthlist.append(monthlist[1])

for i in range(0, len(templist)-2):
     if (templist[i] > x) and (templist[i+1] > x) and (templist[i+2] > x):
         print str(x) + " is exceeded in " + monthlist[i] + ", " + monthlist[i+1] + ", and " + monthlist[i+2]
