
# ----------------------------------------------------------------------
# Name: find_highest_3_months.py
# Author: Peter J. Marcotullio
# Date: 7 September 2015
# Description: This script iterates through a list of 12 items, performs
# math on that list (in this case a simple sum) for every possible
# 3-month consecutive interval.  It creates a new list and then finds the
# highest value and prints the value and the 3 consecutive months label
# -----------------------------------------------------------------------


temp3list=[] # create an empty list to use later

templist = [6,5,4,3,2,1,1,2,3,4,4,6] # input some list of temperatures

# Add the first 2 months to the input list
for x in templist[:2]:
    templist.append(x)

# Do math to the input list over three-month intervals(addition in this case)
for x in range(12):
    sum_3 = sum(templist[0:3])
    temp3list.append(sum_3) # Append the math to the new list
    templist = templist[1:] # remove the first item of the last list

# Find the max temp in the new list - What if there are two similar highs?
# create a variable of that max and find index in the list
max_temp = max(temp3list)
x = temp3list.index(max_temp)

# A list of date labels
dates = ["January-March", "February-April", "March-May", "April-June",
         "May-July", "June-August", "July-September", "August-October",
         "September-November", "October-December", "November-January",
         "December-February"]

# Identify the label for the max temp
date = dates[x]

print "The hottest 3 months of the year were %s with a temp of %d" %(date, max_temp)

# Below a list comprehensive way to identify index value
# [i for i, y in enumerate(dates) if y = x] # untested
