# Checks if all 20 files for a given country are present after a simulation run

import os
import fnmatch

d = '/Volumes/Solid Guy/Sandbox/GlobCover/SSP5'

lastc = ""

for fn in os.listdir(d):
    parts = fn.split("-")
    c = parts[0]

    if c != lastc:
        count = 0
        for f in os.listdir(d):
            if fnmatch.fnmatch(f, c+'-*'):
                count = count + 1

        if count != 20:
            print str(parts[0])+": "+str(count)

        lastc = c
