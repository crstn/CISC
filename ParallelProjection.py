# coding: utf-8
#!/usr/bin/env python

import sys, multiprocessing, subprocess, os, time, os.path


# we'll load the country IDs to run into this one in a bit
runCountries = []

def loadTasks():
    for filename in os.listdir(os.path.expanduser('~') + '/Dropbox/CISC Data/IndividualCountries'):
        if filename.endswith(".npy"):
            # the year is between the dashes
            end = filename.find('.0-')
            if filename[:end] not in runCountries:
                runCountries.append(filename[:end])

def abort():
    print """There are three ways to run this script:
    1. Just pass the number of repititions and the target folder as an argument.
       This will run the simulations both for GRUMP and GlobCover, for all SSPs:

        python ParallelProjection.py 99 /Users/Donald/simulations/output

    2. If you want to run a specific combination of urbanization model and SSP, pass those in addition:

        python ParallelProjection.py 99 /Users/Donald/simulations/output SSP3 GRUMP

    3. If you just want to run specific countries for a given urbanization model and SSP, add the country IDs:

        python ParallelProjection.py 99 /Users/Donald/simulations/output SSP3 GRUMP 156 376 [...]"""
    sys.exit()



# we need at least 2 arguments:
if len(sys.argv) < 3:
    abort()

elif len(sys.argv) == 3: # run the whole world with all scenarios
    loadTasks()
    runs = int(sys.argv[1])
    target = sys.argv[2]
    urbanmodels = ["GRUMP", "GlobCover"]
    ssps = ["SSP1", "SSP2", "SSP3", "SSP4", "SSP5"]

elif len(sys.argv) == 4:
    abort()

elif len(sys.argv) == 5:
    loadTasks()
    runs = int(sys.argv[1])
    target = sys.argv[2]
    ssps = [sys.argv[3]]
    urbanmodels = [sys.argv[4]]

else:
    runs = int(sys.argv[1])
    target = sys.argv[2]
    ssps = [sys.argv[3]]
    urbanmodels = [sys.argv[4]]
    runCountries = sys.argv[5:]

# add a trailing slash to the target folder name, if not there:
if target.strip()[-1] != '/':
    target = target + '/'


cpus = multiprocessing.cpu_count()

print "Running %s runs on %s CPUs to %s with the following parameters:" % (runs, cpus, target)
print ssps
print urbanmodels
print runCountries

# empty list that will hold distionaries describing each individual task:
tasks = []

# add all tasks to one big array:
for run in range(runs):
    for urbanmodel in urbanmodels:
        for ssp in ssps:
            for c in runCountries:
                # append the dictionary for this specific task:
                tasks.append({'run': run,
                              'urbanmodel': urbanmodel,
                              'ssp': ssp,
                              'country': c})


# keep track of the tasks that have been started
started = []

while (len(tasks) > 0):
    # check if any of the started processes is done yet:
    for i in range(len(started)):
        # there was a weird error where i would sometimes be out of bounds of the
        # list ...which should not be possible when running the code below,
        # but it still came up. Anyway, this solves it. ¯\_(ツ)_/¯
        if i < len(started):
            s = started[i]
            feil = target + str(s['run']) +"/" + s['urbanmodel'] + "/" + s['ssp'] + "/" + s['country'] +"-2100-pop.npy"
            if(os.path.isfile(feil)):
                print feil + " is complete"
                # complete, remove from started
                del(started[i])

    # check if we have a free CPU to start a new process
    if(len(started) < cpus):
        t = tasks[0]
        subprocess.Popen(["python", "ProjectPopulationOneCountry.py", t['country'], t['ssp'], t['urbanmodel'], target + str(t['run'])])
        # move to "started" list
        started.append(tasks[0])
        del(tasks[0])

    # wait a sec before we try again
    time.sleep(1)
