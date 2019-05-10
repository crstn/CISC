# coding: utf-8
#!/usr/bin/env python

import sys, multiprocessing, subprocess, os, time, os.path


# we'll load the country IDs to run into this one in a bit
runCountries = []

# overwrite existing simulation data?
overwrite = True

# Generate summaries (mean pop and chance of urbanization) for each country based on simulations?
summarize = True

# also reassemble the individual files to GeoTIFFs and delete the original .npy arrays?
reassemble = True



def loadTasks():
    for filename in os.listdir(os.path.expanduser('~') + '/Dropbox/CISCdata/IndividualCountries'):
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
for urbanmodel in urbanmodels:
    for ssp in ssps:
        for c in runCountries:
            for run in range(runs):
                # append the dictionary for this specific task:
                tasks.append({'script': 'ProjectPopulationOneCountry.py',
                              'outputfile': target + str(run) +"/" + urbanmodel + "/" + ssp + "/" + c +"-2100-pop.npy", # done when this file is there
                              'parameters': [c, ssp, urbanmodel, target + str(run)]})
                            #   'run': run,
                            #   'urbanmodel': urbanmodel,
                            #   'ssp': ssp,
                            #   'country': c})

            if summarize:
                tasks.append({'script': 'Summarize.py',
                              'outputfile': target + "summaries/" + urbanmodel + "/" + ssp + "/" + c +"-2100-urbanization.npy", # done when this file is there
                              'parameters': [c, str(runs), ssp, urbanmodel, target]})

    #append a task for this run that will reassemble the simulations for this run into GeoTIFFs:
    if reassemble:
        tasks.append({'script': 'ReassambleCountries.py',
                      'outputfile': target + "summaries/" + urbanmodel + "/" + ssp + "/urbanRural-2100.tiff", # done when this file is there
                      'parameters': [target + "summaries/", ssp, urbanmodel] + runCountries})

# keep track of the tasks that have been started
started = []

while (len(tasks) > 0):
    # check if any of the started processes is done yet:
    for i in range(len(started)):
        if i < len(started):
            s = started[i]
            if(os.path.isfile(s['outputfile'])):
                print s['outputfile'] + " is complete"
                # complete, remove from started
                del(started[i])


    # check if we have a free CPU to start a new process
    while((len(started) < cpus) and (len(tasks) > 0)):
        t = tasks[0]
        # only start the task IF:
        # 1. the destination file does not exist
        # OR
        # 2. the file exists, but overwrite is set tu True:
        if (not (os.path.isfile(t['outputfile']))) or (os.path.isfile(t['outputfile']) and overwrite):
            if os.path.isfile(t['outputfile']):
                print "Overwriting " + t['outputfile']
            subprocess.Popen(["python", t['script']] + t['parameters'])
            # # move to "started" list
            started.append(tasks[0])
        else:
            print "Overwriting is turned off, skipping " + t['outputfile']

        # either way, this task can be removed from the task list
        del(tasks[0])

    # wait a sec before we try again
    time.sleep(1)
