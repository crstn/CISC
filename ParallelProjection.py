# coding: utf-8
#!/usr/bin/env python

import sys, multiprocessing, subprocess, os, time, os.path

# target = os.path.expanduser('~') + "/Dropbox/CISC Data/IndividualCountries/Projections/GRUMP/"
target = os.path.expanduser('~') + "/Dropbox/CISC Data/IndividualCountries/Projections/Test3/GlobCover/"
# target = '/Volumes/Solid Guy/Sandbox/GlobCover/'

# if this script is called without arguments, throw an error:
if len(sys.argv) <= 2:
    print "Please provide at least the scenario name and the urbanization dataset name (GRUMP or GlobCover) as parameter to run, e.g.:"
    print "python ParallelProjection.py SSP3 GRUMP"
    print "This will run the whole world for that scenario. Optionally, also provide the IDs of specific countries to run:"
    print "python ParallelProjection.py SSP1 GlobCover 156 376 [...]"
    sys.exit()

elif len(sys.argv) == 3: # specifying just the scenario, run the whole world:
    tasks = []
    for filename in os.listdir(os.path.expanduser('~') + '/Dropbox/CISC Data/IndividualCountries'):
        if filename.endswith(".npy"):
            # the year is between the dashes
            end = filename.find('.0-')
            if filename[:end] not in tasks:
                tasks.append(filename[:end])

else:
    tasks = sys.argv[3:]

print "running the following countries:"
print tasks

started = []

cpus = multiprocessing.cpu_count()
print "Running " +str(cpus)+ " countries in parallel"

while (len(tasks) > 0):
    # check if any of the started processes is done yet:
    for i in range(len(started)):
        # if the population projections for 2100 are there, the process is done
        # TODO: not ideal because an old version of that file might be there;
        # or maybe this is actually a feature, because countries we have run already
        # aren't run again? Only a problem if we change the projection algorithm!

        # there was a weird error where i would sometimes be out of bounds of the # list ...which should not be possible when running the code below,
        # but it still came up. Anyway, this solves it. ¯\_(ツ)_/¯
        if i < len(started):
            feil = target + sys.argv[1] +"/" + str(started[i])+"-2100-pop.npy"
            if(os.path.isfile(feil)):
                # complete, remove from started
                del(started[i])

    # check if we have a free CPU to start a new process
    if(len(started) < cpus):
        subprocess.Popen(["python", "ProjectPopulationOneCountry.py", str(tasks[0]), sys.argv[1], sys.argv[2]])
        # move to "started" list
        started.append(tasks[0])
        del(tasks[0])

    # print "pending tasks:"
    # print tasks
    # print "currently running:"
    # print started

    # wait a sec before we try again
    time.sleep(1)
