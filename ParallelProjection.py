# coding: utf-8
#!/usr/bin/env python

import sys, multiprocessing, subprocess, os, time, os.path

# target = os.path.expanduser('~') + "/Dropbox/CISC Data/IndividualCountries/Projections/"
target = '/Volumes/Solid Guy/SSP2 2017-02-22/'

# if this script is called without arguments, run the whole world:
if len(sys.argv) == 1:
    tasks = []
    for filename in os.listdir(os.path.expanduser('~') + '/Dropbox/CISC Data/IndividualCountries'):
        if filename.endswith(".npy"):
            # the year is between the dashes
            end = filename.find('.0-')
            if filename[:end] not in tasks:
                tasks.append(filename[:end])

else:
    tasks = sys.argv[1:]

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
            feil = target + str(started[i])+"-2100-pop.npy"
            if(os.path.isfile(feil)):
                # complete, remove from started
                del(started[i])

    # check if we have a free CPU to start a new process
    if(len(started) < cpus):
        subprocess.Popen(["python", "ProjectPopulationOneCountry.py", str(tasks[0])])
        # move to "started" list
        started.append(tasks[0])
        del(tasks[0])

    # print "pending tasks:"
    # print tasks
    # print "currently running:"
    # print started

    # wait a sec before we try again
    time.sleep(1)
