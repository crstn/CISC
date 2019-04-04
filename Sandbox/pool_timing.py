from multiprocessing import Pool
import multiprocessing
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import timeit

def multiply(a, z):
    #return((a*a)+z, int(z))

    # deliberately take a slow function:
    return [x+z for x in a]

def split_by_zone(a,z):

    """Splits a by zones defined in z, and returns:
    - the sub-array of a for each zone
    - the zone IDs
    """

    zone_ids = np.unique(z)
    results = list()

    for zone in zone_ids:
        results.append((a[z == zone], zone))

    return results


def make_zones(like):

    """Generates 5 zones for a 2d array "like": one for each corner,
    plus one for the center."""

    x,y = like.shape

    zones = np.zeros_like(like)
    zones[x//2:x,0:x//2] = 1
    zones[x//2:x,y//2:y] = 2
    zones[x//4:(x//4)*3,y//4:(y//4)*3] = 3

    return zones




def spawn(x,y,cores,plot=False):

    print("Running " + str(x)+"*"+str(y)+" on "+str(cores)+" cores.")

    in_arr = np.random.rand(x*y).reshape([x,y])

    zones = make_zones(in_arr)


    if(plot):
        plt.imshow(in_arr)
        plt.colorbar()
        plt.savefig("in.png")
        plt.clf() # clear figure

        plt.imshow(zones)
        plt.colorbar()
        plt.savefig("zones.png")
        plt.clf() # clear figure


        #pool = Pool(cores)

        # pool.starmap allows us to pass multiple arguments to the function
        # executed by the pool. The function split_by_zone does exactly
        # that: it spits out a list of tuples, where each tuple consists
        # of the zone ID and an array with the values within that zone.

        with Pool(cores) as p:
            out_arr, out_zone = p.starmap(multiply,split_by_zone(in_arr, zones))
            in_arr[zones == out_zone] = out_arr

        # for res in pool.starmap(multiply,split_by_zone(in_arr, zones)):
        #     # replace the values for the current zone with the
        #     # processed values from the spawned process:
        #     out_arr, out_zone = res
        #     in_arr[zones == out_zone] = out_arr


    if(plot):
        plt.imshow(in_arr)
        plt.colorbar()
        plt.savefig("out.png")


# let's time the function: How does it do with different sizes of arrays
# and different numbers of cores?
if __name__ == '__main__':

    sizes = [10,100,1000,10000]
    # have to add 1 to each number, because range starts at 0...
    cores =  [x+1 for x in list(range(multiprocessing.cpu_count()))]

    outsizes = []
    outcores = []
    times = []

    for s in sizes:
        for c in cores:

            start = timeit.default_timer()
            spawn(s,s,c)
            times.append(timeit.default_timer() - start)
            outsizes.append(s*s)
            outcores.append(c)

    # prepare for plotting: put everything into a dataframe:

    timings = pd.DataFrame(
    {'time': times,
     'size': outsizes,
     'cores': outcores
    })

    print(timings)


    plot = sns.lineplot(x="size", y="time", hue="cores", data=timings)
    plot.figure.savefig("timings.pdf")
