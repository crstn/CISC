import os, random, datetime
import numpy as np
import pandas as pd

def main(path, countryCode, urb_cell, rur_cell, pop_array, indexed_WUP, indexed_WTP, RUNS):


    saveLoc = path+"/Output_"+countryCode

    # os.chdir(os.path.expanduser(path))

    #=================================================================================================
    # Define functions for population growth
    #=================================================================================================

    def Start_Urban_Choice(IndexedWUP, Urb2000):
        x = int(countryCode)
        UrbanChange =  (((indexed_WUP.loc[x, "2010"]*1000)) - Urb2000)
        return UrbanChange


    def Start_Rural_Choice(IndexedWUP, IndexedWTP, Rur2000):
        rural_pop_2010 = ((indexed_WTP.loc[int(countryCode), "2010"]) - (indexed_WUP.loc[int(countryCode), "2010"]))*1000
        RuralChange = (rural_pop_2010 - Rur2000)
        return RuralChange

    def Grow_Urban_Start(popArray, UrbanChangeCell):
        popArray[UrbanChangeCell] += 1000
        return popArray

    def Shrink_Urban_Start(popArray, UrbanChangeCell):
        popArray[UrbanChangeCell] -= 1000
        return popArray

    def Grow_Rural_Start(popArray, RuralChangeCell):
        popArray[RuralChangeCell] += 1000
        return popArray

    def Shrink_Rural_Start(popArray, RuralChangeCell):
        popArray[RuralChangeCell] -= 1000
        return popArray

    def Urban_Change_Choice(IndexedWUP, countryCode, year):
        x = int(countryCode)
        previous_year = (year - 5)
        UrbanChange =  ((indexed_WUP.loc[x, str(year)]) - (indexed_WUP.loc[x, str(previous_year)]))*1000
        return UrbanChange

    def Rural_Change_Choice(IndexedWUP, IndexedWTP, countryCode, year):
        previous_year = (year - 5)
        x = int(countryCode)
        total_pop_change = ((indexed_WTP.loc[x, str(year)]) - (indexed_WTP.loc[x, str(previous_year)]))*1000
        urban_pop_change = ((indexed_WUP.loc[x, str(year)]) - (indexed_WUP.loc[x, str(previous_year)]))*1000
        RuralChange = total_pop_change - urban_pop_change
        return RuralChange


    def Select_Random_Urban_Cell(UrbanCellList):
        UrbanChangeCell = UrbanCellList[random.randint(0, (len(UrbanCellList) - 1))]
        return UrbanChangeCell


    def Select_Random_Rural_Cell(RuralCellList):
        RuralChangeCell = RuralCellList[random.randint(0, (len(RuralCellList) - 1))]
        return RuralChangeCell


    def Grow_Urban_Population(popArray, UrbanChangeCell):
        popArray[UrbanChangeCell] += 1.0
        return popArray


    def Shrink_Urban_Population(popArray, UrbanChangeCell):
        popArray[UrbanChangeCell] -= 1.0
        return popArray


    def Grow_Rural_Population(popArray, RuralChangeCell):
        popArray[RuralChangeCell] += 1.0
        return popArray


    def Shrink_Rural_Population(popArray, RuralChangeCell):
        popArray[RuralChangeCell] -= 1.0
        return popArray


    def Export_Array_for_Year(popArray, year, run):
        os.chdir(saveLoc)
        np.save("Pop_"+countryCode+"_"+str(year)+"_"+str(run), popArray)


    #=================================================================================================
    # Running the  program
    #=================================================================================================

    print "Beginning Analysis for country code ", countryCode, " at:"
    print datetime.datetime.now().strftime("%A, %d. %B %Y %I:%M%p")
    print ""


    print "Aggregating urban and rural populations for 2000."
    urban_pop_list00 = []
    for i in urb_cell:
        urban_pop_list00.append(pop_array[i])
    urban_pop_00 = int(sum(urban_pop_list00))
    print "Urban Population 2000:", urban_pop_00
    total_pop_00 = int(np.sum(pop_array))
    rural_pop_00 = total_pop_00 - urban_pop_00
    print "Rural Population 2000:", rural_pop_00
    print "2000 populations established."

    #=================================================================================================
    #Grow the population
    #=================================================================================================

    print "Growing Population"
    for runCount in range(RUNS):
        print "Run no. ", runCount

        try:  # some country codes are not in the csv, we just skip over them here:

            start_urban_change = Start_Urban_Choice(indexed_WUP, urban_pop_00)
            print start_urban_change

            if start_urban_change > 0:
                for person in xrange(start_urban_change):
                    urban_change_cell = Select_Random_Urban_Cell(urb_cell)

                    pop_array = Grow_Urban_Population(pop_array, urban_change_cell)

            if start_urban_change < 0:
                print start_urban_change
                counter = 0
                print "Shrinking urban population to reconcile discrpeancy"
                while counter < abs(start_urban_change):
                    urban_change_cell = Select_Random_Urban_Cell(urb_cell)
                    if pop_array[urban_change_cell] > 0:
                        pop_array = Shrink_Urban_Population(pop_array, urban_change_cell)
                        counter += 1
                    else:
                        continue
            if start_urban_change == 0:
                pass

            start_rural_change = Start_Rural_Choice(indexed_WUP, indexed_WTP, rural_pop_00)
            print start_rural_change
            if start_rural_change > 0:
                print "Growing rural population to reconcile discrepancy"
                for person in xrange(start_rural_change):
                    rural_change_cell = Select_Random_Rural_Cell(rur_cell)

                    pop_array = Grow_Rural_Population(pop_array, rural_change_cell)

            if start_rural_change < 0:
                print "Shrinking rural population to reconcile discrepancy"
                counter = 0
                while counter < abs(start_rural_change):
                    rural_change_cell = Select_Random_Rural_Cell(rur_cell)
                    if pop_array[rural_change_cell] > 0:
                        pop_array = Shrink_Rural_Population(pop_array, rural_change_cell)
                        counter += 1
                    else:
                        continue
            if start_rural_change == 0:
                pass
            print "Done reconciling pop_array - UN_DESA discrepancy"


            year = 2015
            print year
            while year <= 2050:
               urban_change =  Urban_Change_Choice(indexed_WUP, countryCode, year)
               print "For country code", countryCode, "in year", year, "the urban population change is", urban_change

               if urban_change > 0:
                    for person in xrange(urban_change):
                        urban_change_cell = Select_Random_Urban_Cell(urb_cell)

                        pop_array = Grow_Urban_Population(pop_array, urban_change_cell)

               if urban_change < 0:
                    counter = 0
                    while counter < abs(urban_change):
                        urban_change_cell = Select_Random_Urban_Cell(urb_cell)
                        if (pop_array[urban_change_cell] > 0):
                            pop_array = Shrink_Urban_Population(pop_array, urban_change_cell)
                            counter += 1
                        else:
                            continue

               if urban_change == 0:
                    pass


               rural_change = Rural_Change_Choice(indexed_WUP, indexed_WTP, countryCode, year)
               print "For country code", countryCode, "in year", year, "the rural population change is", rural_change

               if rural_change > 0:
                   for person in xrange(rural_change):
                       rural_change_cell = Select_Random_Rural_Cell(rur_cell)

                       pop_array = Grow_Rural_Population(pop_array, rural_change_cell)

               if rural_change < 0:
                    counter = 0
                    while counter < abs(rural_change):
                        # print "Rural growth counter:", counter
                        rural_change_cell = Select_Random_Rural_Cell(rur_cell)
                        if (pop_array[rural_change_cell] > 0):
                            pop_array = Shrink_Rural_Population(pop_array, rural_change_cell)
                            counter += 1
                        else:
                            continue
               if rural_change == 0 :
                   pass
               Export_Array_for_Year(pop_array, year, runCount)

               print "Done with growth for year", year, ", run", runCount
               year += 5

        except Exception as error:
            print " --- "
            print " "
            print "Oops! Something crashed. Maybe the country code was not in our CSV files?"
            print " "
            print "Error message: ", error
            print " "

    print "Done"



if __name__ == '__main__':
    print "Country code:", i
    main(path, i, urb_cell, rur_cell, pop_array)
