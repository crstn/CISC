import os, random, datetime, logging
import numpy as np
import pandas as pd

#=================================================================================================
# Define population growth factors
#=================================================================================================

popGrowShrinkFactor = 50.0 # using a static value for now 


#=================================================================================================
# Define functions for population growth
#=================================================================================================

def Start_Urban_Choice(indexed_WUP, Urb2000, countryCode):
    x = int(countryCode)
    UrbanChange =  ((indexed_WUP.loc[x, "2010"]*1000) - Urb2000)
    return UrbanChange

def Start_Rural_Choice(indexed_WUP, indexed_WTP, Rur2000, countryCode):
    rural_pop_2010 = ((indexed_WTP.loc[int(countryCode), "2010"]) - (indexed_WUP.loc[int(countryCode), "2010"]))*1000
    RuralChange = (rural_pop_2010 - Rur2000)
    return RuralChange

def Grow_Urban_Start(pop_array, urban_change_cell):
    pop_array[urban_change_cell] += 1000
    return pop_array

def Shrink_Urban_Start(pop_array, urban_change_cell):
    pop_arrayg[urban_change_cell] -= 1000
    return pop_array

def Grow_Rural_Start(pop_array, rural_change_cell):
    pop_array[rural_change_cell] += 1000
    return pop_array

def Shrink_Rural_Start(pop_array, rural_change_cell):
    pop_array[rural_change_cell] -= 1000
    return pop_array

def Urban_Change_Choice(indexed_WUP, countryCode, year):
    x = int(countryCode)
    previous_year = (year - 5)
    UrbanChange =  ((indexed_WUP.loc[x, str(year)]) - (indexed_WUP.loc[x, str(previous_year)]))*1000
    return UrbanChange

def Rural_Change_Choice(indexed_WUP, indexed_WTP, countryCode, year):
    previous_year = (year - 5)
    x = int(countryCode)
    total_pop_change = ((indexed_WTP.loc[x, str(year)]) - (indexed_WTP.loc[x, str(previous_year)]))*1000
    urban_pop_change = ((indexed_WUP.loc[x, str(year)]) - (indexed_WUP.loc[x, str(previous_year)]))*1000
    RuralChange = total_pop_change - urban_pop_change
    return RuralChange


def Select_Random_Cell(CellList):
    return CellList[random.randint(0, (len(CellList) - 1))]

def Export_Array_for_Year(popArray, year, run, countryCode, saveLoc):
    os.chdir(saveLoc)
    np.save("Pop_"+countryCode+"_"+str(year)+"_"+str(run), popArray)

def main(path, countryCode, urb_cell, rur_cell, pop_array, indexed_WUP, indexed_WTP, RUNS):

    #=================================================================================================
    # Running the  program
    #=================================================================================================

    logging.info( "Beginning Analysis for country " + str(countryCode))


    logging.info( "Aggregating urban and rural populations for 2000.")
    urban_pop_list00 = []
    for i in urb_cell:
        urban_pop_list00.append(pop_array[i])
    urban_pop_00 = int(sum(urban_pop_list00))
    logging.info( "Urban Population 2000: " + str(urban_pop_00))
    total_pop_00 = int(np.sum(pop_array))
    rural_pop_00 = total_pop_00 - urban_pop_00
    logging.info( "Rural Population 2000: " + str(rural_pop_00))
    logging.info( "2000 populations established.")

    #=================================================================================================
    #Grow the population
    #=================================================================================================

    logging.info( "Growing Population")
    for runCount in range(RUNS):
        logging.info( "Run no. " + str(runCount))

        start_urban_change = Start_Urban_Choice(indexed_WUP, urban_pop_00, countryCode)
        logging.info( str(start_urban_change) )

        if start_urban_change > 0:
            counter = 0
            while counter < start_urban_change:
                urban_change_cell = Select_Random_Cell(urb_cell)
                # grow urban population
                pop_array[urban_change_cell] += popGrowShrinkFactor
                counter += popGrowShrinkFactor

        if start_urban_change < 0:
            # logging.info( str(start_urban_change) )
            counter = 0
            logging.info( "Shrinking urban population to reconcile discrepancy")
            while counter < start_urban_change:
                urban_change_cell = Select_Random_Cell(urb_cell)
                # we are skipping over cells that don't have at least the number of our current factor in it
                if pop_array[urban_change_cell] >= popGrowShrinkFactor:
                    # shrinking urban population
                    pop_array[urban_change_cell] -= popGrowShrinkFactor
                    # check if we have negative population now:
                    counter += popGrowShrinkFactor
        #        else:
        #            continue
        # if start_urban_change == 0:
        #     pass

        start_rural_change = Start_Rural_Choice(indexed_WUP, indexed_WTP, rural_pop_00, countryCode)
        logging.info( str(start_rural_change) )
        if start_rural_change > 0:
            counter = 0;
            logging.info( "Growing rural population to reconcile discrepancy" )
            while counter < start_rural_change:
                rural_change_cell = Select_Random_Cell(rur_cell)

                # grow rural population
                pop_array[rural_change_cell] += popGrowShrinkFactor
                counter += popGrowShrinkFactor

        if start_rural_change < 0:
            logging.info( "Shrinking rural population to reconcile discrepancy" )
            counter = 0
            while counter < start_rural_change:
                rural_change_cell = Select_Random_Cell(rur_cell)
                # we are skipping over cells that don't have at least the number of our current factor in it
                if pop_array[rural_change_cell] >= popGrowShrinkFactor:
                    # shrink rural population
                    pop_array[rural_change_cell] -= popGrowShrinkFactor
                    counter += popGrowShrinkFactor
                # else:
                #     continue
        if start_rural_change == 0:
            pass
        logging.info( "Done reconciling pop_array - UN_DESA discrepancy" )


        year = 2015
        logging.info( str(year) )
        while year <= 2050:
            urban_change =  Urban_Change_Choice(indexed_WUP, countryCode, year)
            logging.info( "For country code " + str(countryCode) + " in year " + str(year) + " the urban population change is " + str(urban_change) )

            if urban_change > 0:
                counter = 0
                while counter < urban_change:
                    urban_change_cell = Select_Random_Cell(urb_cell)

                    # grow urban population
                    pop_array[urban_change_cell] += popGrowShrinkFactor
                    counter += popGrowShrinkFactor

            if urban_change < 0:
                counter = 0
                while counter < urban_change:
                    urban_change_cell = Select_Random_Cell(urb_cell)
                    # we are skipping over cells that don't have at least the number of our current factor in it
                    if (pop_array[urban_change_cell] >= popGrowShrinkFactor):
                        # shrinking urban population
                        pop_array[urban_change_cell] -= popGrowShrinkFactor
                        counter += popGrowShrinkFactor
                    # else:
                    #     continue

            if urban_change == 0:
                pass


            rural_change = Rural_Change_Choice(indexed_WUP, indexed_WTP, countryCode, year)
            logging.info( "For country code " + str(countryCode) + " in year " + str(year) + " the rural population change is " + str(rural_change) )

            if rural_change > 0:
                counter = 0
                while counter < rural_change:
                    rural_change_cell = Select_Random_Cell(rur_cell)

                    # grow rural population
                    pop_array[rural_change_cell] += popGrowShrinkFactor
                    counter += popGrowShrinkFactor

            if rural_change < 0:
                counter = 0
                while counter < rural_change:
                    # logging.info( "Rural growth counter: " + str(counter))
                    rural_change_cell = Select_Random_Cell(rur_cell)
                    # we are skipping over cells that don't have at least the number of our current factor in it
                    if (pop_array[rural_change_cell] >= popGrowShrinkFactor):
                        # shrink rural population
                        pop_array[rural_change_cell] -= popGrowShrinkFactor
                        counter += popGrowShrinkFactor
                    # else:
                    #     continue
            if rural_change == 0 :
                pass
            # save the output:
            saveLoc = path+"/Output_"+countryCode
            Export_Array_for_Year(pop_array, year, runCount, countryCode, saveLoc)
           
            logging.info( "Done with growth for year " + str(year) + ", run " + str(runCount) )
            year += 5

    logging.info( "Done" )



if __name__ == '__main__':
    logging.basicConfig(filename='output.log',
                        filemode='w',
                        level=logging.INFO,
                        format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s')

    logging.info( "Country code:" + str(i) )
    main(path, i, urb_cell, rur_cell, pop_array)
