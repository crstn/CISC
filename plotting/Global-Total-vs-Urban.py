import matplotlib, os
from matplotlib import pyplot
from matplotlib.backends.backend_pdf import PdfPages

years = [2015, 2020, 2025, 2030, 2035, 2040, 2045, 2050, 2055, 2060, 2065, 2070, 2075, 2080, 2085, 2090, 2095, 2100]
urban = [3954989974, 4352048614, 4752825054, 5169602329, 5576636814, 5976875806, 6373371599, 6745545902, 7083695933, 7413818351, 7730079370, 8005108796, 8270329657, 8514498334, 8748916633, 8967800809, 9173638206, 9357913577]
total = [7349472000, 7758157000, 8141661000, 8500766000, 8838908000, 9157234000, 9453892000, 9725148000, 9968809000, 10184290000, 10375719000, 10547989000, 10701653000, 10836635000, 10953525000, 11055270000, 11142461000, 11213317000]

def tobillions(x):
    return x/1000000000.0

matplotlib.style.use('fivethirtyeight')

pyplot.plot(years, map(tobillions, total), label="Global total pop (DESA Medium)", linewidth = 1.0)
pyplot.plot(years, map(tobillions, urban), label="Global urban pop (Peter Mean)", linewidth = 1.0)

pyplot.xlabel('Year')
pyplot.ylabel('People')
pyplot.title('Global total vs. urban population')
pyplot.legend(loc='lower right')
pyplot.savefig(os.path.expanduser('~') + '/Dropbox/Code/CISC/plotting/GlobalVSTotal.pdf', bbox_inches='tight', dpi=300)
