import matplotlib, os, numpy as np
from matplotlib import pyplot
from matplotlib.backends.backend_pdf import PdfPages
from osgeo import gdal, osr

blocksize = 120 # 12 = 1 degree
maxLat = 84.0 # northern boundary of the TIFFs
resolution = 0.00833333 # size of a cell in the TIFF, in degrees

os.chdir(os.path.expanduser('~') + '/Dropbox/CISC Data/IndividualCountries/Projections/Global');

def openTIFFasNParray(file):
    src = gdal.Open(file, gdal.GA_Update)
    band = src.GetRasterBand(1)
    return np.array(band.ReadAsArray())

def rowToLat(row):
    global maxLat, resolution
    return maxLat - (row * resolution)

# these were generated by GenerateNormalization.py
# total area and land area per 1 degree band, starting at 84N, in sq km:
totalareas = [508297.764629487, 586047.126878042, 663603.953480684, 740942.870226812, 818038.613418311, 894866.042323521, 971400.151450679, 1047616.08262695, 1123489.13686176, 1198994.78598627, 1274108.68404536, 1348806.67843324, 1423064.8207575, 1496859.37741718, 1570166.83988446, 1642963.93467768, 1715227.63301347, 1786935.16013278, 1858064.00428516, 1928591.9253697, 1998496.96322251, 2067757.44554219, 2136351.99545481, 2204259.53870571, 2271459.31048204, 2337930.86185854, 2403654.06586724, 2468609.12319104, 2532776.56747822, 2596137.27028528, 2658672.44564067, 2720363.65424402, 2781192.80729349, 2841142.16995436, 2900194.36446909, 2958332.37291749, 3015539.53963304, 3071799.57328422, 3127096.54862629, 3181414.90793687, 3234739.46214133, 3287055.39163961, 3338348.24684511, 3388603.94844809, 3437808.78741165, 3485949.42471844, 3533012.89087423, 3578986.58518589, 3623858.27482504, 3667616.09369082, 3710248.5410873, 3751744.48022575, 3792093.13657015, 3831284.09603751, 3869307.30306749, 3906153.05857563, 3941812.01780585, 3976275.18809419, 4009533.92655926, 4041579.93773326, 4072405.27114659, 4102002.31888074, 4130363.81310283, 4157482.82359244, 4183352.75527789, 4207967.34579041, 4231320.66305024, 4253407.10289706, 4274221.38677323, 4293758.55947461, 4312013.98697576, 4328983.35434185, 4344662.6637355, 4359048.23252776, 4372136.69152173, 4383924.983296, 4394410.36067587, 4403590.38533814, 4411462.92655589, 4418026.16008838, 4423278.56722088, 4427218.93395818, 4429846.35037545, 4431160.210129, 4431160.210129, 4429846.35037545, 4427218.93395818, 4423278.56722088, 4418026.16008838, 4411462.92655589, 4403590.38533814, 4394410.36067587, 4383924.983296, 4372136.69152173, 4359048.23252776, 4344662.6637355, 4328983.35434185, 4312013.98697576, 4293758.55947461, 4274221.38677323, 4253407.10289706, 4231320.66305024, 4207967.34579041, 4183352.75527789, 4157482.82359244, 4130363.81310283, 4102002.31888074, 4072405.27114659, 4041579.93773326, 4009533.92655926, 3976275.18809419, 3941812.01780585, 3906153.05857563, 3869307.30306749, 3831284.09603751, 3792093.13657015, 3751744.48022575, 3710248.5410873, 3667616.09369082, 3623858.27482504, 3578986.58518589, 3533012.89087423, 3485949.42471844, 3437808.78741165, 3388603.94844809, 3338348.24684511, 3287055.39163961, 3234739.46214133, 3181414.90793687, 3127096.54862629, 3071799.57328422, 3015539.53963304, 2958332.37291749, 2900194.36446909, 2841142.16995436, 2781192.80729349, 2720363.65424402, 2658672.44564067, 2596137.27028528, 2532776.56747822, 2468609.12319104]

landareas = [617.163708106176, 29473.0271612372, 46075.169699729, 49461.8617097094, 69540.9644289871, 71897.2757482392, 50585.7190621218, 60655.9326059245, 74228.9810797237, 38433.3750318941, 89819.3377532452, 129837.476582303, 137728.166520426, 205102.218946564, 309678.484990005, 322205.713527686, 382496.316699499, 250324.716458408, 448797.597066268, 643763.093509109, 368964.090303949, 511635.216874408, 548313.585816696, 553009.250946551, 528691.866010831, 394079.140800285, 401246.252779636, 340139.741102038, 402732.933074952, 451556.510802938, 610677.833757735, 710229.481480406, 780648.676694237, 216455.29380865, 1815506.24749525, 1282273.27048642, 1401855.96405389, 1294391.34916003, 1343349.40138334, 1256877.61608807, 1312164.59985771, 1177571.64218723, 634246.104131145, 1526038.60431272, 1534197.89859695, 1456894.05836676, 1201567.11902483, 1562722.71998493, 1558101.49628892, 1455238.84566661, 1417867.84182047, 1515542.74148025, 1519831.16862104, 1681740.15666678, 1743232.41431521, 1409784.27394607, 1352811.59719565, 1302791.46700836, 1325285.87103808, 1249783.17942245, 1301026.69024615, 1264685.52855074, 1198923.34222345, 1194757.23518356, 1185182.96546885, 1197715.42135113, 1140103.34346708, 1064209.44954229, 979230.023983409, 953049.971278701, 889143.178731181, 813800.551844844, 802634.362647154, 915748.754704213, 987174.623238131, 1042094.29832602, 1018648.24436273, 971551.669825232, 988536.35240679, 941215.179483993, 895617.841231708, 925085.199735872, 958890.464561541, 984532.163060272, 961233.826608034, 1032704.94451857, 1089807.65895712, 1157921.3715144, 1096982.95761864, 1074285.8270762, 1066750.98585948, 1062647.11418179, 993182.197675773, 822110.138697233, 837431.627127811, 827331.194764915, 857142.16675275, 888460.861959951, 944538.304303268, 995447.528960615, 1009280.61354538, 1062339.4485832, 1003873.64025309, 1015445.90135847, 1081455.01013984, 1073000.03484616, 1067487.74122388, 1022699.48110489, 998118.001454197, 907919.69894393, 897263.293905722, 885520.196028972, 821574.053245395, 783500.951737009, 757075.126664176, 395821.345687054, 556950.27242771, 516136.94032083, 364071.594508312, 281956.274137413, 273718.782505557, 259486.875440992, 195763.815421292, 136201.714102211, 123768.081514234, 136210.25549882, 130242.973819544, 107747.180427682, 102333.032056737, 94086.5834539324, 78632.2486041846, 72976.8181160366, 68692.6620614621, 64650.6430837979, 45897.3226264987, 38969.8898944794, 32681.2045499876, 29480.8278026611, 33224.6974156389, 7189.58581935647, 0.1]

pop2100 = openTIFFasNParray('pop-2100.tiff')
pop2010 = openTIFFasNParray('pop-2010.tiff')

matplotlib.style.use('fivethirtyeight')

# make the font smaller, so that the legends don't take up too much space
matplotlib.rcParams.update({'font.size': 8})

# replace NAN with 0
pop2100[pop2100 < 0 ] = 0
pop2010[pop2010 < 0 ] = 0

rows = pop2100.shape[0]
cols = pop2100.shape[1]

sumsPerBlock = []
blocks = range(0, rows, blocksize)
for row in blocks:
    sm2100 = np.sum(pop2100[row:row+blocksize,])
    sm2010 = np.sum(pop2010[row:row+blocksize,])
    diff = (sm2100 - sm2010) / 1000000
    sumsPerBlock.append(diff)

# now calculate the latitutde for every row that we have a number for and use those on the y axis:
latblocks = []
for row in blocks:
    latblocks.append(rowToLat(row))

axes = pyplot.gca()
axes.set_ylim([-90,90])
axes.set_xlim([-50,350])
axes.set_axis_bgcolor('white')

pyplot.plot(sumsPerBlock, latblocks, label="Population difference", linewidth = 1.0)

pyplot.xlabel('Difference between 2010 and 2100 in million people')
pyplot.ylabel('Degrees latitude')
pyplot.title('Population difference between 2010 and 2100 per 1 degree latitude band')
pyplot.legend(loc='upper right')
pyplot.savefig(os.path.expanduser('~') + '/Dropbox/Code/CISC/plotting/PopDiffByLat.pdf', bbox_inches='tight', dpi=300, facecolor='white', transparent=True)

# clear this figure, start a new one:
pyplot.clf()