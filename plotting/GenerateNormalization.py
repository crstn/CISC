# This module generates 2 sets of normalization values for our
# latitude population plots: One contains the area per one degree
# latitude stripe covered by our study ara, the other one contains
# the LAND area in the same stripe. The two sets of values are
# saved as np arrays and used by LineChartByLatitude.py. Values
# in sq km.

# Need a running PostGIS DB (called 'cisc' in this case) on localhost,
# with the Natural Earth Amin 0 data (countries without lakes, see # http://www.naturalearthdata.com/http//www.naturalearthdata.com/download/10m/cultural/ne_10m_admin_0_countries_lakes.zip) loaded. Before loading the shp into PostGIS, clip
# to the bouding box of our study area, otherwise PostGIS will trip because the
# bounding box extends to the poles. Clip
# using ogr2ogr like so:

# ogr2ogr -clipsrc -180.0 -57.0 180.0 85.0 ne_10m_admin_0_countries_lakes_clipped.shp ne_10m_admin_0_countries_lakes.shp

# The table containing the clipped data is called "land" here.

import psycopg2

conn = psycopg2.connect("dbname='cisc' host='localhost'")

cur = conn.cursor()
lat = 84.0

totalareas = []
landareas = []

while lat > -57:
    # Get total surface area (land+water) between those two latitudes
    cur.execute("""select sum(st_area(ST_GeographyFromText('POLYGON((1.0 %s, 2.0 %s, 2.0 %s, 1.0 %s, 1.0 %s))'))) / 1000000 * 360;""", (lat-1.0, lat-1.0, lat, lat, lat-1.0))
    totalareas.append(cur.fetchone()[0])

    # Get total land area between those two latitudes by intersecting the latitude strip with the
    # countries from Natural Earth.
    # The 1 meter buffer is a dirty hack to fix a little self-intersection polygon in the Natural Earth
    # data that I couldn't fix any other way. Makes the whole thing a bit slow and minimally exagerrates
    # the land area, but that shouldn't make any noticable difference.
    cur.execute("""select sum(st_area(st_intersection(ST_GeographyFromText('POLYGON((-180.0 %s, 180.0 %s, 180.0 %s, -180.0 %s, -180.0 %s))'), st_buffer(geom::geography, 1)))) / 1000000 from land;""", (lat-1.0, lat-1.0, lat, lat, lat-1.0))
    landareas.append(cur.fetchone()[0])

    print lat
    lat = lat-1.0;

print len(totalareas)
print len(landareas)

print totalareas
print landareas
