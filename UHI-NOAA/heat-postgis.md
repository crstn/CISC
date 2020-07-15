# Urban Effect Analysis in PostGIS

## Load stations into PostGIS

Set up DB:

```sql
-- DROP DATABASE heat;

CREATE DATABASE heat
    WITH 
    OWNER = carsten
    ENCODING = 'UTF8'
    LC_COLLATE = 'C'
    LC_CTYPE = 'C'
    TABLESPACE = pg_default
    CONNECTION LIMIT = -1;

-- Connect to DB heat, then:

CREATE EXTENSION postgis;
```

Stations were loaded into QGIS, filtered to stations in the tropic zone (see above; that leaves about 16,000 stations) and then imported to PostGIS using the database manager into a table called ```stations``` (with a spatial index), at the same time projecting the dataset to Mollweide ([ESPG:54009](https://epsg.io/54009)) to get an equal area projection.

🔥 insert screenshot!

To remove the columns that we don't need:

```sql
ALTER TABLE stations
	DROP COLUMN field_1,
	DROP COLUMN popdens2015,
	DROP COLUMN id,
	DROP COLUMN "NN",
	DROP COLUMN "NN_dist",
	DROP COLUMN "NN_elev",
	DROP COLUMN nn_elev_diff,
	DROP COLUMN "NN_lat",
	DROP COLUMN "NN_popdens2015";
```

## Identify area (range in lat N/S) that we want to cover

To keep things simple, we'll focus on the Tropical Zone: 23.5N to 23.5S (that's also what the special issue is about, and what we said in our abstract).

## Make 50km resolution grids in [MMQGIS](https://plugins.qgis.org/plugins/mmqgis/)

In QGIS, go to *MMGIS > Create > Create Grid Layer*. In the plugin window, select *Rectangles* as *Geometry Type*, 50000 as *X Spacing* and *Y Spacing*, and set *Extent* to *Layer Extent* with *Layer* set to stations (loaded from PostGIS). Save to a file and then import to PostGIS as ```grid50``` (again with spatial index).

## Join stations to grid cells

```sql
ALTER TABLE stations
ADD COLUMN g50 integer;

-- spatial join

UPDATE stations
SET g50 = grid50.id
FROM grid50
WHERE ST_INTERSECTS(stations.geom, grid50.geom);

--- count stations per cell 

ALTER TABLE grid50
ADD COLUMN stations integer;

UPDATE grid50
SET stations = (SELECT COUNT(g50)
FROM stations
WHERE stations.g50 = grid50.id);

-- delete the grid cells with no or just 1 station
-- before: 82244 grid cells

DELETE FROM grid50
WHERE stations < 2;

-- after: 2475

```

## For each grid cell, identify the _median_ population density for each of the four years of population data

See [this tutorial](https://www.skillslogic.com/blog/dashboards-data-warehousing/calculating-medians-in-postgresql-with-percentile_cont) for explanations:

```sql
-- 1975
ALTER TABLE grid50
ADD COLUMN median1975 double precision;

UPDATE grid50
SET median1975 = (SELECT percentile_cont(0.5) WITHIN GROUP ( ORDER BY pop1975 )
	FROM
		stations
	WHERE stations.g50 = grid50.id);
	
```

Any grid cells where the median is 0 are useless for our analysis, because they only have stations within cells that have no population. Let's get rid of those:

```sql

-- before: 2475
DELETE FROM grid50 
WHERE median1975 = 0;
-- after: 1339
```


🤔 Medians are very low when there are a bunch of stations with no population in the grid cell → **use mean instead?** (problem with that is that it doesn't give us a 50/50 split of the stations within the cell, so let's keep it like that for now.)

🔥 Repeat for other 3 years

## Calculate the mean population for each group

```sql
-- avg pop of the upper 50% percentile, i.e. above the median

ALTER TABLE grid50
ADD COLUMN avg_hi1975 double precision;

UPDATE grid50
SET avg_hi1975 = (SELECT AVG(stations.pop1975)
FROM stations
WHERE stations.g50 = grid50.id
AND stations.pop1975 >= grid50.median1975
GROUP BY stations.g50);

--... and below the median

ALTER TABLE grid50
ADD COLUMN avg_lo1975 double precision;

UPDATE grid50
SET avg_lo1975 = (SELECT AVG(stations.pop1975)
FROM stations
WHERE stations.g50 = grid50.id
AND stations.pop1975 <= grid50.median1975
GROUP BY stations.g50);
```

🔥 Repeat for other 3 years (everything from here...)!

1. Load the temperature data for the 5 years "around" the population snapshot, keeping only TMIN

Download the data for [1973](https://www1.ncdc.noaa.gov/pub/data/ghcn/daily/by_year/1973.csv.gz), [1974](https://www1.ncdc.noaa.gov/pub/data/ghcn/daily/by_year/1974.csv.gz), [1975](https://www1.ncdc.noaa.gov/pub/data/ghcn/daily/by_year/1975.csv.gz), [1976](https://www1.ncdc.noaa.gov/pub/data/ghcn/daily/by_year/1976.csv.gz), and [1977](https://www1.ncdc.noaa.gov/pub/data/ghcn/daily/by_year/1977.csv.gz) first. Then we'll create a "staging" table where we park the imported data, before we copy over just the data that we need to the "real" table:

```sql
CREATE TABLE import
(
    station "varchar",
    dat "varchar",
    typ "varchar",
    val integer,
    a "varchar",
    b "varchar",
    c "varchar",
    d "varchar"
);

ALTER TABLE import
    OWNER to carsten;
```

Then import:

```
COPY import FROM '/Users/carsten/Dropbox/Downloads/1973.csv' DELIMITER ',' CSV;

-- only keep TMIN:
DELETE FROM import WHERE typ != 'TMIN';

-- repeat:
COPY import FROM '/Users/carsten/Dropbox/Downloads/1974.csv' 
DELIMITER ',' CSV;
DELETE FROM import WHERE typ != 'TMIN';
COPY import FROM '/Users/carsten/Dropbox/Downloads/1975.csv' 
DELIMITER ',' CSV;
DELETE FROM import WHERE typ != 'TMIN';
COPY import FROM '/Users/carsten/Dropbox/Downloads/1976.csv' 
DELIMITER ',' CSV;
DELETE FROM import WHERE typ != 'TMIN';
COPY import FROM '/Users/carsten/Dropbox/Downloads/1977.csv' 
DELIMITER ',' CSV;
DELETE FROM import WHERE typ != 'TMIN';
```

Remove the columns we don't need:

```sql
ALTER TABLE import
DROP COLUMN typ,
DROP COLUMN a,
DROP COLUMN b,
DROP COLUMN c,
DROP COLUMN d;
```

Turn dates into actual dates and values into actual degree Celsius:

```sql
-- add new columns
ALTER TABLE import
	ADD COLUMN date date,
	ADD COLUMN min_temp double precision;

-- populate
UPDATE import
	SET date = TO_DATE(dat, 'YYYYMMDD'),
	min_temp = (val/10.0);

-- remove old columns
ALTER TABLE import
	DROP COLUMN val,
	DROP COLUMN dat;

```

 
### For each day (1.1.–31.12.), calculate the mean MIN temperature across the five years for both groups of stations

To make this a bit easier, we'll first create a new column that holds the information whether that station is in the high population density or low population density group for the given year:

```sql
-- add column
ALTER TABLE stations
	ADD COLUMN dens1975 varchar;
	
-- populate
UPDATE stations
SET dens1975 = (SELECT 
				CASE WHEN pop1975 >= median1975 THEN 'hi'
                ELSE 'lo'
                END
FROM grid50
WHERE stations.g50 = grid50.id);


```

Then we'll calculate the averages by lo/hi group for each given cell and day/month across all 5 years and write that to a new table called ```averages```:

```sql
SELECT stations.g50, 
         stations.dens1975,
		 EXTRACT(DAY FROM import.date) AS day,
		 EXTRACT(MONTH FROM import.date) AS month,
		 1975 as period,
		 AVG(import.min_temp) AS avg_temp
INTO averages
FROM import, stations
WHERE import.station = stations.station
AND dens1975 IS NOT NULL
GROUP BY stations.g50, 
         stations.dens1975,
		 EXTRACT(DAY FROM import.date),
		 EXTRACT(MONTH FROM import.date);
```

The column with the density for the current cell is now called ```dens1975```, let's rename it – it is going to hold the appropriate density values for the other periods later, so the name will be confusing:

```sql
ALTER TABLE averages 
RENAME dens1975 TO pop_dens;
```

# Repeat for 1990

First, delete the ```grid50``` table: 

```sql
DROP TABLE grid50;
```

... and then manually import it from QGIS again.

Now just repeat all the steps as before, but for 1990. Before we start, make sure to download the temperature data for
Download the data for [1988](https://www1.ncdc.noaa.gov/pub/data/ghcn/daily/by_year/1988.csv.gz), [1989](https://www1.ncdc.noaa.gov/pub/data/ghcn/daily/by_year/1989.csv.gz), [1990](https://www1.ncdc.noaa.gov/pub/data/ghcn/daily/by_year/1990.csv.gz), [1991](https://www1.ncdc.noaa.gov/pub/data/ghcn/daily/by_year/1991.csv.gz), and [1992](https://www1.ncdc.noaa.gov/pub/data/ghcn/daily/by_year/1992.csv.gz) first.

The run this whole thing to prepare the data for the period around 1990:

```sql
ALTER TABLE grid50
ADD COLUMN stations integer;

UPDATE grid50
SET stations = (SELECT COUNT(g50)
FROM stations
WHERE stations.g50 = grid50.id);

DELETE FROM grid50
WHERE stations < 2;

ALTER TABLE grid50
ADD COLUMN median1990 double precision;

UPDATE grid50
SET median1990 = (SELECT percentile_cont(0.5) WITHIN GROUP ( ORDER BY pop1990 )
	FROM stations
	WHERE stations.g50 = grid50.id);

DELETE FROM grid50 
WHERE median1990 = 0;

ALTER TABLE grid50
ADD COLUMN avg_hi1990 double precision;

UPDATE grid50
SET avg_hi1990 = (SELECT AVG(stations.pop1990)
FROM stations
WHERE stations.g50 = grid50.id
AND stations.pop1990 >= grid50.median1990
GROUP BY stations.g50);

ALTER TABLE grid50
ADD COLUMN avg_lo1990 double precision;

UPDATE grid50
SET avg_lo1990 = (SELECT AVG(stations.pop1990)
FROM stations
WHERE stations.g50 = grid50.id
AND stations.pop1990 <= grid50.median1990
GROUP BY stations.g50);

DROP TABLE import;

CREATE TABLE import
(
    station "varchar",
    dat "varchar",
    typ "varchar",
    val integer,
    a "varchar",
    b "varchar",
    c "varchar",
    d "varchar"
);

ALTER TABLE import
    OWNER to carsten;

COPY import FROM '/Users/carsten/Dropbox/Downloads/1988.csv' 
DELIMITER ',' CSV;
DELETE FROM import WHERE typ != 'TMIN';
COPY import FROM '/Users/carsten/Dropbox/Downloads/1989.csv' 
DELIMITER ',' CSV;
DELETE FROM import WHERE typ != 'TMIN';
COPY import FROM '/Users/carsten/Dropbox/Downloads/1990.csv' 
DELIMITER ',' CSV;
DELETE FROM import WHERE typ != 'TMIN';
COPY import FROM '/Users/carsten/Dropbox/Downloads/1991.csv' 
DELIMITER ',' CSV;
DELETE FROM import WHERE typ != 'TMIN';
COPY import FROM '/Users/carsten/Dropbox/Downloads/1992.csv' DELIMITER ',' CSV;
DELETE FROM import WHERE typ != 'TMIN';

ALTER TABLE import
DROP COLUMN typ,
DROP COLUMN a,
DROP COLUMN b,
DROP COLUMN c,
DROP COLUMN d;

ALTER TABLE import
	ADD COLUMN date date,
	ADD COLUMN min_temp double precision;

UPDATE import
	SET date = TO_DATE(dat, 'YYYYMMDD'),
	min_temp = (val/10.0);

ALTER TABLE import
	DROP COLUMN val,
	DROP COLUMN dat;

ALTER TABLE stations
	ADD COLUMN dens1990 varchar;
	
UPDATE stations
SET dens1990 = (SELECT 
				CASE WHEN pop1990 >= median1990 THEN 'hi'
                ELSE 'lo'
                END
FROM grid50
WHERE stations.g50 = grid50.id);

INSERT INTO averages
SELECT stations.g50, 
         stations.dens1990,
		 EXTRACT(DAY FROM import.date) AS day,
		 EXTRACT(MONTH FROM import.date) AS month,
		 1990 as period,
		 AVG(import.min_temp) AS avg_temp
FROM import, stations
WHERE import.station = stations.station
AND dens1990 IS NOT NULL
GROUP BY stations.g50, 
         stations.dens1990,
		 EXTRACT(DAY FROM import.date),
		 EXTRACT(MONTH FROM import.date);

```

# Repeat for 2000

On to 2000. Before we start, make sure to download the temperature data for
Download the data for [1998](https://www1.ncdc.noaa.gov/pub/data/ghcn/daily/by_year/1998.csv.gz), [1999](https://www1.ncdc.noaa.gov/pub/data/ghcn/daily/by_year/1999.csv.gz), [2000](https://www1.ncdc.noaa.gov/pub/data/ghcn/daily/by_year/2000.csv.gz), [2001](https://www1.ncdc.noaa.gov/pub/data/ghcn/daily/by_year/2001.csv.gz), and [2002](https://www1.ncdc.noaa.gov/pub/data/ghcn/daily/by_year/2002.csv.gz) first.

```sql
DROP TABLE grid50;
```

... and then manually import it from QGIS again.

The run this whole thing to prepare the data for the period around 2000:

```sql
ALTER TABLE grid50
ADD COLUMN stations integer;

UPDATE grid50
SET stations = (SELECT COUNT(g50)
FROM stations
WHERE stations.g50 = grid50.id);

DELETE FROM grid50
WHERE stations < 2;

ALTER TABLE grid50
ADD COLUMN median2000 double precision;

UPDATE grid50
SET median2000 = (SELECT percentile_cont(0.5) WITHIN GROUP ( ORDER BY pop2000 )
	FROM stations
	WHERE stations.g50 = grid50.id);

DELETE FROM grid50 
WHERE median2000 = 0;

ALTER TABLE grid50
ADD COLUMN avg_hi2000 double precision;

UPDATE grid50
SET avg_hi2000 = (SELECT AVG(stations.pop2000)
FROM stations
WHERE stations.g50 = grid50.id
AND stations.pop2000 >= grid50.median2000
GROUP BY stations.g50);

ALTER TABLE grid50
ADD COLUMN avg_lo2000 double precision;

UPDATE grid50
SET avg_lo2000 = (SELECT AVG(stations.pop2000)
FROM stations
WHERE stations.g50 = grid50.id
AND stations.pop2000 <= grid50.median2000
GROUP BY stations.g50);

DROP TABLE import;

CREATE TABLE import
(
    station "varchar",
    dat "varchar",
    typ "varchar",
    val integer,
    a "varchar",
    b "varchar",
    c "varchar",
    d "varchar"
);

ALTER TABLE import
    OWNER to carsten;

COPY import FROM '/Users/carsten/Dropbox/Downloads/1998.csv' 
DELIMITER ',' CSV;
DELETE FROM import WHERE typ != 'TMIN';
COPY import FROM '/Users/carsten/Dropbox/Downloads/1999.csv' 
DELIMITER ',' CSV;
DELETE FROM import WHERE typ != 'TMIN';
COPY import FROM '/Users/carsten/Dropbox/Downloads/2000.csv' 
DELIMITER ',' CSV;
DELETE FROM import WHERE typ != 'TMIN';
COPY import FROM '/Users/carsten/Dropbox/Downloads/2001.csv' 
DELIMITER ',' CSV;
DELETE FROM import WHERE typ != 'TMIN';
COPY import FROM '/Users/carsten/Dropbox/Downloads/2002.csv' DELIMITER ',' CSV;
DELETE FROM import WHERE typ != 'TMIN';

ALTER TABLE import
DROP COLUMN typ,
DROP COLUMN a,
DROP COLUMN b,
DROP COLUMN c,
DROP COLUMN d;

ALTER TABLE import
	ADD COLUMN date date,
	ADD COLUMN min_temp double precision;

UPDATE import
	SET date = TO_DATE(dat, 'YYYYMMDD'),
	min_temp = (val/10.0);

ALTER TABLE import
	DROP COLUMN val,
	DROP COLUMN dat;

ALTER TABLE stations
	ADD COLUMN dens2000 varchar;
	
UPDATE stations
SET dens2000 = (SELECT 
				CASE WHEN pop2000 >= median2000 THEN 'hi'
                ELSE 'lo'
                END
FROM grid50
WHERE stations.g50 = grid50.id);

INSERT INTO averages
SELECT stations.g50, 
         stations.dens2000,
		 EXTRACT(DAY FROM import.date) AS day,
		 EXTRACT(MONTH FROM import.date) AS month,
		 2000 as period,
		 AVG(import.min_temp) AS avg_temp
FROM import, stations
WHERE import.station = stations.station
AND dens2000 IS NOT NULL
GROUP BY stations.g50, 
         stations.dens2000,
		 EXTRACT(DAY FROM import.date),
		 EXTRACT(MONTH FROM import.date);
```

# Repeat for 2015

On to 2015. Before we start, make sure to download the temperature data for
Download the data for [2013](https://www1.ncdc.noaa.gov/pub/data/ghcn/daily/by_year/2013.csv.gz), [2014](https://www1.ncdc.noaa.gov/pub/data/ghcn/daily/by_year/2014.csv.gz), [2015](https://www1.ncdc.noaa.gov/pub/data/ghcn/daily/by_year/2015.csv.gz), [2016](https://www1.ncdc.noaa.gov/pub/data/ghcn/daily/by_year/2016.csv.gz), and [2017](https://www1.ncdc.noaa.gov/pub/data/ghcn/daily/by_year/2017.csv.gz) first.

```sql
DROP TABLE grid50;
```

... and then manually import it from QGIS again.

The run this whole thing to prepare the data for the period around 2015:

```sql
ALTER TABLE grid50
ADD COLUMN stations integer;

UPDATE grid50
SET stations = (SELECT COUNT(g50)
FROM stations
WHERE stations.g50 = grid50.id);

DELETE FROM grid50
WHERE stations < 2;

ALTER TABLE grid50
ADD COLUMN median2015 double precision;

UPDATE grid50
SET median2015 = (SELECT percentile_cont(0.5) WITHIN GROUP ( ORDER BY pop2015 )
	FROM stations
	WHERE stations.g50 = grid50.id);

DELETE FROM grid50 
WHERE median2015 = 0;

ALTER TABLE grid50
ADD COLUMN avg_hi2015 double precision;

UPDATE grid50
SET avg_hi2015 = (SELECT AVG(stations.pop2015)
FROM stations
WHERE stations.g50 = grid50.id
AND stations.pop2015 >= grid50.median2015
GROUP BY stations.g50);

ALTER TABLE grid50
ADD COLUMN avg_lo2015 double precision;

UPDATE grid50
SET avg_lo2015 = (SELECT AVG(stations.pop2015)
FROM stations
WHERE stations.g50 = grid50.id
AND stations.pop2015 <= grid50.median2015
GROUP BY stations.g50);

DROP TABLE import;

CREATE TABLE import
(
    station "varchar",
    dat "varchar",
    typ "varchar",
    val integer,
    a "varchar",
    b "varchar",
    c "varchar",
    d "varchar"
);

ALTER TABLE import
    OWNER to carsten;

COPY import FROM '/Users/carsten/Dropbox/Downloads/2013.csv' 
DELIMITER ',' CSV;
DELETE FROM import WHERE typ != 'TMIN';
COPY import FROM '/Users/carsten/Dropbox/Downloads/2014.csv' 
DELIMITER ',' CSV;
DELETE FROM import WHERE typ != 'TMIN';
COPY import FROM '/Users/carsten/Dropbox/Downloads/2015.csv' 
DELIMITER ',' CSV;
DELETE FROM import WHERE typ != 'TMIN';
COPY import FROM '/Users/carsten/Dropbox/Downloads/2016.csv' 
DELIMITER ',' CSV;
DELETE FROM import WHERE typ != 'TMIN';
COPY import FROM '/Users/carsten/Dropbox/Downloads/2017.csv' DELIMITER ',' CSV;
DELETE FROM import WHERE typ != 'TMIN';

ALTER TABLE import
DROP COLUMN typ,
DROP COLUMN a,
DROP COLUMN b,
DROP COLUMN c,
DROP COLUMN d;

ALTER TABLE import
	ADD COLUMN date date,
	ADD COLUMN min_temp double precision;

UPDATE import
	SET date = TO_DATE(dat, 'YYYYMMDD'),
	min_temp = (val/10.0);

ALTER TABLE import
	DROP COLUMN val,
	DROP COLUMN dat;

ALTER TABLE stations
	ADD COLUMN dens2015 varchar;
	
UPDATE stations
SET dens2015 = (SELECT 
				CASE WHEN pop2015 >= median2015 THEN 'hi'
                ELSE 'lo'
                END
FROM grid50
WHERE stations.g50 = grid50.id);

INSERT INTO averages
SELECT stations.g50, 
         stations.dens2015,
		 EXTRACT(DAY FROM import.date) AS day,
		 EXTRACT(MONTH FROM import.date) AS month,
		 2015 as period,
		 AVG(import.min_temp) AS avg_temp
FROM import, stations
WHERE import.station = stations.station
AND dens2015 IS NOT NULL
GROUP BY stations.g50, 
         stations.dens2015,
		 EXTRACT(DAY FROM import.date),
		 EXTRACT(MONTH FROM import.date);

```

## We'll do the regression analysis in Python

... by [creating a Pandas DF directly from the database](https://towardsdatascience.com/python-and-postgresql-how-to-access-a-postgresql-database-like-a-data-scientist-b5a9c5a0ea43).

1. Calculate the difference in mean MIN temperature and in population per cell
1. Regression?
