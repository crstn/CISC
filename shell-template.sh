# replace the SSP, the urban/rural layer (GRUMP or GlobCover), and the folder to cd into before rm
# also move around the "&" to set how many reassmbly tasks are run in parallel
python ParallelProjection.py SSP5 GlobCover; python ReassambleCountries.py 2010 SSP5 GlobCover & python ReassambleCountries.py 2020 SSP5 GlobCover; python ReassambleCountries.py 2030 SSP5 GlobCover & python ReassambleCountries.py 2040 SSP5 GlobCover; python ReassambleCountries.py 2050 SSP5 GlobCover & python ReassambleCountries.py 2060 SSP5 GlobCover; python ReassambleCountries.py 2070 SSP5 GlobCover & python ReassambleCountries.py 2080 SSP5 GlobCover; python ReassambleCountries.py 2090 SSP5 GlobCover & python ReassambleCountries.py 2100 SSP5 GlobCover; cd /Volumes/Solid\ Guy/Sandbox/GlobCover/SSP5; rm *.npy; cd ~/Dropbox/Code/CISC/
