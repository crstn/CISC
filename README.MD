Modify run.sh to tell the script how many times to run the model, then execute via
```
  sh run.sh
```
on terminal.

Alternatively, to generate a benchmark profile for a single model run, execute
```
  python profile.py 1 1 1 304
```
This will generate a detailed profile of all function calls made during a single model run for a single country (304) on a single processor.
