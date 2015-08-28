modelruns=1 # just set this parameter, everything else is automatic

processors=`getconf _NPROCESSORS_ONLN`  # automatically figures out the number of processor cores
for i in $(seq 1 $processors);
  do python ScriptInProgress.py $processors $i $modelruns > run-$processors-$i.log &
done
