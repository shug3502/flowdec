#!/bin/bash
FILES=/harrisondata/*.ome.tif
for f in $FILES
do
  echo "Deconvolving file $f ..."  
  python /harrisondata/run_deconvolution.py $f
done
