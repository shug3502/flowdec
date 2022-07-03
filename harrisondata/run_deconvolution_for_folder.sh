#!/bin/bash
FILES=/harrisondata/*.ome.tif
for f in $FILES
do
  echo "Deconvolving file $f ..."  
  python /harrisondata/run_deconvolution.py $f
#  python /harrisondata/run_deconvolution.py $f --bead '/harrisondata/OS_LLSM_210620_MC194_HURP_GFP_HaLo_CENPA_PSF_bead_image_488_green_sigma.csv'
done
