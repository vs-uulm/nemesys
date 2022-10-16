#!/usr/bin/env bash
#
# NEMEFTR: Optimal-segmentation baseline

input="input/maxdiff-fromOrig/*-100*.pcap"


L2PROTOS="input/awdl-* input/au-* input/wlan-beacons-*"

prefix="tft"

numpad="200"
for f in reports/${prefix}-* ; do
  if [ -e "$f" ] ; then
    numnext=$(expr 1 + $(ls -d reports/${prefix}-* | sed "s/^.*${prefix}-\([0-9]*\)-.*$/\1/" | sort | tail -1))
    numpad=$(printf "%03d" ${numnext})
  fi
  break
done
currcomm=$(git log -1 --format="%h")
report=reports/${prefix}-${numpad}-clustering-${currcomm}
mkdir ${report}


for fn in ${input} ; do
  # relative to IP layer
  optargs="-r"
  for proto in ${L2PROTOS} ; do
    if [[ "${fn}" == ${proto} ]] ; then
      # replace
      optargs="-l 2"
    fi
  done

  python src/nemeftr_cluster-true-fields_iterate-eps.py ${optargs} ${fn}
done




mv reports/*.csv ${report}/
mv reports/*.pdf ${report}/






spd-say "Bin fertig!"
