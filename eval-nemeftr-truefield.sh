#!/usr/bin/env bash
#
# Calls the reference implementation of the field type clustering of segments on similarity with ground truth
# described in our DSN 2022 paper (NEMEFTR: Optimal-segmentation baseline). The parameters in this script are
# the same as used in the paper's evaluation.


#input=input/*-100*.pcap
#input=input/*-1000.pcap
input="input/maxdiff-fromOrig/*-100*.pcap"
#input="input/maxdiff-fromOrig/ntp_SMIA-20111010_maxdiff-100.pcap"


L1PROTOS="input/ari_*"
L2PROTOS="input/awdl-* input/au-* input/wlan-beacons-*"

prefix="tft"

numpad="350"
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
  for proto in ${L1PROTOS} ; do
    if [[ "${fn}" == ${proto} ]] ; then
      # replace
      optargs="-l 1"
    fi
  done

  # add -p to write plots  ###  add -p for plots
  python src/nemeftr_cluster-true-fields.py ${optargs} ${fn}
done



for fn in ${input} ; do
    bn=$(basename -- ${fn})
    strippedname="${bn%.*}"
    mv reports/${strippedname}/ ${report}/
done
mv reports/*.csv ${report}/
mv reports/*.pdf ${report}/






spd-say "Bin fertig!"
