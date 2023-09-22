#!/usr/bin/env bash

#input=input/*-100.pcap
#input=input/*-1000.pcap
#input="input/*-100.pcap input/*-1000.pcap"
#input=input/maxdiff-filtered/*-1000.pcap
input="input/maxdiff-fromOrig/*-100*.pcap"
#input="input/maxdiff-fromOrig/ntp_SMIA-20111010_maxdiff-100.pcap"


L2PROTOS="input/awdl-* input/au-*"

prefix="netzob-format"

numpad="206"
for f in reports/${prefix}-* ; do
  if [ -e "$f" ] ; then
    numnext=$(expr 1 + $(ls -d reports/${prefix}-* | sed "s/^.*${prefix}-\([0-9]*\)-.*$/\1/" | sort | tail -1))
    numpad=$(printf "%03d" ${numnext})
  fi
  break
done
currcomm=$(git log -1 --format="%h")
report=reports/${prefix}-${numpad}-fms-${currcomm}
mkdir ${report}

smin=57

pids=()
for fn in ${input} ; do
  # relative to IP layer
  optargs="-r"  # --smax 80
    for proto in ${L2PROTOS} ; do
    if [[ "${fn}" == ${proto} ]] ; then
      # replace at layer 2 absolute
      optargs="-l 2"
      # optargs="-l 2 --smax 75"
    fi
  done
  for proto in ${L1PROTOS} ; do
    if [[ "${fn}" == ${proto} ]] ; then
      # replace
      optargs="-l1"
    fi
  done
#   python src/netzob_fms.py --smin ${smin} ${optargs} ${fn} > "${report}/$(basename -s .pcap ${fn}).log" &
  python src/netzob_fms.py ${optargs} ${fn} >> "${report}/$(basename -s .pcap ${fn}).log" &
  pids+=( $! )
done

for pid in "${pids[@]}"; do
        printf 'Waiting for %d...' "$pid"
        wait $pid
        echo 'done.'
done

mv reports/*clByAlign* ${report}/
python reports/combine-nemesys-fms.py ${report}/


spd-say "Bin fertig!"
