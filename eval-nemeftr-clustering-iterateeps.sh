#!/usr/bin/env bash
#
# Calls the reference implementation of the field type clustering of segments on similarity without ground truth
# described in our DSN 2022 paper (NEMEFTR-full mode 1). This script iterates the epsilon parameter for clustering
# for evaluation over the configured range.
# The parameters in this script are the same as used in the paper's evaluation.

input="input/maxdiff-fromOrig/*-100*.pcap"


segmenters="nemesys zeros"

# Nemesys options
# refines="original nemetyl PCA1 PCAmoco zerocharPCAmocoSF emzcPCAmocoSF"

# Zeros options
#refines="none PCA1 PCAmocoSF"

refines="PCAmocoSF nemetyl emzcPCAmocoSF"

L2PROTOS="input/awdl-* input/wlan-beacons-*"
LEPROTOS="input/awdl-* input/wlan-beacons-* input/smb* input/*/smb*"

prefix="cft"

cftnpad="245"
for f in reports/${prefix}-* ; do
  if [ -e "$f" ] ; then
    cftnext=$(expr 1 + $(ls -d reports/${prefix}-* | sed "s/^.*${prefix}-\([0-9]*\)-.*$/\1/" | sort | tail -1))
    cftnpad=$(printf "%03d" ${cftnext})
  fi
  break
done
currcomm=$(git log -1 --format="%h")
report=reports/${prefix}-${cftnpad}-clustering-${currcomm}
mkdir ${report}


for seg in ${segmenters} ; do
  for ref in ${refines} ; do
      if [[ ${seg} == "zeros" ]] && [[ ! ${ref} =~ ^(none|PCA1|PCAmocoSF)$ ]] ; then
          echo ${ref} not suited for zeros segmenter. Ignoring.
          continue
      fi

      pids=()
      for fn in ${input} ; do
          optargs="-r"
          for proto in ${L2PROTOS} ; do
            if [[ "${fn}" == ${proto} ]] ; then
              # replace
              optargs="-l 2"
            fi
          done
          for proto in ${LEPROTOS} ; do
            if [[ "${fn}" == $proto ]] ; then
              # append
              optargs="${optargs} -e"  # -e: little endian
            fi
          done
          # fixed sigma 1.2 (nemeftr-paper: "constant σ of 1.2")
          python src/nemeftr_cluster-segments_iterate-eps.py -t ${seg} -s 1.2 -p ${optargs} -f ${ref} ${fn} >> "${report}/$(basename -s .pcap ${fn}).log" &
          pids+=( $! )
      done

      for pid in "${pids[@]}"; do
              printf 'Waiting for %d...' "$pid"
              wait $pid
              echo 'done.'
      done

      mkdir ${report}-${seg}-${ref}
#      mv reports/*.pdf ${report}-${seg}-${ref}/
      for fn in ${input};
      do
          bn=$(basename -s .pcap ${fn})
          mv reports/${bn}* ${report}-${seg}-${ref}/
      done
  done
done

python src/transform_cluster-statistics.py
mv reports/*.csv ${report}/

spd-say "Bin fertig!"
