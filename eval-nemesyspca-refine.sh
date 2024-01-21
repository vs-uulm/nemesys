#!/usr/bin/env bash

#input=input/*-100.pcap
#input=input/*-1000.pcap
#input="input/*-100.pcap input/*-1000.pcap"

# input=input/maxdiff-fromOrig/*-100.pcap
input=input/maxdiff-fromOrig/*-100*.pcap

# full
#sigmas="0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 1.1 1.2 1.3 1.4 1.5 1.6 1.7 1.8 1.9 2.0 2.1 2.2 2.3"
# default
#sigmas="1.2"
sigmas="0.9 1.2"
#sigmas="0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.2"

# full
# "original" is the WOOT refinement
#refines="base original nemetyl PCA PCA1 PCAmoco zerocharPCAmoco zerocharPCAmocoSF"
# default
#refines="original nemetyl zerocharPCAmocoSF emzcPCAmocoSF"
refines="original"

L1PROTOS="input/ari_*"
L2PROTOS="input/awdl-* input/au-* input/wlan-beacons-*"
# for reference see nemesys-le-compare.ods
LEPROTOS="input/awdl-* input/au-* input/smb* input/*/smb* input/wlan-beacons-*"

prefix="nemesys"

cftnpad="250"
for f in reports/${prefix}-* ; do
  if [ -e "$f" ] ; then
    cftnext=$(expr 1 + $(ls -d reports/${prefix}-* | sed "s/^.*${prefix}-\([0-9]*\)-.*$/\1/" | sort | tail -1))
    cftnpad=$(printf "%03d" ${cftnext})
  fi
  break
done
currcomm=$(git log -1 --format="%h")
report=reports/${prefix}-${cftnpad}-refinement-${currcomm}
mkdir ${report}

for ref in ${refines} ; do
  for sig in ${sigmas} ; do

    pids=()
    for fn in ${input} ; do
      optargs="-r" # with plots add: -p
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
      for proto in ${LEPROTOS} ; do
        if [[ "${fn}" == $proto ]] ; then
          # echo "Compare with and without little endian optimization"
          # then append
          optargs="${optargs} -e"  # -e: little endian
        fi
      done

      bn=$(basename -- ${fn})
      strippedname="${bn%.*}"
      python src/nemesys_pca-refinement.py -s ${sig} ${optargs} -f ${ref} ${fn} >> "${report}/${strippedname}.log" &
      pids+=( $! )

      # Compare with and without little endian optimization
      #python src/nemesys_pca-refinement.py -s ${sig} ${optargs} -e -f ${ref} ${fn}

      # dynamic sigma
      #python src/nemesys_pca-refinement.py -f ${ref} -r ${fn}
    done

    for pid in "${pids[@]}"; do
            printf 'Waiting for %d...' "$pid"
            wait $pid
            echo 'done.'
    done
  done
done



# copy all lines of the "ScoreStatistics.csv" files together into one per trace
mv reports/*.pdf ${report}/
for fn in ${input};
do
    bn=$(basename -- ${fn})
    strippedname="${bn%.*}"
    mv reports/${strippedname} ${report}/
done

# combine PCA refinement condition statistics into a single  "...-combined.csv"
cd ${report}/
ln -s ../combine-stats.sh .
./combine-stats.sh
cd -
# combine all ScoreStatistics into one file with a sheet for each trace in a xls file
python reports/combine-nemesys-fms.py ${report}/


echo "Report written to folder ${report}"
spd-say "Bin fertig!"
