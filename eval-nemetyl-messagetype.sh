#!/bin/bash

#input=input/*-100.pcap
#input=input/*-1000.pcap
#input="input/*-100.pcap input/*-1000.pcap"
#input="input/ntp_SMIA-20111010_deduped-1000.pcap input/smb_SMIA20111010-one_deduped-1000.pcap"

input="input/maxdiff-fromOrig/ntp_SMIA-20111010_maxdiff-100.pcap"
#input="input/maxdiff-fromOrig/*-100*.pcap"


#sigmas="0.6 0.8 1.0 1.2"
# default
#sigmas="0.9 1.2"
sigmas="1.2"

# full
#segmenters="nemesys"
segmenters="nemesys"

# full
#refines="none original base nemetyl"

# Nemesys options
# refines="original nemetyl"
# default
# refines="original nemetyl"
refines="nemetyl"


L2PROTOS="input/awdl-* input/au-* input/wlan-beacons-*"

prefix="nemetyl"

cftnpad="229"
for f in reports/${prefix}-* ; do
  if [ -e "$f" ] ; then
    cftnext=$(expr 1 + $(ls -d reports/${prefix}-* | sed "s/^.*${prefix}-\([0-9]*\)-.*$/\1/" | sort | tail -1))
    cftnpad=$(printf "%03d" ${cftnext})
  fi
  break
done
currcomm=$(git log -1 --format="%h")
report=reports/${prefix}-${cftnpad}-clustering-${currcomm}
#echo ${report}
#exit
mkdir ${report}


for fn in ${input} ; do
    optargs="-r"  # relative to IP layer
    for proto in ${L2PROTOS} ; do
      if [[ "${fn}" == ${proto} ]] ; then
        # replace
        optargs="-l 2"
      fi
    done
    echo -e "\n\ntshark: ${fn}"
#    echo "$fn -t tshark ${optargs} --with-plots"
#    exit
    python src/nemetyl_align-segments.py $fn -t tshark ${optargs} --with-plots
done
for fn in ${input} ; do
    optargs="-r"
    for proto in ${L2PROTOS} ; do
      if [[ "${fn}" == ${proto} ]] ; then
        # replace
        optargs="-l 2"
      fi
    done
    echo -e "\n\n4bytesfixed: ${fn}"
    python src/nemetyl_align-segments.py $fn -t 4bytesfixed ${optargs} --with-plots
done

for seg in ${segmenters} ; do
  for sig in ${sigmas} ; do
      for ref in ${refines} ; do
         if [[ ${seg} == "zeros" ]] && [[ ! ${ref} =~ ^(none|PCA1|PCAmocoSF)$ ]] ; then
            echo ${ref} not suited for zeros segmenter. Ignoring.
            continue
        fi
          for fn in ${input} ; do
              optargs="-r"
              for proto in ${L2PROTOS} ; do
                if [[ "${fn}" == ${proto} ]] ; then
                  # replace
                  optargs="-l 2"
                fi
              done
              echo -e "\n${seg}, sigma ${sig} (${refines}): ${fn}"
              python src/nemetyl_align-segments.py ${fn} -f ${ref} -t ${seg} -s ${sig} ${optargs} --with-plots
          done
      done
  done
done
for fn in ${input} ; do
    bn=$(basename -- ${fn})
    strippedname="${bn%.*}"
    mv reports/${strippedname}/ ${report}/
done
mv reports/*.csv ${report}/

# collect the "messagetype-combined-cluster-statistics.csv" of multiple independent nemetyl-runs
# We don't need this anymore, after the enhancement of the reportWriter module!
# python reports/combine-nemetyl-results.py ${report}

spd-say "Bin fertig!"
