#!/usr/bin/env bash

#input=input/*-100.pcap
#input=input/*-1000.pcap
#
#input=input/maxdiff-fromOrig/*-1000.pcap
input="input/maxdiff-fromOrig/*-100*.pcap"

#refines="base PCA PCA1 PCAmoco PCAmocoSF"
refines="PCAmocoSF"

L1PROTOS="input/ari_*"
L2PROTOS="input/awdl-* input/au-* input/wlan-beacons-*"
# never perform LE optimization for zeropca! see nemesys-le-compare.ods
LEPROTOS=""
#LEPROTOS="input/awdl-* input/au-* input/smb* input/*/smb* input/wlan-beacons-*"

prefix="zeropca"

cftnpad="350"
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

for fn in ${input} ; do
  optargs="-r -p" # with plots add: -p
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
      # append
      optargs="${optargs} -e"  # -e: little endian
    fi
  done

  for ref in ${refines} ; do
    python src/nemezero_pca-refinement.py $* ${optargs} -f ${ref} ${fn}
  done

#  # # # #
#  # # the wrong branch: apply LE optimization to the LE and BE protocols
#  for ref in ${refines} ; do
#    python src/nemezero_pca-refinement.py $* ${optargs} -f ${ref} ${fn}
#    python src/nemezero_pca-refinement.py $* ${optargs} -e -f ${ref} ${fn}
#  done
done


mv reports/*.pdf ${report}/
for fn in ${input};
do
    bn=$(basename -- ${fn})
    strippedname="${bn%.*}"
    mv reports/${strippedname} ${report}/
done

cd ${report}/
../combine-stats.sh
cd -
python reports/combine-nemesys-fms.py ${report}/



echo "Report written to folder ${report}"
spd-say "Bin fertig!"
