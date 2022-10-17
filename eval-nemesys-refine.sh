#!/usr/bin/env bash
#
# Calls the reference implementation of NEMESYS described in our WOOT 2018 paper.
# The parameters in this script are the same as used in the paper's evaluation.


# input=input/*-100.pcap
# input=input/*-1000.pcap
input="input/maxdiff-fromOrig/*-100.pcap"
#input="input/maxdiff-fromOrig/ntp_SMIA-20111010_maxdiff-100.pcap"


# full
#sigmas="0.6 0.7 0.8 0.9 1.0 1.1 1.2 2.4"
sigmas="0.9 1.2"

L2PROTOS="input/awdl-* input/au-*"
LEPROTOS="input/awdl-* input/au-* input/smb* input/*/smb*"

prefix="nemesys"

cftnpad="200"
for f in reports/${prefix}-* ; do
  if [ -e "$f" ] ; then
    cftnext=$(expr 1 + $(ls -d reports/${prefix}-* | sed "s/^.*${prefix}-\([0-9]*\)-.*$/\1/" | sort | tail -1))
    cftnpad=$(printf "%03d" ${cftnext})
  fi
  break
done
currcomm=$(git log -1 --format="%h")
report=reports/${prefix}-${cftnpad}-original-${currcomm}
mkdir ${report}

#for fn in ${input} ; do python src/nemesys_fms.py -r ${fn} ; done
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

  for sig in ${sigmas} ; do
    python src/nemesys_fms.py ${optargs} -s ${sig} ${fn}
  done
done


#mv reports/*.pdf ${report}/
for fn in ${input};
do
    bn=$(basename -s .pcap ${fn})
    mv reports/${bn}* ${report}/
done

python reports/combine-nemesys-fms.py ${report}/



spd-say "Bin fertig!"
