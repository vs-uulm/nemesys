#!/usr/bin/env bash
#
# NEMEFTR pre
# ===========
#
# Topology plot: template centers of true field types
# Histogram: type-separation per true field type
#
# Histograms used in nemeftr-full

#input=input/*-100.pcap
#input=input/*-1000.pcap
#input="input/*-100.pcap input/*-1000.pcap"
#input=input/maxdiff-filtered/*-1000.pcap
input="input/maxdiff-fromOrig/*-100*.pcap"
#input="input/maxdiff-fromOrig/ntp_SMIA-20111010_maxdiff-100.pcap"


L2PROTOS="input/awdl-* input/au-*"
LEPROTOS="input/awdl-* input/au-* input/smb* input/*/smb*"

prefix="ftrvisualize"

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
  optargs="-r" # for varying epsilon add: -e
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
  python src/visualize_fieldtype_separation.py ${optargs} ${fn}
done


# legacy ?
# for fn in ${input} ; do python src/visualize_fieldtype_separation.py ${optargs} ${fn} ; done

mv reports/*.csv ${report}/
mv reports/*.pdf ${report}/






spd-say "Bin fertig!"
