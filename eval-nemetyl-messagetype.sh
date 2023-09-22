#!/bin/bash
#
# Calls the reference implementation of NEMETYL described in our INFOCOM 2020 paper.
# The parameters in this script are the same as used in the paper's evaluation.

#input=input/*-100.pcap
#input=input/*-1000.pcap
#input="input/*-100.pcap input/*-1000.pcap"
#input="input/ntp_SMIA-20111010_deduped-1000.pcap input/smb_SMIA20111010-one_deduped-1000.pcap"

#input="input/wlan-beacons-priv_maxdiff-100.pcapng"
#input="input/maxdiff-fromOrig/ntp_SMIA-20111010_maxdiff-100*.pcap"
#input="input/maxdiff-fromOrig/*-100*.pcap"

# > 41 GB Memory needed for PCAmoco!
# input="input/awdl-filtered.pcap"
# > 30 GB Memory needed for PCAmoco!
# input="inputl/awdl-filtered_maxdiff-500.pcap"

# input="input/awdl-filtered_maxdiff-100.pcap input/awdl-filtered_maxdiff-250.pcap input/awdl-filtered_maxdiff-350.pcap input/maxdiff-fromOrig/*-100*.pcap"
#input="input/maxdiff-fromOrig/*-100*.pcap input/awdl-filtered_maxdiff-100.pcap input/awdl-filtered.pcap"
# input="input/awdl-filtered_maxdiff-250.pcap input/au-wifi-filtered.pcap"
# input="input/awdl-filtered.pcap"
#input="input/awdl-filtered_maxdiff-100.pcap input/awdl-filtered.pcap input/au-wifi-filtered.pcap"
input="input/ari_syslog_corpus_maxdiff-99.pcapng input/ari_syslog_corpus_maxdiff-999.pcapng"

#sigmas="0.6 0.8 1.0 1.2"
# default
#sigmas="0.9 1.2"
sigmas="1.2"

# full
#segmenters="nemesys zeros"
segmenters="zeros"

# full
#refines="none original base nemetyl PCA1 PCAmoco PCAmocoSF zerocharPCAmocoSF emzcPCAmocoSF"

# Nemesys options
# refines="original nemetyl PCA1 PCAmoco zerocharPCAmocoSF"
# default
# refines="original nemetyl zerocharPCAmocoSF"
#refines="zerocharPCAmocoSF"

# Zeros options
# refines="none PCA1 PCAmocoSF"
# default
# refines="PCA1 PCAmocoSF"

# all-segmenter defaults
# refines="PCA1 PCAmocoSF original nemetyl zerocharPCAmocoSF emzcPCAmocoSF"

refines="nemetyl PCAmocoSF zerocharPCAmocoSF emzcPCAmocoSF"

L2PROTOS="input/awdl-* input/au-* input/wlan-beacons-*"
L1PROTOS="input/ari_*"
LEPROTOS="input/awdl-* input/au-* input/smb* input/*/smb* input/wlan-beacons-*"

prefix="nemetyl"

cftnpad="414"
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
    for proto in ${L1PROTOS} ; do
      if [[ "${fn}" == ${proto} ]] ; then
        # replace
        optargs="-l 1"
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
    for proto in ${L1PROTOS} ; do
      if [[ "${fn}" == ${proto} ]] ; then
        # replace
        optargs="-l 1"
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
              for proto in ${L1PROTOS} ; do
                if [[ "${fn}" == ${proto} ]] ; then
                  # replace
                  optargs="-l 1"
                fi
              done
              # comment out for "the wrong branch"
              for proto in ${LEPROTOS} ; do
                if [[ "${fn}" == $proto ]] ; then
                  # append
                  optargs="${optargs} -e"  # -e: little endian
                  echo -e "\nlitte endian"
                fi
              done
              echo -e "\n${seg}, sigma ${sig} (${refines}): ${fn}"
              python src/nemetyl_align-segments.py ${fn} -f ${ref} -t ${seg} -s ${sig} ${optargs} --with-plots
#              # # # #
#              # the wrong branch: apply LE optimization to the BE protocols
#              echo -e "\nforced litte endian"
#              echo -e "\n${seg}, sigma ${sig} (${refines}): ${fn}"
#              python src/nemetyl_align-segments.py ${fn} -f ${ref} -t ${seg} -s ${sig} -e ${optargs} --with-plots
#              # # # #
          done
  #        mkdir ${report}/sig${sig}-${ref}
  #        mv reports/*.pdf ${report}/sig${sig}-${ref}/
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
