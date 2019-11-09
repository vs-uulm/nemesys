#!/usr/bin/env bash

#input=input/*-100.pcap
#input=input/*-1000.pcap
#input="input/*-100.pcap input/*-1000.pcap"
# input="input/ntp_SMIA-20111010_deduped-1000.pcap input/smb_SMIA20111010-one_deduped-1000.pcap"
#input="input/ntp_SMIA-20111010_deduped-100.pcap"
input=input/smb_SMIA20111010-one_deduped-100.pcap

#sigmas="0.6 0.7 0.8 0.9 1.0 1.1 1.2"
sigmas="0.6 0.8 1.0 1.2"
#sigmas="0.9"

#refines="base original PCA PCAmoco"
refines="PCA PCAmoco"
#refines="PCAmoco"


cftnext=$(expr 1 + $(ls -d reports/cft-* | sed "s/^.*cft-\([0-9]*\)-.*$/\1/" | sort | tail -1))
cftnpad=$(printf "%03d" ${cftnext})
currcomm=$(git log -1 --format="%h")
report=reports/cft-${cftnpad}-clustering-${currcomm}
mkdir ${report}


#for fn in ${input} ; do
#for sig in ${sigmas} ; do
##python src/cluster_segments.py -s ${sig} -r PCAmoco ${fn}
#python src/cluster_segments.py -s ${sig} -r PCA ${fn}
##python src/cluster_segments.py -s ${sig} -r base ${fn}
##python src/cluster_segments.py -s ${sig} -r original ${fn}
#
##python src/cluster_segments.py -p -s ${sig} -r PCAmoco ${fn}
#done
#done


for ref in ${refines} ; do
    for fn in ${input} ; do
        python src/cluster_segments.py -p -r ${ref} ${fn}
    done


    mkdir ${report}-${ref}
    mv reports/*.pdf ${report}-${ref}/
    for fn in ${input};
    do
        bn=$(basename -s .pcap ${fn})
        mv reports/${bn}* ${report}-${ref}/
    done
done

python src/transform_cluster-statistics.py
mv reports/*.csv ${report}/

spd-say "Bin fertig!"