#!/bin/bash

#input=input/*-100.pcap
input=input/*-1000.pcap
#input="input/*-100.pcap input/*-1000.pcap"
#input="input/ntp_SMIA-20111010_deduped-1000.pcap input/smb_SMIA20111010-one_deduped-1000.pcap"

sigmas="0.6 0.8 1.0 1.2"
#sigmas="0.9"

refines="base original"
#refines="base"



cftnext=$(expr 1 + $(ls -d reports/nemetyl-* | sed "s/^.*nemetyl-\([0-9]*\)-.*$/\1/" | sort | tail -1))
cftnpad=$(printf "%03d" ${cftnext})
currcomm=$(git log -1 --format="%h")
report=reports/nemetyl-${cftnpad}-clustering-${currcomm}
mkdir ${report}




for fn in ${input} ; do python src/nemetyl_align-segments.py $fn -t tshark --with-plots ; done;
for fn in ${input} ; do python src/nemetyl_align-segments.py $fn -t 4bytesfixed --with-plots ; done;

mv reports/*.pdf ${report}/
mv reports/*.csv ${report}/

for sig in ${sigmas} ; do
    for ref in ${refines} ; do
        for fn in ${input} ; do
            python src/nemetyl_align-segments.py ${fn} -r ${ref} -t nemesys --with-plots
        done

        mkdir ${report}/sig${sig}-${ref}
        mv reports/*.pdf ${report}/sig${sig}-${ref}/
        mv reports/*.csv ${report}/sig${sig}-${ref}/
    done
done



spd-say "Bin fertig!"