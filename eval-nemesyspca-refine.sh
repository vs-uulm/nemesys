#!/usr/bin/env bash

#input=input/*-100.pcap
#input=input/*-1000.pcap
#input="input/*-100.pcap input/*-1000.pcap"
# input="input/ntp_SMIA-20111010_deduped-1000.pcap input/smb_SMIA20111010-one_deduped-1000.pcap"
#input=input/dns_ictf2010-new-deduped-100.pcap
#
input=input/maxdiff-fromOrig/*-1000.pcap
#input=input/maxdiff-fromOrig/dns_ictf2010-new_maxdiff-100.pcap
#input=input/maxdiff-fromOrig/ntp_SMIA-20111010_maxdiff-1000.pcap
#input="input/maxdiff-fromOrig/dhcp_SMIA2011101X-filtered_maxdiff-100.pcap input/maxdiff-fromOrig/smb_SMIA20111010-one-rigid1_maxdiff-100.pcap"

#sigmas="0.6 0.7 0.8 0.9 1.0 1.1 1.2"
#sigmas="1.2"
sigmas="0.9 1.2"

#refines="base original PCA PCA1 PCAmoco"
#refines="zerocharPCAmoco"
#refines="PCA1 zeroPCA base"
#refines="zeroPCA PCAmoco"
refines="zero base"


cftnext=$(expr 1 + $(ls -d reports/nemesys-* | sed "s/^.*nemesys-\([0-9]*\)-.*$/\1/" | sort | tail -1))
cftnpad=$(printf "%03d" ${cftnext})
currcomm=$(git log -1 --format="%h")
report=reports/nemesys-${cftnpad}-refinement-${currcomm}
mkdir ${report}

#for fn in ${input} ; do python src/nemesys_pca-refinement.py ${fn} ; done
for fn in ${input} ; do
for ref in ${refines} ; do
for sig in ${sigmas} ; do
python src/nemesys_pca-refinement.py -s ${sig} -r ${ref} ${fn}
done
done
done


# copy all lines of the "ScoreStatistics.csv" files together into one per trace
mv reports/*.pdf ${report}/
for fn in ${input};
do
    bn=$(basename -s .pcap ${fn})
    mv reports/${bn} ${report}/
    ln -r -s reports/combine-fms.sh ${report}/${bn}/
    cd ${report}/${bn}/
    ./combine-fms.sh
    cd -
done

# combine PCA refinement condition statistics into a single  "...-combined.csv"
cd ${report}/
ln -s ../combine-stats.sh .
./combine-stats.sh
cd -
# combine all ScoreStatistics into one file with a sheet for each trace in a xls file
python reports/combine-fms.py ${report}/


echo "Report written to folder ${report}"
spd-say "Bin fertig!"
