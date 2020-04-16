#!/usr/bin/env bash

#input=input/*-100.pcap
#input=input/*-1000.pcap
#input="input/*-100.pcap input/*-1000.pcap"
#input="input/ntp_SMIA-20111010_deduped-1000.pcap input/smb_SMIA20111010-one_deduped-1000.pcap"
#input=input/dns_ictf2010-new-deduped-100.pcap
#
input=input/maxdiff-fromOrig/*-100.pcap
#input=input/maxdiff-fromOrig/ntp_SMIA-20111010_maxdiff-100.pcap
#input="input/maxdiff-fromOrig/dhcp_SMIA2011101X-filtered_maxdiff-100.pcap"
#input="input/maxdiff-fromOrig/smb_SMIA20111010-one-rigid1_maxdiff-1000.pcap"
#input="input/maxdiff-fromOrig/dhcp_SMIA2011101X-filtered_maxdiff-1000.pcap input/maxdiff-fromOrig/dns_ictf2010_maxdiff-1000.pcap input/maxdiff-fromOrig/dns_ictf2010-new_maxdiff-1000.pcap input/maxdiff-fromOrig/ntp_SMIA-20111010_maxdiff-1000.pcap input/maxdiff-fromOrig/nbns_SMIA20111010-one_maxdiff-1000.pcap"
#input=input/maxdiff-fromOrig/nbns_SMIA20111010-one_maxdiff-1000.pcap


#refines="base PCA PCA1 PCAmoco"
#refines="PCA PCAmoco"
refines="PCAmoco"
#refines="PCA1"


cftnext=$(expr 1 + $(ls -d reports/zeropca-* | sed "s/^.*zeropca-\([0-9]*\)-.*$/\1/" | sort | tail -1))
cftnpad=$(printf "%03d" ${cftnext})
currcomm=$(git log -1 --format="%h")
report=reports/zeropca-${cftnpad}-refinement-${currcomm}
mkdir ${report}

for fn in ${input} ; do
  for ref in ${refines} ; do
    # with plots: -p
    python src/nemezero_pca-refinement.py $* -r ${ref} ${fn}
  done
done


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

cd ${report}/
ln -s ../combine-stats.sh .
./combine-stats.sh
cd -
python reports/combine-fms.py ${report}/


echo "Report written to folder ${report}"
spd-say "Bin fertig!"
