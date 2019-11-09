#!/usr/bin/env bash

input=input/*-100.pcap
#input=input/*-1000.pcap
#input="input/*-100.pcap input/*-1000.pcap"
# input="input/ntp_SMIA-20111010_deduped-1000.pcap input/smb_SMIA20111010-one_deduped-1000.pcap"
#input=input/dns_ictf2010-new-deduped-100.pcap

#sigmas="0.6 0.7 0.8 0.9 1.0 1.1 1.2"
sigmas="0.6 0.8 1.0 1.2"
#sigmas="0.9"

#refines="base original PCA PCA1 PCAmoco"
refines="PCA PCA1"


cftnext=$(expr 1 + $(ls -d reports/cft-* | sed "s/^.*cft-\([0-9]*\)-.*$/\1/" | sort | tail -1))
cftnpad=$(printf "%03d" ${cftnext})
currcomm=$(git log -1 --format="%h")
report=reports/cft-${cftnpad}-refinement-${currcomm}
mkdir ${report}

# for fn in ${input} ; do python src/cluster_segments.py ${fn} -p ; done
#for fn in ${input} ; do python src/nemesys-pca_refinement.py ${fn} ; done
for fn in ${input} ; do
for ref in ${refines} ; do
for sig in ${sigmas} ; do
python src/nemesys-pca_refinement.py -s ${sig} -r ${ref} ${fn}
done
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



spd-say "Bin fertig!"