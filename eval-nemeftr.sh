#!/usr/bin/env bash

#input=input/*-100.pcap
#input=input/*-1000.pcap
#input="input/*-100.pcap input/*-1000.pcap"
#input="input/ntp_SMIA-20111010_deduped-1000.pcap input/smb_SMIA20111010-one_deduped-1000.pcap"
#input=input/ntp_SMIA-20111010_deduped-1000.pcap
#input=input/smb_maccdc2012_maxdiff-1000.pcap
#input=input/maxdiff-filtered/*-1000.pcap
input=input/maxdiff-fromOrig/*-1000.pcap
#input="input/maxdiff-fromOrig/smb_SMIA20111010-one-rigid1_maxdiff-1000.pcap
#  input/maxdiff-fromOrig/ntp_SMIA-20111010_maxdiff-1000.pcap"
#  input=input/maxdiff-fromOrig/dhcp_SMIA2011101X-filtered_maxdiff-1000.pcap
#input=input/maxdiff-fromOrig/ntp_SMIA-20111010_maxdiff-100.pcap

tftnext=$(expr 1 + $(ls -d reports/tft-* | sed "s/^.*tft-\([0-9]*\)-.*$/\1/" | sort | tail -1))
tftnpad=$(printf "%03d" ${tftnext})
currcomm=$(git log -1 --format="%h")
report=reports/tft-${tftnpad}-clustering-${currcomm}
mkdir ${report}



# for fn in ${input} ; do python src/visualize_fieldtype_separation.py ${fn} ; done

#for fn in ${input} ; do python src/nemeftr_cluster-true-fields.py ${fn} ; done
for fn in ${input} ; do python src/nemeftr_cluster-true-fields.py -f ${fn} ; done

# for fn in ${input} ; do python src/nemeftr_field-recognition.py ${fn} ; done



mv reports/*.csv ${report}/
mv reports/*.pdf ${report}/
















# for ana in bcd progdiff progcumudelta value; do python src/characterize_fieldtypes.py input/binaryprotocols_merged_500.pcap $ana; done



spd-say "Bin fertig!"
