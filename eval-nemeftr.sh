#!/usr/bin/env bash

#input=input/*-100.pcap
#input=input/*-1000.pcap
input="input/*-100.pcap input/*-1000.pcap"
# input="input/ntp_SMIA-20111010_deduped-1000.pcap input/smb_SMIA20111010-one_deduped-1000.pcap"


# for fn in ${input} ; do python src/visualize_fieldtype_separation.py ${fn} ; done
# mv reports/*.pdf reports/tft-X01-template-centers/

for fn in ${input} ; do python src/fieldtype-aware_distances.py ${fn} -f ; done
#for fn in ${input} ; do python src/fieldtype-aware_distances.py ${fn} ; done
# mv reports/*.pdf reports/tft-X02-cluster-fieldtypes/



# for fn in ${input} ; do python src/field_recognition.py ${fn} ; done
# mv reports/*.pdf reports/tft-X03-recognized-fields
















# for ana in bcd progdiff progcumudelta value; do python src/characterize_fieldtypes.py input/binaryprotocols_merged_500.pcap $ana; done