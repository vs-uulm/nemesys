#!/bin/bash


prefix="netzob_messagetype"

numpad="200"
for f in reports/${prefix}-* ; do
  if [ -e "$f" ] ; then
    numnext=$(expr 1 + $(ls -d reports/${prefix}-* | sed "s/^.*${prefix}-\([0-9]*\)-.*$/\1/" | sort | tail -1))
    numpad=$(printf "%03d" ${numnext})
  fi
  break
done
currcomm=$(git log -1 --format="%h")
report=reports/${prefix}-${numpad}-${currcomm}
mkdir ${report}


## dhcp_SMIA2011101X_deduped-1000.pcap
#python src/netzob_messagetypes.py input/dhcp_SMIA2011101X_deduped-1000.pcap -r --smin 77 --smax 77
#python src/netzob_messagetypes.py input/dhcp_SMIA2011101X_deduped-1000.pcap -r --smin 78 --smax 78
#python src/netzob_messagetypes.py input/dhcp_SMIA2011101X_deduped-1000.pcap -r --smin 79 --smax 79
#
## dns_ictf2010_deduped-982-1000.pcap
#python src/netzob_messagetypes.py input/dns_ictf2010_deduped-982-1000.pcap -r --smin 49 --smax 51
#
## nbns_SMIA20111010-one_deduped-1000.pcap
#python src/netzob_messagetypes.py input/nbns_SMIA20111010-one_deduped-1000.pcap -r --smin 57 --smax 59
#
## ntp_SMIA-20111010_deduped-1000.pcap
#python src/netzob_messagetypes.py input/ntp_SMIA-20111010_deduped-1000.pcap -r --smin 56 --smax 58
#
## ntp_SMIA-20111010_deduped-100.pcap
#python src/netzob_messagetypes.py input/ntp_SMIA-20111010_deduped-100.pcap -r --smin 56 --smax 58
#
## smb_SMIA20111010-one_deduped-1000.pcap
#python src/netzob_messagetypes.py input/nbns_SMIA20111010-one_deduped-1000.pcap -r --smin 54 --smax 55
#python src/netzob_messagetypes.py input/nbns_SMIA20111010-one_deduped-1000.pcap -r --smin 56 --smax 56

#python src/netzob_messagetypes.py input/maxdiff-fromOrig/dhcp_SMIA2011101X-filtered_maxdiff-100.pcap   -r --smin 76
#python src/netzob_messagetypes.py input/maxdiff-fromOrig/dhcp_SMIA2011101X-filtered_maxdiff-1000.pcap  -r --smin 76
#python src/netzob_messagetypes.py input/maxdiff-fromOrig/dns_ictf2010-new_maxdiff-100.pcap             -r --smin 50
#python src/netzob_messagetypes.py input/maxdiff-fromOrig/dns_ictf2010-new_maxdiff-1000.pcap            -r --smin 50
#python src/netzob_messagetypes.py input/maxdiff-fromOrig/nbns_SMIA20111010-one_maxdiff-100.pcap        -r --smin 53
#python src/netzob_messagetypes.py input/maxdiff-fromOrig/nbns_SMIA20111010-one_maxdiff-1000.pcap       -r --smin 53
#python src/netzob_messagetypes.py input/maxdiff-fromOrig/ntp_SMIA-20111010_maxdiff-100.pcap            -r --smin 66
#python src/netzob_messagetypes.py input/maxdiff-fromOrig/ntp_SMIA-20111010_maxdiff-1000.pcap           -r --smin 66
#python src/netzob_messagetypes.py input/maxdiff-fromOrig/smb_SMIA20111010-one-rigid1_maxdiff-100.pcap  -r --smin 53
#python src/netzob_messagetypes.py input/maxdiff-fromOrig/smb_SMIA20111010-one-rigid1_maxdiff-1000.pcap -r --smin 53

## find similarity threshold
#python src/netzob_messagetypes.py input/awdl-filtered_maxdiff-100.pcap -l2 --smin 45 --smax 80
#python src/netzob_messagetypes.py input/au-wifi-filtered.pcap -l2 --smin 45 --smax 80
## regular run
#python src/netzob_messagetypes.py input/awdl-filtered_maxdiff-500.pcap -l2 --smin 65
#python src/netzob_messagetypes.py input/awdl-filtered_maxdiff-350.pcap -l2 --smin 65
#python src/netzob_messagetypes.py input/awdl-filtered_maxdiff-250.pcap -l2 --smin 65
#python src/netzob_messagetypes.py input/awdl-filtered_maxdiff-100.pcap -l2 --smin 57
#python src/netzob_messagetypes.py input/awdl-filtered.pcap -l2 --smin 57

mv reports/*.csv ${report}/
mv reports/*.pdf ${report}/

spd-say "Bin fertig!"
