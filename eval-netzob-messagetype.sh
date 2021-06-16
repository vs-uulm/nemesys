#!/bin/bash

# dhcp_SMIA2011101X_deduped-1000.pcap
python src/netzob_messagetypes.py input/dhcp_SMIA2011101X_deduped-1000.pcap -r --smin 77 --smax 77
python src/netzob_messagetypes.py input/dhcp_SMIA2011101X_deduped-1000.pcap -r --smin 78 --smax 78
python src/netzob_messagetypes.py input/dhcp_SMIA2011101X_deduped-1000.pcap -r --smin 79 --smax 79


# dns_ictf2010_deduped-982-1000.pcap
python src/netzob_messagetypes.py input/dns_ictf2010_deduped-982-1000.pcap -r --smin 49 --smax 51


# nbns_SMIA20111010-one_deduped-1000.pcap
python src/netzob_messagetypes.py input/nbns_SMIA20111010-one_deduped-1000.pcap -r --smin 57 --smax 59


# ntp_SMIA-20111010_deduped-1000.pcap
python src/netzob_messagetypes.py input/ntp_SMIA-20111010_deduped-1000.pcap -r --smin 56 --smax 58


# smb_SMIA20111010-one_deduped-1000.pcap
python src/netzob_messagetypes.py input/nbns_SMIA20111010-one_deduped-1000.pcap -r --smin 54 --smax 55
python src/netzob_messagetypes.py input/nbns_SMIA20111010-one_deduped-1000.pcap -r --smin 56 --smax 56

