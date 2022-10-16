#!/bin/bash
#
# Split pcaps ending with "-1100.pcap" into disjunct 100 and 1000 parts.

ext="-1100.pcap"

for fn in ${@} ; do 
	bn=$(basename -s "${ext}" "${fn}")
	editcap -r ${fn} "${bn}-100.pcap" 1-100
	editcap -r ${fn} "${bn}-1000.pcap" 101-1100
done
