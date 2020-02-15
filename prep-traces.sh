
## from filtered traditional traces
#pcaps="input/dhcp_SMIA2011101X_deduped-10000.pcap
#  input/dns_ictf2010_deduped-9911-10000.pcap
#  input/dns_ictf2010-new-deduped-10000.pcap
#  input/nbns_SMIA20111010-one_deduped-10000.pcap
#  input/ntp_SMIA-20111010_deduped-9995-10000.pcap
#  input/smb_SMIA20111010-one_deduped-10000.pcap"


# # from full traditional traces
#pcaps="/home/stephan/REUP-common/trace-collection/sources/A_SMIA/SMIA_2011-10-10+11/dhcp_SMIA2011101X-filtered.pcap
#  /home/stephan/REUP-common/trace-collection/sources/A_SMIA/SMIA_2011-10-10_08_632834000_file1-splits/nbns/nbns_SMIA20111010-one.pcap
#  /home/stephan/REUP-common/trace-collection/sources/A_SMIA/SMIA_2011-10-10_08_632834000_file1-splits/smb/smb_SMIA20111010-one-rigid1.pcap
#  /home/stephan/REUP-common/trace-collection/sources/A_SMIA/ntp_SMIA-20111010.pcap
#  /home/stephan/Dokumente/git.lab-vs/REUP/nemesys/input/hide/dns_ictf2010.pcap
#  /home/stephan/Dokumente/git.lab-vs/REUP/nemesys/input/hide/dns_ictf2010-new.pcap"

# # MACCDC2012
#pcaps="/media/usb0/project-raw-files/traces/MACCDC2012/smb_maccdc2012.pcap"


# TODO:
## # digitalcorpora traces
#pcaps="/home/stephan/Dokumente/REUP-common/trace-collection/sources/A_digitalcorpora/net-2009-11-13-09_24_dns.pcap
#  /home/stephan/Dokumente/REUP-common/trace-collection/sources/A_digitalcorpora/net-2009-11-13-09_24_ntp.pcap"
#
## # iCFT/USS 2017
#pcaps="/home/stephan/Dokumente/REUP-common/trace-collection/sources/B_USS-ictf2017/trace-iCTF-USS2017_dns.pcap"


for fn in ${pcaps} ; do
	python src/prep_filter-maxdiff-trace.py ${fn} -p1100
done

spd-say "Bin fertig!"
