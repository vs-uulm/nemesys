# Specimen Sources

"Representative" Protocols:

* binary: DNS, NTP, DHCP, SMB, NBNS
    * missing specimens for: NFS, RPC, Modbus, ZeroAccess (Botnet-Protocol used by Netzob)
    * non-Application-Layer: TCP, ICMP, ARP

## Tools
* Filter traces by tshark: `tshark -r infile -w outfile`
* Concatenate pcaps: `mergecap -F pcap -w OUTFILE INFILES`
* Change encapulation: `editcap -F pcap -T ENCTYPE INFILE OUTFILE`
* Deduplicate and truncate to fixed size: `prep_deduplicate-trace.py PCAP --p [N]`
* **maxdiff** `python ./src/prep_filter-maxdiff-trace.py -p [N] -r -f valcom PCAP`



## SMIA
[Netresec-Page](http://download.netresec.com/pcap/smia-2011/SMIA_2011-10-10_08%253A03%253A19_CEST_632834000_file1.pcap)

### NTP
* ntp_SMIA-20111010.pcap
    * from SMIA_2011-10-10_08-03-19_CEST_632834000_file1.pcap
    * filtered by `tshark -2 -R "ntp && !icmp" ... -F pcap`
    * by `python src/prep_deduplicate-trace.py ntp_SMIA-20111010.pcap --p [N]`
    * filter out: `!ntp.flags.mode == 6 && !ntp.flags.mode == 7`
      (for mode == 7 the dissector is incomplete)

### DHCP
* dhcp_SMIA-20111010_deduped-100.pcap
    * from SMIA_2011-10-10_08_632834000_file1-splits/dhcp
    * filtered for `bootp`
    * by `python src/prep_deduplicate-trace.py dhcp_SMIA-20111010.pcap --p [N]`
    * **loops > 100h when printing a symbol after netzob inference**
* dhcp_SMIA20111010-one-clean_deduped-100.pcap
    * dhcp-SMIA20111010-one.pcap (which is SMIA_2011-10-10_08_632834000 filtered for bootp)
    * filtered by `!bootp.option.user_class && !icmp`
* dhcp_SMIA20111010-one_deduped-995.pcap
    * dhcp-SMIA20111010-one.pcap (which is SMIA_2011-10-10_08_632834000 filtered for bootp)
    * filtered for `bootp.dhcp`
* dhcp_SMIA2011101X_deduped-10000.pcap
    * from SMIA_2011-10-10_08_632834000_file1-splits/dhcp
      merged with dhcp_SMIA_2011-10-11_07-38-27_CEST_961090000_file1-filtered.pcap
      to REUP-common/trace-collection/sources/A_SMIA/SMIA_2011-10-10+11 (?!)
    * filtered by: `!bootp.option.user_class && !icmp` and for `bootp.dhcp`
* maxdiff-fromOrig/dhcp_SMIA2011101X-filtered_maxdiff-1100.pcap
    * from trace-collection/sources/A_SMIA/SMIA_2011-10-10+11/dhcp_SMIA2011101X-filtered.pcap

**For newer Wireshark versions this is now: `dhcp && !dhcp.option.user_class && !icmp`**

### Netbios Name Server
* nbns_SMIA20111010-one_deduped-100.pcap
    * from SMIA_2011-10-10_08_632834000_file1-splits/nbns
    * filtered for `nbns && !icmp && !nbns.type == 33`
    * deduplicated and truncated by `python src/prep_deduplicate-trace.py`...

### SMB
* smb_SMIA20111010-one_deduped-100.pcap
    * from SMIA_2011-10-10_08_632834000_file1-splits/smb
    * filtered for `smb && !smb2 && !lanman && !smb.dfs.referral.version && !mailslot && !smb.mincount && !dcerpc && !nbdgm && !nbss.continuation_data && !smb.remaining == 1024`
        * afterwards filter repeatedly by `!(smb.trans2.cmd && _ws.expert.group == "Sequence")`
        * alternatively: `!smb.trans2.cmd || (smb.trans2.cmd > 0 && smb.trans2.cmd < 0xffff)`
    * -- multiple find_first2 files
    * deduplicated and truncated by `python src/prep_deduplicate-trace.py`...
* maxdiff-fromOrig/smb_SMIA20111010-one-rigid1_maxdiff-1100.pcap
    * from trace-collection/sources/A_SMIA/SMIA_2011-10-10_08_632834000_file1-splits/smb/smb_SMIA20111010-one-rigid1.pcap
    * filtered by adapted /home/stephan/REUP-common/trace-collection/sources/filters/filter-smb.sh
* smb_maccdc2012_maxdiff-1100.pcap
    * from /media/usb0/project-raw-files/traces/MACCDC2012/smb_maccdc2012.pcap
    * packets of trace are 802.11q VLAN encapsulated, strip to IP:  
      python3 strip_encapsulation.py smb_maccdc2012_000*-f2.pcap
    * filtered by /home/stephan/REUP-common/trace-collection/sources/filters/filter-smb.sh
    * download command and additional infos: /media/usb0/project-raw-files/traces/MACCDC2012/source.txt



## iCTF 2010
[UCSB](http://ictf.cs.ucsb.edu/ictfdata/2010/dumps/ictf2010pcap.tar.gz)

### IRC
* irc_ictf2010-42.pcap
    * from file ictf2010.pcap42
    * filtered by `tshark -2 -R "irc && !(icmp || tcp.analysis.retransmission || _ws.expert || _ws.malformed) && frame.len > 44 && frame.len < 1400 && !irc contains 20:20:0d:0a && !irc contains 20:20:20:20"`
    * by `python src/prep_deduplicate-trace.py irc_ictf2010-42.pcap --p [N]`

### DNS
* dns_ictf2010.pcap
    * from original file ictf2010.pcap
    * filtered for `dns && !icmp`

* dns_ictf2010_deduped-[N].pcap
    * from ictf2010_dns-f2.pcap
    * by `python src/prep_deduplicate-trace.py dns_ictf2010.pcap --p [N]`
    * filtered by `!_ws.malformed`

#### new DNS

* filter non-error responses: `(dns.flags.response == 1) && (dns.flags.rcode == 0)`
* write to `dns_ictf2010-responses.pcap`
* deduplicate: `python src/prep_deduplicate-trace.py -p 10000 input/hide/dns_ictf2010-responses.pcap`
* For 100/1000/10000s, each do:
    * get 66, 666, 6666 packets to merge: `python src/prep_deduplicate-trace.py -p 66  input/dns_ictf2010_deduped-9911-10000.pcap`
    * merge: `mergecap -w dns_ictf2010-new_1000.pcap hide/dns_ictf2010_deduped-666.pcap hide/dns_ictf2010-responses_deduped.pcap`
    * truncate: `python src/prep_deduplicate-trace.py -p 1000 input/dns_ictf2010-new_1000.pcap`
    * rename to: `dns_ictf2010-new-deduped-100.pcap`


## Random
Validation to find structure: Generated PCAPs with no structure (random byte sequences):

generate_random_pcap.py
with parameters: 

* -l 100
* -c 100 and 10000
* with and without -f


## binaryprotocols_merged
* from dhcp, dns, nbns, ntp, smb
* add missing Ethernet encapsulation by scapy:
  ```python 
  dnspcap = rdpcap("dns_ictf2010_deduped-982.pcap")
  outpakets = [Ether()/a for a in dnspcap]
  wrpcap("dns_ictf2010_deduped-982-ether.pcap", outpakets)
  ```
* with `mergecap -F pcap -w binaryprotocols_merged_XXX.pcap INFILES`

## Private/Own recordings

* wlan monitor captures wardriving through Biberach
* C_SEEMOO/wlan-mgt-priv.pcapng merged from C_SEEMOO/wlan-mgt by mergecap
* from this is filtered: wlan-beacons-priv.pcapng 
  * `wlan.fc.type_subtype == 0x0008 && !_ws.expert`
  * (very common SSIDs could be reduced by `!(wlan.ssid == "HZN241577234" || wlan.ssid == "Fritzle")` ) but we didn't
  * `python ~/Dokumente/git.lab-vs/REUP/nemesys/src/prep_filter-maxdiff-trace.py -l2 -p100[|0|00] wlan-beacons-priv.pcapng`

