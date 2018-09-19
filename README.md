# NEMESYS
NEtwork MEssage Syntax analysYS

Author: Stephan Kleber, Ulm University



## Release: WOOT18
https://www.usenix.org/conference/woot18/presentation/kleber


## Sample scripts

### prep_*
Detect identical payloads and de-duplicate traces in this regard.

### check_*
Parse a PCAP file and print its dissection for testing.

### nemesys_*
Infer messages from PCAPs by the NEMESYS approach (BCDG-segmentation)
and write FMS and other evaluation data to reports and plots.

### netzob_*
Infer PCAP with Netzob and compare the result to the tshark dissector of the protocol.




## Globally available options
Select layer to analyze by -l [#] and optionally -r

