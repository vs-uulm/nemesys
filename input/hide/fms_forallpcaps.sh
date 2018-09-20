for d in input/*.pcap ; do 
python src/nemesys_fms.py -s0.9 $d ; done

