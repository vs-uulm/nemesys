#!/usr/bin/python3
"""
Limit the number of message in a trace to a fixed value of unique packets (PACKET_LIMIT).
This way, generates comparable traces as evaluation input.
"""

import logging  # hide warnings of scapy: https://stackoverflow.com/questions/24812604/hide-scapy-warning-message-ipv6
logging.getLogger("scapy.runtime").setLevel(logging.ERROR)
from scapy.all import *
import argparse
from os.path import exists,isfile,splitext
from collections import OrderedDict

from validation.messageParser import ParsingConstants

PACKET_LIMIT = 1000

class Deduplicate(object):
    """Class to detect identical payloads and de-duplicate traces in this regard."""

    def __init__(self, targetlayer, packetlimit):
        """
        Setup the de-duplicator.

        :param targetlayer: Layer of interest to have unique content
        :param packetlimit: The number of messages to limit the output trace to
        """
        self.TARGETLAYER = targetlayer
        self.PACKET_LIMIT = packetlimit
        # use an ordered dictionary to remove identical payloads and still retain the ordering of messages
        self.unique_packets = OrderedDict()


    def dedup(self, packet):
        """
        Execute the deduplication in combination with Scapy's sniff:
        Works as a filter method for Scapy's sniff(..., stop_filter=dedup, ...)
        Saves each input packet into self.unique_packets and returns False (meaning to continue), as long as
            1. packet's TARGETLAYER content is not already in self.unique_packets
            2. self.unique_packets is not larger then PACKET_LIMIT

        :param packet: packet data
        :return: True if packet sniffing should continue, False if number of unique messages reaches PACKET_LIMIT
        """
        try:
            # select target network layer
            if isinstance(self.TARGETLAYER, int):
                targetpacket = packet[self.TARGETLAYER]
            else:
                targetpacket = packet[self.TARGETLAYER][2]


            self.unique_packets[str(targetpacket)] = packet
            if len(self.unique_packets) >= PACKET_LIMIT:
                return True
        except IndexError:
            if isinstance(TARGETLAYER, str):
                layername = TARGETLAYER + ' + 2'
            else:
                layername = TARGETLAYER
            print('Protocol layer ' + str(layername) + ' not available in the following packet:')
            print('\n\n' + repr(packet) + '\n\n')
        return False


def main(filename, outputfile, targetlayer, packetlimit):
    dedup = Deduplicate(targetlayer, packetlimit)

    # TODO sniff waits indefinitely if the input-pcap file contains less # packets < PACKET_LIMIT; break with ctrl+c
    # sniff has the advantage to NOT read the whole file into the memory initially. This saves memory for huge pcaps.
    sniff(offline=filename,stop_filter=dedup.dedup,store=0)

    # get the first packet (we assume all have the same linktype)
    eplpkt = next(iter(dedup.unique_packets.values()))
    if isinstance(eplpkt, Ether):
        lt = ParsingConstants.LINKTYPES["ETHERNET"] # 0x1
    elif isinstance(eplpkt, IP):
        lt = ParsingConstants.LINKTYPES["RAW_IP"] # 0x65
    else:
        raise Exception("Check linktype.")

    wrpcap(outputfile, dedup.unique_packets.values(), linktype=lt)
    print("Deduplication of {:s} of pcap written to {:s}".format(
        str(TARGETLAYER) if not isinstance(TARGETLAYER, str) else TARGETLAYER + ' + 2',
        outfile))



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=
                                     'Limit number of packets in pcap outfile to fixed number of unique packets.')
    parser.add_argument('pcapfilename', help='pcapfilename')
    parser.add_argument('-l', '--layernumber', nargs='?', type= int,
                        help='layernumber (default: IP+2)', default='-1')
    parser.add_argument('-p', '--packetcount', nargs='?', type= int,
                        help='packet count (default: {:d})'.format(PACKET_LIMIT), default=PACKET_LIMIT)
    args = parser.parse_args()

    FILENAME = args.pcapfilename # 'file.pcap'
    PACKET_LIMIT = args.packetcount
    if args.layernumber >= 0:
        TARGETLAYER = args.layernumber # use 'IP' as flag for "IP+2"
    else:
        TARGETLAYER = 'IP'

    if not isfile(FILENAME):
        print('File not found: ' + FILENAME)
        exit(1)

    infile,ext = splitext(FILENAME)
    outfile = infile + "_deduped-{:d}".format(PACKET_LIMIT) + ext
    if exists(outfile):
        print('Output file exists: ' + outfile)
        exit(1)

    main(FILENAME, outfile, TARGETLAYER, PACKET_LIMIT)


















