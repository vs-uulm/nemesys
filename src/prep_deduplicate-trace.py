#!/usr/bin/python3
"""
Limit the number of message in a trace to a fixed value of unique packets (PACKET_LIMIT).
This way, generates comparable traces as evaluation input.
"""

import argparse
from os.path import exists,isfile,splitext
from collections import OrderedDict
from collections.abc import Sequence

import logging  # hide warnings of scapy: https://stackoverflow.com/questions/24812604/hide-scapy-warning-message-ipv6

from scapy.layers.dot11 import RadioTap

logging.getLogger("scapy.runtime").setLevel(logging.ERROR)
from scapy.layers.inet import IP
from scapy.layers.l2 import Ether
from scapy.all import sniff, wrpcap

from nemere.validation.messageParser import ParsingConstants

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
                targetpacket = packet[self.TARGETLAYER[0]][self.TARGETLAYER[1]]


            self.unique_packets[str(targetpacket)] = packet
            if len(self.unique_packets) >= PACKET_LIMIT:
                return True
        except IndexError:
            if isinstance(self.TARGETLAYER, str):
                layername = self.TARGETLAYER
            elif isinstance(self.TARGETLAYER, Sequence):
                layername = f'{self.TARGETLAYER[0]} + {self.TARGETLAYER[1]}'
            else:
                layername = self.TARGETLAYER
            print('Protocol layer ' + str(layername) + ' not available in the following packet:')
            print('\n\n' + repr(packet) + '\n\n')
        return False


def main(filename, outputfile, targetlayer, packetlimit):
    dedup = Deduplicate(targetlayer, packetlimit)

    # TODO sniff waits indefinitely if the input-pcap file contains less # packets < PACKET_LIMIT; break with ctrl+c
    # sniff has the advantage to NOT read the whole file into the memory initially. This saves memory for huge pcaps.
    sniff(offline=filename, stop_filter=dedup.dedup, store=0)

    # get the first packet (we assume all have the same linktype)
    eplpkt = next(iter(dedup.unique_packets.values()))
    if isinstance(eplpkt, Ether):
        lt = ParsingConstants.LINKTYPES["ETHERNET"] # 0x1
    elif isinstance(eplpkt, IP):
        lt = ParsingConstants.LINKTYPES["RAW_IP"] # 0x65
    elif isinstance(eplpkt, RadioTap):
         lt = ParsingConstants.LINKTYPES["IEEE802_11_RADIO"]  # 0x7f
    else:
        raise Exception("Check linktype ({}).".format(eplpkt.name))

    wrpcap(outputfile, dedup.unique_packets.values(), linktype=lt)
    print("Deduplication of {:s} of pcap written to {:s}".format(
        targetlayer if isinstance(targetlayer, str) else
            f"{targetlayer[0]} + {targetlayer[1]}" if isinstance(targetlayer, Sequence) else str(targetlayer),
        outfile))



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=
                                     'Limit number of packets in pcap outfile to fixed number of unique packets.')
    parser.add_argument('pcapfilename', help='pcapfilename')
    parser.add_argument('-l', '--layer', type=int, default=2,
                        help='Protocol layer relative to IP to consider. Default is 2 layers above IP '
                             '(typically the payload of a transport protocol).')
    parser.add_argument('-r', '--relativeToIP', default=False, action='store_true')
    parser.add_argument('-p', '--packetcount', nargs='?', type= int,
                        help='packet count (default: {:d})'.format(PACKET_LIMIT), default=PACKET_LIMIT)
    args = parser.parse_args()

    FILENAME = args.pcapfilename # 'file.pcap'
    PACKET_LIMIT = args.packetcount
    if not args.relativeToIP:
        TARGETLAYER = (0, args.layer)
    else:
        TARGETLAYER = ('IP', args.layer)

    if not isfile(FILENAME):
        print('File not found: ' + FILENAME)
        exit(1)

    infile,ext = splitext(FILENAME)
    outfile = infile + "_deduped-{:d}".format(PACKET_LIMIT) + ext
    if exists(outfile):
        print('Output file exists: ' + outfile)
        exit(1)

    main(FILENAME, outfile, TARGETLAYER, PACKET_LIMIT)






