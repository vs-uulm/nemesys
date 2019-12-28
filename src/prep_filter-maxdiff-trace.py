import argparse
import bisect
from dataclasses import dataclass
from os.path import isfile, splitext, exists
from itertools import chain
from collections import Counter
from typing import List

import IPython
import numpy
import scapy.all as sy
from tabulate import tabulate

from netzob.Model.Vocabulary.Messages.RawMessage import RawMessage, AbstractMessage
from netzob.Model.Vocabulary.Messages.L4NetworkMessage import L4NetworkMessage
from netzob.Model.Vocabulary.Messages.L3NetworkMessage import L3NetworkMessage
from netzob.Model.Vocabulary.Messages.L2NetworkMessage import L2NetworkMessage

from inference.segmentHandler import bcDeltaGaussMessageSegmentation, zeroBaseRefinements
from utils.loader import SpecimenLoader, BaseLoader
from validation.dissectorMatcher import MessageComparator
from validation.messageParser import ParsedMessage

PACKET_LIMIT = 100
sigma = 1.2


@dataclass
class MessageValueCommonality:
    commonality : float
    message : AbstractMessage

    def __typecheck(self, other):
        if not isinstance(other, type(self)):
            raise TypeError("Non-comparable objects.")

    def __lt__(self, other):
        self.__typecheck(other)
        return self.commonality < other.commonality

    def __le__(self, other):
        self.__typecheck(other)
        return self.commonality <= other.commonality

    def __ge__(self, other):
        self.__typecheck(other)
        return self.commonality >= other.commonality

    def __gt__(self, other):
        self.__typecheck(other)
        return self.commonality <= other.commonality



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=
                                     'Filter packets in PCAP outfile.')
    parser.add_argument('pcapfilename', help='pcapfilename')
    parser.add_argument('-l', '--layernumber', nargs='?', type= int,
                        help='layernumber relative to IP (default: IP+2)', default='2')
    parser.add_argument('-p', '--packetcount', nargs='?', type= int,
                        help='packet count (default: {:d})'.format(PACKET_LIMIT), default=PACKET_LIMIT)
    parser.add_argument('-i', '--interactive', help='Show interactive plot instead of writing output to file and '
                                                    'open ipython prompt after finishing the analysis.',
                        action="store_true")
    args = parser.parse_args()

    pcapfilename = args.pcapfilename
    packetcount = args.packetcount
    layer = args.layernumber

    if not isfile(pcapfilename):
        print('File not found: ' + pcapfilename)
        exit(1)

    infile,ext = splitext(pcapfilename)
    outfile = infile + "_deduped-{:d}".format(packetcount) + ext
    if exists(outfile):
        print('Output file exists: ' + outfile)
        exit(1)

    # get segments from messages and their common values
    specimens = SpecimenLoader(pcapfilename, layer, True)
    segmentsPerMsg = bcDeltaGaussMessageSegmentation(specimens, sigma)
    refinedPerMsg = zeroBaseRefinements(segmentsPerMsg)
    segments = chain.from_iterable(refinedPerMsg)
    valCounter = Counter(s.bytes for s in segments)

    # # number of messages supporting the 100 most common segment values
    # msgCounter = dict()
    # for b, c in valCounter.most_common(100):
    #     for msg in refinedPerMsg:
    #         if b in (r.bytes for r in msg):
    #             if b not in msgCounter:
    #                 msgCounter[b] = 0
    #             msgCounter[b] += 1

    # sorted mean commonality of each messages' segments
    valueCommonalityPerMsg = list()  # type: List[MessageValueCommonality]
    for msg in refinedPerMsg:
        valCom = float(numpy.mean([valCounter[seg.bytes] for seg in msg]))
        mvc = MessageValueCommonality(valCom, msg[0].message)
        bisect.insort(valueCommonalityPerMsg, mvc)

    # TODO deduplicate

    # the selected messages of most "uncommon messages"
    filteredMsgs = valueCommonalityPerMsg[:packetcount]
    filteredSpecimens = BaseLoader(
        (fm.message for fm in filteredMsgs), (specimens.messagePool[fm.message] for fm in filteredMsgs))
    filteredComparator = MessageComparator(filteredSpecimens, layer, True)

    # Statistics about selected message types
    print("Message types found in trace:")
    pms = [pm for pm in filteredComparator.parsedMessages.values()]  # type: List[ParsedMessage]
    msgtypes = Counter((pm.protocolname, pm.messagetype) for pm in pms)
    print(tabulate((*pm,c) for pm,c in msgtypes.most_common()))


    # packetList = sy.rdpcap(pcapfilename)
    #
    # sorted([a.date for a in specimens.messagePool.values()]) == sorted([a.time for a in packetList])
    #
    # packet = packetList[0]
    # rawmsg = next(iter(specimens.messagePool.values()))
    # for packet, rawmsg in zip(packetList, specimens.messagePool.values()):
    #     assert bytes(packet) == rawmsg.data



    if args.interactive:
        # globals().update(locals())
        IPython.embed()