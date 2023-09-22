"""
Filter a PCAP for the subset of packets that have the maximum difference to all other messages.
For the maximum difference, multiple approaches are conceivable. Here we implement three for comparison and apply
the metric of the average least common segment values per message.
"""

import logging  # hide warnings of scapy: https://stackoverflow.com/questions/24812604/hide-scapy-warning-message-ipv6

from nemere.validation.messageParser import DissectionIncomplete

logging.getLogger("scapy.runtime").setLevel(logging.ERROR)


import argparse, bisect, csv, IPython, numpy, time
import scapy.all as sy
from dataclasses import dataclass
from os.path import isfile, splitext, exists, join
from itertools import chain
from collections import Counter, OrderedDict
from typing import List
from tabulate import tabulate

from netzob.Model.Vocabulary.Messages.RawMessage import AbstractMessage

from nemere.inference.segmentHandler import bcDeltaGaussMessageSegmentation, baseRefinements
from nemere.inference.segments import MessageSegment
from nemere.inference.templates import DelegatingDC, Template
from nemere.inference.analyzers import Value
from nemere.utils.loader import SpecimenLoader, BaseLoader
from nemere.utils.evaluationHelpers import reportFolder
from nemere.validation.dissectorMatcher import MessageComparator

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


# noinspection PyShadowingNames
def canberraDissimFilter(specimens: BaseLoader, packetcount: int):
    """
    Interpret each message as one segment, calculate their Canberra dissimilarity and filter for the largest mean
    dissimilarities in the resulting DC matrix.

    The result mostly looks alright at first glance, but is heavily computation intense.
    """
    oneSegPerMsg = [MessageSegment(Value(msg), 0, len(msg.data)) for msg in specimens.messagePool.keys()]
    dc = DelegatingDC(oneSegPerMsg)
    # get packetcount largest mean dissimilarities from matrix -> msg-indices.
    preFilteredMsgs = [msg for msg, meanDiss in
        sorted(((msg, meanDiss) for msg, meanDiss in zip(dc.segments, dc.distanceMatrix.mean(axis=0))),
               key=lambda x: x[1])[-packetcount:]]
    # replace all templates by one of their base segments. The other messages are duplicates and should be removed.
    filteredMsgs = [msg.baseSegments[0] if isinstance(msg, Template) else msg for msg in preFilteredMsgs]
    return filteredMsgs

# noinspection PyShadowingNames
def cosineCommonalityFilter(refinedPerMsg: List[List[MessageSegment]], packetcount: int):
    """
    unfinished filter, originally intended to validate simpler filters.

    Calculates the similarity of messages by the cosine similarity from the segment commonalities used as feature
    vectors and filters the most dissimilar messages.

    Is comparably computation intense and, from the looks of it, seems to give a skewed result.
    """
    from sklearn.metrics.pairwise import cosine_similarity

    # all the non-zero-only segment
    valCounter = Counter(s.bytes for s in chain.from_iterable(refinedPerMsg) if set(s.bytes) != b"\x00")
    # mapping from segment values to an index for it and its global count
    valIdx = {val: (idx,cnt) for idx,(val,cnt) in enumerate(valCounter.most_common())}

    # represent global counts of values in a feature matrix
    vectors = numpy.zeros((len(refinedPerMsg), len(valCounter)))
    for mid,msg in enumerate(refinedPerMsg):
        for seg in msg:
            sid,cnt = valIdx[seg.bytes]
            vectors[mid,sid] = cnt

    cosim = cosine_similarity(vectors)

    # get packetcount lowest mean cosine similarities from matrix -> msg-indices.
    filteredMsgs = [MessageValueCommonality(meanDiss, msg) for msg, meanDiss in
        sorted(((msgSegs[0].message, meanSimi) for msgSegs, meanSimi in zip(refinedPerMsg, cosim.mean(axis=0))),
               key=lambda x: x[1])[-packetcount:]]

    return filteredMsgs

# noinspection PyShadowingNames
def valueCommonalityFilter(refinedPerMsg: List[List[MessageSegment]], packetcount: int):
    """
    Filter for the messages with the least common segments on average. The average is calculated by the median since it
    is sensitive to the extremes of singular segment values.
    """
    print("Count Segment values...")
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

    # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # sorted mean commonality of each messages' segments
    print("Determine messages' commonalities...")
    valueCommonalityPerMsg = list()  # type: List[MessageValueCommonality]
    for msg in refinedPerMsg:
        if len(msg) < 1:
            print("Message ignored, since empty?!")  # TODO investigate cause of error
            continue
        valCom = float(numpy.median([valCounter[seg.bytes] for seg in msg]))
        mvc = MessageValueCommonality(valCom, msg[0].message)
        bisect.insort(valueCommonalityPerMsg, mvc)

    # Deduplicate
    uniqueMsgs = OrderedDict()
    for valCom in valueCommonalityPerMsg:
        if valCom.message.data in uniqueMsgs:
            continue  # skip the existing packet
        uniqueMsgs[valCom.message.data] = valCom

    # the selected messages of most "uncommon messages"
    filteredMsgs = list(uniqueMsgs.values())[:packetcount]
    # filteredMsgs = list(uniqueMsgs.values())[-packetcount:]  # TODO
    return filteredMsgs


filterOptions = ["candis", "coscom", "valcom"]

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=
                                     'Filter a PCAP for the subset of packets that have the maximum difference to all '
                                     'and write these packets to a new file with name: '
                                     '[pcapfilename]_maxdiff-[packetcount].pcap')
    parser.add_argument('pcapfilename', help='pcapfilename')
    parser.add_argument('-l', '--layer', type=int, default=2,
                        help='Protocol layer relative to IP to consider. Default is 2 layers above IP '
                             '(typically the payload of a transport protocol).')
    parser.add_argument('-r', '--relativeToIP', default=False, action='store_true',
                        help="Interpret --layer relative to the IP layer.")
    parser.add_argument('-f', '--filter', choices=filterOptions, default=filterOptions[-1],
                        help="Filter to apply for optimizing the trace.")
    parser.add_argument('-p', '--packetcount', nargs='?', type= int,
                        help='packet count (default: {:d})'.format(PACKET_LIMIT), default=PACKET_LIMIT)
    parser.add_argument('-i', '--interactive', help='Show interactive plot instead of writing output to file and '
                                                    'open ipython prompt after finishing the analysis.',
                        action="store_true")
    args = parser.parse_args()

    pcapfilename = args.pcapfilename
    packetcount = args.packetcount

    if not isfile(pcapfilename):
        print('File not found: ' + pcapfilename)
        exit(1)

    infile,ext = splitext(pcapfilename)
    outfile = infile + "_maxdiff-" + args.filter + "-{:d}".format(packetcount) + ext  # TODO _maxdiff  _mindiff
    if exists(outfile):
        print('Output file exists: ' + outfile)
        exit(1)

    print("Loading", pcapfilename)
    # get segments from messages and their common values
    specimens = SpecimenLoader(pcapfilename, args.layer, args.relativeToIP)
    if args.filter in filterOptions[1:]:  # only the second two filters need segments.
        segmentsPerMsg = bcDeltaGaussMessageSegmentation(specimens, sigma)
        refinedPerMsg = baseRefinements(segmentsPerMsg)

    print("Filter messages...")
    filterduration = time.time()
    if args.filter == filterOptions[0]:
        filteredMsgs = canberraDissimFilter(specimens, packetcount)
    elif args.filter == filterOptions[1]:
        filteredMsgs = cosineCommonalityFilter(refinedPerMsg, packetcount)
    elif args.filter == filterOptions[2]:
        filteredMsgs = valueCommonalityFilter(refinedPerMsg, packetcount)
    else:
        raise RuntimeError(f"Unknown filter {args.filter} selected.")
    filterduration = time.time() - filterduration
    print(f"Filtered in {filterduration:.2f} s")
    filteredSpecimens = BaseLoader(  # also used for resolving messages from l4 to raw
        (fm.message for fm in filteredMsgs), (specimens.messagePool[fm.message] for fm in filteredMsgs),
        baselayer=specimens.getBaseLayerOfPCAP()
    )

    #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # Statistics about selected message types
    try:
        print("Get groundtruth from tshark...")
        comparator = MessageComparator(specimens, args.layer, args.relativeToIP)

        print("\nMessage types found in trace vs. filter result:")
        originalMsgtypes = Counter(
            (pm.protocolname, pm.messagetype) for pm in [pm for pm in comparator.parsedMessages.values()])
        filteredMsgtypes = Counter(
            (comparator.parsedMessages[fm].protocolname, comparator.parsedMessages[fm].messagetype)
            for fm in filteredSpecimens.messagePool.values())
        stats = [(*pm,c,filteredMsgtypes[pm]) for pm,c in originalMsgtypes.most_common()]
        headers = ["Protocol", "Message Type", "Original Count", "Filtered Count"]
        # write print and write statistic into csv
        print(tabulate(stats, headers=headers) + "\n")
        # outbase, ext = splitext(outfile)
        csvfile = join(reportFolder, "prep_filter-maxdiff_" + args.filter + ".csv")
        writeHead = True
        if exists(csvfile):
            print('CSV file exists: ' + csvfile, "\nAppending data.")
            writeHead = False
        with open(csvfile, "a") as cf:
            cw = csv.writer(cf)
            if writeHead:
                cw.writerow(headers)
            cw.writerows(stats)
    except (NotImplementedError, DissectionIncomplete) as e:
        print("Groundtruth not available for unknown protocol, comparison aborted.\n"
              "Original exception was: ", e)
    # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    #
    # write back the packets
    print("Re-read trace with scapy...")
    packetList = sy.rdpcap(pcapfilename)
    # # The order of packets is not constent for netzob's PCAPImporter and scapy's rdpcap, despite the following is true
    # sorted([a.date for a in specimens.messagePool.values()]) == sorted([a.time for a in packetList])
    packetMap = {bytes(packet): packet for packet in packetList}
    filteredPackets = [packetMap[rawmsg.data] for rawmsg in filteredSpecimens.messagePool.values()]
    sortedPackets = sorted(filteredPackets, key=lambda x: x.time)
    print("Write filtered trace to", outfile)
    sy.wrpcap(outfile, sortedPackets, linktype=specimens.getBaseLayerOfPCAP())

    if args.interactive:
        IPython.embed()
