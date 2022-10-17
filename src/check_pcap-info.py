"""
Parse PCAP, print some statistics and infos about it, and open an IPython shell.
"""
from itertools import chain
from typing import List, Sequence

import IPython, numpy
from argparse import ArgumentParser
from os.path import isfile, basename
from tabulate import tabulate

from nemere.utils.loader import SpecimenLoader


def countByteFrequency():
    from collections import Counter
    from itertools import chain

    bytefreq = Counter(chain.from_iterable(msg.data for msg in specimens.messagePool.keys()))
    return bytefreq.most_common()


def meanByteDiff(messages: Sequence) -> List[List[float]]:
    return [[numpy.diff(list(msg.data)).mean()] for msg in messages]


if __name__ == '__main__':
    parser = ArgumentParser(
        description='arse PCAP, print some statistics and infos about it and open a IPython shell.')
    parser.add_argument('pcapfilename', help='pcapfilename')
    parser.add_argument('-l', '--targetlayer', type=int)
    parser.add_argument('-r', '--relativeToIP', default=False, action='store_true')
    parser.add_argument('-i', '--interactive', help='Open IPython prompt after output of the PCAP infos.',
                        action="store_true")
    args = parser.parse_args()
    if not isfile(args.pcapfilename):
        print('File not found: ' + args.pcapfilename)
        exit(1)
    if args.targetlayer:
        specimens = SpecimenLoader(args.pcapfilename, args.targetlayer, args.relativeToIP)
    else:
        specimens = SpecimenLoader(args.pcapfilename)
    # pkt = list(specimens.messagePool.keys())

    print("Filename:", basename(args.pcapfilename))
    print("PCAP base layer is:", specimens.getBaseLayerOfPCAP())
    print("Longest message without its encapsulation:", specimens.maximumMessageLength)
    print("Sum of message payload bytes:", specimens.cumulatedMessageLength)
    print("Most frequent byte values:")
    print(tabulate(
        ((hex(b), o) for b, o in countByteFrequency()[:10])
        , headers=["byte value", "occurrences"]))
    print("Mean difference between bytes per message:",
          numpy.mean(list(chain.from_iterable(meanByteDiff(specimens.messagePool.keys())))))
    # print(tabulate(meanByteDiff(specimens.messagePool.keys())))
    print()

    if args.interactive:
        print('Loaded PCAP in: specimens')
        # start python shell
        IPython.embed()

