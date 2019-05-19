"""
Recognize fields by subsequence frequency analysis
(and recognize field types by templates)
"""

import argparse

from os.path import isfile, basename
from typing import Tuple, Dict, List
import numpy
from itertools import chain
from collections import Counter
from tabulate import tabulate
import IPython
from kneed import KneeLocator
import matplotlib.pyplot as plt

from netzob.Model.Vocabulary.Messages.AbstractMessage import AbstractMessage

from inference.analyzers import *
from utils.loader import SpecimenLoader
from inference.templates import DBSCANsegmentClusterer, DelegatingDC, FieldTypeMemento

class BIDEracker(object):
    """
    Dynamic occurrance tracking BIDE-based closed sequence mining.
    """
    def __init__(self, messages, percentile=80):
        """

        :param percentile: The percentile of occurrances per prefix-extension interation as threshold for being frequent.
        """
        self._percentile = percentile
        self._occurranceDrop = 60
        # length: subsequence: (count, message: offset)  (last dict is "support" of length - 1)
        self._subsequenceLookup = dict()
        self.fillL1(messages)
        self.printExtensions()
        self.iterateExtensions()


    def countAllOccurences(self):
        allcounts = {byteValue: count for freqSeq in self._subsequenceLookup.values() for byteValue, (count, occurrences) in freqSeq.items()}
        return sorted(allcounts.items(), key=lambda x: x[1])


    def fillL1(self, messages):
        # fill L1 (all one-byte sequences and their positions)
        self._subsequenceLookup[1] = dict()
        for message in messages:
            for offset, intValue in enumerate(message.data):
                byteValue = bytes([intValue])
                if not byteValue in self._subsequenceLookup[1]:
                    self._subsequenceLookup[1][byteValue] = [0, dict()]
                if not message in self._subsequenceLookup[1][byteValue][1]:
                    self._subsequenceLookup[1][byteValue][1][message] = list()
                self._subsequenceLookup[1][byteValue][0] += 1  # count
                self._subsequenceLookup[1][byteValue][1][message].append(offset)  # location

        # # initial pruning
        # allocc = countAllOccurences(subsequenceLookup)
        # # print([count for byteValue, count in allocc])
        # knee = numpy.percentile([count for byteValue, count in allocc], 66)
        # print("k", 1, "knee", knee)
        # for byteValue in list(subsequenceLookup[1]):  # since we change the dict, iter over copy of keys
        #     count, occurrences = subsequenceLookup[1][byteValue]
        #     if knee > count:  # threshold for being frequent
        #         # remove subsequence just added in k if it is not frequent
        #         del subsequenceLookup[1][byteValue]


    def iterateExtensions(self):
        k = 2
        # iterate ks as long as there are common sequences with length k
        while len(self._subsequenceLookup[k - 1]) > 0:
            self._subsequenceLookup[k] = dict()

            # determine k's min support by counting all occurrences
            allocc = self.countAllOccurences()
            knee = numpy.percentile([count for byteValue, count in allocc], 80)
            print("k", k, "knee", knee)

            # if 8 < k < 12:
            #     print("next iteration")
            #     print(tabulate([(bv.hex(), co, lok) for lok, bco in self._subsequenceLookup.items()
            #                     for bv, (co, oc) in bco.items() if bv[:2] == b"\x81\x82"]))
            #
            everythingfrequent = {(message, o): locK for locK, freqSeq in self._subsequenceLookup.items() for
                                  count, occurrences in freqSeq.values()
                                  for message, offsets in occurrences.items() for o in offsets if
                                  knee <= count}  # threshold for being frequent
            # sortedoccurences = sorted(subsequenceLookup[k - 1].items(), key=lambda x: x[1][0],
            #                           reverse=True)  # key: bytevalue's count

            # extend frequent prefixes known from k-1
            # for byteValue, (count, occurrences) in sortedoccurences:  # type: bytes, (int, Dict[AbstractMessage, List[int]])
            for message, offset in everythingfrequent.keys():  # search for all the frequent strings in k's supporters
                # for message, offsets in occurrences.items():  # search for all the frequent strings in k's supporters
                #     for o in sorted(offset):
                if len(message.data) < offset + k \
                        or (message, offset) not in everythingfrequent \
                        or not any((message, k_1o) in everythingfrequent and everythingfrequent[(message, k_1o)] >= k_1l
                               for k_1o, k_1l in zip(range(offset+1, offset + k + 1), range(k-1, 0, -1))):
                    # message does not contain an extension of prefex at position o for sequence length k
                    #  ... or no frequent extension
                    continue

                byteValue = message.data[offset:offset + k]
                if not byteValue in self._subsequenceLookup[k]:
                    self._subsequenceLookup[k][byteValue] = [0, dict()]
                if not message in self._subsequenceLookup[k][byteValue][1]:
                    self._subsequenceLookup[k][byteValue][1][message] = list()
                # add all frequent occurrences in k's supporters
                self._subsequenceLookup[k][byteValue][0] += 1  # count
                self._subsequenceLookup[k][byteValue][1][message].append(offset)  # location

            # print(tabulate(sorted([(cn, bv.hex()) for bv, (cn, sp) in subsequenceLookup[k].items()], key=lambda x: x[0])))

            # pruning
            # all of the new sequences that are frequent will cause the occurrences in their supporters to be removed
            if len(self._subsequenceLookup[k]) > 0:
                for byteValue, (count, occurrences) in self._subsequenceLookup[k].items():
                    # print(count, byteValue)
                    for message, offsets in occurrences.items():
                        for ofs in offsets:
                            assert message.data[ofs:ofs + k] == byteValue

                # newlyfrequent = {(message, o): k for count, occurrences in self._subsequenceLookup[k].values()
                #                  for message, offsets in occurrences.items() for o in offsets if
                #                  knee <= count}  # threshold for being frequent

                # iterate all frequent sequences newly found in k
                for byteValue in list(self._subsequenceLookup[k].keys()):  # since we change the dict, iter over copy of keys
                    count, occurrences = self._subsequenceLookup[k][byteValue]
                    # threshold for being absolutely frequent
                    if knee > count:
                        # remove subsequence just added in k if it is not frequent
                        del self._subsequenceLookup[k][byteValue]
                        continue  # k's sequence to be infrequent causes its support in k-1 to remain valid (and potentially frequent)
                    # if byteValue[:2] == b"\x63\x82\x53\x63":
                    #     print(count, byteValue.hex())

                    # ... and for being locally frequent (prefix + extension)
                    for ext in range(1, k):
                        # print(byteValue[:-ext], ext)
                        if byteValue[:-ext] in self._subsequenceLookup[k-ext]:
                            if count < self._subsequenceLookup[k-ext][byteValue[:-ext]][0] * self._occurranceDrop / 100:
                                # remove subsequence just added in k if it is not frequent
                                if byteValue in self._subsequenceLookup[k]:
                                    del self._subsequenceLookup[k][byteValue]
                                # if byteValue[:2] == b"\x81\x82":
                                #    print("remove", byteValue)
                                continue
                            break
                        else:
                            pass
                            # print(byteValue[:-1].hex(), "not in", k, "-1")

                    # prefix = byteValue[:-1]
                    # find all occurences in all supporters in k-1 for k's frequent string and remove from k-1
                    for message, offsets in occurrences.items():
                        for o in sorted(offsets):
                            # print("newly:", message.data[o:o+k])
                            # remove only if frequent from supporter is completely contained in new frequent
                            #   meaning: remove all occurences in any supporters for any k_1o >= o and k_1l < k
                            for k_1o, k_1l in zip(range(o, o + k + 1), range(k, 0, -1)):
                                if (message, k_1o) in everythingfrequent and everythingfrequent[
                                    (message, k_1o)] <= k_1l:
                                    locK = everythingfrequent[(message, k_1o)]
                                    byVa = message.data[k_1o:k_1o + locK]
                                    # print("rem sup occ:", k_1o, " - ", message.data[k_1o:k_1o+locK])
                                    # print(subsequenceLookup[locK][byVa][1][message])
                                    if k_1o in self._subsequenceLookup[locK][byVa][1][message]:
                                        self._subsequenceLookup[locK][byVa][1][message].remove(k_1o)
                                    self._subsequenceLookup[locK][byVa][0] -= 1
                #
                # if 8 < k < 12:
                #     print("after pruning")
                #     print(tabulate([(bv.hex(), co, lok) for lok, bco in self._subsequenceLookup.items()
                #                     for bv, (co, oc) in bco.items() if bv[:2] == b"\x81\x82"]))
                #
                # prune k-1's supporters which's occurrence count have fallen to zero as they have been accounted for in k (in the previous loop)
                for locK, bco in self._subsequenceLookup.items():
                    if locK == k:
                        continue
                    for byteValue in list(bco):  # since we change the dict, iter over copy of keys
                        count, occurrences = self._subsequenceLookup[locK][byteValue]
                        if count <= 0:  # may be negative is multiple longer frequent strings have had this one as supporting occurence
                            del self._subsequenceLookup[locK][byteValue]
                        else:
                            for message in list(occurrences):
                                if len(occurrences[message]) == 0:
                                    del self._subsequenceLookup[locK][byteValue][1][message]
            k += 1

    def printExtensions(self, start=1):
        for kLoc in range(start, max(self._subsequenceLookup.keys())):
            print("### k:", kLoc)
            print(tabulate(
                sorted([(cn, bv.hex()) for bv, (cn, sp) in self._subsequenceLookup[kLoc].items()], key=lambda x: x[0])))


    def mostFrequent(self, start=1):
        # bytevalue: (count, occurences)
        retVal = dict()
        for kLoc in range(start, max(self._subsequenceLookup.keys())):
            for bv in self._subsequenceLookup[kLoc].keys():
                if bv in retVal:
                    print("collision!")
            retVal.update(self._subsequenceLookup[kLoc].items())
        return retVal






if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Recognize fields by subsequence frequency analysis.')
    parser.add_argument('pcapfilename', help='pcapfilename') # , nargs='+')
    parser.add_argument('-i', '--interactive', help='show interactive plot instead of writing output to file.',
                        action="store_true")
    parser.add_argument('--count', '-c', help='Count common values of features.',
                        action="store_true")
    args = parser.parse_args()

    if not isfile(args.pcapfilename):
        print('File not found: ' + args.pcapfilename)
        exit(1)

    pcapbasename = basename(args.pcapfilename)

    print("Loading messages...")
    specimens = SpecimenLoader(args.pcapfilename, layer=2, relativeToIP=True)


    if args.count:
        subsfreq = BIDEracker(specimens.messagePool.keys())

        # subsfreq.printExtensions(2)
        print(tabulate(sorted([(cnt, bv.hex()) for bv, (cnt, occ) in subsfreq.mostFrequent(2).items()])))

    # TODO search for persisted fieldtypeTemplates in specimens
    fieldtypeTemplates = [
        FieldTypeMemento(numpy.array([0.0, 0.0, 107.29166666666667]), numpy.array([0.0, 0.0, 47.78029507257382]), 'int',
                         Value, (), MessageAnalyzer.U_BYTE),
        FieldTypeMemento(numpy.array([0.0, 0.0, 4.75, 127.84375]),
                         numpy.array([0.0, 0.0, 4.743416490252569, 39.259640038307786]), 'int', Value, (),
                         MessageAnalyzer.U_BYTE),
        FieldTypeMemento(numpy.array(
            [204.86521739130436, 63.56086956521739, 24.269565217391303, 122.07826086956521, 133.07792207792207,
             133.67965367965368, 146.88311688311688, 119.74891774891775]), numpy.array(
            [28.9997617485814, 19.980278083333324, 53.41011494157094, 67.32576841468352, 72.22609394647127,
             73.30645791589188, 73.21296233242103, 76.760645502516]), 'timestamp', Value, (), MessageAnalyzer.U_BYTE),
        FieldTypeMemento(numpy.array([0.0, 0.08080808080808081, 23.828282828282827, 113.43434343434343]),
                         numpy.array([0.0, 0.7060598835273263, 16.235663701838636, 76.9843433664198]), 'float', Value,
                         (), MessageAnalyzer.U_BYTE),
        FieldTypeMemento(numpy.array([172.0, 18.928571428571427, 2.3095238095238093, 26.071428571428573]),
                         numpy.array([0.0, 1.4374722712498649, 0.6721711530234811, 24.562256039881753]), 'ipv4', Value,
                         (), MessageAnalyzer.U_BYTE),
        FieldTypeMemento(numpy.array([122.91489361702128, 122.76595744680851, 136.2340425531915, 116.25531914893617]),
                         numpy.array([69.99721264579163, 69.05662578282282, 74.7862806755008, 81.35382873860368]), 'id',
                         Value, (), MessageAnalyzer.U_BYTE),
        FieldTypeMemento(numpy.array(
            [0.0, 11.586206896551724, 39.58620689655172, 132.55172413793105, 118.93103448275862, 120.51724137931035]),
                         numpy.array([0.0, 2.1895872919155233, 7.481089914044711, 72.63398378497232, 79.721468157183,
                                      70.24176019398028]), 'macaddr', Value, (), MessageAnalyzer.U_BYTE),
    ]


    if args.interactive:
        IPython.embed()



