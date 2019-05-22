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
from pprint import pprint

from netzob.Model.Vocabulary.Messages.AbstractMessage import AbstractMessage

from validation.dissectorMatcher import MessageComparator
from utils.loader import SpecimenLoader
from inference.analyzers import *
from inference.segments import TypedSegment
from inference.templates import DBSCANsegmentClusterer, DelegatingDC, FieldTypeMemento
from utils.evaluationHelpers import analyses, annotateFieldTypes


# fix the analysis method to VALUE
analysisTitle = 'value'
analyzerType = analyses[analysisTitle]
analysisArgs = None




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


class FieldTypeRecognizer(object):
    fieldtypeTemplates = [
        # # macaddr
        # FieldTypeMemento(numpy.array([0.0, 12.0, 41.0, 137.28571428571428, 123.17857142857143, 124.82142857142857]), numpy.array([0.0, 0.0, 0.0, 69.38446570834607, 77.84143845546744, 67.62293227439753]), numpy.array([[0.022103573766497436, 0.0046327614942338, -0.006504195407397688, 0.6591927243117947, -0.644376065770878, 0.9913785886639866], [0.0046327614942338, 0.021488881156709244, -0.001382945195792973, -1.1536744995748671, 5.150600064885553, 0.3555595274599619], [-0.006504195407397688, -0.001382945195792973, 0.026359085778108426, 1.0184065358245054, 0.6723490088910922, -0.3966508786750666], [0.6591927243117947, -1.1536744995748671, 1.0184065358245054, 4992.5079365079355, 746.2804232804233, -19.428571428571455], [-0.644376065770878, 5.150600064885553, 0.6723490088910922, 746.2804232804233, 6283.70767195767, 452.44047619047615], [0.9913785886639866, 0.3555595274599619, -0.3966508786750666, -19.428571428571455, 452.44047619047615, 4742.22619047619]]), 'macaddr', Value, (), MessageAnalyzer.U_BYTE) ,
        # # timestamp
        # FieldTypeMemento(numpy.array([205.75982532751092, 63.838427947598255, 24.375545851528383, 122.61135371179039, 133.65652173913043, 134.2608695652174, 147.52173913043478, 120.2695652173913]), numpy.array([25.70183334947299, 19.57641190835309, 53.50246711271789, 66.9864879316446, 71.84675313281585, 72.93261773499316, 72.72710729917598, 76.51925835899723]), numpy.array([[663.4815368114683, -472.49073010036244, -423.75589902704553, 139.7702826936337, 9.80247835746574, 3.6199532674480817, -242.52683291197417, 505.4362598636329], [-472.49073010036244, 384.91676243009425, 296.96445261625695, -111.35254347659536, 75.25739293648974, -4.666034628054849, 166.73218800275805, -345.0736612273047], [-423.75589902704553, 296.96445261625695, 2875.068873056001, 318.1860683367806, 261.37767179958604, 84.96889603922475, 157.02409407798976, -593.1297594422736], [139.7702826936337, -111.35254347659536, 318.1860683367806, 4506.870221405036, 442.0494330805179, -212.5338044893896, -498.76463265149766, 382.4062667585993], [9.80247835746574, 75.25739293648974, 261.37767179958604, 442.0494330805179, 5184.497228023539, -456.02790962597265, -181.6759065881906, 370.75238276058485], [3.6199532674480817, -4.666034628054849, 84.96889603922475, -212.5338044893896, -456.02790962597265, 5342.394531991645, -240.0144294664894, 118.43155496487557], [-242.52683291197417, 166.73218800275805, 157.02409407798976, -498.76463265149766, -181.6759065881906, -240.0144294664894, 5312.329219669637, -311.37269793051064], [505.4362598636329, -345.0736612273047, -593.1297594422736, 382.4062667585993, 370.75238276058485, 118.43155496487557, -311.37269793051064, 5880.76544522499]]), 'timestamp', Value, (), MessageAnalyzer.U_BYTE) ,
        # # id
        # FieldTypeMemento(numpy.array([122.91489361702128, 122.76595744680851, 136.2340425531915, 116.25531914893617]), numpy.array([69.99721264579163, 69.05662578282282, 74.7862806755008, 81.35382873860368]), numpy.array([[5006.123034227568, 1258.0448658649402, -284.06660499537475, 412.17437557816834], [1258.0448658649402, 4872.487511563368, -406.5527289546716, 674.6262719703982], [-284.06660499537475, -406.5527289546716, 5714.574468085107, 456.6345975948195], [412.17437557816834, 674.6262719703982, 456.6345975948195, 6762.324699352451]]), 'id', Value, (), MessageAnalyzer.U_BYTE) ,
        # float
        FieldTypeMemento(numpy.array([0.0, 2., 24., 113.]),
                         numpy.array([0.0, 0.7060598835273263, 16.235663701838636, 76.9843433664198]), numpy.array(
                [[0.021138394040146703, 0.018424928271229466, 0.2192364920149479, 0.32239294646470784],
                 [0.018424928271229466, 0.5036075036075054, 0.03442589156874873, 7.556380127808698],
                 [0.2192364920149479, 0.03442589156874873, 266.28653885796797, -84.0164914450628],
                 [0.32239294646470784, 7.556380127808698, -84.0164914450628, 5987.06452277881]]), 'float', Value, (),
                         MessageAnalyzer.U_BYTE),
        # # FieldTypeMemento(numpy.array([0.0, 0.08080808080808081, 23.828282828282827, 113.43434343434343]), numpy.array([0.0, 0.7060598835273263, 16.235663701838636, 76.9843433664198]), numpy.array([[0.021138394040146703, 0.018424928271229466, 0.2192364920149479, 0.32239294646470784], [0.018424928271229466, 0.5036075036075054, 0.03442589156874873, 7.556380127808698], [0.2192364920149479, 0.03442589156874873, 266.28653885796797, -84.0164914450628], [0.32239294646470784, 7.556380127808698, -84.0164914450628, 5987.06452277881]]), 'float', Value, (), MessageAnalyzer.U_BYTE) ,
        # # # int
        # # FieldTypeMemento(numpy.array([0.0, 0.0, 107.29166666666667]), numpy.array([0.0, 0.0, 47.78029507257382]), numpy.array([[0.027493310152260777, -0.005639908063087656, -0.07922934857447499], [-0.005639908063087656, 0.018914313768380903, -2.358032476778992], [-0.07922934857447499, -2.358032476778992, 2382.215579710145]]), 'int', Value, (), MessageAnalyzer.U_BYTE) ,
        # FieldTypeMemento(numpy.array([0.0, 0.0, 4.75, 127.84375]), numpy.array([0.0, 0.0, 4.743416490252569, 39.259640038307786]), numpy.array([[0.021960602414756426, 0.0007935582831639521, -0.22956507460690648, 0.6994209166124543], [0.0007935582831639521, 0.017292996674435825, -0.06886282798507015, 1.917984540011557], [-0.22956507460690648, -0.06886282798507015, 23.225806451612904, -30.427419354838708], [0.6994209166124543, 1.917984540011557, -30.427419354838708, 1591.039314516129]]), 'int', Value, (), MessageAnalyzer.U_BYTE) ,
        # # ipv4
        # FieldTypeMemento(numpy.array([172.0, 18.928571428571427, 2.3095238095238093, 26.071428571428573]), numpy.array([0.0, 1.4374722712498649, 0.6721711530234811, 24.562256039881753]), numpy.array([[0.022912984623871414, 0.03544250639873077, -0.0004582497404991295, -0.4596803115368301], [0.03544250639873077, 2.1167247386759582, 0.022648083623693426, 2.2491289198606266], [-0.0004582497404991295, 0.022648083623693426, 0.4628339140534262, 10.19686411149826], [-0.4596803115368301, 2.2491289198606266, 10.19686411149826, 618.0191637630664]]), 'ipv4', Value, (), MessageAnalyzer.U_BYTE) ,
    ]

    def __init__(self, analyzer: MessageAnalyzer):
        self._analyzer = analyzer


    @property
    def message(self):
        return self._analyzer.message


    def findInMessage(self, fieldtypeTemplate: FieldTypeMemento):
        """

        :param fieldtypeTemplate:
        :return: list of (position, confidence) for all offsets.
        """
        from scipy.spatial import distance
        assert fieldtypeTemplate.analyzer == type(self._analyzer)

        # position, confidence
        posCon = list()
        ftlen = len(fieldtypeTemplate.mean)
        for offset in range(len(self._analyzer.values) - ftlen):
            ngram = self._analyzer.values[offset:offset+ftlen]
            if set(ngram) == {0}:  # zero values do not give any information
                posCon.append(99)
            else:
                confidence = distance.mahalanobis(fieldtypeTemplate.mean, ngram, fieldtypeTemplate.picov)
                posCon.append(confidence)

        return posCon


    def recognizedFields(self, confidenceThreshold = 2):
        """
        Most probable inferred field structure: The field template positions with the highest confidence for a match.
        TODO How to decide which of any overlapping fields should be the recognized one?
        TODO How to decide which confidence value is high enough to assume a match if no concurring comparison values
            ("no relevant matches") are in this message?
        :return:
        """
        mostConfident = dict()
        for ftMemento in FieldTypeRecognizer.fieldtypeTemplates:
            confidences = self.findInMessage(ftMemento)
            mostConfident[ftMemento] = [(pos, con) for pos, con in enumerate(confidences) if con < confidenceThreshold]
        return mostConfident


def calConfidence(ftMemento: FieldTypeMemento, ftRecognizer: FieldTypeRecognizer):
    msgsegs = segmentedMessages[ftRecognizer.message]  # type: Tuple[TypedSegment]
    offsets = [msgseg.offset for msgseg in msgsegs if msgseg.fieldtype == ftMemento.fieldtype]
    confidences = ftRecognizer.findInMessage(ftMemento)
    if confidences:
        mostConfident = int(numpy.argmin(confidences))
        # if confidences[mostConfident] < 0.1:
        print(" ".join(["|" if i in offsets else " " for i in range(len(ftRecognizer.message.data))]))
        print(ftRecognizer.message.data.hex())
        print(" "*(2*mostConfident), bytes(ftMemento.mean.round().astype(int).tolist()).hex(), sep="")
        print(tabulate([offsets, [confidences[o] for o in offsets]]))
        print("mostConfident", mostConfident, "{:.3f}".format(confidences[mostConfident]))
        print()
    return confidences




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
    else:
        # fetch ground truth
        comparator = MessageComparator(specimens, 2, True, debug=False)
        segmentedMessages = {msgseg[0].message: msgseg
                             for msgseg in annotateFieldTypes(analyzerType, analysisArgs, comparator)
                             }
        # get analysis values
        analyzers = [MessageAnalyzer.findExistingAnalysis(analyzerType, MessageAnalyzer.U_BYTE, l4msg, analysisArgs)
                     for l4msg in specimens.messagePool.keys()]

        # TODO search for persisted fieldtypeTemplates in specimens
        ftRecognizer = list()
        for analyzer in analyzers:
            ftRecognizer.append(FieldTypeRecognizer(analyzer))

        # test per message with all FieldTypeRecognizers
        recognized = list()
        # for msgids in (189, 243, 333, 262):
        for msgids in range(500):
            ftr = ftRecognizer[msgids]
            recognized.append((ftr.message, ftr.recognizedFields()))

        truePos = dict()
        falsePos = dict()
        for msg, ftmposcon in recognized:
            truePos[msg] = dict()
            falsePos[msg] = dict()
            for ftm, poscon in ftmposcon.items():
                ftyOffsets = {msgseg.offset for msgseg in segmentedMessages[msg] if msgseg.fieldtype == ftm.fieldtype}
                recOffsets = {pos for pos, con in poscon}

                truePos[msg][ftm] = ftyOffsets.intersection(recOffsets)
                falsePos[msg][ftm] = recOffsets.difference(ftyOffsets)

        from visualization.simplePrint import tabuSeqOfSeg

        for msg, ftmposcon in recognized:
            fpftmoff = falsePos[msg]
            tpftmoff = truePos[msg]

            if all([len(fpoff) == 0 for fpoff in fpftmoff.values()]) \
                    and all([len(fpoff) == 0 for fpoff in tpftmoff.values()]):
                continue

            print("\n=========================")
            tabuSeqOfSeg([segmentedMessages[msg]])

            for ftm, offs in fpftmoff.items():
                ftmlen = len(ftm.mean)
                poscon4ftm = ftmposcon[ftm]

                print("=========================")
                print("true  ft:", ftm.fieldtype)
                print("ftm mean:", bytes(ftm.mean.astype(int).tolist()).hex())
                if len(offs) == 0:
                    print("no false positives")
                    continue
                else:
                    print("false positives:")

                for off in offs:
                    overlapSegs = [(msgseg.offset, msgseg.fieldtype) for msgseg in segmentedMessages[msg]
                                   if off <= msgseg.offset < off+ftmlen]
                    con4off = [con for pos,con in poscon4ftm if pos == off][0]
                    print("  ", off, msg.data[off:off+ftmlen].hex(), "{:.3f}".format(con4off))
                    print("  ", overlapSegs)
                    print()


            # pprint(segmentedMessages[msg])
            for ftm, offs in tpftmoff.items():
                ftmlen = len(ftm.mean)
                poscon4ftm = ftmposcon[ftm]

                print("=========================")
                print("true  ft:", ftm.fieldtype)
                print("ftm mean:", bytes(ftm.mean.astype(int).tolist()).hex())
                if len(offs) == 0:
                    print("no true positives")
                    continue
                else:
                    print("true positives:")

                for off in offs:
                    overlapSegs = [(msgseg.offset, msgseg.fieldtype) for msgseg in segmentedMessages[msg]
                                   if off <= msgseg.offset < off+ftmlen]
                    if overlapSegs:
                        con4off = [con for pos,con in poscon4ftm if pos == off][0]
                        print("  ", off, msg.data[off:off+ftmlen].hex(), "{:.3f}".format(con4off))
                        print("  ", overlapSegs)
                    print()

        # # get segmented message for first recognition
        # segmentedMessages[ftRecognizer[0].message]






        # # test per FieldTypeTemplate for single messages
        # for ftm in FieldTypeRecognizer.fieldtypeTemplates:
        #     print(ftm.fieldtype)
        #     # near and far messages for int 189 243
        #     # near and far messages for ipv4 333 262
        #     # for ftr in ftRecognizer[:100]:  # TODO for testing some
        #     for msgids in (189,243,333,262):
        #         ftr = ftRecognizer[msgids]
        #         calConfidence(ftm, ftr)






    if args.interactive:
        print("\n")
        IPython.embed()



