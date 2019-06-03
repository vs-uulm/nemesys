"""
Recognize fields by subsequence frequency analysis
(and recognize field types by templates)
"""

import argparse
from collections import OrderedDict

from os.path import isfile, basename
from tabulate import tabulate
import IPython

from validation.dissectorMatcher import MessageComparator
from utils.loader import SpecimenLoader
from inference.analyzers import *
from inference.segments import TypedSegment
from inference.fieldTypes import FieldTypeMemento, FieldTypeRecognizer, FieldTypeQuery, RecognizedField
from utils.evaluationHelpers import analyses, annotateFieldTypes
from visualization.simplePrint import segmentFieldTypes, tabuSeqOfSeg, printFieldContext
import visualization.bcolors as bcolors

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


def evaluateCharIDOverlaps(ftQueries: [FieldTypeQuery]):
    """
    Recognized ID fields often overlap chars and themselves. Therefore, existing IDs at (most of) a char sequence
    could be used as heuristic to improve confidence for chars.
    This needs more effort to evaluate and utilizer, thus, currently its not used.

    :param ftQueries:
    :return:
    """
    # # find all RecognizedFields that overlap with any chars. per offset
    # [(recogf.message,
    #   [ftq.retrieve4position(o) for o in range(recogf.position, recogf.end) if ftq.retrieve4position(o) != recogf])
    #  for ftq in ftQueries for recogf in ftq.resolveConflicting() if recogf.template == "chars"]

    # find all RecognizedFields that overlap with any chars. per message
    charOverlaps = list()
    for enf, ftq in enumerate(ftQueries):
        for recogf in ftq.resolveConflicting():
            if recogf.template.fieldtype == "chars":
                msgEntry = (enf, recogf, set())  # (ftq.message, recogf, list())
                for o in range(recogf.position, recogf.end):
                    recogsAtPos = ftq.retrieve4position(o)
                    for rap in recogsAtPos:
                        if rap != recogf:
                            msgEntry[2].add(rap)
                charOverlaps.append(msgEntry)

    # filter all the ids from the RecognizedFields that overlap with any chars
    charOverlappingIds = [(mid, rcgchr, [rf for rf in rcgflst if
                                         isinstance(rf.template, FieldTypeMemento) and rf.template.fieldtype == "id"])
                          for mid, rcgchr, rcgflst in charOverlaps]

    # positions in char sequences not overlapping with a recognized id field
    nococharpos = [(mid, rcgchr, [o for o in range(rcgchr.position, rcgchr.end) if
                                  o not in {offs for rf in rcgflst for offs in range(rf.position, rf.end)}]) for
                   mid, rcgchr, rcgflst in charOverlappingIds]

    print(charOverlappingIds)
    print("=================")
    print(nococharpos)


def printMarkedBytesInMessage(message: AbstractMessage, markStart, markEnd, subStart=0, subEnd=None):
    if subEnd is None:
        subEnd = len(message.data)
    assert markStart >= subStart
    assert markEnd <= subEnd
    sub = message.data[subStart:subEnd]
    relMarkStart = markStart-subStart
    relMarkEnd = markEnd-subStart
    colored = \
        sub[:relMarkStart].hex() + \
        bcolors.colorizeStr(
            sub[relMarkStart:relMarkEnd].hex(),
            10
        ) + \
        sub[relMarkEnd:].hex()
    print(colored)


def printRecogInSeg(recSegTup: Tuple[RecognizedField, TypedSegment]):
    printMarkedBytesInMessage(recSegTup[0].message,
                              recSegTup[0].position, recSegTup[0].end,
                              min(recSegTup[1].offset, recSegTup[0].position),
                              max(recSegTup[1].nextOffset, recSegTup[0].end))
    print(recSegTup[1].fieldtype)


# # test messages
# near and far messages for int 189 243
# near and far messages for ipv4 333 262
testMessages = (189, 243, 333, 262)


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
        print("Fetch ground truth...")
        comparator = MessageComparator(specimens, 2, True, debug=False)
        segmentedMessages = {msgseg[0].message: msgseg
                             for msgseg in annotateFieldTypes(analyzerType, analysisArgs, comparator)
                             }
        # get analysis values
        analyzers = [MessageAnalyzer.findExistingAnalysis(analyzerType, MessageAnalyzer.U_BYTE, l4msg, analysisArgs)
                     for l4msg in specimens.messagePool.keys()]

        # prepare persisted fieldtypeTemplates as FieldTypeRecognizers for each message
        ftRecognizer = list()
        for analyzer in analyzers:
            ftRecognizer.append(FieldTypeRecognizer(analyzer))

        # recognize fieldtypeTemplates per message with all known FieldTypeRecognizers
        print("Recognize fields...")
        recognized = OrderedDict()
        # for msgids in testMessages:
        for msgids in range(len(ftRecognizer)):
            ftr = ftRecognizer[msgids]
            recognized[ftr.message] = ftr.recognizedFields()



        # # # # # # # # # # # # # # # # # # # # # # # # # # #
        # # test per FieldTypeTemplate for single messages
        # for ftm in FieldTypeRecognizer.fieldtypeTemplates:
        #     print(ftm.fieldtype)
        #     # near and far messages for int 189 243
        #     # near and far messages for ipv4 333 262
        #     # for ftr in ftRecognizer[:100]:  # for testing some
        #     for msgids in (189,243,333,262):
        #         ftr = ftRecognizer[msgids]
        #         calConfidence(ftm, ftr)

        # # # # # # # # # # # # # # # # # # # # # # # # # # #
        # # Determine and print true and false positive matches
        # noinspection PyUnreachableCode
        if False:
            truePos = dict()
            falsePos = dict()
            for msg, ftmposcon in recognized:
                truePos[msg] = dict()
                falsePos[msg] = dict()
                for ftm, poscon in ftmposcon.items():
                    ftyOffsets = {msgseg.offset for msgseg in segmentedMessages[msg] if msgseg.fieldtype == ftm.fieldtype}
                    recOffsets = {pos for pos, con in poscon}

                    # TODO take field length == ftm length into account
                    truePos[msg][ftm] = ftyOffsets.intersection(recOffsets)
                    falsePos[msg][ftm] = recOffsets.difference(ftyOffsets)

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
        # # # # # # # # # # # # # # # # # # # # # # # # # # #

        # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        # # # Print recognized field type templates per message
        # for msg, ftmposcon in recognized.items():
        #     segmentFieldTypes(segmentedMessages[msg], ftmposcon)
        #     print()
        # # # # # # # # # # # # # # # # # # # # # # # # # # # #

        # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        # one specific
        # a = 92; segmentFieldTypes(segmentedMessages[recognized[a][0]], recognized[a][1])
        # one test message to query
        # ftq = FieldTypeQuery(ftRecognizer[-1])
        # # # # # # # # # # # # # # # # # # # # # # # # # # # #




        # # # # # # # # # # # # # # # # # # # # # # # #
        ftQueries = [FieldTypeQuery(ftr) for ftr in ftRecognizer]
        # evaluateCharIDOverlaps(ftQueries)
        # # # # # # # # # # # # # # # # # # # # # # # #

        # # # # # # # # # # # # # # # # # # # # # # # #
        # sum of true and false posititves count across all messages for each type
        matchStatistics = dict()
        for query in ftQueries:
            for ftName, ftStats in query.matchStatistics(segmentedMessages[query.message]).items():
                if not ftName in matchStatistics:
                    matchStatistics[ftName] = ([], [], [])
                # (truePos, falsePos, falseNeg)
                matchStatistics[ftName][0].extend(ftStats[0])
                matchStatistics[ftName][1].extend(ftStats[1])
                matchStatistics[ftName][2].extend(ftStats[2])
        mstattab = [(ftName, len(matchStatistics[ftName][0]),
                     len(matchStatistics[ftName][1]), len(matchStatistics[ftName][2]))
                    for ftName in sorted(matchStatistics.keys())]
        print()
        print(tabulate(mstattab, headers=("ftName", "truePos", "falsePos", "falseNeg")))
        print()
        # # compare true and recognized fieldtypes of false positives.
        # [(a.template.fieldtype, b.fieldtype, a.position, b.offset) for a, b in matchStatistics["id"][1]]
        # [(b.fieldtype, a.position, b.offset, b.bytes.hex()) for a, b in matchStatistics["id"][1]]
        # # # # # # # # # # # # # # # # # # # # # # # #





        # # # # # # # # # # # # # # # # # # # # # # # #
        # falsePositiveIDs = sorted([idfp for idfp in matchStatistics["id"][1]], key=lambda o: o[0].confidence)
        # for fpid, seg in falsePositiveIDs:
        #     printFieldContext(segmentedMessages, fpid)
        #
        # # the confidence of
        # timestamp = FieldTypeRecognizer.fieldtypeTemplates[1]
        # # at position
        # offsAtFpO = falsePositiveIDs[0][1].offset
        # # that overlaps with
        # recogAtFpO = falsePositiveIDs[0][0]
        # # which became the recognized field type with confidence 0.98
        # msg4FpO = next((ftr for ftr in ftRecognizer if ftr.message == recogAtFpO.message), None)
        # confAtFpO = msg4FpO.findInMessage(timestamp)[offsAtFpO]
        # # is 1.48
        # # # # # # # # # # # # # # # # # # # # # # # #

        # # # # # # # # # # # # # # # # # # # # # # # #
        falsePositiveFlags = sorted([flagsfp for flagsfp in matchStatistics["flags"][1]], key=lambda o: o[0].confidence)
        for fpFlags, seg in falsePositiveFlags[:20]:
            printFieldContext(segmentedMessages, fpFlags)

        # the confidence of
        timestamp = FieldTypeRecognizer.fieldtypeTemplates[1]
        # at position
        offsAtFpO = falsePositiveFlags[0][1].offset
        # that overlaps with
        recogAtFpO = falsePositiveFlags[0][0]
        # which became the recognized field type with confidence 0.98
        msg4FpO = next((ftr for ftr in ftRecognizer if ftr.message == recogAtFpO.message), None)
        confAtFpO = msg4FpO.findInMessage(timestamp)[offsAtFpO]
        # is 1.48
        # # # # # # # # # # # # # # # # # # # # # # # #





        # # # # # # # # # # # # # # # # # # # # # # # #

        # TODO evaluate bytes with only single (2-3?) bits set => flags?  (confidence?)



        # # # # # # # # # # # # # # # # # # # # # # # #
        # noinspection PyUnreachableCode
        if False:
            # # # # # # # # # # # # # # # # # # # # # # # #
            # Isolated, individual evaluation of field type recognition
            #   see FieldTypes.ods
            # ftString = "macaddr"
            # ftMemento = FieldTypeRecognizer.fieldtypeTemplates[0]
            # ftString = "timestamp"
            # ftMemento = FieldTypeRecognizer.fieldtypeTemplates[1]
            ftString = "id"
            ftMemento = FieldTypeRecognizer.fieldtypeTemplates[2]
            # ftString = "ipv4"
            # ftMemento = FieldTypeRecognizer.fieldtypeTemplates[3]
            assert ftString == ftMemento.fieldtype
            # for all messages
            truePos = dict()
            falsePos = dict()
            falseNeg = dict()
            for ftr in ftRecognizer:
                msg = ftr.message
                msgsegs = segmentedMessages[msg]

                confidences = ftr.findInMessage(ftMemento)
                sortRecogMacs = sorted([RecognizedField(msg, ftMemento, pos, con) for pos, con in enumerate(confidences)],
                       key=lambda x: x.confidence)

                # get true and false positives and false negatives and their recognized fields
                fieldOffsets = {msgseg.offset for msgseg in msgsegs if msgseg.fieldtype == ftMemento.fieldtype
                                and msgseg.offset <= msgsegs[-1].nextOffset - len(ftMemento)}
                recogOffsets = {recogFtm.position: recogFtm for recogFtm in sortRecogMacs}
                # globals().update(locals())
                truePos[msg] = [recogOffsets[tpfo] for tpfo in fieldOffsets.intersection(set(recogOffsets.keys()))]
                falsePos[msg] = [recogOffsets[fpfo] for fpfo in set(recogOffsets.keys()).difference(fieldOffsets)]
                falseNeg[msg] = [recogOffsets[fnfo] for fnfo in fieldOffsets.difference(set(recogOffsets.keys()))]

                # TODO filter interesting stuff:  see FieldTypes.ods
                #   low (actually high) confidence - false positives  <-- what do they represent that looks like the template?
                #   high confidence - false negatives  <-- why do they not resemble the template?

            # get confidence values for true and false positives
            tphistoconf = numpy.histogram([recog.confidence for msgtp in truePos.values() for recog in msgtp])
            # # median fails as a heuristic for a good confidence threshold
            # tpmedianconf = numpy.median([recog.confidence for msgtp in truePos.values() for recog in msgtp])
            tpsortconf = sorted([(recog.confidence, recog) for msgtp in truePos.values() for recog in msgtp], key = lambda x: x[0])
            # minimum bin size to find reasonable confidence threshold to work with
            tpminbin = tphistoconf[1][tphistoconf[0].argmin()]
            tpthresh = max(recog.confidence for msgtp in truePos.values() for recog in msgtp if recog.confidence < tpminbin) + 0.0001

            # histogram of fp confidences with skewed bins. Interesting but not generally helpful
            fpconf = [recog.confidence for msgfp in falsePos.values() for recog in msgfp]
            fpbins = [0, tpthresh, tpthresh * 2, tpthresh * 4, tpthresh * 8]
            if fpbins[-1] < max(fpconf) / 2:
                fpbins.append(max(fpconf) / 2)
            if fpbins[-1] < max(fpconf):
                fpbins.append(max(fpconf))
            fphistoconf = numpy.histogram(fpconf, bins=fpbins)

            ntimesthresh = 4
            # best confidences (lower ntimesthresh times the first block if filled tp bins) sorted and with instance
            fpsortconf = sorted([(recog.confidence, recog) for msgfp in falsePos.values()
                                 for recog in msgfp if recog.confidence < ntimesthresh*tpthresh], key=lambda x: x[0])
            # # no false negatives since every position is a positive (with varying confidence)
            # fnconf = [recog.confidence for msgfn in falseNeg.values() for recog in msgfn]
            fpminbin = fphistoconf[1][fphistoconf[0].argmin()]
            likelyGTerrors = [recog.confidence for msgfp in falsePos.values() for recog in msgfp if recog.confidence < fpminbin]
            fpthresh = max(likelyGTerrors) if len(likelyGTerrors) > 0 else fpsortconf[0][1]

            # numpy.argmin([recog.confidence for recog in falsePos[ftMemento]])
            # fpminrecog = fpsortconf[0][1]


            for conf, recog in fpsortconf[:100]:
                printFieldContext(segmentedMessages, recog)

                # TODO get macaddr, ipv4, ... statistics and print/analyze confidence outliers, ... and false positives
                #   solve to note below by it.

            print("Evaluating", ftString)
            print("Histogram of true positive confidences (auto bins)")
            print(tabulate(tphistoconf))
            # print("Median:", tpmedianconf)
            print("Low tp threshold:", tpthresh)
            print("Max fp conf below", ntimesthresh, "times threshold", max(conf for conf, recog in fpsortconf))
            print("Histogram of false positive confidences (manual bins)")
            print(tabulate(fphistoconf))
            print("Max fp conf below the bin separating likely gt errors from actual fps", fpthresh)

        # # # # # # # # # # # # # # # # # # # # # # # #

        # TODO statistics of how well mahalanobis separates fieldtypes (select a threshold) per ftm:
        #   for all fieldtpye memento:
        #       for all recognized positions that match the true fieldtype (true positives):
        #           determine mahalanobis distance to memento
        #       collect and sort all dists for this fieldtype
        #       plot as histogram/function
        #       |
        #       for all recognized positions that not match the true fieldtype (false positives):
        #           determine mahalanobis distance to memento
        #       collect and sort all dists for this fieldtype
        #       overlay plot from before as histogram/function (see the one for segment type distance distributions)

        # # # # # # # # # # # # # # # # # # # # # # # #



    if args.interactive:
        print("\n")
        IPython.embed()



