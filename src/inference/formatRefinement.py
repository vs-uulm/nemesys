import csv
from abc import ABC, abstractmethod
from os.path import join, exists
from typing import List, Dict, Tuple, Sequence

import numpy
from kneed import KneeLocator
from tabulate import tabulate
import IPython

from inference.segments import MessageSegment
from inference.templates import FieldTypeContext, DBSCANsegmentClusterer, DistanceCalculator, Template, \
    ClusterAutoconfException
from validation.dissectorMatcher import MessageComparator


def isPrintableChar(char: int):
    if 0x20 <= char <= 0x7e or char in ['\t', '\n', '\r']:
        return True
    return False


def isPrintable(bstring: bytes) -> bool:
    """
    A bit broader definition of printable than python string's isPrintable()

    :param bstring: a string of bytes
    :return: True if bytes contains only \t, \n, \r or is between >= 0x20 and <= 0x7e
    """
    for bchar in bstring:
        if isPrintableChar(bchar):
            continue
        else:
            return False
    return True


def locateNonPrintable(bstring: bytes) -> List[int]:
    """
    A bit broader definition of printable than python string's isPrintable()

    :param bstring: a string of bytes
    :return: position of bytes not in \t, \n, \r or between >= 0x20 and <= 0x7e
    """
    npr = list()
    for idx, bchar in enumerate(bstring):
        if isPrintableChar(bchar):
            continue
        else:
            npr.append(idx)
    return npr


class MessageModifier(ABC):
    _debug = False

    def __init__(self, segments: List[MessageSegment]):
        """
        :param segments: The segments of one message in offset order
        """
        self.segments = segments



class Merger(MessageModifier, ABC):
    """
    Base class to merge segments based on a variable condition.
    """

    def merge(self):
        """
        Perform the merging.

        :return: a new set of segments after the input has been merged
        """
        mergedSegments = self.segments[0:1]
        if len(self.segments) > 1:
            for segl, segr in zip(self.segments[:-1], self.segments[1:]):
                # TODO check for equal analyzer, requires implementing a suitable equality-check in analyzer
                # from inference.MessageAnalyzer import MessageAnalyzer
                if segl.offset + segl.length == segr.offset and self.condition(segl, segr):
                    mergedSegments[-1] = MessageSegment(mergedSegments[-1].analyzer, mergedSegments[-1].offset,
                                                        mergedSegments[-1].length + segr.length)
                    if self._debug:
                        print("Merged segments: \n{} and \n{} into \n{}".format(segl, segr, mergedSegments[-1]))
                else:
                    mergedSegments.append(segr)
        return mergedSegments


    @staticmethod
    @abstractmethod
    def condition(segl: MessageSegment, segr: MessageSegment) -> bool:
        """
        A generic condition called to determine whether a merging is necessary.

        :param segl: left segment
        :param segr: right segment
        :return: True if merging is required, False otherwise.
        """
        pass


class MergeConsecutiveChars(Merger):
    """
    Merge consecutive segments completely consisting of printable-char values into a text field.
    Printable chars are defined as: \t, \n, \r, >= 0x20 and <= 0x7e

    >>> from inference.segmentHandler import bcDeltaGaussMessageSegmentation
    >>> from utils.loader import SpecimenLoader
    >>> import inference.formatRefinement as refine
    >>> from tabulate import tabulate
    >>> sl = SpecimenLoader('../input/dns_ictf2010_deduped-100.pcap', layer=0, relativeToIP=True)
    >>> segmentsPerMsg = bcDeltaGaussMessageSegmentation(sl)
    Segmentation by inflections of sigma-0.6-gauss-filtered bit-variance.
    >>> for messageSegments in segmentsPerMsg:
    ...     mcc = MergeConsecutiveChars(messageSegments)
    ...     mccmg = mcc.merge()
    ...     if mccmg != messageSegments:
    ...         sgms = b''.join([m.bytes for m in mccmg])
    ...         sgss = b''.join([m.bytes for m in messageSegments])
    ...         if sgms != sgss:
    ...             print("Mismatch!")
    """

    @staticmethod
    def condition(segl: MessageSegment, segr: MessageSegment):
        """
        Check whether both segments consist of printable characters.
        """
        return isPrintable(segl.bytes) and isPrintable(segr.bytes)


class RelocateSplits(MessageModifier, ABC):
    """
    Relocate split locations based on properties of adjacent segments.
    """

    def split(self):
        """
        Perform the splitting of the segments.

        :return: List of segments splitted from the input.
        """
        segmentStack = list(reversed(self.segments))
        mangledSegments = list()
        if len(self.segments) > 1:
            while segmentStack:
                # TODO check for equal analyzer, requires equality-check in analyzer
                # from inference.MessageAnalyzer import MessageAnalyzer

                segc = segmentStack.pop()
                # TODO: this is char specific only!
                if not isPrintable(segc.bytes):
                    # cancel split relocation
                    mangledSegments.append(segc)
                    continue

                if mangledSegments:
                    # integrate segment to the left into center
                    segl = mangledSegments[-1]
                    if segl.offset + segl.length == segc.offset:
                        splitpos = self.toTheLeft(segl)
                        # segment to the left ends with chars, add them to the center segment
                        if splitpos < segl.length:
                            if splitpos > 0:
                                mangledSegments[-1] = MessageSegment(mangledSegments[-1].analyzer,
                                                                 mangledSegments[-1].offset, splitpos)
                            else: # segment to the left completely used up in center
                                del mangledSegments[-1]
                            restlen = segl.length - splitpos
                            if self._debug:
                                print("Recombined segments: \n{} and {} into ".format(segl, segc))
                            segc = MessageSegment(segc.analyzer, segc.offset - restlen,
                                                             segc.length + restlen)
                            if self._debug:
                                print("{} and {}".format(mangledSegments[-1] if mangledSegments else 'Empty', segc))

                if segmentStack:
                    # integrate segment to the right into center
                    segr = segmentStack[-1]
                    if segc.offset + segc.length == segr.offset:
                        splitpos = self.toTheRight(segr)
                        # segment to the right starts with chars, add them to the center segment
                        if splitpos > 0:
                            if segr.length - splitpos > 0:
                                segmentStack[-1] = MessageSegment(segr.analyzer, segr.offset + splitpos,
                                                                 segr.length - splitpos)
                            else: # segment to the right completely used up in center
                                del segmentStack[-1]
                            if self._debug:
                                print("Recombined segments: \n{} and {} into ".format(segc, segr))
                            segc = MessageSegment(segc.analyzer, segc.offset,
                                                              segc.length + splitpos)
                            if self._debug:
                                print("{} and {}".format(segc, segmentStack[-1] if segmentStack else 'Empty'))

                mangledSegments.append(segc)
        return mangledSegments

    @staticmethod
    @abstractmethod
    def toTheLeft(segl: MessageSegment) -> int:
        """
        :param segl: The current segment
        :return: The relative position of the new split to the left of the current segment.
        """
        pass

    @staticmethod
    @abstractmethod
    def toTheRight(segr: MessageSegment) -> int:
        """
        :param segr: The current segment
        :return: The relative position of the new split to the right of the current segment.
        """
        pass


class ResplitConsecutiveChars(RelocateSplits):
    """
    Split segments to keep consecutive chars together.
    """

    @staticmethod
    def toTheLeft(segl: MessageSegment) -> int:
        """
        :param segl:
        :return: the count of printable chars at the end of the segment
        """
        splitpos = segl.length
        for char in reversed(segl.bytes):
            if isPrintableChar(char):
                splitpos -= 1
            else:
                break
        return splitpos

    @staticmethod
    def toTheRight(segr: MessageSegment) -> int:
        """

        :param segr:
        :return: the count of printable chars at the beginning of the segment
        """
        splitpos = 0
        for char in segr.bytes:
            if isPrintableChar(char):
                splitpos += 1
            else:
                break
        return splitpos


class Resplit2LeastFrequentPair(MessageModifier):
    """
    Search for value pairs at segment (begin|end)s; and one byte pair ahead and after.
    If the combination across the border is more common than either ahead-pair or after-pair, shift the border to
    cut at the least common value combination

    Hypothesis: Field values are more probable to be identical than bytes across fields.

    Hypothesis is wrong in general. FMS drops in many cases. Drop in average:
     * dhcp: 0.011
     * ntp: -0.070 (improves slightly)
     * dns:  0.012

    """
    __pairFrequencies = None
    __CHUNKLEN = 2

    @staticmethod
    def countPairFrequencies(allMsgsSegs: List[List[MessageSegment]]):
        """
        Given the segment bounds: | ..YZ][AB.. |
        -- search for ZA, YZ, AB of all segments in all messages and count the occurrence frequency of each value pair.

        Needs only to be called once before all segments of one inference pass can be refined.
        A different inference required to run this method again before refinement by this class.

        >>> from inference.segmentHandler import bcDeltaGaussMessageSegmentation
        >>> from utils.loader import SpecimenLoader
        >>> import inference.formatRefinement as refine
        >>> from tabulate import tabulate
        >>> sl = SpecimenLoader('../input/random-100-continuous.pcap', layer=0, relativeToIP=True)
        >>> segmentsPerMsg = bcDeltaGaussMessageSegmentation(sl)
        Segmentation by inflections of sigma-0.6-gauss-filtered bit-variance.
        >>> messageSegments = segmentsPerMsg[0]
        >>> # Initialize Resplit2LeastFrequentPair class
        >>> refine.Resplit2LeastFrequentPair.countPairFrequencies(segmentsPerMsg)
        >>> replitSegments = refine.Resplit2LeastFrequentPair(messageSegments).split()
        >>> segbytes = [[],[]]
        >>> for a, b in zip(messageSegments, replitSegments):
        ...     if a != b:
        ...         segbytes[0].append(a.bytes.hex())
        ...         segbytes[1].append(b.bytes.hex())
        >>> print(tabulate(segbytes))
        --------------------  --------  --------  ------  ----------  --------  --------  --------  --------  ------
        780001000040007c837f  0000017f  000001    6f9fca  9de16a3b    5af87abf  108735    4b574410  9b9f      e59f5d
        780001000040007c83    7f000001  7f000001  6f9f    ca9de16a3b  5af87a    bf108735  4b5744    109b9fe5  9f5d
        --------------------  --------  --------  ------  ----------  --------  --------  --------  --------  ------

        """
        from collections import Counter
        Resplit2LeastFrequentPair.__pairFrequencies = Counter()

        for segsList in allMsgsSegs:
            # these are all segments of one message
            offsets = [segment.offset for segment in segsList]  # here we simply assume there is no gap between segments
            msgbytes = segsList[0].message.data  # we assume that all segments in the list are from one message only
            msglen = len(msgbytes)
            for fieldboundary in offsets:
                if fieldboundary == 0 or fieldboundary == msglen:
                    continue
                if fieldboundary < Resplit2LeastFrequentPair.__CHUNKLEN \
                        or fieldboundary + Resplit2LeastFrequentPair.__CHUNKLEN > msglen:
                    continue
                clh = Resplit2LeastFrequentPair.__CHUNKLEN // 2
                across = msgbytes[fieldboundary - clh:fieldboundary + clh]
                before = msgbytes[fieldboundary - Resplit2LeastFrequentPair.__CHUNKLEN:fieldboundary]
                after  = msgbytes[fieldboundary    :fieldboundary + Resplit2LeastFrequentPair.__CHUNKLEN]
                # print(msgbytes[fieldboundary:fieldboundary+1])
                # print(across)
                # print(before)
                # print(after)
                assert len(across) == Resplit2LeastFrequentPair.__CHUNKLEN \
                       and len(before) == Resplit2LeastFrequentPair.__CHUNKLEN \
                       and len(after) == Resplit2LeastFrequentPair.__CHUNKLEN
                Resplit2LeastFrequentPair.__pairFrequencies.update([across, before, after])
        if Resplit2LeastFrequentPair._debug:
            from tabulate import tabulate
            print('Most common byte pairs at boundaries:')
            print(tabulate([(byteval.hex(), count)
                            for byteval, count in Resplit2LeastFrequentPair.__pairFrequencies.most_common(5)]))

    @staticmethod
    def frequencies():
        return Resplit2LeastFrequentPair.__pairFrequencies


    def split(self):
        """
        Perform the splitting of the segments.

        :return: List of segments splitted from the input.
        """
        segmentStack = list(reversed(self.segments[1:]))
        mangledSegments = [self.segments[0]]
        if len(self.segments) > 1:
            while segmentStack:
                segc = segmentStack.pop()
                segl = mangledSegments[-1]
                if segl.offset + segl.length == segc.offset:
                    # compare byte pairs' frequency
                    splitshift = self.lookupLeastFrequent(segc)
                    if ( 0 > splitshift >= -segl.length) \
                        or (0 < splitshift <= segc.length):
                        if segl.length != -splitshift:
                            mangledSegments[-1] = MessageSegment(mangledSegments[-1].analyzer,
                                                                 mangledSegments[-1].offset,
                                                                 mangledSegments[-1].length + splitshift)
                        else: # segment to the left completely used up in center
                            del mangledSegments[-1]
                        if self._debug:
                            print("Recombined segments: \n{} and {} into ".format(segl, segc))
                        segc = MessageSegment(segc.analyzer, segc.offset + splitshift,
                                                         segc.length - splitshift)
                        if self._debug:
                            print("{} and {}".format(mangledSegments[-1] if mangledSegments else 'Empty', segc))
                mangledSegments.append(segc)
        return mangledSegments


    @staticmethod
    def lookupLeastFrequent(seg: MessageSegment) -> int:
        """
        Given the occurence frequencies of all segment bounds: | ..YZ][AB.. |
        shift border if ZA is more common than YZ or AB. New split at least common pair.

        :return: the direction to shift to break at the least frequent byte pair
        """
        if seg.offset == 0:
            return 0
        if seg.offset < Resplit2LeastFrequentPair.__CHUNKLEN \
                or seg.offset + Resplit2LeastFrequentPair.__CHUNKLEN > len(seg.message.data):
            return 0

        msgbytes = seg.message.data
        clh = Resplit2LeastFrequentPair.__CHUNKLEN // 2
        across = msgbytes[seg.offset - clh:seg.offset + clh]
        before = msgbytes[seg.offset - Resplit2LeastFrequentPair.__CHUNKLEN:seg.offset]
        after  = msgbytes[seg.offset    :seg.offset + Resplit2LeastFrequentPair.__CHUNKLEN]
        assert len(across) == Resplit2LeastFrequentPair.__CHUNKLEN \
               and len(before) == Resplit2LeastFrequentPair.__CHUNKLEN \
               and len(after) == Resplit2LeastFrequentPair.__CHUNKLEN
        countAcross = Resplit2LeastFrequentPair.__pairFrequencies[across]
        countBefore = Resplit2LeastFrequentPair.__pairFrequencies[before]
        countAfter  = Resplit2LeastFrequentPair.__pairFrequencies[after]
        countMin = min(countAcross, countBefore, countAfter)
        if countMin == countAcross:
            return 0
        if countMin == countBefore:
            return -1
        if countMin == countAfter:
            return 1




class CropDistinct(MessageModifier):
    """
    Split segments into smaller chunks if a given value is contained in the segment.
    The given value is cropped to a segment on its own.
    """
    def __init__(self, segments: List[MessageSegment], mostcommon: List[bytes]):
        """
        :param segments: The segments of one message in offset order.
        :param mostcommon: most common bytes sequences to be searched for and cropped
            (sorted descending from most frequent)
        """
        super().__init__(segments)
        self._moco = mostcommon

    @staticmethod
    def countCommonValues(segmentedMessages: List[List[MessageSegment]]):
        from collections import Counter
        from itertools import chain
        segcnt = Counter([seg.bytes for seg in chain.from_iterable(segmentedMessages)])
        segFreq = segcnt.most_common()
        freqThre = .2 * len(segmentedMessages)
        thre = 0
        while segFreq[thre][1] > freqThre:
            thre += 1
        moco = [fv for fv, ct in segFreq[:thre] if set(fv) != {0}]  # omit \x00-sequences
        return moco

    def split(self):
        newmsg = list()
        for sid, seg in enumerate(self.segments):  # enum necessary to change to in place edit after debug (want to do?)
            didReplace = False
            for comfeat in self._moco:
                comoff = seg.bytes.find(comfeat)
                if comoff == -1:  # comfeat not in moco, continue with next in moco
                    continue

                featlen = len(comfeat)
                if seg.length == featlen:  # its already the concise frequent feature
                    newmsg.append(seg)
                else:
                    if CropDistinct._debug:
                        print("\nReplaced {} by:".format(seg.bytes.hex()), end=" ")

                    absco = seg.offset + comoff
                    if comoff > 0:
                        segl = MessageSegment(seg.analyzer, seg.offset, comoff)
                        newmsg.append(segl)
                        if CropDistinct._debug:
                            print(segl.bytes.hex(), end=" ")

                    segc = MessageSegment(seg.analyzer, absco, featlen)
                    newmsg.append(segc)
                    if CropDistinct._debug:
                        print(segc.bytes.hex(), end=" ")

                    rlen = seg.length - comoff - featlen
                    if rlen > 0:
                        segr = MessageSegment(seg.analyzer, absco + featlen, rlen)
                        newmsg.append(segr)
                        if CropDistinct._debug:
                            print(segr.bytes.hex(), end=" ")

                didReplace = True
                break  # only most common match!? otherwise how to handle subsequent matches after split(s)?
            if not didReplace:
                newmsg.append(seg)
            elif CropDistinct._debug:
                print()

        return newmsg


class CumulativeCharMerger(MessageModifier):
    """
    Merge consecutive segments that toghether fulfill the char conditions in inference.segmentHandler.isExtendedCharSeq
    """

    def merge(self):
        """
        Perform the merging.

        >>> bytes.fromhex("00000000000002")
        >>> bytes.fromhex("613205")

        :return: a new set of segments after the input has been merged
        """
        from inference.segmentHandler import isExtendedCharSeq

        minLen = 6

        segmentStack = list(reversed(self.segments))
        newmsg = list()
        isCharCand = False
        workingStack = list()
        while segmentStack:
            workingStack.append(segmentStack.pop())
            if sum([len(ws.bytes) for ws in workingStack]) < minLen:
                continue

            # now we have 6 bytes
            # and the merge is a new char candidate
            joinedbytes = b"".join([ws.bytes for ws in workingStack])
            if isExtendedCharSeq(joinedbytes) \
                    and b"\x00\x00" not in joinedbytes:
                isCharCand = True
                continue
            # the last segment ended the char candidate
            elif isCharCand:
                isCharCand = False
                if len(workingStack) > 2:
                    newlen = sum([ws.length for ws in workingStack[:-1]])
                    newseg = MessageSegment(workingStack[0].analyzer,
                                            workingStack[0].offset, newlen)
                    newmsg.append(newseg)
                else:
                    # retain the original segment (for equality test and to save creating a new object instance)
                    newmsg.append(workingStack[0])
                if len(workingStack) > 1:
                    segmentStack.append(workingStack[-1])
                workingStack = list()
            # there was not a char candidate
            else:
                newmsg.append(workingStack[0])
                for ws in reversed(workingStack[1:]):
                    segmentStack.append(ws)
                workingStack = list()
        # there are segments in the working stack left
        if len(workingStack) > 1 and isCharCand:
            newlen = sum([ws.length for ws in workingStack])
            newseg = MessageSegment(workingStack[0].analyzer,
                                    workingStack[0].offset, newlen)
            newmsg.append(newseg)
        # there was no char sequence and there are segments in the working stack left
        else:
            newmsg.extend(workingStack)
        return newmsg


class SplitFixed(MessageModifier):
    """
    Split a given segment into chunks of fixed lengths.
    """

    def split(self, segmentID: int, chunkLength: int):
        selSeg = self.segments[segmentID]
        if chunkLength < selSeg.length:
            newSegs = list()
            for chunkoff in range(selSeg.offset, selSeg.nextOffset, chunkLength):
                remainLen = selSeg.nextOffset - chunkoff
                newSegs.append(MessageSegment(selSeg.analyzer, chunkoff, min(remainLen, chunkLength)))
            newmsg = self.segments[:segmentID] + newSegs + self.segments[segmentID + 1:]
            return newmsg
        else:
            return self.segments


class RelocateZeros(MessageModifier):

    def __init__(self, segments: List[MessageSegment], comparator: MessageComparator):
        super().__init__(segments)
        self._parsedMessage = comparator.parsedMessages[comparator.messages[segments[0].message]]
        self.counts = None
        self.doprint = False

    def split(self):
        trueFieldEnds = MessageComparator.fieldEndsFromLength([l for t, l in self._parsedMessage.getFieldSequence()])

        newmsg = self.segments[0:1]
        for segr in self.segments[1:]:
            segl = newmsg[-1]
            # exactly one zero right before or after boundary: no change
            if segl.bytes[-1] == 0 and segr.bytes[0] != 0 or segr.bytes[0] == 0 and segl.bytes[-1] != 0:
                if segr.offset in trueFieldEnds:
                    self.counts["0t"] += 1
                else:
                    self.counts["0f"] += 1
                newmsg.append(segr)
            else:
                # prio 1: zero to the right of boundary
                if segr.length > 2 and segr.bytes[1] == 0 and segr.bytes[0] != 0 and segr.bytes[2] == 0:
                    # shift right
                    newmsg[-1] = MessageSegment(segl.analyzer, segl.offset, segl.length + 1)
                    newmsg.append(MessageSegment(segr.analyzer, segr.offset + 1, segr.length - 1))
                    if newmsg[-1].offset in trueFieldEnds:
                        self.counts["+1tr"] += 1
                    else:
                        self.counts["+1fr"] += 1
                        if segr.offset in trueFieldEnds:
                            self.counts["+1.0"] += 1
                            self.doprint = True
                elif segl.length > 1 and segl.bytes[-1] == 0 and segl.bytes[-2] != 0:
                    # shift left (never happens)
                    newmsg[-1] = MessageSegment(segl.analyzer, segl.offset, segl.length - 1)
                    newmsg.append(MessageSegment(segr.analyzer, segr.offset - 1, segr.length + 1))
                    if newmsg[-1].offset in trueFieldEnds:
                        self.counts["-1tr"] += 1
                    else:
                        self.counts["-1fr"] += 1
                        if segr.offset in trueFieldEnds:
                            self.counts["-1.0"] += 1
                            self.doprint = True
                # prio 2: 2 zeros to the left of boundary
                elif segr.length > 1 and segr.bytes[0] == 0 and segr.bytes[1] != 0 and \
                        segl.length > 1 and segl.bytes[-1] == 0:
                    # shift right (never happens)
                    newmsg[-1] = MessageSegment(segl.analyzer, segl.offset, segl.length + 1)
                    newmsg.append(MessageSegment(segr.analyzer, segr.offset + 1, segr.length - 1))
                    if newmsg[-1].offset in trueFieldEnds:
                        self.counts["+1tl"] += 1
                    else:
                        self.counts["+1fl"] += 1
                        if segr.offset in trueFieldEnds:
                            self.counts["+1.0"] += 1
                            self.doprint = True
                elif segl.length > 2 and segl.bytes[-2] == 0 and segl.bytes[-1] != 0 and \
                        segl.bytes[-3] == 0:
                    # shift left
                    newmsg[-1] = MessageSegment(segl.analyzer, segl.offset, segl.length - 1)
                    newmsg.append(MessageSegment(segr.analyzer, segr.offset - 1, segr.length + 1))
                    if newmsg[-1].offset in trueFieldEnds:
                        self.counts["-1tl"] += 1
                    else:
                        self.counts["-1fl"] += 1
                        if segr.offset in trueFieldEnds:
                            self.counts["-1.0"] += 1
                            self.doprint = True
                else:
                    # no zeros at boundary
                    newmsg.append(segr)
                assert len(newmsg) > 1 and newmsg[-2].nextOffset == newmsg[-1].offset or newmsg[-1].offset == 0
        return newmsg

    def initCounts(self, amounts: Dict[str, int] = None):
        """
        Counts of different kinds of off-by-one situations for zeros

        :return:
        """
        if amounts is None:
            self.counts = {
                "0t":0, "0f":0, "-1tr":0, "-1fr":0, "-1tl":0, "-1fl":0, "+1tr":0, "+1fr":0, "+1tl":0, "+1fl":0,
                "-1.0":0, "+1.0":0
            }  # key: boundary offset change + true/false
        else:
            self.counts = amounts




class RelocatePCA(object):

    # PCA conditions parameters
    minSegLen = 3  # minimum length of the largest cluster member
    screeMinThresh = 10  # threshold for the minimum of a component to be considered principal
    principalCountThresh = .5   # threshold for the maximum number of allowed principal components to start the analysis;
            # interpreted as fraction of the component count as dynamic determination.
    contributionRelevant = 0.1  # 0.01; threshold for the minimum difference from 0 to consider a loading component in the eigenvector as contributing to the variance
    maxLengthDelta = 7  # (max true: 6 in dns-new, min false 9 in dhcp)

    def __init__(self, similarSegments: FieldTypeContext,
                 eigenValuesAndVectors: Tuple[numpy.ndarray, numpy.ndarray]=None,
                 screeKnee: float=None):
        self._similarSegments = similarSegments
        self._eigen = eigenValuesAndVectors if eigenValuesAndVectors is not None \
            else numpy.linalg.eigh(similarSegments.cov)
        self._screeKnee = screeKnee if screeKnee is not None \
            else RelocatePCA.screeKneedle(self._eigen)
        # defines an minimum for any principal component, that is
        #       at least the knee in scree,
        #       not less than one magnitude smaller then the first PC,
        #       and larger than the absolute minimum.
        self._screeThresh = max(self._screeKnee, RelocatePCA.screeMinThresh, max(self._eigen[0]) / 10) \
                                if any(self._eigen[0] < RelocatePCA.screeMinThresh) \
                                else RelocatePCA.screeMinThresh
        self._principalComponents = self._eigen[0] > self._screeThresh
        self._contribution = self._eigen[1][:, self._principalComponents]  # type: numpy.ndarray

        self._testSubclusters = None


    @property
    def similarSegments(self):
        return self._similarSegments


    @staticmethod
    def screeKneedle(eigenValuesAndVectors: Tuple[numpy.ndarray, numpy.ndarray]) -> float:
        """

        :param eigenValuesAndVectors:
        :return: y-value of the knee, 0 if there is no knee.
        """
        scree = list(reversed(sorted(eigenValuesAndVectors[0].tolist())))
        try:
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                kl = KneeLocator(
                    list(range(eigenValuesAndVectors[0].shape[0])),
                    scree, curve='convex', direction='decreasing')
        except ValueError:
            return 0.0
        if not isinstance(kl.knee, int):
            return 0.0

        # if kl.knee > 1:
        #     print(scree)
        #     import matplotlib.pyplot as plt
        #     kl.plot_knee_normalized()
        #     plt.show()
        #     # IPython.embed()

        return scree[kl.knee]



    @staticmethod
    def filterForSubclustering(fTypeContext: Sequence[FieldTypeContext]):
        from inference.segmentHandler import isExtendedCharSeq

        interestingClusters = [
            cid for cid, clu in enumerate(fTypeContext)
            if not clu.length < RelocatePCA.minSegLen
               and not sum([isExtendedCharSeq(seg.bytes) for seg in clu.baseSegments]) > .5 * len(clu.baseSegments)
               and not all(clu.stdev == 0)
            # # moved to second iteration below to make it dynamically dependent on the scree-knee:
            # and not all(numpy.linalg.eigh(clu.cov)[0] <= screeMinThresh)
        ]

        return interestingClusters


    @staticmethod
    def filterRelevantClusters(fTypeContext: Sequence[FieldTypeContext]):
        """
        Filteres clusters that have the required properties to be subclustered. Ensures that the kneedle algorithm
        can be run on each cluster of the resulting subset of clusters.

        :param fTypeContext:
        :return: tuple containing:
            * list of indices of the given fTypeContext that are relevant.
            * dict of eigenvalues and vectors with the fTypeContext indices as keys
            * dict of scree-knee values with the fTypeContext indices as keys
        """
        interestingClusters = RelocatePCA.filterForSubclustering(fTypeContext)

        # leave only clusters of at least minSegLen * 2 (=6)
        minClusterSize = RelocatePCA.minSegLen * 2
        for cid in interestingClusters.copy():
            if len(fTypeContext[cid].baseSegments) < minClusterSize:
                interestingClusters.remove(cid)

        # remove all clusters that have no knees in scree
        eigenVnV = dict()
        screeKnees = dict()
        for cid in interestingClusters.copy():
            eigen = numpy.linalg.eigh(fTypeContext[cid].cov)  # type: Tuple[numpy.ndarray, numpy.ndarray]
            eigenVnV[cid] = eigen
            screeKnees[cid] = RelocatePCA.screeKneedle(eigen)
            if all(eigen[0] <= screeKnees[cid]):
                interestingClusters.remove(cid)

        # for cid, clu in enumerate(fTypeContext):
        #     if cid not in interestingClusters:
        #         eigen = numpy.linalg.eigh(fTypeContext[cid].cov)
        #         print(eigen[0])
        #         IPython.embed()

        # remove all clusters having a segment length difference of more than maxLengthDelta
        for cid in interestingClusters.copy():
            bslen = {bs.length for bs in fTypeContext[cid].baseSegments}
            if max(bslen) - min(bslen) > RelocatePCA.maxLengthDelta:
                interestingClusters.remove(cid)
                print("Removed cluster {} for too high segment length difference of {}".format(
                    fTypeContext[cid].fieldtype,
                    max(bslen) - min(bslen)
                ))

        return interestingClusters, eigenVnV, screeKnees


    @property
    def principalComponents(self):
        return self._principalComponents


    @property
    def screeThresh(self):
        return self._screeThresh


    @property
    def contribution(self):
        return self._contribution


    def getSubclusters(self, dc: DistanceCalculator = None, S:float = None, reportFolder:str = None, trace:str = None):
        """
        Re-Cluster

        :param dc:
        :param S:
        :param reportFolder:
        :param trace:
        :return: A flat list of subclusters. If no subclustering was necessary or possible, returns itself.
        """

        # if reportFolder is not None and trace is not None:
        #     from os.path import join, exists
        #     import csv

        maxAbsolutePrincipals = 4

        # number of principal components
        tooManyPCs = sum(self._eigen[0] > self._screeThresh) > \
            min(maxAbsolutePrincipals, self._eigen[0].shape[0] * RelocatePCA.principalCountThresh)
            # and any(self._eigen[0] < RelocatePCA.screeMinThresh)
        # if there remain too few segments to sub-cluster, return the whole FieldTypeContext for analysis
        enoughSegments = len(self._similarSegments.baseSegments) > RelocatePCA.minSegLen
        # segment length difference is too high
        bslen = {bs.length for bs in self.similarSegments.baseSegments}
        tooHighLenDiff = max(bslen) - min(bslen) > RelocatePCA.maxLengthDelta

        if tooManyPCs:
            print("Cluster {} needs reclustering: too many principal components.".format(
                self._similarSegments.fieldtype))
        if tooHighLenDiff:
            print("Cluster {} needs reclustering: length difference too high.".format(
                self._similarSegments.fieldtype))


        if (tooManyPCs or tooHighLenDiff) and enoughSegments:
            if dc is None:
                print("No dissimilarities available. Ignoring cluster.")
                return list()
            else:
                try:
                    # Re-Cluster
                    clusterer = DBSCANsegmentClusterer(dc, segments=self._similarSegments.baseSegments, S=S)
                    noise, *clusters = clusterer.clusterSimilarSegments(False)
                    print(self._similarSegments.fieldtype,
                          clusterer, "- cluster sizes:", [len(s) for s in clusters], "- noise:", len(noise))

                    # # # # # # # # # # # # # # # # # # # # # # # #
                    # write statistics
                    if reportFolder is not None and trace is not None:
                        fn = join(reportFolder, "subcluster-parameters.csv")
                        writeheader = not exists(fn)
                        with open(fn, "a") as segfile:
                            segcsv = csv.writer(segfile)
                            if writeheader:
                                segcsv.writerow(
                                    ["trace", "cluster label", "cluster size",
                                     "max segment length", "# principals",
                                     "max principal value",
                                     "# subclusters", "noise", "Kneedle S", "DBSCAN eps", "DBSCAN min_samples"
                                     ]
                                )
                            segcsv.writerow([
                                trace, self.similarSegments.fieldtype, len(self.similarSegments.baseSegments),
                                self.similarSegments.length, sum(self.principalComponents),
                                self._eigen[0][self._principalComponents].max(),
                                len(clusters), len(noise), clusterer.S, clusterer.eps, clusterer.min_samples
                            ])


                    # if there remains only noise, continue and try to do the analysis for the whole FieldTypeContext
                    if len(clusters) > 0:
                        # Generate suitable FieldTypeContext objects from the sub-clusters
                        fTypeContext = list()
                        for cLabel, segments in enumerate(clusters):
                            resolvedSegments = list()
                            for seg in segments:
                                if isinstance(seg, Template):
                                    resolvedSegments.extend(seg.baseSegments)
                                else:
                                    resolvedSegments.append(seg)
                            fcontext = FieldTypeContext(resolvedSegments)
                            fcontext.fieldtype = "{}.{}".format(self._similarSegments.fieldtype, cLabel)
                            fTypeContext.append(fcontext)
                        # self._testSubclusters = fTypeContext
                        # Determine clusters with relevant properties for further analysis
                        # interestingClusters, eigenVnV, screeKnees = RelocatePCA.filterRelevantClusters(fTypeContext)
                        # print(interestingClusters, "from", len(clusters), "clusters")

                        # Do PCA on all interesting clusters
                        # retValCollection = dict.fromkeys(interestingClusters, None)  # type: Dict[int, List[List[int], Tuple[numpy.ndarray, numpy.ndarray], float]]
                        # for iC in interestingClusters:
                        #     print("Analyzing sub-cluster", fTypeContext[iC].fieldtype)
                        #     subRpca = RelocatePCA(fTypeContext[iC])
                        #     relocate = subRpca.relocateOffsets(dc, reportFolder, trace)
                        #
                        #     retValCollection[iC] = [relocate, eigenVnV[iC], screeKnees[iC], subRpca]
                        # return retValCollection

                        interestingClusters = RelocatePCA.filterForSubclustering(fTypeContext)
                        subclusters = list()
                        for cid, subcluster in enumerate(fTypeContext):
                            print("Analyzing sub-cluster", subcluster.fieldtype)
                            subRpca = RelocatePCA(subcluster)
                            if cid in interestingClusters:
                                subclusters.extend(subRpca.getSubclusters(dc, S, reportFolder, trace))
                            else:
                                subclusters.append(subRpca)
                        return subclusters
                except ClusterAutoconfException as e:
                    print(e)
                    return [self]

        return [self]


    def relocateOffsets(self, reportFolder:str = None, trace:str = None, comparator: MessageComparator = None,
                        conditionA = True, conditionE1 = True, conditionE2 = True, conditionF = False,
                        conditionG = False):
        """

        :param conditionA: Enable Condition A
        :param conditionE1: Enable Condition E1
        :param conditionE2: Enable Condition E2
        :param conditionF:  Enable Condition F
        :param conditionG:  Enable Condition G
        :param reportFolder: For debugging
        :param trace: For debugging
        :param comparator: For debugging
        :return: positions to be relocated if the PCA for the current cluster has meaningful result,
            otherwise sub-cluster and return the PCA and relocation positions for each sub-cluster.
        """
        import csv

        # # # # # # # # # # # # # # # # # # # # # # # #
        # Count true boundaries for the segments' relative positions
        if comparator:
            from collections import Counter
            trueOffsets = list()
            for bs in self.similarSegments.baseSegments:
                fe = comparator.fieldEndsPerMessage(bs.analyzer.message)
                offs, nxtOffs = self.similarSegments.paddedPosition(bs)
                trueOffsets.extend(o - offs for o in fe if offs <= o <= nxtOffs)
            truOffCnt = Counter(trueOffsets)
            mostCommonTrueBounds = [offs for offs, cnt in truOffCnt.most_common()
                                    if cnt > 0.5 * len(self.similarSegments.baseSegments)]

        # # # # # # # # # # # # # # # # # # # # # # # #
        # "component-analysis"
        #
        # the principal components (i. e. with Eigenvalue > 1) of the covariance matrix is assumed to be
        # towards the end of varying fields with similar content (e.g. counting numbers).
        # The component is near 1 or -1 in the Eigenvector of the respective Eigenvalue.

        relocate = list()  # new boundaries found: relocate the next end to this relative offset
        # relocateFromStart = list()  # new boundaries found: relocate the previous start to this relative offset

        # continue only if we have some principal components
        if not self.principalComponents.any():
            return relocate

        # # Condition a: Covariance ~0 after non-0
        # at which eigenvector component does any of the principal components have a relevant contribution
        contributingLoadingComponents = (abs(self._contribution) > RelocatePCA.contributionRelevant).any(1)


        if conditionA:
            for lc in reversed(range(1, contributingLoadingComponents.shape[0])):
                if not contributingLoadingComponents[lc] and contributingLoadingComponents[lc - 1]:
                    relocate.append(lc)

                    # # # # # # # # # # # # # # # # # # # # # # # #
                    # write statistics
                    if reportFolder is not None and trace is not None:
                        fn = join(reportFolder, "pca-conditions-a.csv")
                        writeheader = not exists(fn)
                        with open(fn, "a") as segfile:
                            segcsv = csv.writer(segfile)
                            if writeheader:
                                segcsv.writerow(
                                    ["trace", "cluster label", "cluster size",
                                     "max segment length", "# principals",
                                     "max principal value",
                                     "is FP",
                                     "first boundary true",
                                     "final boundary true",
                                     "offset", "max contribution before offset", "max contribution at offset"]
                                )
                            # noinspection PyUnboundLocalVariable
                            segcsv.writerow([
                                trace, self.similarSegments.fieldtype, len(self.similarSegments.baseSegments),
                                self.similarSegments.length, sum(self.principalComponents),
                                self._eigen[0][self._principalComponents].max(),
                                repr(lc not in mostCommonTrueBounds) if comparator else "",
                                repr(0 in mostCommonTrueBounds) if comparator else "",
                                repr(self.similarSegments.length in mostCommonTrueBounds) if comparator else "",
                                lc, abs(self._contribution[lc - 1]).max(), abs(self._contribution[lc]).max()
                            ])

        # # # Condition b: Loadings peak in the middle a segment
        # principalLoadingPeak = abs(self._eigen[1][:, self._eigen[0].argmax()]).argmax()
        # if principalLoadingPeak < self._eigen[0].shape[0] - 1:
        #     relocate.append(principalLoadingPeak+1)

        # # # Condition c:
        # tailSize = 1
        # if not contributingLoadingComponents[:-tailSize].any():
        #     for tailP in range(tailSize, 0, -1):
        #         if contributingLoadingComponents[-tailP]:
        #             relocate.append(self._eigen[0].shape[0] - tailSize)
        #             break

        # # Condition d:
        # cancelled

        eigVsorted = list(reversed(sorted([(val, vec) for val, vec in zip(
            self._eigen[0][self.principalComponents], self.contribution.T)],
                                          key=lambda x: x[0])))
        eigVecS = numpy.array([colVec[1] for colVec in eigVsorted]).T


        # TODO caveat: precision of numerical method for cov or eigen does not suffice for near zero resolution
        #  in all cases. e. g. setting nearZero to 0.003 indeterministically results in a false negative for the
        #  condition. Setting it higher, might introduce more false positives.

        # # Condition e: Loading peak of principal component rising from (near) 0.0
        #
        # e1 parameters
        nearZero = 0.003
        notableContrib = 0.66  # or 0.7 (see smb tf04)
        # peak may be (near) +/- 1.0 in most cases, but +/- 0.5 includes also ntp tf06 and smb tf04,
        #   however, it has false positive dhcp tf01 and smb tf00.4

        # e2 parameters
        # also apply to higher nearZero and lower notableContrib if longer (>= 4) sequence of nearZero
        #   precedes notableContrib
        relaxedNearZero = 0.004
        relaxedNZlength = 4
        relaxedNotableContrib = 0.05
        relaxedMaxContrib = 0.66


        # if self.similarSegments.fieldtype == "tf01":
        #     IPython.embed()

        # apply to multiple PCs to get multiple cuts, see smb tf01
        #   leads to only one improvement and one FP in 100s traces. Removed again.
        # for rank in range(eigVecS.shape[1]):
        # rank = 0

        rnzCount = 0
        for lc in range(1, eigVecS.shape[0]):
            # IPython.embed()

            # just one PC
            # pcLoadings = eigVecS[:, rank]

            pcLoadings = eigVecS

            if all(abs(pcLoadings[lc - 1]) < relaxedNearZero):
                rnzCount += 1
            else:
                rnzCount = 0

            if conditionE1 and all(abs(pcLoadings[lc - 1]) < nearZero) and any(abs(pcLoadings[lc]) > notableContrib):
                # # # # # # # # # # # # # # # # # # # # # # # #
                # write statistics
                if reportFolder is not None and trace is not None:
                    fn = join(reportFolder, "pca-conditions-e1.csv")
                    writeheader = not exists(fn)
                    with open(fn, "a") as segfile:
                        segcsv = csv.writer(segfile)
                        if writeheader:
                            segcsv.writerow(
                                ["trace", "cluster label", "cluster size",
                                 "max segment length", "# principals",
                                 "rank", "rank principal value",
                                 "is FP",
                                 "near new bound",
                                 "offset", "principal contribution before offset", "principal contribution at offset"]
                            )
                        segcsv.writerow([
                            trace, self.similarSegments.fieldtype, len(self.similarSegments.baseSegments),
                            self.similarSegments.length, sum(self.principalComponents),
                            # rank, eigVsorted[rank][0],
                            "all", "-",
                            repr(lc not in mostCommonTrueBounds) if comparator else "",
                            repr(any(nearBound in relocate for nearBound in [lc, lc - 1, lc + 1])),
                            lc, max(abs(pcLoadings[lc - 1])), max(abs(pcLoadings[lc]))
                        ])

                relocate.append(lc)

            elif conditionE2 and rnzCount >= relaxedNZlength and any(abs(pcLoadings[lc]) > relaxedNotableContrib) \
                    and all(abs(pcLoadings[lc]) < relaxedMaxContrib):
                    # # that is away more than 1 position from another new cut (see smb tf03), leads to only one FP
                    # not any(nearBound in relocate for nearBound in [lc, lc - 1, lc + 1]):

                # # # # # # # # # # # # # # # # # # # # # # # #
                # write statistics
                if reportFolder is not None and trace is not None:
                    fn = join(reportFolder, "pca-conditions-e2.csv")
                    writeheader = not exists(fn)
                    with open(fn, "a") as segfile:
                        segcsv = csv.writer(segfile)
                        if writeheader:
                            segcsv.writerow(
                                ["trace", "cluster label", "cluster size",
                                 "max segment length", "# principals",
                                 "rank", "rnzCount",
                                 "is FP",
                                 "near new bound",
                                 "offset", "principal contribution before offset", "principal contribution at offset"]
                            )
                        segcsv.writerow([
                            trace, self.similarSegments.fieldtype, len(self.similarSegments.baseSegments),
                            self.similarSegments.length, sum(self.principalComponents),
                            "all", rnzCount,
                            repr(lc not in mostCommonTrueBounds) if comparator else "",
                            repr(any(nearBound in relocate for nearBound in [lc, lc - 1, lc + 1])),
                            lc, max(abs(pcLoadings[lc - 1])), max(abs(pcLoadings[lc]))
                        ])

                relocate.append(lc)

        # # To search for a case of the "caveat: precision of numerical method" (see above):
        # if trace == "dns_ictf2010_deduped-100" and self.similarSegments.fieldtype == "tf03" and 7 not in relocate:
        #     print("#"*40 + "\nThis is the conditionE1-bug!\n" + "#"*40)
        #     IPython.embed()

        # # Condition f: inversion of loading of the first principal component if it has a "notable" loading, i. e.,
        # transition from/to: -0.5 <  --|--  > 0.5
        # just concerns ntp tf01
        if conditionF:
            for lc in range(1, eigVecS.shape[0]):
                pcLoadings = eigVecS[:, 0]
                if pcLoadings[lc - 1] < -relaxedNotableContrib and pcLoadings[lc] > relaxedNotableContrib or \
                        pcLoadings[lc - 1] > relaxedNotableContrib and pcLoadings[lc] < -relaxedNotableContrib:

                    # # # # # # # # # # # # # # # # # # # # # # # #
                    # write statistics
                    if reportFolder is not None and trace is not None:
                        fn = join(reportFolder, "pca-conditions-f.csv")
                        writeheader = not exists(fn)
                        with open(fn, "a") as segfile:
                            segcsv = csv.writer(segfile)
                            if writeheader:
                                segcsv.writerow(
                                    ["trace", "cluster label", "cluster size",
                                     "max segment length", "# principals",
                                     "rank principal value",
                                     "is FP",
                                     "near new bound",
                                     "offset", "principal contribution before offset",
                                     "principal contribution at offset"]
                                )
                            segcsv.writerow([
                                trace, self.similarSegments.fieldtype, len(self.similarSegments.baseSegments),
                                self.similarSegments.length, sum(self.principalComponents),
                                eigVsorted[0][0],
                                repr(lc not in mostCommonTrueBounds) if comparator else "",
                                repr(any(nearBound in relocate for nearBound in [lc, lc - 1, lc + 1])),
                                lc, abs(pcLoadings[lc - 1]), abs(pcLoadings[lc])
                            ])

                    relocate.append(lc)

        # # Condition g:
        if conditionG and eigVecS.shape[1] > 1:
            smallLoadingDelta = 0.5

            for lc in range(1, eigVecS.shape[0]):
                pcLoadings = eigVecS

                if max(pcLoadings[lc - 1]) - min(pcLoadings[lc - 1]) < smallLoadingDelta \
                        < pcLoadings[lc - 1].mean() - pcLoadings[lc].mean() \
                        and max(pcLoadings[lc]) - min(pcLoadings[lc]) < smallLoadingDelta:

                    # # # # # # # # # # # # # # # # # # # # # # # #
                    # write statistics
                    if reportFolder is not None and trace is not None:
                        fn = join(reportFolder, "pca-conditions-g.csv")
                        writeheader = not exists(fn)
                        with open(fn, "a") as segfile:
                            segcsv = csv.writer(segfile)
                            if writeheader:
                                segcsv.writerow(
                                    ["trace", "cluster label", "cluster size",
                                     "max segment length", "# principals",
                                     "is FP",
                                     "near new bound",
                                     "offset",
                                     "contribution delta before offset",
                                     "contribution delta at offset",
                                     "contribution shift at offset",
                                     ]
                                )
                            segcsv.writerow([
                                trace, self.similarSegments.fieldtype, len(self.similarSegments.baseSegments),
                                self.similarSegments.length, sum(self.principalComponents),
                                repr(lc not in mostCommonTrueBounds) if comparator else "",
                                repr(any(nearBound in relocate for nearBound in [lc, lc - 1, lc + 1])),
                                lc,
                                max(pcLoadings[lc - 1]) - min(pcLoadings[lc - 1]),
                                max(pcLoadings[lc]) - min(pcLoadings[lc]),
                                pcLoadings[lc - 1].mean() - pcLoadings[lc].mean()
                            ])

                    relocate.append(lc)


        return relocate



    def relocateBoundaries(self, dc: DistanceCalculator = None, kneedleSensitivity:float = 12.0,
                           comparator: MessageComparator = None, reportFolder:str = None):
        import tabulate as tabmod
        tabmod.PRESERVE_WHITESPACE = True

        collectedSubclusters = list()
        collectedSubclusters.extend(self.getSubclusters(dc, kneedleSensitivity))
        relevantSubclusters, eigenVnV, screeKnees = RelocatePCA.filterRelevantClusters(
            [a.similarSegments for a in collectedSubclusters])
        relocatedBounds = dict()
        relocatedCommons = dict()
        for cid, sc in enumerate(collectedSubclusters):  # type: int, RelocatePCA

            # # TODO evaluate:
            # #  if a cluster has no principal components > the threshold, but ones larger than 0, use the padded
            # #  values [0, len] as bounds for all segments in the cluster.D

            if cid in relevantSubclusters:
                relocate = sc.relocateOffsets()

                # prepare different views on the newly proposed offsets
                paddOffs = {bs: sc.similarSegments.paddedPosition(bs) for bs in sc.similarSegments.baseSegments}
                baseOffs = {bs: sc.similarSegments.baseOffset(bs) for bs in sc.similarSegments.baseSegments}
                endOffs = {bs: sc.similarSegments.baseOffset(bs) + bs.length
                           for bs in sc.similarSegments.baseSegments}
                fromEnd = {bs: sc.similarSegments.maxLen - sc.similarSegments.baseOffset(bs) - bs.length
                           for bs in sc.similarSegments.baseSegments}
                minBase = min(baseOffs.values())

                # translate padded offsets to "local segment-wise offsets"
                segSpecificRel = {bs: sorted({rel - baseOffs[bs] for rel in relocate})
                                  for bs in sc.similarSegments.baseSegments}


                # # # # # # # # # # # # # # # # # # # # # # # #
                # generate the new cuts from the proposed bounds
                newRelativeBounds = dict()
                for seg, rel in segSpecificRel.items():
                    newBounds = list()
                    # move vs. add first segment
                    if len(rel) == 0 or rel[0] > 1:
                        newBounds.append(0)
                    # new boundaries
                    for rend in rel:
                        newBounds.append(rend)
                    # move vs. add last segment
                    if len(rel) == 0 or rel[-1] < seg.length - 1:
                        newBounds.append(seg.length)
                    newRelativeBounds[seg] = newBounds
                newPaddingRelative = {bs: [rbound + baseOffs[bs] for rbound in newRelativeBounds[bs]]
                                      for bs in sc.similarSegments.baseSegments}

                # padding-relative positions of boundary moves from that position and moves to that position
                # based on the starts and ends of the original segment bounds.
                moveFrom = dict()
                moveTo = dict()
                for seg, rel in newRelativeBounds.items():
                    moveFrom[seg] = list()
                    moveTo[seg] = list()
                    if rel[0] > 0:
                        moveFrom[seg].append(baseOffs[seg])
                        moveTo[seg].append(newPaddingRelative[seg][0])
                    if rel[-1] < seg.length:
                        moveFrom[seg].append(endOffs[seg])
                        moveTo[seg].append(newPaddingRelative[seg][-1])
                # # # # # # # # # # # # # # # # # # # # # # # #


                # # # # # # # # # # # # # # # # # # # # # # # #
                # padded range refinement (+ preparation)
                commonBounds = RelocatePCA.CommonBoundUtil(baseOffs, endOffs, moveFrom, moveTo)
                cutsExt = commonBounds.frequentBoundReframing(newPaddingRelative, relocate)
                relocatedCommons.update(cutsExt)

                if comparator and reportFolder:
                    # # # # # # # # # # # # # # # # # # # # # # # #
                    # generate value matrix for visualization
                    valMtrx = list()
                    for seg, rel in newRelativeBounds.items():
                        segVal = [seg.bytes.hex()]

                        emptyRelStart = sum(globRel <= baseOffs[seg] for globRel in relocate)
                        emptyRelStart -= 1 if rel[0] > 0 else 0
                        segVal.extend([""]*emptyRelStart)

                        if rel[0] >= 0 and (len(relocate) == 0 or min(relocate) - baseOffs[seg] > 0):
                            prepend = "  " * (baseOffs[seg] - minBase)
                        else:
                            prepend = ""

                        # segment continues
                        if rel[0] > 0:
                            segVal.append(prepend[:-2] + " ~" + seg.bytes[:rel[0]].hex())
                            prepend = ""

                        # determine translated start and end of new boundaries per segment and cut bytes accordingly.
                        for rstart, rend in zip(rel[:-1], rel[1:]):
                            if rend < 0:
                                segVal.append("")
                                prepend = ""
                                continue
                            if rstart < 0:
                                prepend += "  " * -rstart
                                rstart = 0

                            # values of new segment
                            segVal.append(prepend + seg.bytes[rstart:rend].hex())
                            prepend = ""

                        # segment continues
                        if rel[-1] < seg.length:
                            segVal.append(seg.bytes[rel[-1]:].hex() + "~ ")

                        emptyRelEnd = sum(fromEnd[seg] >= sc.similarSegments.maxLen - globRel for globRel in relocate)
                        segVal.extend([""] * emptyRelEnd)

                        valMtrx.append(segVal + [newPaddingRelative[seg]] + [cutsExt[seg]])

                    valTable = tabulate(valMtrx, showindex=True, tablefmt="orgtbl").splitlines()

                    # # # # # # # # # # # # # # # # # # # # # # # #
                    # write statistics for padded range cutting
                    for num, (seg, rel) in enumerate(newPaddingRelative.items()):
                        from os.path import join, exists, basename, splitext
                        import csv

                        scoFile = "padded-range-cutting.csv"
                        scoHeader = ["trace", "cluster label", "segment", "base offset", "length",
                                     "common offset", "com off freq", "max com off freq", # "com off freq" = "common offset frequency"
                                     "shorter than max len",
                                     "start/end", "true bound", "relocate",
                                     "moved away", "moved to", "all moved", "off-by-one...", # ... from bound
                                     "in range",
                                     "sole...", # ... or more common than neighbor
                                     "relative frequency",
                                     "commonUnchangedOffbyone", # a Common that is unchanged and Offbyone
                                     "uobofreq",
                                     "vals"
                                     ]
                        scoTrace = splitext(basename(comparator.specimens.pcapFileName))[0]
                        fn = join(reportFolder, scoFile)
                        writeheader = not exists(fn)

                        # do not filter out unchanged bounds for "off-by-one", see dns tf03
                        oboRel = rel.copy()
                        # TODO the following four lines create more problems than they solve.
                        #  The adverse effect could be reduced by additional conditions adding a lot of complexity.
                        # if baseOffs[seg] in oboRel:
                        #     oboRel.remove(baseOffs[seg])
                        # if endOffs[seg] in oboRel:
                        #     oboRel.remove(endOffs[seg])
                        # COPY IN frequentBoundReframing
                        relocWmargin = RelocatePCA._offbyone(oboRel)

                        # resolve adjacent most common starts/ends (use more common bound)
                        moCoReSt, moCoReEn = commonBounds.filterOutMoreCommonNeighbors(relocWmargin)

                        # resolve adjacent most common starts/ends (use more common bound)
                        commonUnchangedOffbyone = commonBounds.commonUnchangedOffByOne(seg, relocate) # TODO uobo
                        commonUnchanged = commonBounds.commonUnchanged(seg, relocate) # TODO uobo


                        # TODO validate to be inverse of moveAt*
                        # unchangedBounds = list()
                        # if baseOffs[seg] in rel:
                        #     unchangedBounds.append(baseOffs[seg])
                        # if endOffs[seg] in rel:
                        #     unchangedBounds.append(endOffs[seg])
                        # if not commonBounds.unchangedBounds(seg) == unchangedBounds:
                        #     print(commonBounds.unchangedBounds(seg))
                        #     print(unchangedBounds)
                        #     IPython.embed()
                        # assert commonBounds.unchangedBounds(seg) == unchangedBounds
                        unchangedBounds = commonBounds.unchangedBounds(seg, relocate) # TODO uobo


                        # Separately calculate frequency of off-by-one positions of unchanged bounds
                        # (copy of commonBounds.commonUnchangedOffByOne(seg) internals)
                        uoboFreq = {
                            ub + 1:
                                commonBounds.commonStarts[ub + 1] / sum(commonBounds.commonStarts.values())
                            for ub in unchangedBounds if ub + 1 in commonBounds.commonStarts }
                        uoboFreq.update({
                            ub - 1:
                                max(commonBounds.commonEnds[ub - 1] / sum(commonBounds.commonEnds.values()),
                                    uoboFreq[ub - 1] if ub - 1 in uoboFreq else -1)
                            for ub in unchangedBounds if ub - 1 in commonBounds.commonEnds })
                        # uoboFreq = {uobo: max(commonBounds.commonStarts[uobo] / sum(commonBounds.commonStarts.values()),
                        #                 commonBounds.commonEnds[uobo] / sum(commonBounds.commonEnds.values()))
                        #         for uobo in unchangedOffbyone
                        #         if (uobo in commonBounds.commonStarts)
                        #         or (uobo in commonBounds.commonEnds)}


                        # True boundaries for the segments' relative positions
                        fe = [0] + comparator.fieldEndsPerMessage(seg.analyzer.message)
                        offs, nxtOffs = paddOffs[seg]
                        trueOffsets = [o - offs for o in fe if offs <= o <= nxtOffs]

                        if writeheader:
                            with open(fn, "a") as segfile:
                                segcsv = csv.writer(segfile)
                                segcsv.writerow(scoHeader)
                        for com, cnt in commonBounds.commonStarts.most_common():
                            with open(fn, "a") as segfile:
                                segcsv = csv.writer(segfile)
                                segcsv.writerow([
                                    scoTrace, sc.similarSegments.fieldtype, num, baseOffs[seg], seg.length,
                                    com, cnt, commonBounds.commonStarts.most_common(1)[0][1],
                                    "({})".format(sc.similarSegments.maxLen - com),
                                    "start", repr(com in trueOffsets),
                                    repr(com in rel),
                                    repr(com in moveFrom[seg]),
                                    repr(com in moveTo[seg]),
                                    repr(com in commonBounds.allAreMoved),
                                    repr(com in relocWmargin),
                                    repr(com > min(rel)),
                                    repr(com in moCoReSt),
                                    commonBounds.commonStarts[com] / sum(commonBounds.commonStarts.values()),
                                    repr(com in commonUnchanged), # TODO uobo
                                    uoboFreq[com] if com in uoboFreq else "",
                                    valTable[num],
                                ])
                        for com, cnt in commonBounds.commonEnds.most_common():
                            with open(fn, "a") as segfile:
                                segcsv = csv.writer(segfile)
                                segcsv.writerow([
                                    scoTrace, sc.similarSegments.fieldtype, num, baseOffs[seg], seg.length,
                                    com, cnt, commonBounds.commonEnds.most_common(1)[0][1],
                                    sc.similarSegments.maxLen - com,
                                    "end", repr(com in trueOffsets),
                                    repr(com in rel),
                                    repr(com in moveFrom[seg]),
                                    repr(com in moveTo[seg]),
                                    repr(com in commonBounds.allAreMoved),
                                    repr(com in relocWmargin),
                                    repr(com < max(rel)),
                                    repr(com in moCoReEn),
                                    commonBounds.commonEnds[com]/sum(commonBounds.commonEnds.values()),
                                    repr(com in commonUnchanged), # TODO uobo
                                    uoboFreq[com] if com in uoboFreq else "",
                                    valTable[num],
                                ])


                    print()
                    print(sc.similarSegments.fieldtype)
                    print()
                    print(tabulate(valMtrx, showindex=True, headers=["seg", "original"]
                                                    + ["new"] * (len(valMtrx[0]) - 3)
                                                    + ["newBounds", "cutsExt"]
                                   ))
                    print()
                    print(commonBounds.commonStarts.most_common())
                    print(commonBounds.commonEnds.most_common())

                # if comparator.specimens.pcapFileName == "input/dhcp_SMIA2011101X_deduped-100.pcap" \
                #         and sc.similarSegments.fieldtype == "tf09":
                #     IPython.embed()

                # collect new bounds
                relocatedBounds.update(newRelativeBounds)

        tabmod.PRESERVE_WHITESPACE = False

        # TODO relocate boundaries (create new segments)
        #  and place in relocatedSegments
        relocatedSegments = list()
        # TODO what to do with multiple overlapping/contradicting relocations?
        # relocatedBounds
        # relocatedCommons
        from itertools import chain
        messageBounds = {seg.message: [[], []] for seg in chain(relocatedBounds.keys(), relocatedCommons.keys())}
        for seg in chain(relocatedBounds.keys(), relocatedCommons.keys()):  # type: MessageSegment
            origBounds, newBounds = messageBounds[seg.message]
            origBounds.append(seg.offset)
            origBounds.append(seg.nextOffset)
            newBounds.extend([rb + seg.offset for rb in relocatedBounds[seg]])
            newBounds.extend([rc + seg.offset for rc in relocatedCommons[seg]])

        for msg, (origBounds, newBounds) in messageBounds.items():
            print("\n")
            for off in range(len(msg.data)):
                print("v ", end="") if off in origBounds else print("  ", end="")
            print("\n" + msg.data.hex())
            for off in range(len(msg.data)):
                print("^ ", end="") if off in newBounds else print("  ", end="")

        print("\n")

        IPython.embed()

        return relocatedSegments




    @staticmethod
    def _offbyone(reloc: List[int]):
        """
        :param reloc: A list of integer values.
        :return: A list of integer values with their direct off by one neighbors. Sorted and deduplicated.
        """
        return sorted(set([r - 1 for r in reloc] + [r for r in reloc] + [r + 1 for r in reloc]))





    class CommonBoundUtil(object):

        uoboFreqThresh = 0.4

        """
        ...

        Modify counter to treat moved bounds as actual bases and ends
        """
        def __init__(self, baseOffs: Dict[MessageSegment, int], endOffs: Dict[MessageSegment, int],
                     moveFrom: Dict[MessageSegment, List[int]], moveTo: Dict[MessageSegment, List[int]]):
            from collections import Counter

            self._baseOffs = baseOffs
            self._endOffs = endOffs
            self._moveFrom = moveFrom
            self._moveTo = moveTo

            moveAtStart = {seg: baseOffs[seg] in mofro for seg, mofro in moveFrom.items()}  # mofro[0] == baseOffs[seg]
            moveAtEnd = {seg: endOffs[seg] in mofro for seg, mofro in moveFrom.items()}  # mofro[-1] == endOffs[seg]
            commonStarts = Counter(baseOffs.values())
            commonEnds = Counter(endOffs.values())

            self.allAreMoved = [globrel for globrel in commonStarts.keys()
                           if all(moveAtStart[seg] for seg, sstart in baseOffs.items() if globrel == sstart)
                           ] + \
                          [globrel for globrel in commonEnds.keys()
                           if all(moveAtEnd[seg] for seg, send in endOffs.items() if globrel == send)]

            # if all original bounds that constitute a commonStart/End are moved away in all segments of the
            # type, remove from common bounds.
            self.commonStarts = Counter(base if base not in moveFrom[seg] else moveTo[seg][moveFrom[seg].index(base)]
                                        for seg, base in baseOffs.items())
            self.commonEnds   = Counter(end if end not in moveFrom[seg] else moveTo[seg][moveFrom[seg].index(end)]
                                        for seg, end in endOffs.items())


        def filterOutMoreCommonNeighbors(self, relocWmargin: List[int]) -> Tuple[List[int], List[int]]:
            """
            resolve adjacent most common starts/ends (use more common bound)

            :param relocWmargin:
            :return: More common starts and ends, than their neighboring common starts or ends.
            """
            moCoReSt = [cS for cS, cnt in self.commonStarts.most_common()
                        if (cS + 1 not in set(self.commonStarts.keys()).difference(relocWmargin)
                            or cnt > self.commonStarts[cS + 1])  # cS+1 in relocWmargin or
                        and (cS - 1 not in set(self.commonStarts.keys()).difference(relocWmargin)
                             or cnt > self.commonStarts[cS - 1])]  # cS-1 in relocWmargin or
            moCoReEn = [cS for cS, cnt in self.commonEnds.most_common()
                        if (cS + 1 not in set(self.commonEnds.keys()).difference(relocWmargin)
                            or cnt > self.commonEnds[cS + 1])  # cS+1 in relocWmargin or
                        and (cS - 1 not in set(self.commonEnds.keys()).difference(relocWmargin)
                             or cnt > self.commonEnds[cS - 1])]  # cS-1 in relocWmargin or
            # TODO really annoying FP after this change (i. e., the 4 line ends commented out above)
            #  in ntp_SMIA-20111010_deduped-100 tf02
            # set(common[...].keys()).difference(relocWmargin) solves more common neighbor that is filtered by
            # closeness to new bound (dns-new tf01/6 1,2,4,6,...)

            return moCoReSt, moCoReEn


        def unchangedBounds(self, seg: MessageSegment, reloc: List[int]):
            """
            inverse of moveAt*

            :param seg:
            :param reloc: here we need the padding-global raw "relocate" without the added 0 and length
            :return:
            """
            unchangedBounds = [off for off in (self._baseOffs[seg], self._endOffs[seg])
                               if off not in reloc]
            return unchangedBounds


        def commonUnchangedOffByOne(self, seg: MessageSegment, reloc: List[int]):
            """
            Consider only common bounds that are off-by-one from the original bound

            :param seg:
            :return: List of direct neighbors of the unchanged bounds of seg that are common bounds themselves and are
                more frequent than uoboFreqThresh.
            """
            unchangedBounds = self.unchangedBounds(seg, reloc)

            commonUnchangedOffbyone = [
                                          ub + 1 for ub in unchangedBounds
                        if ub + 1 in self.commonStarts
                                          and self.commonStarts[ub + 1] >
                                          RelocatePCA.CommonBoundUtil.uoboFreqThresh * sum(self.commonStarts.values())
                ] + [
                                          ub - 1 for ub in unchangedBounds
                        if ub - 1 in self.commonEnds
                                          and self.commonEnds[ub - 1] >
                                          RelocatePCA.CommonBoundUtil.uoboFreqThresh * sum(self.commonEnds.values())
                ]

            # unchangedOffbyone = [ub - 1 for ub in unchangedBounds] + [ub + 1 for ub in unchangedBounds]
            # commonUnchangedOffbyone = [uobo for uobo in unchangedOffbyone
            #                            if (uobo in self.commonStarts and
            #                                self.commonStarts[uobo] >
            #                                RelocatePCA.CommonBoundUtil.uoboFreqThresh * sum(self.commonStarts.values()))
            #                            or (uobo in self.commonEnds and
            #                                self.commonEnds[uobo] >
            #                                RelocatePCA.CommonBoundUtil.uoboFreqThresh * sum(self.commonEnds.values())
            #                            )]

            # if any(cb for cb in (self.commonStarts, self.commonEnds)
            #         if uobo in cb and
            #         cb[uobo] > RelocatePCA.CommonBoundUtil.uoboFreqThresh * sum(cb.values())
            #        )
            # ]

            return commonUnchangedOffbyone


        def commonUnchanged(self, seg: MessageSegment, reloc: List[int]):
            """
            Consider all common bounds between the start and the outermost (first/last) relocated bound for a segment.

            :param seg:
            :param reloc: here we need the padding-global raw "relocate" without the added 0 and length
            :return:
            """
            commonUnchanged = \
                  [cs for cs in self.commonStarts.keys()
                   if len(reloc) == 0 or cs < min(reloc) > self._baseOffs[seg]] \
                + [ce for ce in self.commonEnds.keys()
                   if len(reloc) == 0 or self._endOffs[seg] > max(reloc) < ce]
            return [cu for cu in commonUnchanged
                    if (cu + 1 > self._endOffs[seg]
                        or ((cu+1 not in self.commonStarts or self.commonStarts[cu+1] < self.commonStarts[cu])
                        and (cu+1 not in self.commonEnds or self.commonEnds[cu+1] < self.commonEnds[cu])))
                    and (cu - 1 < self._baseOffs[seg]
                         or ((cu-1 not in self.commonStarts or self.commonStarts[cu-1] < self.commonStarts[cu])
                         and (cu-1 not in self.commonEnds or self.commonEnds[cu-1] < self.commonEnds[cu])))
                    ]


        def frequentBoundReframing(self, newPaddingRelative: Dict[MessageSegment, List[int]], relocate: List[int]) \
                -> Dict[MessageSegment, List[int]]:
            """
            Frequent raw segment bound reframing:
            Refine boundaries within the padded range of all segments in the cluster considering frequent common offsets
            and end positions.

            :return: The proposed cut positions per segment based on frequent common raw segment bounds in the cluster.
            """
            cutsExt = dict()
            # # # # # # # # # # # # # # # # # # # # # # # #
            # padded range refinement
            for seg, reloc in newPaddingRelative.items():
                # resolve adjacent most common starts/ends (use more common bound)
                commonUnchanged = self.commonUnchanged(seg, relocate) # TODO uobo
                commonUnchangedOffbyone = self.commonUnchangedOffByOne(seg, relocate)  # TODO uobo

                # Used to determine positions that are more than off-by-one from new bound,
                #  naturally includes: is not a move and not a relocation
                relocWmargin = RelocatePCA._offbyone(reloc + commonUnchangedOffbyone) # TODO uobo   + commonUnchanged
                moCoReSt, moCoReEn = self.filterOutMoreCommonNeighbors(relocWmargin)

                cutsExtStart = sorted(common for common in self.commonStarts
                                      # conditions for reframing by common segment starts
                                      if common > min(reloc) and (
                                              common not in relocWmargin and common in moCoReSt
                                              or common in commonUnchangedOffbyone # TODO uobo
                                      )
                                      )
                cutsExtEnd = sorted(common for common in self.commonEnds
                                    # conditions for reframing by common segment ends
                                    if common < max(reloc) and (
                                            common not in relocWmargin and common in moCoReEn
                                            or common in commonUnchangedOffbyone # TODO uobo
                                    )
                                    )
                cutsExt[seg] = cutsExtStart + cutsExtEnd
            return cutsExt






























