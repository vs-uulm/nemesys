import csv, numpy, IPython, logging
from abc import ABC, abstractmethod
from typing import List, Dict, Tuple, Sequence, Iterable, Set, Union
from os.path import join, exists
from itertools import chain

from bitstring import Bits
from kneed import KneeLocator
from tabulate import tabulate
from pyitlib import discrete_random_variable as drv

from netzob.Model.Vocabulary.Messages.AbstractMessage import AbstractMessage

from nemere.inference.segments import MessageSegment, TypedSegment
from nemere.inference.segmentHandler import isExtendedCharSeq
from nemere.inference.templates import FieldTypeContext, DistanceCalculator, Template, ClusterAutoconfException, \
    OPTICSsegmentClusterer, DBSCANadjepsClusterer
from nemere.validation.dissectorMatcher import MessageComparator


def isPrintableChar(char: int):
    """
    :param char: Character value: integer representation of single byte value).
    :return: True if it is a printable character, False otherwise.
    """
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

def isOverlapping(segA: MessageSegment, segB: MessageSegment) -> bool:
    """
    Determines whether the given segmentS overlap.

    >>> from nemere.inference.formatRefinement import isOverlapping
    >>> from nemere.inference.segments import MessageSegment
    >>> from nemere.inference.analyzers import Value
    >>> from netzob.Model.Vocabulary.Messages.RawMessage import RawMessage
    >>> from itertools import combinations
    >>>
    >>> dummymsg = RawMessage(bytes(list(range(20, 40))))
    >>> dummyana = Value(dummymsg)
    >>> nonoverlapping = [ MessageSegment(dummyana, 0, 2), MessageSegment(dummyana, 5, 3),
    ...                    MessageSegment(dummyana, 8, 6), MessageSegment(dummyana, 17, 2) ]
    >>> overlapping1 = [ MessageSegment(dummyana, 0, 2), MessageSegment(dummyana, 1, 3) ]
    >>> overlapping2 = [ MessageSegment(dummyana, 7, 6), MessageSegment(dummyana, 5, 6) ]
    >>> noncomb = combinations(nonoverlapping, 2)
    >>> for nc in noncomb:
    ...     print(isOverlapping(*nc))
    False
    False
    False
    False
    False
    False
    >>> print(isOverlapping(*overlapping1))
    True
    >>> print(isOverlapping(*overlapping2))
    True
    >>> print(isOverlapping(*reversed(overlapping1)))
    True
    >>> print(isOverlapping(*reversed(overlapping2)))
    True

    :param segA: The segment to check against.
    :param segB: The segment to check against.
    :return: Is overlapping or not.
    """
    if segA.message == segB.message \
            and (segA.offset < segB.nextOffset
             and segB.offset < segA.nextOffset):
        return True
    else:
        return False


def entropyOfBytes(byteData: bytes, n=3):
    """
    :param byteData: byte string to calculate the "stream" entropy for.
    :param n: n-gram length in bits.
    :return: Entropy of the bit n-grams of a byte string
    """
    bitData = Bits(bytes=byteData)
    ngrams = [bitData[offset:offset+n].uint for offset in range(len(bitData)-n+1)]
    return drv.entropy(ngrams)/n


def entropyOfXor(byteDataA: bytes, byteDataB: bytes, n=3):
    """

    :param byteDataA: byte string A
    :param byteDataB: byte string B
    :param n: n-gram length in bits.
    :return: Entropy of the bit n-grams of the XOR of two byte strings
    """
    bitDataA = Bits(bytes=byteDataA)
    bitDataB = Bits(bytes=byteDataB)
    trunc = min(len(bitDataA), len(bitDataB))
    xored = bitDataA[:trunc] ^ bitDataB[:trunc]
    ngrams = [xored[offset:offset+n].uint for offset in range(len(xored)-n+1)]
    return drv.entropy(ngrams)/n


class MessageModifier(ABC):
    """Base class for message segmentation refinements."""
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

    @abstractmethod
    def condition(self, segl: MessageSegment, segr: MessageSegment) -> bool:
        """
        A generic condition called to determine whether a merging is necessary.
        Needs to be implemented in subclasses.

        :param segl: left segment
        :param segr: right segment
        :return: True if merging is required, False otherwise.
        """
        raise NotImplementedError("A condition for merging needs to be defined by subclassing.")


class MergeConsecutiveChars(Merger):
    """
    Merge consecutive segments completely consisting of printable-char values into a text field.
    Printable chars are defined as: \t, \n, \r, >= 0x20 and <= 0x7e

    >>> from nemere.inference.segmentHandler import bcDeltaGaussMessageSegmentation
    >>> from nemere.utils.loader import SpecimenLoader
    >>> import nemere.inference.formatRefinement as refine
    >>> from tabulate import tabulate
    >>> sl = SpecimenLoader('../input/deduped-orig/dns_ictf2010_deduped-100.pcap', layer=0, relativeToIP=True)
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

    def condition(self, segl: MessageSegment, segr: MessageSegment):
        """
        Check whether both segments consist of printable characters.

        :param segl: left segment
        :param segr: right segment
        :return: True if merging is required, False otherwise.
        """
        return isPrintable(segl.bytes) and isPrintable(segr.bytes)


class EntropyMerger(Merger):
    """
    Merge consecutive segments if both have an n-gram entropy higher than a threshold and other conditions are met.
    """

    # min entropy above which consecutive segments are merged if both have an n-gram entropy higher than this.
    consecutiveEntropiesThresh = 0.5
    consecutiveEntropiesDiffThresh = 0.2
    # min entropy of the congruence between two segments beginnings above which merging is allowed.
    congruenceEntropiesThresh = 0.8

    # # conservative thresholds: minimal or change, improves as well as reduces quality.
    # consecutiveEntropiesThresh = 0.7
    # consecutiveEntropiesDiffThresh = 0.05
    # congruenceEntropiesThresh = 0.95

    def condition(self, segl: MessageSegment, segr: MessageSegment):
        """
        Merge segments if entropy conditions (see `self.staticCondition()`) are met.

        :param segl: left segment
        :param segr: right segment
        :return: True if merging is required, False otherwise.
        """
        clearToMerge = type(self).staticCondition(segl, segr)
        if clearToMerge:
            logger = logging.getLogger(__name__)
            logger.debug(f"merge random segments: {segl.bytes.hex()} and {segr.bytes.hex()}")
        return clearToMerge

    @staticmethod
    def staticCondition(segl: MessageSegment, segr: MessageSegment):
        """
        Merge ﬁelds of a message carrying random-looking contents and thus split up too much  into a single segment,
        after identifying them by their local entropy. We deﬁne local entropy to be the normalized entropy of
        consecutive bit n-grams. If two subsequent segments have a high local entropy and the diﬀerence between their
        local entropies is small, we identify both segments as likely random. However, there are instances where two
        subsequent segments look random within themselves, but their start is discernible, like counters or timestamps
        that have similar most signiﬁcant bits. To prevent merging those, we bit-wise XOR subsequent segments and
        calculate the local entropy for this combination. We merge consecutive segments if the local entropy of both
        segments is greater than EntropyMerger.congruenceEntropiesThresh,
        their entropy diﬀerence is below EntropyMerger.consecutiveEntropiesDiffThresh,
        and the entropy of the XOR congruence between both segments’ beginnings is greater than 0.95.
        The thresholds are conservative in that they where empirically determined using our test traces
        to minimize false positives.

        :param segl: left segment
        :param segr: right segment
        :return: The two segments should be merged of True, if not this is False.
        """
        eobl = entropyOfBytes(segl.bytes)
        eobr = entropyOfBytes(segr.bytes)
        diff = abs(eobl - eobr)
        seglRandomEnough = eobl > EntropyMerger.consecutiveEntropiesThresh
        segrRandomEnough = eobr > EntropyMerger.consecutiveEntropiesThresh
        congruenceRandomEnough = entropyOfXor(segl.bytes, segr.bytes) > EntropyMerger.congruenceEntropiesThresh
        consecutiveRandomEnough = diff < EntropyMerger.consecutiveEntropiesDiffThresh
        clearToMerge = (consecutiveRandomEnough and seglRandomEnough and segrRandomEnough and congruenceRandomEnough) \
                       or diff < 0.01
        return clearToMerge


class FrequencyRiseMerger(Merger):
    """start new segment only when frequency rises"""

    def __init__(self, segments: List[TypedSegment], littleEndian=False):
        """Abuses the fieldtype of TypedSegment to hold the occurrence frequency value."""
        super().__init__(segments)
        self._littleEndian = littleEndian
        # hold (retain) the last frequency value for all following unknown segment
        self._holdPeak = dict()
        lastFrequency = 0
        for seg in reversed(segments) if self._littleEndian else segments:
            secondToLast = lastFrequency
            lastFrequency = seg.fieldtype if isinstance(seg.fieldtype, int) else lastFrequency
            lastFrequency = lastFrequency if lastFrequency >= secondToLast \
                else secondToLast - (secondToLast - lastFrequency) // 4  # dampen downwards
            self._holdPeak[seg] = lastFrequency

    def condition(self, segl: TypedSegment, segr: TypedSegment) -> bool:
        """
        Condition check to merge segments if the frequency of byte values rises.

        :param segl: left segment
        :param segr: right segment
        :return: True if merging is required, False otherwise.
        """
        if self._littleEndian:
            return self._holdPeak[segl] <= self._holdPeak[segr]
        else:
            return self._holdPeak[segl] >= self._holdPeak[segr]


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
        else:
            mangledSegments = self.segments
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
        :param segl: The current segment
        :return: The count of printable chars at the end of the segment
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
        :param segr: The current segment
        :return: The count of printable chars at the beginning of the segment
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

        >>> from nemere.inference.segmentHandler import bcDeltaGaussMessageSegmentation
        >>> from nemere.utils.loader import SpecimenLoader
        >>> import nemere.inference.formatRefinement as refine
        >>> from tabulate import tabulate
        >>> sl = SpecimenLoader('../input/hide/random-100-continuous.pcap', layer=0, relativeToIP=True)
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
        """
        :return: The frequencies of all pairs of consecutive values in this set of segments.
        """
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
    Find common values of segments and split/crop other, larger segments if they contain these values.
    """

    minSegmentLength = 2
    frequencyThreshold = 0.1
    """fraction of *messages* to exhibit the value to be considered frequent"""

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
        """
        :param segmentedMessages: The segments to analyze
        :return: The most common byte values of the given segments
            "Most common" is dynamically defined as those with a frequency above
            CropDistinct.frequencyThreshold * len(segmentedMessages)
        """
        from collections import Counter
        from itertools import chain
        segcnt = Counter([seg.bytes for seg in chain.from_iterable(segmentedMessages)])
        segFreq = segcnt.most_common()
        freqThre = CropDistinct.frequencyThreshold * len(segmentedMessages)
        thre = 0
        while thre < len(segFreq) and segFreq[thre][1] > freqThre:
            thre += 1
        # by the "if" in list comprehension: omit \x00-sequences and shorter than {minSegmentLength}-byte long segments
        moco = [fv for fv, ct in segFreq[:thre] if set(fv) != {0} and len(fv) >= CropDistinct.minSegmentLength]
        # return moco

        # omit all sequences that have common subsequences
        mocoShort = [m for m in moco if not any(m != c and m.find(c) > -1 for c in moco)]
        return mocoShort

    def split(self):
        """
        Perform the split of segments that contain frequent substrings.
        :return: The message segments with the segments replaced by refined segments.
        """
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
    Merge consecutive segments that together fulfill the char conditions in inference.segmentHandler.isExtendedCharSeq
    """

    def merge(self):
        """
        Perform the merging.

        >>> from nemere.utils.loader import SpecimenLoader
        >>> from nemere.inference.segmentHandler import bcDeltaGaussMessageSegmentation
        >>> from nemere.inference.formatRefinement import CumulativeCharMerger
        >>> sl = SpecimenLoader('../input/deduped-orig/dns_ictf2010_deduped-100.pcap', layer=0, relativeToIP=True)
        >>> segmentsPerMsg = bcDeltaGaussMessageSegmentation(sl)
        Segmentation by inflections of sigma-0.6-gauss-filtered bit-variance.
        >>> for messageSegments in segmentsPerMsg:
        ...     ccm = CumulativeCharMerger(messageSegments)
        ...     ccmmsg = ccm.merge()
        ...     if ccmmsg != messageSegments:
        ...         sgms = b''.join([m.bytes for m in ccmmsg])
        ...         sgss = b''.join([m.bytes for m in messageSegments])
        ...         if sgms != sgss:
        ...             print("Mismatch!")

        :return: A new set of segments after the input has been merged
        """
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
        """
        :param segmentID: The index of the segment to split within the sequence of segments composing the message
        :param chunkLength: The fixed length of the target segments in bytes
        :return: The message segments with the given segment replaced by multiple segments of the given fixed length.
        """
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
    """Improve its raw segments by relocating inferred boundaries near the beginning and end of sequences of nulls."""

    def __init__(self, segments: List[MessageSegment], comparator: MessageComparator):
        """
        :param segments: The segments of one message in offset order.
        :param comparator: Ground truth
        """
        super().__init__(segments)
        self._parsedMessage = comparator.parsedMessages[comparator.messages[segments[0].message]]
        self.counts = None
        self.doprint = False

    def split(self):
        """
        Split message segments:
        1. If the last bytes before the nulls fulﬁll the character heuristic, assume it is a null-termination for the
           string and allocate the nulls to the end of the character segment;
        1. otherwise, assign the null-bytes to the following segment, assuming they are the
           unset most signiﬁcant bytes of a number value.

        :return: The new segmentations of the messages.
        """
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
        Counts of different kinds of off-by-one situations for zeros. Results are placed in `self.counts`.
        """
        if amounts is None:
            self.counts = {
                "0t":0, "0f":0, "-1tr":0, "-1fr":0, "-1tl":0, "-1fl":0, "+1tr":0, "+1fr":0, "+1tl":0, "+1fl":0,
                "-1.0":0, "+1.0":0
            }  # key: boundary offset change + true/false
        else:
            self.counts = amounts


class RelocatePCA(object):
    """
    Method to refine the segmentation by principle component analysis (PCA) to discover
    linearly correlated variance between sets of message segments.
    """
    # PCA conditions parameters
    minSegLen = 3  # minimum number of cluster members
    maxAbsolutePrincipals = 4  # absolute maximum of significant principal components for PCA/sub-clustering
    screeMinThresh = 10  # threshold for the minimum of a component to be considered principal
    principalCountThresh = .5   # threshold for the maximum number of allowed principal components to start the analysis;
            # interpreted as fraction of the component count as dynamic determination.
    contributionRelevant = 0.1  # 0.12; threshold for the minimum difference from 0
            # to consider a loading component in the eigenvector as contributing to the variance
    # maxLengthDelta = 30  # (max true: 6 in dns-new, min false 9 in dhcp)   # replaced by maxLengthDeltaRatio
    maxLengthDeltaRatio = 0.5  # threshold for the ratio between sortest and longest segment.

    pcDeltaMin = 0.98

    # e1 parameters
    nearZero = 0.030  # 0.003
    notableContrib = 0.75  # 0.66  # or 0.7 (see smb tf04)
    # peak may be (near) +/- 1.0 in most cases, but +/- 0.5 includes also ntp tf06 and smb tf04,
    #   however, it has false positive dhcp tf01 and smb tf00.4

    # e2 parameters
    # also apply to higher nearZero and lower notableContrib if longer (>= 4) sequence of nearZero
    #   precedes notableContrib
    relaxedNearZero = 0.050 # 0.004
    relaxedNZlength = 4
    relaxedNotableContrib = 0.005 # 0.003
    relaxedMaxContrib = 1.0 # 0.66

    def __init__(self, similarSegments: FieldTypeContext,
                 eigenValuesAndVectors: Tuple[numpy.ndarray, numpy.ndarray]=None,
                 screeKnee: float=None, littleEndian: bool=False):
        """
        :param similarSegments: Segments that are similar enough so that
            we do not measure random but systematic variance.
        :param eigenValuesAndVectors: The Eigenvalues and -vectors of the covariance matrix of similarSegments.
        :param screeKnee: The knee in the scree graph of the given Eigenvalues.
        :param littleEndian: If the protocol should be treated as little endian.
        """
        self._similarSegments = similarSegments
        self._eigen = eigenValuesAndVectors if eigenValuesAndVectors is not None \
            else numpy.linalg.eigh(similarSegments.cov)
        """Tuple of eigenvalues (one-dimensional ndarray) and eigenvectors (two-dimensional ndarray)"""
        self._screeKnee = screeKnee if screeKnee is not None \
            else RelocatePCA.screeKneedle(self._eigen)

        self._screeThresh = max(self._screeKnee, RelocatePCA.screeMinThresh, max(self._eigen[0]) / 10) \
                                if any(self._eigen[0] < RelocatePCA.screeMinThresh) \
                                else RelocatePCA.screeMinThresh
        """
            defines a minimum for any principal component, that is
             * at least the knee in scree,
             * not less than one magnitude smaller then the first PC,
             * and larger than the absolute minimum.
        """
        self._principalComponents = self._eigen[0] > self._screeThresh
        """**significant** principal components"""
        self._contribution = self._eigen[1][:, self._principalComponents]  # type: numpy.ndarray
        """contributions of loadings per **significant** principal component"""
        self._littleEndian = littleEndian

        self._testSubclusters = None

    @property
    def similarSegments(self):
        """
        :return: Segments that are contained in this object.
        """
        return self._similarSegments

    @property
    def eigen(self):
        """
        :return: Eigenvalues and -vectors
        """
        return self._eigen

    @staticmethod
    def screeKneedle(eigenValuesAndVectors: Tuple[numpy.ndarray, numpy.ndarray]) -> float:
        """
        Scree is the name of the downward curve of the eigenvalues of the principal components. Typically, it has a
        steep drop from the components with the most significant contribution to the variance towards a leveled tail of
        low eigenvalues. The transition between the principal components with a significant contribution and the
        negligible components is marked by the knee of the curve, according to common analysis methods. To determine
        the knee, we facilitate the Kneedle algorithm (`KneeLocator`).

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

        return scree[kl.knee]

    @staticmethod
    def filterForSubclustering(fTypeContext: Sequence[FieldTypeContext]):
        """
        :param fTypeContext: Input clusters.
        :return: List of cluster IDs that potentially need to be subclustered.
            For the current approach, return all clusters.
        """
        interestingClusters = [
            cid for cid, clu in enumerate(fTypeContext)
            # if not clu.length < RelocatePCA.minSegLen
            #    and not sum([isExtendedCharSeq(seg.bytes) for seg in clu.baseSegments]) > .5 * len(clu.baseSegments)
               # and not all(clu.stdev == 0)
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

        # leave only clusters of at least minSegLen (=6)
        minClusterSize = RelocatePCA.minSegLen
        for cid in interestingClusters.copy():
            # must be at least minClusterSize unique segment values
            if len(set([bs.bytes for bs in fTypeContext[cid].baseSegments])) < minClusterSize:
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

        # # remove all clusters having a segment length difference of more than maxLengthDelta  # TODo removed due to ntp quality in zeropca-016-refinement-79d3ba4-ntp and nemesys-107-refinement-79d3ba4
        # for cid in interestingClusters.copy():
        #     bslen = {bs.length for bs in fTypeContext[cid].baseSegments}
        #     if min(bslen) / max(bslen) < RelocatePCA.maxLengthDeltaRatio:
        #         interestingClusters.remove(cid)
        #         print("Removed cluster {} for too high segment length difference ratio of {}".format(
        #             fTypeContext[cid].fieldtype,
        #             (max(bslen) - min(bslen))/max(bslen)
        #         ))

        return interestingClusters, eigenVnV, screeKnees

    @property
    def principalComponents(self):
        """:return: **significant** principal components (scores)"""
        return self._principalComponents

    @property
    def screeThresh(self):
        """return: The scree-dependent minimum for any principal component."""
        return self._screeThresh

    @property
    def contribution(self):
        """:return: Contributions of loadings per **significant** principal component"""
        return self._contribution

    @staticmethod
    def _preFilter(segments: Sequence[MessageSegment], label: str) -> Tuple[FieldTypeContext, bool]:
        """
        Create FieldTypeContext object from segments and
        apply to decide about the next recursion in sub-clustering.

        Filter conditions are:
            * enoughSegments
            * enoughVariance
            * notOnlyChars

        :param segments: cluster result as raw list of segments
        :param label: cluster label
        :return: tuple of a
            * FieldTypeContext object for the list of segments and
            * the result of the filter, telling whether the cluster can be further analyzed by PCA or sub-clustering:
                exclude if false.
        """
        # resolve templates into their single segments
        resolvedSegments = list()
        for seg in segments:
            if isinstance(seg, Template):
                resolvedSegments.extend(seg.baseSegments)
            else:
                resolvedSegments.append(seg)

        # Align segments. Calc mean, std, cov, ...
        ftContext = FieldTypeContext(resolvedSegments)
        ftContext.fieldtype = label

        return ftContext, RelocatePCA.__preFilter(ftContext)

    @staticmethod
    def __preFilter(ftContext: FieldTypeContext) -> bool:
        """
        Apply to decide about the next recursion in sub-clustering.

        Filter conditions are:
            * enoughSegments
            * enoughVariance
            * notOnlyChars

        :param ftContext: cluster result as FieldTypeContext object
        :return: the result of the filter, telling whether the cluster can be further analyzed by PCA or sub-clustering:
                exclude if false.
        """
        uniqueSegVals = len(set(bs.bytes for bs in ftContext.baseSegments))
        enoughSegments = uniqueSegVals >= RelocatePCA.minSegLen
        enoughVariance = not all(ftContext.stdev == 0)
        notOnlyChars = not all([isExtendedCharSeq(seg.bytes) for seg in ftContext.baseSegments])

        if not enoughSegments:
            print("Cluster {} has not enough unique segments ({}).".format(ftContext.fieldtype, uniqueSegVals))
        if not enoughVariance:
            print("Cluster {} has no variance.".format(ftContext.fieldtype))
        if not notOnlyChars:
            print("Cluster {} has only chars.".format(ftContext.fieldtype))

        return enoughSegments and enoughVariance and notOnlyChars

    def _meetsPCAprerequisites(self) -> bool:
        """
        Apply at the beginning of a new sub-clustering recusion between PCA or further sub-clustering.

        :return: Result of the test, whether all PCA prerequisites are met.
        """
        # number of principal components, including fallback in self._screeThresh if there is no knee
        tooManyPCs = sum(self._eigen[0] > self._screeThresh) > \
            min(RelocatePCA.maxAbsolutePrincipals, self._eigen[0].shape[0] * RelocatePCA.principalCountThresh)
        # segment length difference is too high
        bslen = {bs.length for bs in self.similarSegments.baseSegments}
        tooHighLenDiff = min(bslen) / max(bslen) < RelocatePCA.maxLengthDeltaRatio
        charSegCount = sum([isExtendedCharSeq(seg.bytes) for seg in self._similarSegments.baseSegments])
        tooManyChars = charSegCount > .5 * len(self._similarSegments.baseSegments)

        if tooManyPCs:
            print("Cluster {} needs reclustering: too many principal components ({}/{}).".format(
                self._similarSegments.fieldtype, sum(self._eigen[0] > self._screeThresh), self._eigen[0].shape[0]))
        if tooHighLenDiff:
            print("Cluster {} needs reclustering: length difference too high ({:.2f}).".format(
                self._similarSegments.fieldtype, min(bslen) / max(bslen)))
        if tooManyChars:
            print("Cluster {} needs reclustering: too many char segments ({:d}).".format(
                self._similarSegments.fieldtype, charSegCount))

        return not tooManyPCs and not tooHighLenDiff and not tooManyChars

    def getSubclusters(self, dc: DistanceCalculator = None, S: float = None,
                       reportFolder: str = None, trace: str = None) -> List[Union['RelocatePCA', FieldTypeContext]]:
        """
        Recursive sub-cluster.

        :param dc: Distance calculator for clustering.
        :param S: Kneedle sensitivity parameter for autodetection of DBSCAN clustering parameter.
        :param reportFolder:
        :param trace:
        TODO :param collectEvaluationData: For evaluation: Collect the intermediate (sub-)clusters generated during
            the analysis of the segments.
        :return: A flat list of subclusters. If no subclustering was necessary or possible, returns itself.
        """
        # # # # # # # # # # # # # # # # # # # # # # # #
        # terminate recursion and return self for PCA
        if self._meetsPCAprerequisites():
            return [self]
        # # # # # # # # # # # # # # # # # # # # # # # #

        # no DC available for clustering
        if dc is None:
            print("No dissimilarities available. Ignoring cluster {}.".format(self._similarSegments.fieldtype))
            return []

        # Sub-Cluster
        try:
            clusterer = DBSCANadjepsClusterer(dc, segments=self._similarSegments.baseSegments, S=S)
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

            # if there remains only noise, ignore cluster
            if len(clusters) == 0:
                # TODO test impact
                if RelocatePCA.__preFilter(self._similarSegments):
                    # # # # # # # # # # # # # # # # # # # # # # # #
                    # terminate recursion and return self for PCA as "last resort"
                    print("Use super-cluster {}: only noise.".format(self._similarSegments.fieldtype))
                    return [self]
                else:
                    print("Ignore cluster {}: only noise.".format(self._similarSegments.fieldtype))
                    return []

            subclusters = list()
            for cid, segments in enumerate(clusters):
                cLabel = "{}.{}".format(self._similarSegments.fieldtype, cid)
                # Generate suitable FieldTypeContext objects from the sub-clusters
                ftContext, doRecurse = RelocatePCA._preFilter(segments, cLabel)
                # if basic requirements not met, exclude from PCA analysis
                if not doRecurse:
                    # # # # # # # # # # # # # # # # # # # # # # # #
                    # stop further recursion and ignore this cluster
                    print("Ignore subcluster {} due to pre-filter.".format(ftContext.fieldtype))
                    # performing "common bound" refinement here does not improve the results:
                    # * is triggered in only few cases
                    # * there are just a handfull of segments that may benefit
                    # * there is some probability to produce more FP than TP.
                    continue
                    # # # # # # # # # # # # # # # # # # # # # # # #
                print("Analyzing sub-cluster", ftContext.fieldtype)
                try:
                    subcluster = RelocatePCA(ftContext, littleEndian=self._littleEndian)
                    # # # # # # # # # # # # # # # # # # # # # # # #
                    # descend into recursion
                    subclusters.extend(subcluster.getSubclusters(dc, S, reportFolder, trace))
                    # # # # # # # # # # # # # # # # # # # # # # # #
                except numpy.linalg.LinAlgError:
                    print("Ignore subcluster due to eigenvalues did not converge")
                    print(repr(ftContext.baseSegments))
            return subclusters
        except ClusterAutoconfException as e:
            print(e)
            return [self]

    def relocateOffsets(self, reportFolder:str = None, trace:str = None, comparator: MessageComparator = None,
                        conditionA = True, conditionE1 = False, conditionE2 = True,
                        conditionF = False, conditionG = False):
        """
        Determine candidate offsets for relocation according to the PCA method using the given enabled interpretation
        conditions. Contains a series of debugging options that allow to trace the origin for refinement decisions.

        :param conditionA:  Enable Condition A
        :param conditionE1: Enable Condition E1
        :param conditionE2: Enable Condition E2
        :param conditionF:  Enable Condition F
        :param conditionG:  Enable Condition G
        :param reportFolder: For debugging
        :param trace:        For debugging
        :param comparator:   For debugging
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
        # "component analysis"
        #
        # the principal components (i. e. with Eigenvalue > thresh) of the covariance matrix are assumed to peak
        # towards the end of varying fields with similar content (e. g. counting numbers).
        # The component is near 1 or -1 in the Eigenvector of the respective Eigenvalue.

        relocate = list()  # new boundaries found: relocate the next end to this relative offset
        # relocateFromStart = list()  # new boundaries found: relocate the previous start to this relative offset

        # continue only if we have some principal components
        if not self.principalComponents.any():
            return relocate

        littleEndian = self._littleEndian
        if littleEndian:
            print("Assuming little endian")

        # # Condition a: Covariance ~0 after non-0
        # at which eigenvector component does any of the principal components have a relevant contribution
        contributingLoadingComponents = (abs(self._contribution) > RelocatePCA.contributionRelevant).any(1)
        contribution = self._contribution

        if littleEndian:
            contributingLoadingComponents = numpy.flip(contributingLoadingComponents)
            contribution = numpy.flip(contribution)

        if conditionA:
            # lc (loading component) is equivalent to the byte offset in the message
            for lc in reversed(range(1, contributingLoadingComponents.shape[0])):
                # a "significant" relative drop in covariance
                relPCdelta = (abs(contribution[lc - 1]).max() - abs(contribution[lc]).max()) \
                             / abs(contribution[lc - 1]).max()

                if not contributingLoadingComponents[lc] and contributingLoadingComponents[lc - 1] \
                        and relPCdelta > RelocatePCA.pcDeltaMin:
                    if littleEndian:
                        offset = contributingLoadingComponents.shape[0] - lc
                    else:
                        offset = lc
                    relocate.append(offset)

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
                                repr(offset not in mostCommonTrueBounds) if comparator else "",
                                repr(0 in mostCommonTrueBounds) if comparator else "",
                                repr(self.similarSegments.length in mostCommonTrueBounds) if comparator else "",
                                offset, abs(contribution[lc - 1]).max(), abs(contribution[lc]).max()
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


        # prepare list of eigen vectors sorted by eigen values
        eigVsorted = list(reversed(sorted([(val, vec) for val, vec in zip(
            self._eigen[0][self.principalComponents], self.contribution.T)],
                                          key=lambda x: x[0])))
        eigVecS = numpy.array([colVec[1] for colVec in eigVsorted]).T

        # TODO caveat: precision of numerical method for cov or eigen does not suffice for near zero resolution
        #  in all cases. e. g. setting nearZero to 0.003 indeterministically results in a false negative for the
        #  condition. Setting it higher, might introduce more false positives.

        # # Condition e: Loading peak of principal component rising from (near) 0.0
        #

        # apply to multiple PCs to get multiple cuts, see smb tf01
        #   leads to only one improvement and one FP in 100s traces. Removed again.
        # for rank in range(eigVecS.shape[1]):
        # rank = 0

        rnzCount = 0
        for lc in range(1, eigVecS.shape[0]):
            # alternative: just one PC
            # pcLoadings = eigVecS[:, rank]

            if littleEndian:
                pcLoadings = numpy.flip(eigVecS)
            else:
                pcLoadings = eigVecS
            if all(abs(pcLoadings[lc - 1]) < RelocatePCA.relaxedNearZero):
                rnzCount += 1
            else:
                rnzCount = 0
            relPCdelta = (abs(pcLoadings[lc]).max() - abs(pcLoadings[lc - 1]).max()) / abs(pcLoadings[lc]).max()

            if conditionE1 \
                    and all(abs(pcLoadings[lc - 1]) < RelocatePCA.nearZero) \
                    and any(abs(pcLoadings[lc]) > RelocatePCA.notableContrib):
                if littleEndian:
                    offset = eigVecS.shape[0] - lc
                else:
                    offset = lc
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
                            repr(offset not in mostCommonTrueBounds) if comparator else "",
                            repr(any(nearBound in relocate for nearBound in [offset, offset - 1, offset + 1])),
                            offset, max(abs(pcLoadings[lc - 1])), max(abs(pcLoadings[lc]))
                        ])
                relocate.append(offset)

            # has been close to zero for at least the previous **relaxedNZlength** bytes
            # and any loading is greater than relaxedNotableContrib
            # and all loadings are less than relaxedMaxContrib
            # and ...
            elif conditionE2 \
                    and rnzCount >= RelocatePCA.relaxedNZlength \
                    and any(abs(pcLoadings[lc]) > RelocatePCA.relaxedNotableContrib) \
                    and all(abs(pcLoadings[lc]) < RelocatePCA.relaxedMaxContrib) \
                    and relPCdelta > RelocatePCA.pcDeltaMin:
                    # # that is away more than 1 position from another new cut (see smb tf03), leads to only one FP
                    # not any(nearBound in relocate for nearBound in [lc, lc - 1, lc + 1]):
                if littleEndian:
                    offset = eigVecS.shape[0] - lc
                else:
                    offset = lc
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
                            repr(offset not in mostCommonTrueBounds) if comparator else "",
                            repr(any(nearBound in relocate for nearBound in [offset, offset - 1, offset + 1])),
                            lc, max(abs(pcLoadings[lc - 1])), max(abs(pcLoadings[lc]))
                        ])
                relocate.append(offset)

        # # To search for a case of the "caveat: precision of numerical method" (see above):
        # if trace == "dns_ictf2010_deduped-100" and self.similarSegments.fieldtype == "tf03" and 7 not in relocate:
        #     print("#"*40 + "\nThis is the conditionE1-bug!\n" + "#"*40)
        #     IPython.embed()

        # # Condition f: inversion of loading of the first principal component if it has a "notable" loading, i. e.,
        # transition from/to: -0.5 <  --|--  > 0.5
        # just concerns ntp tf01
        if conditionF:
            # TODO little endian not implemented!
            for lc in range(1, eigVecS.shape[0]):
                pcLoadings = eigVecS[:, 0]
                if pcLoadings[lc - 1] < -RelocatePCA.relaxedNotableContrib \
                            and pcLoadings[lc] > RelocatePCA.relaxedNotableContrib \
                        or pcLoadings[lc - 1] > RelocatePCA.relaxedNotableContrib \
                            and pcLoadings[lc] < -RelocatePCA.relaxedNotableContrib:

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
            # TODO little endian not implemented!
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

    def relocateBoundaries(self, comparator: MessageComparator = None, reportFolder:str = None) \
            -> Dict[MessageSegment, List[int]]:
        """
        Determine new boundaries for all segments in the RelocatePCA object.
        Performed by:
            * relocatedBounds
            * relocatedCommons

        :param comparator: For evaluation: Encapsulated true field bounds to compare results to.
        :param reportFolder: For evaluation: Destination path to write results and statistics to.
        :return: Relocated boundaries.
        """
        from os.path import splitext, basename
        trace = splitext(basename(comparator.specimens.pcapFileName))[0] if comparator else None

        # prepare proposed new bounds
        relocate = self.relocateOffsets(reportFolder, trace, comparator)

        # prepare different views on the newly proposed offsets
        paddOffs = {bs: self.similarSegments.paddedPosition(bs) for bs in self.similarSegments.baseSegments}
        baseOffs = {bs: self.similarSegments.baseOffset(bs) for bs in self.similarSegments.baseSegments}
        endOffs = {bs: self.similarSegments.baseOffset(bs) + bs.length
                   for bs in self.similarSegments.baseSegments}
        fromEnd = {bs: self.similarSegments.maxLen - self.similarSegments.baseOffset(bs) - bs.length
                   for bs in self.similarSegments.baseSegments}
        minBase = min(baseOffs.values())

        # translate padded offsets to "local segment-wise offsets"
        segSpecificRel = {bs: sorted({rel - baseOffs[bs] for rel in relocate})
                          for bs in self.similarSegments.baseSegments}

        # # # # # # # # # # # # # # # # # # # # # # # #
        # generate the new cuts from the proposed bounds
        relocatedBounds = dict()
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
            relocatedBounds[seg] = newBounds
        newPaddingRelative = {bs: [rbound + baseOffs[bs] for rbound in relocatedBounds[bs]]
                              for bs in self.similarSegments.baseSegments}
        # # # # # # # # # # # # # # # # # # # # # # # #

        # # # # # # # # # # # # # # # # # # # # # # # #
        # padded range refinement (+ preparation)
        #
        # padding-relative positions of boundary moves from that position and moves to that position
        # based on the starts and ends of the original segment bounds.
        moveFrom = dict()
        moveTo = dict()
        for seg, rel in relocatedBounds.items():
            moveFrom[seg] = list()
            moveTo[seg] = list()
            if rel[0] > 0:
                moveFrom[seg].append(baseOffs[seg])
                moveTo[seg].append(newPaddingRelative[seg][0])
            if rel[-1] < seg.length:
                moveFrom[seg].append(endOffs[seg])
                moveTo[seg].append(newPaddingRelative[seg][-1])
        # # # # # # # # # # # # # # # # # # # # # # # #
        commonBounds = RelocatePCA.CommonBoundUtil(baseOffs, endOffs, moveFrom, moveTo)
        relocatedCommons = commonBounds.frequentBoundReframing(newPaddingRelative, relocate)
        # # # # # # # # # # # # # # # # # # # # # # # #

        if comparator and reportFolder:
            import tabulate as tabmod
            tabmod.PRESERVE_WHITESPACE = True

            # # # # # # # # # # # # # # # # # # # # # # # #
            # generate value matrix for visualization
            valMtrx = list()
            for seg, rel in relocatedBounds.items():
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

                emptyRelEnd = sum(fromEnd[seg] >= self.similarSegments.maxLen - globRel for globRel in relocate)
                segVal.extend([""] * emptyRelEnd)

                valMtrx.append(segVal + [newPaddingRelative[seg]] + [relocatedCommons[seg]])

            valTable = tabulate(valMtrx, showindex=True, tablefmt="orgtbl").splitlines()

            # # # # # # # # # # # # # # # # # # # # # # # #
            # write statistics for padded range cutting
            for num, (seg, rel) in enumerate(newPaddingRelative.items()):
                from os.path import join, exists, basename, splitext
                import csv

                scoFile = "padded-range-cutting.csv"
                scoHeader = ["trace", "cluster label", "segment", "base offset", "length",
                             "common offset (CO)",
                             "this CO's freq", "max of all CO freq.s", # "CO/com off freq" = "common offset frequency"
                             "shorter than max len", "accepted CO",
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
                # commonUnchangedOffbyone = commonBounds.commonUnchangedOffByOne(seg, relocate)
                commonUnchanged = commonBounds.commonUnchanged(seg, relocate)

                unchangedBounds = commonBounds.unchangedBounds(seg, relocate)

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
                            scoTrace, self.similarSegments.fieldtype, num, baseOffs[seg], seg.length,
                            com, repr(com in relocatedCommons[seg]),
                            cnt, commonBounds.commonStarts.most_common(1)[0][1],
                            "({})".format(self.similarSegments.maxLen - com),
                            "start", repr(com in trueOffsets),
                            repr(com in rel),
                            repr(com in moveFrom[seg]),
                            repr(com in moveTo[seg]),
                            repr(com in commonBounds.allAreMoved),
                            repr(com in relocWmargin),
                            repr(com > min(rel)),
                            repr(com in moCoReSt),
                            commonBounds.commonStarts[com] / sum(commonBounds.commonStarts.values()),
                            repr(com in commonUnchanged),
                            uoboFreq[com] if com in uoboFreq else "",
                            valTable[num],
                        ])
                for com, cnt in commonBounds.commonEnds.most_common():
                    with open(fn, "a") as segfile:
                        segcsv = csv.writer(segfile)
                        segcsv.writerow([
                            scoTrace, self.similarSegments.fieldtype, num, baseOffs[seg], seg.length,
                            com, repr(com in relocatedCommons[seg]),
                            cnt, commonBounds.commonEnds.most_common(1)[0][1],
                            self.similarSegments.maxLen - com,
                            "end", repr(com in trueOffsets),
                            repr(com in rel),
                            repr(com in moveFrom[seg]),
                            repr(com in moveTo[seg]),
                            repr(com in commonBounds.allAreMoved),
                            repr(com in relocWmargin),
                            repr(com < max(rel)),
                            repr(com in moCoReEn),
                            commonBounds.commonEnds[com]/sum(commonBounds.commonEnds.values()),
                            repr(com in commonUnchanged),
                            uoboFreq[com] if com in uoboFreq else "",
                            valTable[num],
                        ])

            tabmod.PRESERVE_WHITESPACE = False

        # collect new bounds
        relocatedBoundaries = {seg: list() for seg in self.similarSegments.baseSegments}
        for segment, newBounds in relocatedBounds.items():
            relocatedBoundaries[segment].extend([int(rb + segment.offset) for rb in newBounds])
            assert len(relocatedBoundaries[segment]) == len(set(relocatedBoundaries[segment]))
        for segment, newCommons in relocatedCommons.items():
            relocatedBoundaries[segment].extend([int(rc + segment.offset) for rc in newCommons
                                                 if int(rc + segment.offset) not in relocatedBoundaries[segment]])
            if not len(relocatedBoundaries[segment]) == len(set(relocatedBoundaries[segment])):
                IPython.embed()
            assert len(relocatedBoundaries[segment]) == len(set(relocatedBoundaries[segment])), \
                repr(relocatedBoundaries[segment]) + "\n" + repr(set(relocatedBoundaries[segment]))

        return relocatedBoundaries

    @staticmethod
    def _offbyone(reloc: List[int]):
        """
        :param reloc: A list of integer values.
        :return: A list of integer values with their direct off by one neighbors. Sorted and deduplicated.
        """
        return sorted(set([r - 1 for r in reloc] + [r for r in reloc] + [r + 1 for r in reloc]))


    class CommonBoundUtil(object):
        """
        Utility class to refine boundaries within the padded range of all segments in a cluster considering
        frequent common offsets and end positions.
        """

        commonFreqThreshold = 0.4
        """Threshold for the frequency of a bound has at least to have 
        to be considered common in CommonBoundUtil#frequentBoundReframing()."""
        uoboFreqThresh = 0.4 # 0.8
        """Threshold for the frequency 
        direct neighbors of unchanged bounds of a segment that are common bounds themselves 
        have to have at least to be considered in CommonBoundUtil#commonUnchangedOffByOne()."""

        def __init__(self, baseOffs: Dict[MessageSegment, int], endOffs: Dict[MessageSegment, int],
                     moveFrom: Dict[MessageSegment, List[int]], moveTo: Dict[MessageSegment, List[int]]):
            """

            :param baseOffs: The relative base offset for all segments to analyze.
            :param endOffs: The relative ends positions for all segments to analyze.
            :param moveFrom: Lists of boundaries that should be replaced by another
                for the segments where this is applicable.
            :param moveTo: Lists of boundaries that should replace others for the segments where this is applicable.
            """
            from collections import Counter

            assert baseOffs.keys() == endOffs.keys() == moveFrom.keys() == moveTo.keys(), \
                "All segments need to have a base offset and end defined."

            self._baseOffs = baseOffs
            self._endOffs = endOffs
            self._moveFrom = moveFrom
            self._moveTo = moveTo

            moveAtStart = {seg: baseOffs[seg] in mofro for seg, mofro in moveFrom.items()}  # mofro[0] == baseOffs[seg]
            moveAtEnd = {seg: endOffs[seg] in mofro for seg, mofro in moveFrom.items()}  # mofro[-1] == endOffs[seg]
            commonStarts = Counter(baseOffs.values())
            commonEnds = Counter(endOffs.values())

            self.allAreMoved = \
                [
                    globrel for globrel in commonStarts.keys()
                    if all(moveAtStart[seg] for seg, sstart in baseOffs.items() if globrel == sstart)
                ] + [
                    globrel for globrel in commonEnds.keys()
                    if all(moveAtEnd[seg] for seg, send in endOffs.items() if globrel == send)
                ]

            # if all original bounds that constitute a commonStart/End are moved away in all segments of the
            # type, remove from common bounds.
            self.commonStarts = Counter(base if base not in moveFrom[seg] else moveTo[seg][moveFrom[seg].index(base)]
                                        for seg, base in baseOffs.items())
            self.commonEnds   = Counter(end if end not in moveFrom[seg] else moveTo[seg][moveFrom[seg].index(end)]
                                        for seg, end in endOffs.items())


        def filterOutMoreCommonNeighbors(self, relocWmargin: List[int]) -> Tuple[List[int], List[int]]:
            """
            Resolve adjacent most common starts/ends (use more common bound).

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
            Inverse of moveAt*

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
            :param reloc: here we need the padding-global raw "relocate" without the added 0 and length
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
                -> Dict[MessageSegment, Set[int]]:
            """
            Frequent raw segment bound reframing:
            Refine boundaries within the padded range of all segments in the cluster considering frequent common offsets
            and end positions.

            :param newPaddingRelative: new relative boundaries, including start and end of a unchanged segment, if any.
            :param relocate: here we need the padding-global raw "relocate" without the added 0 and length
            :return: The proposed cut positions per segment based on frequent common raw segment bounds in the cluster.
            """
            startsFreqSum = sum(self.commonStarts.values())
            endsFreqSum = sum(self.commonEnds.values())
            startsMoreFreqThanThresh = [com for com, cnt in self.commonStarts.items()
                                        if cnt / startsFreqSum > type(self).commonFreqThreshold]
            endsMoreFreqThanThresh = [com for com, cnt in self.commonEnds.items()
                                        if cnt / endsFreqSum > type(self).commonFreqThreshold]

            cutsExt = dict()
            # # # # # # # # # # # # # # # # # # # # # # # #
            # padded range refinement
            for seg, reloc in newPaddingRelative.items():
                # resolve adjacent most common starts/ends (use more common bound)
                # commonUnchanged = self.commonUnchanged(seg, relocate)
                commonUnchangedOffbyone = self.commonUnchangedOffByOne(seg, relocate)

                # Used to determine positions that are more than off-by-one from new bound,
                #  naturally includes: is not a move and not a relocation
                relocWmargin = RelocatePCA._offbyone(reloc + commonUnchangedOffbyone)
                moCoReSt, moCoReEn = self.filterOutMoreCommonNeighbors(relocWmargin)

                cutsExtStart = sorted(common for common in startsMoreFreqThanThresh
                                      # conditions for reframing by common segment starts
                                      if common > min(reloc) and (
                                              common not in relocWmargin and common in moCoReSt
                                              or common in commonUnchangedOffbyone
                                      )
                                      )
                cutsExtEnd = sorted(common for common in endsMoreFreqThanThresh
                                    # conditions for reframing by common segment ends
                                    if common < max(reloc) and (
                                            common not in relocWmargin and common in moCoReEn
                                            or common in commonUnchangedOffbyone
                                    )
                                    )
                cutsExt[seg] = set(cutsExtStart + cutsExtEnd)
            return cutsExt


    @staticmethod
    def segs4bound(segs2bounds: Dict[MessageSegment, List[int]], bound: int) -> List[MessageSegment]:
        """
        Helper for iterating bounds in segment : list-of-bounds structure.

        :param segs2bounds:
        :param bound:
        :return: Yields the segment that is a key in segs2bounds, if it has bound in its value list.
            Yields the same segment for each time bound is in its list.
        """
        for seg, bounds in segs2bounds.items():
            for b in bounds.copy():
                if b == bound:
                    yield seg

    @staticmethod
    def removeSuperfluousBounds(newBounds: Dict[AbstractMessage, Dict[MessageSegment, List[int]]]):
        """
        Iterate the given new bounds per message and remove superfluous ones:
            * bound in segment offsets or nextOffsets
            * de-duplicate bounds that are in scope of more than one segment

        **Changes are made at newBounds in place!**

        :param newBounds: Dict of all input messages, with a Dict mapping each segment to a list of its bounds.
        :return: reference to the newBounds Dict
        """
        from nemere.visualization.simplePrint import markSegmentInMessage

        for message, segsbounds in newBounds.items():
            # Below, by list(chain(...)) create a copy to iterate, so we can delete stuff in the original bound lists.

            # if bound in segment offsets or nextOffsets:
            #   remove bound (also resolves some off-by-one conflicts, giving precedence to moves)
            for bound in list(chain(*segsbounds.values())):
                if bound in (segA.offset for segA in segsbounds.keys()) \
                        or bound in (segA.nextOffset for segA in segsbounds.keys()) \
                        or 0 <= bound >= len(message.data):
                    for lookedupSeg in RelocatePCA.segs4bound(segsbounds, bound):
                        segsbounds[lookedupSeg].remove(bound)  # calls remove as often as there is bound in the list
                        # while bound in segsbounds[lookedupSeg]:

            # if bound in scope of more than one segment: resolve
            for bound in list(chain(*segsbounds.values())):
                lookedupSegs = list(RelocatePCA.segs4bound(segsbounds, bound))
                if len(lookedupSegs) > 1:
                    segsNotHavingBoundInScope = [seg for seg in lookedupSegs if
                                                 seg.offset > bound or bound < seg.nextOffset]
                    if len(lookedupSegs) - len(segsNotHavingBoundInScope) == 1:
                        for segOutScope in segsNotHavingBoundInScope:
                            while bound in segsbounds[segOutScope]:
                                segsbounds[segOutScope].remove(bound)
                    elif len(lookedupSegs) - len(segsNotHavingBoundInScope) == 0:
                        # just leave one arbitrary reference to the bound.
                        for segOutScope in segsNotHavingBoundInScope[1:]:
                            while bound in segsbounds[segOutScope]:
                                segsbounds[segOutScope].remove(bound)
                    else:
                        # multiple segments truly have bound in scope
                        # TODO replace by exception - what can we do before failing?
                        print("Bound {} is in scope of multiple segments:".format(bound))
                        for lookedupSeg in RelocatePCA.segs4bound(segsbounds, bound):
                            markSegmentInMessage(lookedupSeg)
                        print("Needs resolving!")
                        print()
                        # IPython.embed()

            # if bound in other segment is as close as one position away: resolve
            flatSegsboundsCopy = [(seg, bound) for seg, bounds in segsbounds.items() for bound in bounds]
            for seg, bound in flatSegsboundsCopy:
                for neighbor in [bound - 1, bound + 1]:
                    if neighbor in (b for segA, bounds in segsbounds.items() if segA != seg for b in bounds):
                        # retain seg/neighborSeg that has bound in scope, delete other(s)
                        inScopeSeg = seg.offset <= bound < seg.nextOffset
                        neiSeg = [segA for segA, bounds in segsbounds.items() if segA != seg and neighbor in bounds]
                        inScopeNei = [segA.offset <= neighbor < segA.nextOffset for segA in neiSeg]

                        if sum([inScopeSeg] + inScopeNei) <= 1:  # just one neighbor remains, good
                            if not inScopeSeg:
                                while bound in segsbounds[seg]:
                                    segsbounds[seg].remove(bound)
                            for segN, inScope in zip(neiSeg, inScopeNei):
                                if not inScope:
                                    while neighbor in segsbounds[segN]:
                                        segsbounds[segN].remove(neighbor)
                            continue

                        # TODO replace by exception
                        print("There are off-by-one neighbors:")
                        for lookedupSeg in RelocatePCA.segs4bound(segsbounds, bound):
                            markSegmentInMessage(lookedupSeg)
                            print("bound: {} - neighbor: {}".format(bound, neighbor))
                        print("Needs resolving!")
                        print()
                        IPython.embed()

        return newBounds

    @staticmethod
    def refineSegmentedMessages(inferredSegmentedMessages: Iterable[Sequence[MessageSegment]],
                                newBounds: Dict[AbstractMessage, Dict[MessageSegment, List[int]]]):
        """

        :param inferredSegmentedMessages: List of messages split into inferred segments.
        :param newBounds: Mapping of messages to the hitherto segments and
            the new bounds in each segment's scope, if any.
        :return: List of messages split into the refined segments, including unchanged segments to always form valid
            segment representations of all messages.
        """
        margin = 1
        refinedSegmentedMessages = list()  # type: List[List[MessageSegment]]
        # iterate sorted message segments
        for msgsegs in inferredSegmentedMessages:

            # TODO happens during RelocatePCA.refineSegments(charPass1, refinementDC) for dns sigma 2.4
            if len(msgsegs) < 1:
                print("Empty message. Investigate!")

                continue

            msg = msgsegs[0].message
            if msg not in newBounds:
                refinedSegmentedMessages.append(msgsegs)
                continue

            # sort new bounds and ensure they are in within the message (offset >= 0 and <= len)
            newMsgBounds = sorted({nb for nb in chain(*newBounds[msg].values()) if 0 <= nb <= len(msg.data)})
            lastBound = 0
            currentBoundID = 0
            refinedSegmentedMessages.append(list())
            for segInf in sorted(msgsegs, key=lambda x: x.offset):
                # for condition tracing
                ifs = list()

                # finish if lastBound reached message end (is hit in case of the msgsegs[-1] is replaced with
                #   a segment starting in scope of msgsegs[-2]
                if len(refinedSegmentedMessages[-1]) > 0 and \
                        lastBound == refinedSegmentedMessages[-1][-1].nextOffset == len(msg.data):
                    break

                # add skipped bytes to next segment if next_segment.nextOffset < bound
                # create segment from last bound to min(next_segment.nextOffset, bound);
                if lastBound < segInf.offset:
                    assert len(newMsgBounds) > 0  # should never happen otherwise

                    if currentBoundID < len(newMsgBounds) and newMsgBounds[
                        currentBoundID] <= segInf.nextOffset + margin:
                        nextOffset = newMsgBounds[currentBoundID]
                        currentBoundID += 1
                    else:
                        nextOffset = segInf.nextOffset

                    assert not len(refinedSegmentedMessages[-1]) > 0 or \
                           refinedSegmentedMessages[-1][-1].nextOffset == lastBound, \
                        "Segment sequence error: add skipped bytes"
                    ifs.append("skipped")

                    refinedSegmentedMessages[-1].append(
                        MessageSegment(segInf.analyzer, lastBound, nextOffset - lastBound)
                    )
                    lastBound = refinedSegmentedMessages[-1][-1].nextOffset
                    if nextOffset >= segInf.nextOffset:
                        continue

                # if no bounds in scope of segment: add old segment and continue
                if lastBound == segInf.offset and (
                        len(newMsgBounds) == 0 or currentBoundID >= len(newMsgBounds)
                        or newMsgBounds[currentBoundID] > segInf.nextOffset + margin):
                    assert not len(refinedSegmentedMessages[-1]) > 0 \
                           or refinedSegmentedMessages[-1][-1].nextOffset == segInf.offset, \
                        "Segment sequence error: no bounds in scope of segment"
                    ifs.append("no bounds")

                    refinedSegmentedMessages[-1].append(segInf)
                    lastBound = segInf.nextOffset
                    continue
                # if bound in scope of segment:
                #   create segment from segment offset or last bound (which is larger) to bound
                for bound in [nmb for nmb in newMsgBounds[currentBoundID:] if nmb < segInf.nextOffset]:
                    newOffset = max(segInf.offset, lastBound)

                    assert not len(refinedSegmentedMessages[-1]) > 0 or \
                           refinedSegmentedMessages[-1][-1].nextOffset == newOffset, \
                        "Segment sequence error: bound in scope of segment"
                    ifs.append("bounds")

                    refinedSegmentedMessages[-1].append(
                        MessageSegment(segInf.analyzer, newOffset, bound - newOffset)
                    )
                    lastBound = newMsgBounds[currentBoundID]
                    currentBoundID += 1

                # no further bounds (at least until segment end)
                if segInf.nextOffset - lastBound <= margin and len(msg.data) - segInf.nextOffset > 0:
                    continue

                # if no further bounds for message or bound > segment next offset+1 and resulting segment longer than 1:
                #   create segment from last bound to inferred segment's next offset;
                if currentBoundID >= len(newMsgBounds) or (
                        newMsgBounds[currentBoundID] > segInf.nextOffset + 1):

                    assert not len(refinedSegmentedMessages[-1]) > 0 or \
                           refinedSegmentedMessages[-1][-1].nextOffset == lastBound, \
                        "Segment sequence error: if no further bounds"
                    # try:
                    #     print(ifs)
                    #     MessageSegment(segInf.analyzer, lastBound, segInf.nextOffset - lastBound)
                    # except:
                    #     IPython.embed()
                    ifs.append("no further bounds")

                    refinedSegmentedMessages[-1].append(
                        MessageSegment(segInf.analyzer, lastBound, segInf.nextOffset - lastBound)
                    )
                    lastBound = refinedSegmentedMessages[-1][-1].nextOffset
                    # do not advance currentBoundID bound (in case there is another bound
                    #   so we need to consider it in scope of a later segment)

                # bound == next offset+1 and resulting segment longer than 1: create segment from last bound to bound
                elif newMsgBounds[currentBoundID] == segInf.nextOffset + 1 and newMsgBounds[
                    currentBoundID] - lastBound > 1:

                    assert not len(refinedSegmentedMessages[-1]) > 0 or \
                           refinedSegmentedMessages[-1][-1].nextOffset == lastBound, \
                        "Segment sequence error: bound == next offset+1"
                    ifs.append("bound == next offset+1")

                    refinedSegmentedMessages[-1].append(
                        MessageSegment(segInf.analyzer, lastBound, newMsgBounds[currentBoundID] - lastBound)
                    )
                    lastBound = refinedSegmentedMessages[-1][-1].nextOffset
                    currentBoundID += 1

            # final assertion of complete representation of message by the new segments
            msgbytes = b"".join([seg.bytes for seg in refinedSegmentedMessages[-1]])
            if not msgbytes == msg.data:
                print(msg.data.hex())
                print(msgbytes.hex())
                print(msgsegs)
                IPython.embed()
            assert msgbytes == msg.data, "segment sequence does not match message bytes"
        return refinedSegmentedMessages

    @staticmethod
    def refineSegments(inferredSegmentedMessages: Iterable[Sequence[MessageSegment]], dc: DistanceCalculator,
                       initialKneedleSensitivity: float=10.0, subclusterKneedleSensitivity: float=5.0,
                       comparator: MessageComparator = None, reportFolder: str = None,
                       collectEvaluationData: Union[List['RelocatePCA'], bool]=False, retClusterer=False,
                       littleEndian=False, trace: str=None) \
            -> Iterable[Sequence[MessageSegment]]:
        """
        Main method to conduct PCA refinement for a set of segments.

        :param inferredSegmentedMessages: List of messages split into inferred segments.
        :param subclusterKneedleSensitivity: sensitivity of the initial clustering autodetection.
        :param initialKneedleSensitivity: use reduced sensitivity (from 10 to 5)
            due to large dissimilarities in clusters (TODO more evaluation!).
        :param dc: Distance calculator representing the segments to be analyzed and refined.
        :param comparator: For evaluation: Encapsulated true field bounds to compare results to.
        :param reportFolder: For evaluation: Destination path to write results and statistics to.
        :param collectEvaluationData: For evaluation: Collect the intermediate (sub-)clusters generated during
            the analysis of the segments.
        :param retClusterer: For evaluation: If a list is provided, any used clusterer instances are appended to it.

        :return: List of segments grouped by the message they are from.
        :raise ClusterAutoconfException: In case no clustering can be performed due to failed parameter autodetection.
        """
        # include only segments that are not just 0 and longer than 1 byte
        relevantSegments = [rs for rs in dc.rawSegments if set(rs.values) != {0} and len(rs.values) > 1]
        try:
            clusterer = DBSCANadjepsClusterer(dc, relevantSegments, S=initialKneedleSensitivity)
        except ClusterAutoconfException as e:
            logger = logging.getLogger(__name__)
            logger.warning(repr(e) + ". Falling back to OPTICS clusterer.")
            clusterer = OPTICSsegmentClusterer(dc, relevantSegments)
        noise, *clusters = clusterer.clusterSimilarSegments(False)
        print("Initial clustering:",
              clusterer, "- cluster sizes:", [len(s) for s in clusters], "- noise:", len(noise))
        if isinstance(retClusterer, List):
            retClusterer.append(clusterer)

        # if there remains only noise, ignore clustering
        if len(clusters) == 0:
            print("No refinement possible: clustering returns only noise.")
            return inferredSegmentedMessages

        newBounds = dict()  # type: Dict[AbstractMessage, Dict[MessageSegment, List[int]]]
        for cLabel, segments in enumerate(clusters):
            # Generate suitable FieldTypeContext objects from the sub-clusters
            ftContext, suitedForAnalysis = RelocatePCA._preFilter(segments, "tf{:02d}".format(cLabel))
            # if basic requirements not met, exclude from PCA analysis
            if not suitedForAnalysis:
                # # # # # # # # # # # # # # # # # # # # # # # #
                # stop further recursion and ignore this cluster
                print("Ignore subcluster {} due to pre-filter.".format(ftContext.fieldtype))
                # performing "common bound" refinement here does not improve the results:
                # * is triggered in only few cases
                # * there are just a handfull of segments that may benefit
                # * there is some probability to produce more FP than TP.
                continue
                # # # # # # # # # # # # # # # # # # # # # # # #

            print("Analyzing cluster", ftContext.fieldtype)
            try:
                cluster = RelocatePCA(ftContext, littleEndian=littleEndian)
                # # # # # # # # # # # # # # # # # # # # # # # #
                # start recursion
                collectedSubclusters = cluster.getSubclusters(dc, subclusterKneedleSensitivity, reportFolder, trace)
                # # # # # # # # # # # # # # # # # # # # # # # #
            except numpy.linalg.LinAlgError:
                print("Ignore cluster due to eigenvalues did not converge")
                print(repr(ftContext.baseSegments))
                continue

            if isinstance(collectEvaluationData, list):
                collectEvaluationData.extend(collectedSubclusters)

            # relocateBoundaries for all collectedSubclusters
            for sc in collectedSubclusters:
                if isinstance(sc, RelocatePCA):
                    clusterBounds = sc.relocateBoundaries(comparator, reportFolder)
                else:
                    # => this is never executed!
                    # performing "common bound" refinement here does not improve the results:
                    # * is triggered in only few cases
                    # * there are just a handfull of segments that may benefit
                    # * there is some probability to produce more FP than TP.
                    # so, no such subclusters are returned by getSubclusters and on "if not suitedForAnalysis"
                    # => this is never executed!
                    baseOffs = {bs: sc.baseOffset(bs) for bs in sc.baseSegments}
                    endOffs = {bs: sc.baseOffset(bs) + bs.length
                               for bs in sc.baseSegments}
                    commonBounds = RelocatePCA.CommonBoundUtil(
                        baseOffs, endOffs, dict.fromkeys(baseOffs.keys(), []), dict.fromkeys(baseOffs.keys(), []))
                    allOffs = {bs: [baseOffs[bs], endOffs[bs]] for bs in sc.baseSegments}
                    relocatedCommons = commonBounds.frequentBoundReframing(allOffs, [])
                    relocatedBoundaries = {seg: list() for seg in sc.baseSegments}
                    for segment, newCommons in relocatedCommons.items():
                        relocatedBoundaries[segment].extend([int(rc + segment.offset) for rc in newCommons
                                                             if int(rc + segment.offset) not in relocatedBoundaries[segment]])
                        if not len(relocatedBoundaries[segment]) == len(set(relocatedBoundaries[segment])):
                            IPython.embed()
                        assert len(relocatedBoundaries[segment]) == len(set(relocatedBoundaries[segment])), \
                            repr(relocatedBoundaries[segment]) + "\n" + repr(set(relocatedBoundaries[segment]))
                    clusterBounds = relocatedBoundaries

                for segment, bounds in clusterBounds.items():
                    if segment.message not in newBounds:
                        newBounds[segment.message] = dict()
                    elif segment in newBounds[segment.message] and newBounds[segment.message][segment] != bounds:
                        # TODO replace by exception or solution
                        print("\nSame segment was PCA-refined multiple times. Needs resolving. Segment is:\n",
                              segment, "Concurrent bounds are:\n",
                              newBounds[segment.message][segment], "and\n",
                              bounds)
                        print()
                        IPython.embed()
                    newBounds[segment.message][segment] = bounds

        # remove from newBounds, in place
        RelocatePCA.removeSuperfluousBounds(newBounds)

        return RelocatePCA.refineSegmentedMessages(inferredSegmentedMessages, newBounds)



class BlendZeroSlices(MessageModifier):
    """
    Generate zero-bounded segments from bytes of given message (i.e. `self.segments[0].message.data`).
    Blend these segments with the given segments (i.e. `self.segments`), with zero bounds have precedence.
    """
    def __init__(self, segments: List[MessageSegment]):
        """
        :param segments: The segments of one message in offset order
        """
        super().__init__(segments)
        if BlendZeroSlices._debug:
            self.zeroSegments = list()

    def blend(self, ignoreSingleZeros=False, littleEndian=False):
        """
        :param ignoreSingleZeros: ignore single zero bytes (except after char sequences),
            or if False, generate segments only from zero sequences of at least two bytes.
        :return: List of segments blended together from the segments of the object as basis and the zero,
            together forming the message.
        """
        zeroBounds = list()
        mdata = self.segments[0].message.data  # type: bytes

        # all transitions from 0 to any and vice versa + message start and end
        for bi, bv in enumerate(mdata[1:], 1):
            if bv == 0 and mdata[bi-1] != 0 \
                    or bv != 0 and mdata[bi-1] == 0:
                zeroBounds.append(bi)
        zeroBounds = [0] + zeroBounds + [len(mdata)]

        # remove boundaries of short zero sequences to merge to previous or next non-zero segment
        minCharLen = 6  # 6
        zeroBounds = sorted(set(zeroBounds))
        zBCopy = zeroBounds.copy()
        for zi, zb in enumerate(zBCopy[:-1]):  # omit message end bound
            if mdata[zb] == 0:
                # next zero bound
                nzb = zBCopy[zi + 1]

                # TODO should that be done after the char `if`?
                # if the next bound (nzb) is only one byte ahead and we should ignore single zeros, remove both bounds.
                if ignoreSingleZeros and zb + 1 == nzb:
                    # if the current bound is not the message start
                    if zb > 0:
                        zeroBounds.remove(zb)
                    # if the next bound is not the message end
                    if nzb < len(mdata):
                        zeroBounds.remove(nzb)
                    continue

                # ... there are only one or two zeros in a row ...
                if zb + 2 >= nzb:
                    # if chars are preceding or its little endian, add zero to previous
                    if isExtendedCharSeq(mdata[max(0,zb-minCharLen):zb], minLen=minCharLen) or littleEndian: # \
                            # or zb > 0 and MessageAnalyzer.nibblesFromBytes(mdata[zb-1:zb])[1] == 0:
                            # # or the least significant nibble of the preceding byte is zero
                        if zb in zeroBounds:
                            zeroBounds.remove(zb)
                    # otherwise to next
                    elif nzb < len(mdata):
                        if nzb in zeroBounds:
                            zeroBounds.remove(nzb)

        if BlendZeroSlices._debug:
            # generate zero-bounded segments from bounds
            ms = list()
            for segStart, segEnd in zip(zeroBounds[:-1], zeroBounds[1:]):
                ms.append(MessageSegment(self.segments[0].analyzer, segStart, segEnd - segStart))
            self.zeroSegments.append(ms)


        # integrate original inferred bounds with zero segments, zero bounds have precedence
        combinedMsg = list()  # type: List[MessageSegment]
        infMarginOffsets = [infs.nextOffset for infs in self.segments
                            if infs.nextOffset - 1 not in zeroBounds and infs.nextOffset + 1 not in zeroBounds]
        remZeroBounds = [zb for zb in zeroBounds if zb not in infMarginOffsets]
        combinedBounds = sorted(infMarginOffsets + remZeroBounds)
        if combinedBounds[0] != 0:
            combinedBounds = [0] + combinedBounds
        if combinedBounds[-1] != len(mdata):
            combinedBounds += [len(mdata)]
        startEndMap = {(seg.offset, seg.nextOffset) : seg for seg in self.segments}
        analyzer = self.segments[0].analyzer
        for bS, bE in zip(combinedBounds[:-1], combinedBounds[1:]):
            # unchanged
            if (bS, bE) in startEndMap:
                combinedMsg.append(startEndMap[(bS, bE)])
            else:
                nseg = MessageSegment(analyzer, bS, bE-bS)
                combinedMsg.append(nseg)
        # final assertion of complete representation of message by the new segments
        msgbytes = b"".join([seg.bytes for seg in combinedMsg])
        assert msgbytes == mdata, "segment sequence does not match message bytes"

        return combinedMsg




class CropChars(MessageModifier):
    """
    Identify probable char sequences using `inference.fieldTypes.FieldTypeRecognizer.charsInMessage`, which in turn uses
    `inference.segmentHandler.isExtendedCharSeq`, to find these.
    Generates segments from those char sequences and the given segments of the message (see constructor).
    """
    def split(self):
        """
        Split a message into char segments and blend them with the segments given in the object instance's segment
        variable.

        :return: List of char and non-char segments for the message
        """
        from nemere.inference.fieldTypeLibrary import BinaryProtocolsFieldTypeRecognizer

        ftrecog = BinaryProtocolsFieldTypeRecognizer(self.segments[0].analyzer)
        # RecognizedFields or type char using isExtendedCharSeq
        charsRecog = [cr.toSegment() for cr in ftrecog.charsInMessage()]

        blendedSegments = self.blend(charsRecog)

        # final assertion of complete representation of message by the new segments
        newbytes = b"".join([seg.bytes for seg in blendedSegments])
        msgbytes = b"".join([seg.bytes for seg in self.segments])
        assert msgbytes == newbytes, "segment sequence does not match message bytes:\nshould: " \
                                     + msgbytes.hex() + "\nis:     " + newbytes.hex()

        return blendedSegments

    def blend(self, mixin: List[MessageSegment]):
        """
        :param mixin: list of segments to blend into the segments of the object
        :return: List of segments blended together from the segments of the object as basis and the mixin,
            together forming the message.
        """
        stack = sorted(mixin, key=lambda x: -x.offset)
        newSegSeq = list()  # type: List[MessageSegment]
        # shifting from stack (from mixin) and inserting from self.segments (msg) to newSegSeq
        #       v---------------,
        # newSegSeq <- msg   stack/mixin
        #  |:::|      |   |   |   |
        #  |---|      |:::|   |   |
        #  |---|      |:::|   |---|
        #  |:::|      |:::|   |---|
        #  +---+      +---+   +---+
        for seg in self.segments:
            while len(stack) > 0 and stack[-1].offset < seg.nextOffset:  # stack peek
                lastOffset = newSegSeq[-1].nextOffset if len(newSegSeq) > 0 else 0
                if stack[-1].offset > lastOffset:  # prepend/fill gap to char
                    newSegSeq.append(MessageSegment(seg.analyzer, lastOffset, stack[-1].offset - lastOffset))
                newSegSeq.append(stack.pop())  # append char
            lastOffset = newSegSeq[-1].nextOffset if len(newSegSeq) > 0 else 0
            if lastOffset == seg.offset:  # append unchanged segment
                newSegSeq.append(seg)
            elif lastOffset < seg.nextOffset:  # append gap to next segment
                newSegSeq.append(MessageSegment(seg.analyzer, lastOffset, seg.nextOffset - lastOffset))
            # else nothing to do since no bytes/segments left
        return newSegSeq

    @staticmethod
    def isOverlapping(segA: MessageSegment, segB: MessageSegment) -> bool:
        """
        Determines whether the given segmentS overlap.

        :param segA: The segment to check against.
        :param segB: The segment to check against.
        :return: Is overlapping or not.
        """
        return isOverlapping(segA, segB)

