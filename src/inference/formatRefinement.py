from typing import List
from abc import ABC, abstractmethod, ABCMeta

from inference.segments import MessageSegment

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


class Merger(ABC):
    """
    Base class to merge segments based on a variable condition.
    """
    __debug = False

    def __init__(self, segments: List[MessageSegment]):
        """
        :param segments: in offset order
        """
        self.segments = segments


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
                    if self.__debug:
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


class RelocateSplits(ABC, metaclass=ABCMeta):
    """
    Relocate split locations based on properties of adjacent segments.
    """
    __debug = False

    def __init__(self, segments: List[MessageSegment]):
        """
        :param segments: The segments of one message in offset order.
        """
        self.segments = segments

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
                    mangledSegments.append(segc)
                    continue

                if mangledSegments:
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
                            if self.__debug:
                                print("Recombined segments: \n{} and {} into ".format(segl, segc))
                            segc = MessageSegment(segc.analyzer, segc.offset - restlen,
                                                             segc.length + restlen)
                            if self.__debug:
                                print("{} and {}".format(mangledSegments[-1] if mangledSegments else 'Empty', segc))

                if segmentStack:
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
                            if self.__debug:
                                print("Recombined segments: \n{} and {} into ".format(segc, segr))
                            segc = MessageSegment(segc.analyzer, segc.offset,
                                                              segc.length + splitpos)
                            if self.__debug:
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


class Resplit2LeastFrequentPair(object):
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
    __debug = False
    __pairFrequencies = None
    __CHUNKLEN = 2

    def __init__(self, segments: List[MessageSegment]):
        """
        :param segments: in offset order
        """
        self.segments = segments

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
        if Resplit2LeastFrequentPair.__debug:
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
                        if self.__debug:
                            print("Recombined segments: \n{} and {} into ".format(segl, segc))
                        segc = MessageSegment(segc.analyzer, segc.offset + splitshift,
                                                         segc.length - splitshift)
                        if self.__debug:
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




class CropDistinct(object):
    """
    Split segments into smaller chunks if a given value is contained in the segment.
    The given value is cropped to a segment on its own.
    """
    __debug = False

    def __init__(self, segments: List[MessageSegment], mostcommon: List[bytes]):
        """
        :param segments: The segments of one message in offset order.
        :param mostcommon: most common bytes sequences to be searched for and cropped
            (sorted descending from most frequent)
        """
        self.segments = segments
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
                    if CropDistinct.__debug:
                        print("\nReplaced {} by:".format(seg.bytes.hex()), end=" ")

                    absco = seg.offset + comoff
                    if comoff > 0:
                        segl = MessageSegment(seg.analyzer, seg.offset, comoff)
                        newmsg.append(segl)
                        if CropDistinct.__debug:
                            print(segl.bytes.hex(), end=" ")

                    segc = MessageSegment(seg.analyzer, absco, featlen)
                    newmsg.append(segc)
                    if CropDistinct.__debug:
                        print(segc.bytes.hex(), end=" ")

                    rlen = seg.length - comoff - featlen
                    if rlen > 0:
                        segr = MessageSegment(seg.analyzer, absco + featlen, rlen)
                        newmsg.append(segr)
                        if CropDistinct.__debug:
                            print(segr.bytes.hex(), end=" ")

                didReplace = True
                break  # only most common match!? otherwise how to handle subsequent matches after split(s)?
            if not didReplace:
                newmsg.append(seg)
            elif CropDistinct.__debug:
                print()

        return newmsg
