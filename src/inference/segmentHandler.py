"""
Batch handling of multiple segments.
"""

import numpy
import copy
from typing import List, Dict, Tuple, Union, Sequence, TypeVar, Iterable

from netzob.Model.Vocabulary.Symbol import Symbol, Field

from inference.segments import MessageSegment, HelperSegment, TypedSegment
from inference.analyzers import MessageAnalyzer
from inference.templates import TypedTemplate, DistanceCalculator


def segmentMeans(segmentsPerMsg: List[List[MessageSegment]]):
    """
    :param segmentsPerMsg: List of Lists of Segments per message
    :return: List of means of the values of each message
    """
    meanSegments = list()
    for perMessage in segmentsPerMsg:
        meanMessage = list()
        for a in perMessage:
            meanS = HelperSegment(a.analyzer, a.offset, a.length)
            meanS.values = a.mean()
            meanMessage.append(meanS)
        meanSegments.append(meanMessage)
    return meanSegments


def segmentStdevs(segmentsPerMsg: List[List[MessageSegment]]):
    """
    :param segmentsPerMsg: List of Lists of Segments per message
    :return: List of deviations of the values of each message
    """
    meanSegments = list()
    for perMessage in segmentsPerMsg:
        varMessage = list()
        for a in perMessage:
            varS = HelperSegment(a.analyzer, a.offset, a.length)
            varS.values = a.stdev()
            varMessage.append(varS)
        meanSegments.append(varMessage)
    return meanSegments


def symbolsFromSegments(segmentsPerMsg: Iterable[Sequence[MessageSegment]]) -> List[Symbol]:
    """
    Generate a list of Netzob Symbols from the given lists of segments for each message.

    :param segmentsPerMsg: List of messages, represented by lists of segments.
    :return: list of Symbols, one for each entry in the given iterable of lists.
    """
    return [Symbol( [Field(segment.bytes) for segment in sorted(segSeq, key=lambda f: f.offset)],
                    messages=[segSeq[0].message])
        for segSeq in segmentsPerMsg ]


def segmentsFromLabels(analyzer, labels) -> Tuple[TypedSegment]:
    """
    Segment messages according to true fields from the labels
    and mark each segment with its true type.

    :param analyzer: An Analyzer for/with a message
    :param labels: The labels of the true format
    :return: Segments of the analyzer's message according to the true format
    """
    segments = list()
    offset = 0
    for ftype, flen in labels:
        segments.append(TypedSegment(analyzer, offset, flen, ftype))
        offset += flen
    return tuple(segments)


# TODO replace parameter comparator by specimens
def segmentsFixed(length: int, comparator,
                  analyzerType: type, analysisArgs: Union[Tuple, None], unit=MessageAnalyzer.U_BYTE, padded=False) \
        -> List[Tuple[MessageSegment]]:
    """
    Segment messages into fixed size chunks.

    >>> from utils.loader import SpecimenLoader
    >>> from validation.dissectorMatcher import MessageComparator
    >>> from inference.analyzers import Value
    >>> from inference.segmentHandler import segmentsFixed
    >>> specimens = SpecimenLoader("../input/ntp_SMIA-20111010_deduped-100.pcap", 2, True)
    >>> comparator = MessageComparator(specimens, 2, True, debug=False)
    >>> segmentedMessages = segmentsFixed(4, comparator, Value, None)
    >>> areIdentical = True
    >>> for msgsegs in segmentedMessages:
    ...     msg = msgsegs[0].message
    ...     msgbytes = b"".join([seg.bytes for seg in msgsegs])
    ...     areIdentical = areIdentical and msgbytes == msg.data
    >>> print(areIdentical)
    True

    :param length: Fixed length for all segments. Overhanging segments at the end that are shorter than length
        will be padded with NANs.
    :param comparator: Comparator that contains the payload messages.
    :param analyzerType: Type of the analysis. Subclass of inference.analyzers.MessageAnalyzer.
    :param analysisArgs: Arguments for the analysis method.
    :param unit: Base unit for the analysis. Either MessageAnalyzer.U_BYTE or MessageAnalyzer.U_NIBBLE.
    :param padded: Toggle to pad the last segment to the requested fixed length or leave the last segment to be
        shorter than length if the message length is not an exact multiple of the segment length.
    :return: Segments of the analyzer's message according to the true format.
    """
    segments = list()
    for l4msg, rmsg in comparator.messages.items():
        if len(l4msg.data) % length == 0:
            lastOffset = len(l4msg.data)
        else:
            lastOffset = (len(l4msg.data) // length) * length

        originalAnalyzer = MessageAnalyzer.findExistingAnalysis(analyzerType, unit, l4msg, analysisArgs)
        sequence = [
            MessageSegment(originalAnalyzer, offset, length) for offset in range(0, lastOffset, length)
        ]

        if len(l4msg.data) > lastOffset:  # append the overlap
            if padded:
                # here are nasty hacks!
                # TODO Better define a new subclass of MessageSegment that internally padds values
                #  (and bytes? what are the guarantees?) to a given length that exceeds the message length
                residuepadd = lastOffset + length - len(l4msg.data)
                newMessage = copy.copy(originalAnalyzer.message)
                newMessage.data = newMessage.data + b'\x00' * residuepadd
                newAnalyzer = type(originalAnalyzer)(newMessage, originalAnalyzer.unit)  # type: MessageAnalyzer
                newAnalyzer.setAnalysisParams(*originalAnalyzer.analysisParams)
                padd = [numpy.nan] * residuepadd
                newAnalyzer._values = originalAnalyzer.values + padd
                newSegment = MessageSegment(newAnalyzer, lastOffset, length)
                for seg in sequence:  # replace all previous analyzers to make the sequence homogeneous for this message
                    seg.analyzer = newAnalyzer
                sequence.append(newSegment)
            else:
                newSegment = MessageSegment(originalAnalyzer, lastOffset, len(l4msg.data) - lastOffset)
                sequence.append(newSegment)

        segments.append(tuple(sequence))

    return segments


def groupByLength(segmentedMessages: Iterable) -> Dict[int, List[MessageSegment]]:
    """
    Regroup a list of lists of segments into groups of segments that have equal length

    :param segmentedMessages:
    :return: dict with length: List[segments] pairs
    """
    from itertools import chain
    segsByLen = dict()
    for seg in chain.from_iterable(segmentedMessages):  # type: MessageSegment
        seglen = len(seg.bytes)
        if seglen not in segsByLen:
            segsByLen[seglen] = list()
        segsByLen[seglen].append(seg)
    return segsByLen


def segments2types(segments: Iterable[TypedSegment]) -> Dict[str, List[TypedSegment]]:
    """
    Rearrange a list of typed segments into a dict of type: list(segments of that type)

    :param segments: Supports also non TypedSegment input by placing such segments into a [mixed] group.
    :return: A dict of
        fieldtype (str) : segments of this type (list)
    """
    typegroups = dict()
    for seg in segments:
        fieldtype = seg.fieldtype if isinstance(seg, (TypedSegment, TypedTemplate)) else '[unknown]'
        if fieldtype in typegroups:
            typegroups[fieldtype].append(seg)
        else:
            typegroups[fieldtype] = [seg]
    return typegroups


def bcDeltaGaussMessageSegmentation(specimens, sigma=0.6) -> List[List[MessageSegment]]:
    """
    Segment message by determining inflection points of gauss-filtered bit congruence deltas.

    >>> from utils.loader import SpecimenLoader
    >>> sl = SpecimenLoader('../input/random-100-continuous.pcap', layer=0, relativeToIP=True)
    >>> segmentsPerMsg = bcDeltaGaussMessageSegmentation(sl)
    Segmentation by inflections of sigma-0.6-gauss-filtered bit-variance.
    >>> for spm in segmentsPerMsg:
    ...     if b''.join([seg.bytes for seg in spm]).hex() != spm[0].message.data.hex():
    ...         print("Mismatch!")

    :return: Segmentation of the specimens in the pool.
    """
    from inference.analyzers import BitCongruenceDeltaGauss

    print('Segmentation by inflections of sigma-{:.1f}-gauss-filtered bit-variance.'.format(
        sigma
    ))
    msgSeg = list()
    for l4msg, rmsg in specimens.messagePool.items():
        analyzer = BitCongruenceDeltaGauss(l4msg)
        analyzer.setAnalysisParams(sigma)
        analyzer.analyze()
        msgSeg.append(analyzer.messageSegmentation())
    return msgSeg



def refinements(segmentsPerMsg: List[List[MessageSegment]], dc: DistanceCalculator) -> List[List[MessageSegment]]:
    """
    Refine the segmentation using specific improvements for the feature:
    Inflections of gauss-filtered bit-congruence deltas.

    TODO reevaluate all usages!

    :param segmentsPerMsg: a list of one list of segments per message.
    :return: refined segments in on list per message
    """
    import inference.formatRefinement as refine

    print("Refine segmentation...")

    refinedPerMsg = list()
    for msg in segmentsPerMsg:
        # merge consecutive segments of printable-char values (\t, \n, \r, >= 0x20 and <= 0x7e) into one text field.
        charsMerged = refine.MergeConsecutiveChars(msg).merge()
        charSplited = refine.ResplitConsecutiveChars(charsMerged).split()
        refinedPerMsg.append(charSplited)

    # for tests use test_segment-refinements.py
    moco = refine.CropDistinct.countCommonValues(refinedPerMsg)
    newstuff = list()
    for msg in refinedPerMsg:
        croppedMsg = refine.CropDistinct(msg, moco).split()
        charmerged = refine.CumulativeCharMerger(croppedMsg).merge()
        splitfixed = refine.SplitFixed(charmerged).split(0, 1)
        newstuff.append(splitfixed)

    return newstuff


def baseRefinements(segmentsPerMsg: Sequence[Sequence[MessageSegment]]) -> List[List[MessageSegment]]:
    """
    Refine the segmentation using specific improvements for the feature:
    Inflections of gauss-filtered bit-congruence deltas.

    :param segmentsPerMsg: a list of one list of segments per message.
    :return: refined segments in list per message
    """
    import inference.formatRefinement as refine

    print("Refine segmentation (base refinements)...")

    refinedPerMsg = list()
    for msg in segmentsPerMsg:
        # merge consecutive segments of printable-char values (\t, \n, \r, >= 0x20 and <= 0x7e) into one text field.
        charsMerged = refine.MergeConsecutiveChars(msg).merge()
        charSplited = refine.ResplitConsecutiveChars(charsMerged).split()
        refinedPerMsg.append(charSplited)

    # for tests use test_segment-refinements.py
    moco = refine.CropDistinct.countCommonValues(refinedPerMsg)
    newstuff = list()
    for msg in refinedPerMsg:
        croppedMsg = refine.CropDistinct(msg, moco).split()
        charmerged = refine.CumulativeCharMerger(croppedMsg).merge()
        newstuff.append(charmerged)

    return newstuff


def nemetylRefinements(segmentsPerMsg: Sequence[Sequence[MessageSegment]]) -> List[List[MessageSegment]]:
    """
    Refine the segmentation using specific improvements for the feature:
    Inflections of gauss-filtered bit-congruence deltas.

    :param segmentsPerMsg: a list of one list of segments per message.
    :return: refined segments in list per message
    """
    import inference.formatRefinement as refine

    print("Refine segmentation (nemetyl refinements)...")

    refinedPerMsg = list()
    for msg in segmentsPerMsg:
        # merge consecutive segments of printable-char values (\t, \n, \r, >= 0x20 and <= 0x7e) into one text field.
        charsMerged = refine.MergeConsecutiveChars(msg).merge()
        charSplited = refine.ResplitConsecutiveChars(charsMerged).split()
        refinedPerMsg.append(charSplited)

    # for tests use test_segment-refinements.py
    moco = refine.CropDistinct.countCommonValues(refinedPerMsg)
    newstuff = list()
    for msg in refinedPerMsg:
        croppedMsg = refine.CropDistinct(msg, moco).split()
        charmerged = refine.CumulativeCharMerger(croppedMsg).merge()
        splitfixed = refine.SplitFixed(charmerged).split(0, 1)
        newstuff.append(splitfixed)

    return newstuff


def charRefinements(segmentsPerMsg: Sequence[Sequence[MessageSegment]]) -> List[Sequence[MessageSegment]]:
    """
    Refine the segmentation using specific improvements for the feature:
    Inflections of gauss-filtered bit-congruence deltas.

    Copy of inference.segmentHandler.refinements without
        * frequency reinforced segments (CropDistinct) and
        * splitting of first segment (SplitFixed)

    :param segmentsPerMsg: a list of one list of segments per message.
    :return: refined segments in list per message
    """
    import inference.formatRefinement as refine

    print("Refine segmentation (char refinements)...")

    refinedPerMsg = list()
    for msg in segmentsPerMsg:
        # merge consecutive segments of printable-char values (\t, \n, \r, >= 0x20 and <= 0x7e) into one text field.
        charsMerged = refine.MergeConsecutiveChars(msg).merge()
        charSplited = refine.ResplitConsecutiveChars(charsMerged).split()
        refinedPerMsg.append(charSplited)

    # for tests use test_segment-refinements.py
    newstuff = list()
    for msg in refinedPerMsg:
        charmerged = refine.CumulativeCharMerger(msg).merge()
        newstuff.append(charmerged)

    return newstuff


def originalRefinements(segmentsPerMsg: Sequence[Sequence[MessageSegment]]) -> List[List[MessageSegment]]:
    """
    Refine the segmentation according to the WOOT2018 paper method using specific improvements for the feature:
    Inflections of gauss-filtered bit-congruence deltas.

    :param segmentsPerMsg: a list of one list of segments per message.
    :return: refined segments in list per message
    """
    import inference.formatRefinement as refine

    print("Refine segmentation (WOOT18 refinements)...")

    refinedPerMsg = list()
    for msg in segmentsPerMsg:
        # merge consecutive segments of printable-char values (\t, \n, \r, >= 0x20 and <= 0x7e) into one text field.
        charsMerged = refine.MergeConsecutiveChars(msg).merge()
        charSplited = refine.ResplitConsecutiveChars(charsMerged).split()
        refinedPerMsg.append(charSplited)

    return refinedPerMsg


T = TypeVar('T')
def matrixFromTpairs(distances: Iterable[Tuple[T,T,float]], segmentOrder: Sequence[T], identity=0, incomparable=1,
                     simtrx: numpy.ndarray=None) -> numpy.ndarray:
    """
    Arrange the representation of the pairwise similarities of the input parameter in an symmetric array.
    The order of the matrix elements in each row and column is the same as in self._segments.

    Used in constructor.

    :param simtrx: provide a ndarray object to use as matrix to fill instead of creating a new one.
    :param distances: The pairwise similarities to arrange.
        0. T: segA
        1. T: segB
        2. float: distance
    :param segmentOrder: The segments in ordering they should be represented in the matrix
    :param identity: The value pairs of identical segments should receive in the matrix
    :param incomparable: The value incomparable segment pairs should get in the matrix
    :return: The distance matrix for the given similarities.
        1 for each undefined element, 0 in the diagonal, even if not given in the input.
    """
    numsegs = len(segmentOrder)
    if simtrx is not None:
        assert simtrx.shape == (numsegs, numsegs)
        print("Use provided matrix:", type(simtrx).__name__, simtrx.shape, simtrx.dtype)
        simtrx.fill(incomparable)
    else:
        # reduce memory footprint by limiting precision to 16 bit float
        # https://docs.scipy.org/doc/numpy-1.13.0/user/basics.types.html?highlight=float16
        simtrx = numpy.ones((numsegs, numsegs), dtype=numpy.float16)
        if incomparable != 1:
            simtrx.fill(incomparable)
    numpy.fill_diagonal(simtrx, identity)
    # fill matrix with pairwise distances
    for intseg in distances:
        row = segmentOrder.index(intseg[0])
        col = segmentOrder.index(intseg[1])
        simtrx[row, col] = intseg[2]
        simtrx[col, row] = intseg[2]
        # check that the diagonal is only populated by the identity value
        if row == col and intseg[2] != identity:
            print("Warning: Identity value at {},{} was overwritten by {}".format(row, col, intseg[2]))
    return simtrx



def filterSegments(segments: Iterable[MessageSegment]) -> List[MessageSegment]:
    """
    Filter input segment for only those segments that are adding relevant information for further analysis:
    * filter out segments shorter than 3 bytes
    * filter out all-zero byte sequences
    * filter out segments that resulted in no relevant feature data, i. e.,
      (0, .., 0) | (nan, .., nan) | or a mixture of both
    * filter out identical segments to leave only one representative

    (as an more advanced alternative see inference.templates.DelegatingDC)

    :param segments: list of segments to filter.
    :return: Sorted list of Segments that remained after applying all filter conditions.
    """
    # filter out segments shorter than 3 bytes
    filteredSegments = [t for t in segments if t.length > 2]
    # filteredSegments = segments

    # filter out segments that contain no relevant byte data, i. e., all-zero byte sequences
    filteredSegments = [t for t in filteredSegments if t.bytes.count(b'\x00') != len(t.bytes)]

    # filter out segments that resulted in no relevant feature data, i. e.,
    # (0, .., 0) | (nan, .., nan) | or a mixture of both
    filteredSegments = [s for s in filteredSegments if
                        numpy.count_nonzero(s.values) - numpy.count_nonzero(numpy.isnan(s.values)) > 0]

    # filter out identical segments
    uniqueFeatures = set()
    fS = filteredSegments
    filteredSegments = list()
    for s in fS:
        svt = tuple(s.values)
        if svt not in uniqueFeatures:
            uniqueFeatures.add(svt)
            filteredSegments.append(s)

    # sorted only for visual representation in heatmap or similar
    filteredSegments = sorted(filteredSegments, key=lambda x: x.length)

    return filteredSegments

def isExtendedCharSeq(values: bytes, meanCorridor=(50, 115), minLen=6):
    from inference.formatRefinement import locateNonPrintable

    vallen = len(values)
    nonzeros = [v for v in values if v > 0x00]
    return (vallen >= minLen
                and any(values)
                and numpy.max(tuple(values)) < 0x7f
                and meanCorridor[0] <= numpy.mean(nonzeros) <= meanCorridor[1]
                and 0.33 > len(locateNonPrintable(bytes(nonzeros))) / vallen
                # # above solution evaluated with only smb having a minor increase in false positives compared to below
                # and 0.66 > len(locateNonPrintable(values)) / vallen  # from smb one-char-many-zeros segments
            )

def filterChars(segments: Iterable[MessageSegment], meanCorridor=(50, 115), minLen=6):
    """
    Filter segments by some hypotheses about what might be a char sequence:
        1. Segment is larger than minLen
        2. Segment has not only 0x00 values
        3. All values are < 127 (0x7f)
        4. The sequence's values have a mean of between n and m, e. g. if 0x20 <= char <= 0x7e (printable chars)
        5. The ratio of nonprintables is less than 2/3 of all values

    :param segments: List of segments to be filtered
    :param meanCorridor: Corridor of mean value that denotes a probable char sequence.
        A meanCorridor=(0x20, 0x7e) would ensure to include even segments of repetitions
        of " " (bottommost printable) or "~" (topmost printable)
    :param minLen: Minimum length of a segment to be condidered for hypothesis testing
    :return: Filtered segments: segments that hypothetically are chars
    """

    filtered = [seg for seg in segments
                if isExtendedCharSeq(seg.bytes, meanCorridor, minLen)
                ]
    return filtered
