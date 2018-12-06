"""
Batch handling of multiple segments.
"""

import numpy
from typing import List, Dict, Tuple, Union, Sequence, TypeVar

from inference.segments import MessageSegment, HelperSegment, TypedSegment
from inference.analyzers import MessageAnalyzer




def segmentMeans(segmentsPerMsg: List[List]):
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


def segmentStdevs(segmentsPerMsg: List[List]):
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


def symbolsFromSegments(segmentsPerMsg):
    from netzob.Model.Vocabulary.Symbol import Symbol, Field
    return [Symbol([Field(segment.bytes) for segment in sorted(segSeq, key=lambda f: f.offset)], messages=[segSeq[0].message]) for segSeq in segmentsPerMsg ]


def segmentsFromLabels(analyzer, labels) -> List[TypedSegment]:
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
    return segments


def annotateFieldTypes(analyzerType: type, analysisArgs: Union[Tuple, None], comparator,
                       unit=MessageAnalyzer.U_BYTE) -> List[List[TypedSegment]]:
    """
    :return: list of lists of segments that are annotated with their field type.
    """
    segmentedMessages = [segmentsFromLabels(
        MessageAnalyzer.findExistingAnalysis(analyzerType, unit,
                                             l4msg, analysisArgs), comparator.dissections[rmsg])
        for l4msg, rmsg in comparator.messages.items()]
    return segmentedMessages


def segmentsFixed(analyzerType: type, analysisArgs: Union[Tuple, None], comparator, length: int,
                       unit=MessageAnalyzer.U_BYTE) -> List[Tuple[MessageSegment]]:
    """
    Segment messages into fixed size chunks.

    :param length: The length for all the segments. Overhanging final segments shorter than length will be padded with
        nans.
    :return: Segments of the analyzer's message according to the true format
    """
    segments = list()
    for l4msg, rmsg in comparator.messages.items():
        if len(l4msg.data) % length == 0:  # exclude the overlap
            lastOffset = len(l4msg.data)
        else:
            lastOffset = (len(l4msg.data) // length) * length
        sequence = [
            MessageSegment(
            MessageAnalyzer.findExistingAnalysis(analyzerType, unit,
                                                 l4msg, analysisArgs),
            offset, length)
            for offset in range(0, lastOffset, length)
        ]
        if len(l4msg.data) > lastOffset:  # append the overlap
            # TODO here are nasty hacks!
            # Better define a new subclass of MessageSegment that internally padds values
            # (and bytes? what are the guarantees?) to a given length that exceeds the message length
            residuepadd = lastOffset + length - len(l4msg.data)
            originalAnalyzer = MessageAnalyzer.findExistingAnalysis(analyzerType, unit,
                                                 l4msg, analysisArgs)
            import copy
            newMessage = copy.copy(originalAnalyzer.message)
            newMessage.data = newMessage.data + b'\x00' * residuepadd
            newAnalyzer = type(originalAnalyzer)(newMessage, originalAnalyzer.unit)  # type: MessageAnalyzer
            newAnalyzer.setAnalysisParams(*originalAnalyzer.analysisParams)
            padd = [numpy.nan] * residuepadd
            newAnalyzer._values = originalAnalyzer.values + padd
            newSegment = MessageSegment(newAnalyzer, lastOffset+1, length)
            for seg in sequence:  # replace all previous analyzers to make the sequence homogeneous for this message
                seg.analyzer = newAnalyzer
            sequence.append(newSegment)
        segments.append(tuple(sequence))
    return segments


def groupByLength(segmentedMessages) -> Dict[int, List[MessageSegment]]:
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


def segments2types(segments: List[TypedSegment]) -> Dict[str, List[TypedSegment]]:
    """
    Rearrange a list of typed segments into a dict of type: list(segments of that type)

    :param segments:
    :return:
    """
    typegroups = dict()
    for seg in segments:
        if seg.fieldtype in typegroups:
            typegroups[seg.fieldtype].append(seg)
        else:
            typegroups[seg.fieldtype] = [seg]
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


def refinements(segmentsPerMsg: List[List[MessageSegment]]):
    """
    Refine the segmentation using specific improvements for the feature:
    Inflections of gauss-filtered bit-congruence deltas.

    :param segmentsPerMsg: a list of one list of segments per message.
    :return: refined segments in on list per message
    """
    import inference.formatRefinement as refine

    print("Refine segmentation...")

    refinedPerMsg = [
            # merge consecutive segments of printable-char values (\t, \n, \r, >= 0x20 and <= 0x7e) into one text field.
            refine.ResplitConsecutiveChars(
                refine.MergeConsecutiveChars(m).merge()
            ).split()
        for m in segmentsPerMsg]
    return refinedPerMsg


T = TypeVar('T')
def matrixFromTpairs(distances: List[Tuple[T,T,float]], segmentOrder: Sequence[T], identity=0, incomparable=1) -> numpy.ndarray:
    """
    Arrange the representation of the pairwise similarities of the input parameter in an symmetric array.
    The order of the matrix elements in each row and column is the same as in self._segments.

    Used in constructor.

    :param distances: The pairwise similarities to arrange.
        0. T: segA
        1. T: segB
        2. float: distance
    :return: The distance matrix for the given similarities.
        1 for each undefined element, 0 in the diagonal, even if not given in the input.
    """
    numsegs = len(segmentOrder)
    simtrx = numpy.ones((numsegs, numsegs))
    if incomparable != 1:
        simtrx.fill(incomparable)
    numpy.fill_diagonal(simtrx, identity)
    # fill matrix with pairwise distances
    for intseg in distances:
        row = segmentOrder.index(intseg[0])
        col = segmentOrder.index(intseg[1])
        simtrx[row, col] = intseg[2]
        simtrx[col, row] = intseg[2]
    return simtrx


