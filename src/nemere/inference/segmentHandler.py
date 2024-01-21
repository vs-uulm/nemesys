"""
Batch handling of multiple segments.
"""
from itertools import chain

import numpy
import copy
from typing import List, Dict, Tuple, Union, Sequence, TypeVar, Iterable

from netzob.Model.Vocabulary.Symbol import Symbol, Field

from nemere.utils.loader import BaseLoader
from nemere.inference.segments import MessageSegment, HelperSegment, TypedSegment, AbstractSegment
from nemere.inference.analyzers import MessageAnalyzer, Value
from nemere.inference.templates import AbstractClusterer, TypedTemplate, DelegatingDC, MemmapDC


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

    >>> from nemere.inference.segmentHandler import symbolsFromSegments
    >>> from nemere.inference.segments import MessageSegment
    >>> from nemere.inference.analyzers import Value
    >>> from netzob.Model.Vocabulary.Symbol import Symbol
    >>> from netzob.Model.Vocabulary.Messages.RawMessage import RawMessage
    >>> # prevent Netzob from producing debug output.
    >>> import logging
    >>> logging.getLogger().setLevel(30)
    >>>
    >>> dummymsg = RawMessage(bytes(list(range(50, 70))))
    >>> dummyana = Value(dummymsg)
    >>> testgapped = [[ MessageSegment(dummyana, 0, 2), MessageSegment(dummyana, 5, 2), MessageSegment(dummyana, 7, 6),
    ...                MessageSegment(dummyana, 17, 2) ]]
    >>> symbol = symbolsFromSegments(testgapped)[0]
    >>> print(symbol.str_data())
    Field | Field | Field | Field    | Field  | Field | Field
    ----- | ----- | ----- | -------- | ------ | ----- | -----
    '23'  | '456' | '78'  | '9:;<=>' | '?@AB' | 'CD'  | 'E'...
    ----- | ----- | ----- | -------- | ------ | ----- | -----


    Intermediately produces:
    ```
    from pprint import pprint
    pprint(filledSegments)
    [[MessageSegment 2 bytes at (0, 2): 1415 | values: (20, 21),
      MessageSegment 3 bytes at (2, 5): 161718 | values: (22, 23, 24),
      MessageSegment 2 bytes at (5, 7): 191a | values: (25, 26),
      MessageSegment 6 bytes at (7, 13): 1b1c1d1e1f20 | values: (27, 28, 29...,
      MessageSegment 4 bytes at (13, 17): 21222324 | values: (33, 34, 35...,
      MessageSegment 2 bytes at (17, 19): 2526 | values: (37, 38)]]
    ````

    :param segmentsPerMsg: List of messages, represented by lists of segments.
    :return: list of Symbols, one for each entry in the given iterable of lists.
    """
    sortedSegments = (sorted(segSeq, key=lambda f: f.offset) for segSeq in segmentsPerMsg)
    filledSegments = list()
    for segSeq in sortedSegments:
        assert len(segSeq) > 0
        filledGaps = list()
        for segment in segSeq:
            lastoffset = filledGaps[-1].nextOffset if len(filledGaps) > 0 else 0
            if segment.offset > lastoffset:
                gaplength = segment.offset - lastoffset
                filledGaps.append(MessageSegment(segment.analyzer, lastoffset, gaplength))
            filledGaps.append(segment)
        # check for required trailing segment
        lastoffset = filledGaps[-1].nextOffset
        msglen = len(filledGaps[-1].message.data)
        if lastoffset < msglen:
            gaplength = msglen - lastoffset
            filledGaps.append(MessageSegment(filledGaps[-1].analyzer, lastoffset, gaplength))
        filledSegments.append(filledGaps)

    return [ Symbol( [Field(segment.bytes) for segment in segSeq],
                     messages=[segSeq[0].message], name=f"nemesys Symbol {i}" )
             for i,segSeq in enumerate(filledSegments) ]


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


def segmentsFromSymbols(symbols: List[Symbol]):
    """
    :param symbols: List of Netzob symbols.
    :return: List of messages represented by a list of segments each.
    """
    msgflds = [(msg,flds) for s in symbols for msg,flds in s.getMessageCells().items()]
    segmentedMessages = []
    for msg,flds in msgflds:
        analyzer = Value(msg)
        msgSegs = []
        pointer = 0
        for bv in flds:
            length = len(bv)
            if length == 0:
                continue
            msgSegs.append(MessageSegment(analyzer, pointer, length))
            pointer += length
        assert pointer == len(msg.data)
        segmentedMessages.append(msgSegs)
    return segmentedMessages


def fixedlengthSegmenter(length: int, specimens: BaseLoader,
                         analyzerType: type, analysisArgs: Union[Tuple, None], unit=MessageAnalyzer.U_BYTE, padded=False) \
        -> List[Tuple[MessageSegment]]:
    """
    Segment messages into fixed size chunks.

    >>> from nemere.utils.loader import SpecimenLoader
    >>> from nemere.validation.dissectorMatcher import MessageComparator
    >>> from nemere.inference.analyzers import Value
    >>> from nemere.inference.segmentHandler import fixedlengthSegmenter
    >>> specimens = SpecimenLoader("../input/deduped-orig/ntp_SMIA-20111010_deduped-100.pcap", 2, True)
    >>> comparator = MessageComparator(specimens, 2, True, debug=False)
    >>> segmentedMessages = fixedlengthSegmenter(4, specimens, Value, None)
    >>> areIdentical = True
    >>> for msgsegs in segmentedMessages:
    ...     msg = msgsegs[0].message
    ...     msgbytes = b"".join([seg.bytes for seg in msgsegs])
    ...     areIdentical = areIdentical and msgbytes == msg.data
    >>> print(areIdentical)
    True

    :param length: Fixed length for all segments. Overhanging segments at the end that are shorter than length
        will be padded with NANs.
    :param specimens: Loader utility class that contains the payload messages.
    :param analyzerType: Type of the analysis. Subclass of inference.analyzers.MessageAnalyzer.
    :param analysisArgs: Arguments for the analysis method.
    :param unit: Base unit for the analysis. Either MessageAnalyzer.U_BYTE or MessageAnalyzer.U_NIBBLE.
    :param padded: Toggle to pad the last segment to the requested fixed length or leave the last segment to be
        shorter than length if the message length is not an exact multiple of the segment length.
    :return: Segments of the analyzer's message according to the true format.
    """
    segments = list()
    for l4msg, rmsg in specimens.messagePool.items():
        if len(l4msg.data) % length == 0:  # exclude the overlap
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
                # TODO Better define a new subclass of MessageSegment that internally pads values
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
    from nemere.utils.evaluationHelpers import unknown
    typegroups = dict()
    for seg in segments:
        fieldtype = seg.fieldtype if isinstance(seg, (TypedSegment, TypedTemplate)) else unknown
        if fieldtype in typegroups:
            typegroups[fieldtype].append(seg)
        else:
            typegroups[fieldtype] = [seg]
    return typegroups


def bcDeltaGaussMessageSegmentation(specimens, sigma=0.6) -> List[List[MessageSegment]]:
    """
    Segment message by determining inflection points of gauss-filtered bit congruence deltas.

    >>> from nemere.utils.loader import SpecimenLoader
    >>> sl = SpecimenLoader('../input/hide/random-100-continuous.pcap', layer=0, relativeToIP=True)
    >>> segmentsPerMsg = bcDeltaGaussMessageSegmentation(sl)
    Segmentation by inflections of sigma-0.6-gauss-filtered bit-variance.
    >>> for spm in segmentsPerMsg:
    ...     # noinspection PyUnresolvedReferences
    ...     if b''.join([seg.bytes for seg in spm]).hex() != spm[0].message.data.hex():
    ...         print("Mismatch!")

    :return: Segmentation of the specimens in the pool.
    """
    from nemere.inference.analyzers import BitCongruenceDeltaGauss

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

def bcDeltaGaussMessageSegmentationLE(specimens, sigma=0.6) -> List[List[MessageSegment]]:
    """
    Little Endian version of
    Segment message by determining inflection points of gauss-filtered bit congruence deltas.

    >>> from nemere.utils.loader import SpecimenLoader
    >>> sl = SpecimenLoader('../input/hide/random-100-continuous.pcap', layer=0, relativeToIP=True)
    >>> segmentsPerMsg = bcDeltaGaussMessageSegmentationLE(sl)
    Segmentation by inflections of sigma-0.6-gauss-filtered bit-variance for little endian.
    >>> for spm in segmentsPerMsg:
    ...     if b''.join([seg.bytes for seg in spm]).hex() != spm[0].message.data.hex():
    ...         print("Mismatch!")

    :return: Segmentation of the specimens in the pool.
    """
    from nemere.inference.analyzers import BitCongruenceDeltaGaussLE

    print('Segmentation by inflections of sigma-{:.1f}-gauss-filtered bit-variance for little endian.'.format(
        sigma
    ))
    msgSeg = list()
    for l4msg, rmsg in specimens.messagePool.items():
        analyzer = BitCongruenceDeltaGaussLE(l4msg)
        analyzer.setAnalysisParams(sigma)
        analyzer.analyze()
        msgSeg.append(analyzer.messageSegmentation())
    return msgSeg


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # #  Start: Refinements # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# noinspection PyUnusedLocal
def refinements(segmentsPerMsg: List[List[MessageSegment]], **kwargs) -> List[List[MessageSegment]]:
    """
    Refine the segmentation using specific improvements for the feature:
    Inflections of gauss-filtered bit-congruence deltas.

    TODO reevaluate all usages!

    :param segmentsPerMsg: a list of one list of segments per message.
    :return: refined segments in list per message
    """
    # for compatibility with different refinment methods with and without kwargs, remove dummy argument
    if list(kwargs.keys()) == ["unused"]:
        kwargs = {}
    return originalRefinements(segmentsPerMsg, **kwargs)


def pcaMocoRefinements(segmentsPerMsg: List[List[MessageSegment]], **kwargs) -> List[List[MessageSegment]]:
    """
    Refine the segmentation using specific improvements for the feature:
    Inflections of gauss-filtered bit-congruence deltas.

    :param segmentsPerMsg: a list of one list of segments per message.
    :return: refined segments in list per message
    """
    from itertools import chain
    from nemere.inference.formatRefinement import RelocatePCA, CropDistinct

    print("Refine segmentation (2xPCA, moco refinements)...")

    if "collectedSubclusters" in kwargs:
        kwargs["collectEvaluationData"] = kwargs["collectedSubclusters"]
        del kwargs["collectedSubclusters"]

    # charPass1 = charRefinements(segmentsPerMsg)
    # refinedSegmentedMessages = RelocatePCA.refineSegments(charPass1, dc)

    # pcaRound = charRefinements(segmentsPerMsg)
    pcaRound = MessageAnalyzer.convertAnalyzers(segmentsPerMsg, Value)
    for _ in range(2):
        refinementDC = DelegatingDC(list(chain.from_iterable(pcaRound)))
        pcaRound = RelocatePCA.refineSegments(pcaRound, refinementDC, **kwargs)

    # additionally perform most common values refinement
    moco = CropDistinct.countCommonValues(pcaRound)
    print([m.hex() for m in moco])
    refinedSM = list()
    for msg in pcaRound:
        croppedMsg = CropDistinct(msg, moco).split()
        refinedSM.append(croppedMsg)

    charPass2 = charRefinements(refinedSM)

    return charPass2


def pcaMocoDoubleCharRefinements(segmentsPerMsg: Sequence[Sequence[MessageSegment]], **kwargs
                       ) -> List[List[MessageSegment]]:
    """
    perform the char refinements after 2 passes of PCA and moco.
    TODO Is this better than pcaMocoRefinements

    :param segmentsPerMsg: A list of lists of segments for each message.
    :param kwargs: Arguments to forward to the `RelocatePCA.refineSegments()` function
    :return: refined segments in list per message
    """
    import nemere.inference.formatRefinement as refine

    print("Refine segmentation (2xPCA, moco refinements)...")

    if "collectedSubclusters" in kwargs:
        kwargs["collectEvaluationData"] = kwargs["collectedSubclusters"]
        del kwargs["collectedSubclusters"]

    valueSegsPerMsg = MessageAnalyzer.convertAnalyzers(segmentsPerMsg, Value)

    pcaRound = charRefinements(valueSegsPerMsg)
    for _ in range(2):
        refinementDC = DelegatingDC(list(chain.from_iterable(pcaRound)))
        pcaRound = refine.RelocatePCA.refineSegments(pcaRound, refinementDC, **kwargs)

    # additionally perform most common values refinement
    moco = refine.CropDistinct.countCommonValues(pcaRound)
    print([m.hex() for m in moco])
    refinedSM = list()
    for msg in pcaRound:
        croppedMsg = refine.CropDistinct(msg, moco).split()
        refinedSM.append(croppedMsg)
    return charRefinements(refinedSM)


def pcaRefinements(segmentsPerMsg: Sequence[Sequence[MessageSegment]], **kwargs) -> List[List[MessageSegment]]:
    """
    Refine the segmentation using specific improvements for the feature:
    Inflections of gauss-filtered bit-congruence deltas.

    :param segmentsPerMsg: a list of one list of segments per message.
    :param kwargs: Is forwarded to `RelocatePCA.refineSegments()`
    :return: refined segments in list per message
    """
    from itertools import chain
    from nemere.inference.formatRefinement import RelocatePCA

    if "collectedSubclusters" in kwargs:
        kwargs["collectEvaluationData"] = kwargs["collectedSubclusters"]
        del kwargs["collectedSubclusters"]

    print("Refine segmentation (PCA refinements)...")

    valueSegsPerMsg = MessageAnalyzer.convertAnalyzers(segmentsPerMsg, Value)

    # char refinement before and after
    charPass1 = charRefinements(valueSegsPerMsg)
    refinementDC = DelegatingDC(list(chain.from_iterable(charPass1)))
    refinedSM = RelocatePCA.refineSegments(charPass1, refinementDC, **kwargs)
    charPass2 = charRefinements(refinedSM)

    return charPass2


def pcaPcaRefinements(segmentsPerMsg: Sequence[Sequence[MessageSegment]], **kwargs) -> List[List[MessageSegment]]:
    """
    Refine the segmentation using specific improvements for the feature:
    Inflections of gauss-filtered bit-congruence deltas.

    :param segmentsPerMsg: a list of one list of segments per message.
    :param kwargs: is forwarded to `RelocatePCA.refineSegments()`
    :return: refined segments in list per message
    """
    from itertools import chain
    from nemere.inference.formatRefinement import RelocatePCA

    if "collectedSubclusters" in kwargs:
        kwargs["collectEvaluationData"] = kwargs["collectedSubclusters"]
        del kwargs["collectedSubclusters"]

    print("Refine segmentation (2xPCA refinements)...")

    # char refinement before and after
    # pcaRound = charRefinements(segmentsPerMsg)
    pcaRound = MessageAnalyzer.convertAnalyzers(segmentsPerMsg, Value)

    for _ in range(2):
        refinementDC = DelegatingDC(list(chain.from_iterable(pcaRound)))
        pcaRound = RelocatePCA.refineSegments(pcaRound, refinementDC, **kwargs)
    refinedSM = charRefinements(pcaRound)

    return refinedSM


def baseRefinements(segmentsPerMsg: Sequence[Sequence[MessageSegment]]) -> List[List[MessageSegment]]:
    """
    Refine the segmentation using specific improvements for the feature:
    Inflections of gauss-filtered bit-congruence deltas.

    :param segmentsPerMsg: a list of one list of segments per message.
    :return: refined segments in list per message
    """
    import nemere.inference.formatRefinement as refine

    print("Refine segmentation (base refinements)...")

    refinedPerMsg = list()
    for msg in segmentsPerMsg:
        # merge consecutive segments of printable-char values (\t, \n, \r, >= 0x20 and <= 0x7e) into one text field.
        charsMerged = refine.MergeConsecutiveChars(msg).merge()
        charSplited = refine.ResplitConsecutiveChars(charsMerged).split()
        refinedPerMsg.append(charSplited)

    moco = refine.CropDistinct.countCommonValues(refinedPerMsg)
    newstuff = list()
    for msg in refinedPerMsg:
        croppedMsg = refine.CropDistinct(msg, moco).split()
        charmerged = refine.CumulativeCharMerger(croppedMsg).merge()
        newstuff.append(charmerged)

    return newstuff


def zeroBaseRefinements(segmentsPerMsg: Sequence[Sequence[MessageSegment]], **kwargs) -> List[List[MessageSegment]]:
    """
    Refine segmentation by slicing messages at null-bytes using
    nemere.inference.formatRefinement.BlendZeroSlices
    and subsequently apply baseRefinements.

    :param segmentsPerMsg: A list of lists of segments for each message.
    :param kwargs: Flags for collectedSubclusters and/or littleEndian
    :return: refined segments in list per message
    """
    import nemere.inference.formatRefinement as refine

    if "collectedSubclusters" in kwargs:
        kwargs["collectEvaluationData"] = kwargs["collectedSubclusters"]
        del kwargs["collectedSubclusters"]
    littleEndian = "littleEndian" in kwargs and kwargs["littleEndian"] == True

    print("Refine segmentation (zero-slices refinements)...")

    valueSegsPerMsg = MessageAnalyzer.convertAnalyzers(segmentsPerMsg, Value)

    combinedRefinedSegments = [refine.BlendZeroSlices(list(msg)).blend(littleEndian=littleEndian)
                               for msg in valueSegsPerMsg]
    return baseRefinements(combinedRefinedSegments)


def zeroPCARefinements(segmentsPerMsg: Sequence[Sequence[MessageSegment]]) -> List[List[MessageSegment]]:
    """
    Refine segmentation by slicing messages at null-bytes using zeroBaseRefinements
    (i. a., nemere.inference.formatRefinement.BlendZeroSlices)
    and subsequently apply pcaRefinements.

    :param segmentsPerMsg: a list of one list of segments per message.
    :return: refined segments in list per message
    """
    return pcaRefinements(zeroBaseRefinements(segmentsPerMsg))


def nemetylRefinements(segmentsPerMsg: Sequence[Sequence[MessageSegment]]) -> List[List[MessageSegment]]:
    """
    Refine the segmentation using specific improvements for the feature:
    Inflections of gauss-filtered bit-congruence deltas.

    :param segmentsPerMsg: a list of one list of segments per message.
    :return: refined segments in list per message
    """
    import nemere.inference.formatRefinement as refine

    print("Refine segmentation (nemetyl refinements)...")

    refinedPerMsg = list()
    for msg in segmentsPerMsg:
        # merge consecutive segments of printable-char values (\t, \n, \r, >= 0x20 and <= 0x7e) into one text field.
        charsMerged = refine.MergeConsecutiveChars(msg).merge()
        charSplited = refine.ResplitConsecutiveChars(charsMerged).split()
        refinedPerMsg.append(charSplited)

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

    Note: This refinement alone actually performes considerably worse than originalRefinements!

    :param segmentsPerMsg: a list of one list of segments per message.
    :return: refined segments in list per message
    """
    import nemere.inference.formatRefinement as refine

    print("Refine segmentation (char refinements)...")

    refinedPerMsg = list()
    for msg in segmentsPerMsg:
        # merge consecutive segments of printable-char values (\t, \n, \r, >= 0x20 and <= 0x7e) into one text field.
        charsMerged = refine.MergeConsecutiveChars(msg).merge()
        charSplited = refine.ResplitConsecutiveChars(charsMerged).split()
        refinedPerMsg.append(charSplited)
        # assert correct result
        if msg[0].offset != charSplited[0].offset or msg[-1].nextOffset != charSplited[-1].nextOffset:
            raise RuntimeError("Segment bytes where lost!")

    newstuff = list()
    for msg in refinedPerMsg:
        charmerged = refine.CumulativeCharMerger(msg).merge()
        newstuff.append(charmerged)
        # assert correct result
        if msg[0].offset != charmerged[0].offset or msg[-1].nextOffset != charmerged[-1].nextOffset:
            raise RuntimeError("Segment bytes where lost!")

    return newstuff


def originalRefinements(segmentsPerMsg: Sequence[Sequence[MessageSegment]]) -> List[List[MessageSegment]]:
    """
    Refine the segmentation according to the WOOT2018 paper method using specific improvements for the feature:
    Inflections of gauss-filtered bit-congruence deltas.

    :param segmentsPerMsg: a list of one list of segments per message.
    :return: refined segments in list per message
    """
    import nemere.inference.formatRefinement as refine

    print("Refine segmentation (original WOOT18 refinements)...")

    refinedPerMsg = list()
    for msg in segmentsPerMsg:
        # merge consecutive segments of printable-char values (\t, \n, \r, >= 0x20 and <= 0x7e) into one text field.
        charsMerged = refine.MergeConsecutiveChars(msg).merge()
        charSplited = refine.ResplitConsecutiveChars(charsMerged).split()
        refinedPerMsg.append(charSplited)

    return refinedPerMsg


def zerocharPCAmocoRefinements(segmentsPerMsg: Sequence[Sequence[MessageSegment]], **kwargs
                       ) -> List[List[MessageSegment]]:
    """
    Refine segmentation by slicing messages at null-bytes using zeroBaseRefinements
    (i. a., nemere.inference.formatRefinement.BlendZeroSlices)
    and subsequently apply PCA refinement and most-common-values refinement.

    :param segmentsPerMsg: a list of one list of segments per message.
    :param kwargs: Is forwarded to `RelocatePCA.refineSegments()`
        and also directly uses collectedSubclusters and littleEndian flags
    :return: refined segments in list per message
    """
    import nemere.inference.formatRefinement as refine

    if "collectedSubclusters" in kwargs:
        kwargs["collectEvaluationData"] = kwargs["collectedSubclusters"]
        del kwargs["collectedSubclusters"]
    littleEndian = "littleEndian" in kwargs and kwargs["littleEndian"] == True

    print("Refine segmentation (zero-slices, char, 2xPCA, moco refinements)...")

    valueSegsPerMsg = MessageAnalyzer.convertAnalyzers(segmentsPerMsg, Value)

    # .blend(True) to omit single zeros: in most cases (dns, nbns, smb), quality deteriorates.
    zeroSlicedMessages = [refine.BlendZeroSlices(list(msg)).blend(False, littleEndian) for msg in valueSegsPerMsg]
    pcaRound = [refine.CropChars(segs).split() for segs in zeroSlicedMessages]
    for _ in range(2):
        refinementDC = MemmapDC(list(chain.from_iterable(pcaRound)))
        pcaRound = refine.RelocatePCA.refineSegments(pcaRound, refinementDC, **kwargs)

    # additionally perform most common values refinement
    moco = refine.CropDistinct.countCommonValues(pcaRound)
    print([m.hex() for m in moco])
    refinedSM = list()
    for msg in pcaRound:
        croppedMsg = refine.CropDistinct(msg, moco).split()
        refinedSM.append(croppedMsg)

    # decreases FMS slightly for dhcp and smb
    # refinedSM = charRefinements(refinedSM)

    return refinedSM

    # now needs recalculation of segment distances in dc


def pcaMocoSFrefinements(segmentsPerMsg: Sequence[Sequence[MessageSegment]], **kwargs
                                 ) -> List[List[MessageSegment]]:
    """
    Refine the segmentation according to the (unpublished) NEMEPCA paper method using specific improvements
    for the feature: Inflections of gauss-filtered bit-congruence deltas.
        * PCA
        * CropDistinct
        * SplitFixedv2

    :param segmentsPerMsg: a list of one list of segments per message.
    :param kwargs: For evaluation:
        * comparator: Encapsulated true field bounds to compare results to.
        * reportFolder: For evaluation: Destination path to write results and statistics to.
        * collectedSubclusters: For evaluation: Collect the intermediate (sub-)clusters generated during
        the analysis of the segments.
        * and others accepted by refine.RelocatePCA.refineSegments()
    :return: refined segments in list per message

    :raises ClusterAutoconfException: In case no clustering can be performed due to failed parameter autodetection.
    """
    import nemere.inference.formatRefinement as refine

    if "collectedSubclusters" in kwargs:
        kwargs["collectEvaluationData"] = kwargs["collectedSubclusters"]
        del kwargs["collectedSubclusters"]

    print("Refine segmentation (PCA, CropDistinct, SplitFixedv2 refinements)...")

    pcaRound = MessageAnalyzer.convertAnalyzers(segmentsPerMsg, Value)
    for _ in range(1):
        refinementDC = MemmapDC(list(chain.from_iterable(pcaRound)))
        pcaRound = refine.RelocatePCA.refineSegments(pcaRound, refinementDC, **kwargs)

    # additionally perform most common values refinement
    moco = refine.CropDistinct.countCommonValues(pcaRound)
    print([m.hex() for m in moco])
    refinedSM = list()
    for msg in pcaRound:
        croppedMsg = refine.CropDistinct(msg, moco).split()
        # and: first segments that are longer than 2 and at least two bytes are less than \x10
        # (that have the first two bytes being non-zero)
        if croppedMsg[0].length > 2 and sum(b < 0x10 for b in croppedMsg[0].bytes) >= 2:
            splitfixed = refine.SplitFixed(croppedMsg).split(0, 1)
            refinedSM.append(splitfixed)
        else:
            refinedSM.append(croppedMsg)
    return refinedSM

def zerocharPCAmocoSFrefinements(segmentsPerMsg: Sequence[Sequence[MessageSegment]], **kwargs
                                 ) -> List[List[MessageSegment]]:
    """
    Refine the segmentation with a slightly simpler method than in the NEMEPCA paper (DSN 2022)
    using specific improvements for the feature: Inflections of gauss-filtered bit-congruence deltas.
        * NEMESYS
        * NullBytes
        * CropChars
        * PCA
        * CropDistinct
        * SplitFixedv2

    :param segmentsPerMsg: a list of one list of segments per message.
    :param kwargs: For evaluation:
        * comparator: Encapsulated true field bounds to compare results to.
        * reportFolder: For evaluation: Destination path to write results and statistics to.
        * collectedSubclusters: For evaluation: Collect the intermediate (sub-)clusters generated during
        the analysis of the segments.
        * and others accepted by refine.RelocatePCA.refineSegments()
    :return: refined segments in list per message

    :raises ClusterAutoconfException: In case no clustering can be performed due to failed parameter autodetection.
    """
    import nemere.inference.formatRefinement as refine

    if "collectedSubclusters" in kwargs:
        kwargs["collectEvaluationData"] = kwargs["collectedSubclusters"]
        del kwargs["collectedSubclusters"]
    littleEndian = "littleEndian" in kwargs and kwargs["littleEndian"] == True

    print("Refine segmentation (NullBytes, CropChars)...")

    valueSegsPerMsg = MessageAnalyzer.convertAnalyzers(segmentsPerMsg, Value)

    # .blend(True) to omit single zeros: in most cases (dns, nbns, smb), quality deteriorates.
    zeroSlicedMessages = [refine.BlendZeroSlices(list(msg)).blend(False, littleEndian) for msg in valueSegsPerMsg]
    pcaRound = [refine.CropChars(segs).split() for segs in zeroSlicedMessages]
    return pcaMocoSFrefinements(pcaRound, **kwargs)

def entropymergeZeroCharPCAmocoSFrefinements(segmentsPerMsg: Sequence[Sequence[MessageSegment]], **kwargs
                                 ) -> List[List[MessageSegment]]:
    """
    Refine the segmentation according to the NEMEPCA (CNS 2022 paper) method using specific improvements
    for the feature: Inflections of gauss-filtered bit-congruence deltas.
        * NEMESYS
        * EntropyMerge
        * NullBytes
        * CropChars
        * PCA
        * CropDistinct
        * SplitFixedv2

    :param segmentsPerMsg: a list of one list of segments per message.
    :param kwargs: For evaluation:
        * comparator: Encapsulated true field bounds to compare results to.
        * reportFolder: For evaluation: Destination path to write results and statistics to.
        * collectedSubclusters: For evaluation: Collect the intermediate (sub-)clusters generated during
        the analysis of the segments.
        * and others accepted by refine.RelocatePCA.refineSegments()
    :return: refined segments in list per message

    :param segmentsPerMsg:
    :param kwargs:
    :return:
    """
    import nemere.inference.formatRefinement as refine

    print("Refine segmentation (Merge consecutive random segments)...")
    valueSegsPerMsg = MessageAnalyzer.convertAnalyzers(segmentsPerMsg, Value)
    entropyMergedMessages = None
    newMergedMessages = valueSegsPerMsg
    # repeat merging as long as there are segments to merge left that match the conditions
    while newMergedMessages != entropyMergedMessages:
        entropyMergedMessages = newMergedMessages
        newMergedMessages = [refine.EntropyMerger(list(msg)).merge() for msg in entropyMergedMessages]
    refinedMessages = zerocharPCAmocoSFrefinements(newMergedMessages, **kwargs)
    return refinedMessages
    # return [refine.EntropyMerger(list(msg)).merge() for msg in refinedMessages]


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # #  End: Refinements # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


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


def segments2clusteredTypes(clusterer: AbstractClusterer, analysisTitle: str,
                            singularTemplates=True, charSegments:List[AbstractSegment]=None) \
        -> List[Tuple[str, List[Tuple[str, List[Tuple[str, TypedSegment]]]]]]:
    """
    Cluster segments according to the distance of their feature vectors.
    Keep and label segments classified as noise.

    :param clusterer: Clusterer object that contains all the segments to be clustered
    :param analysisTitle: the string to be used as label for the result
    :param singularTemplates: Flag to separate singular values into their own templates.
    :param charSegments: list of char segments if they should not be clustered regularly.
    :return: List/Tuple structure of annotated analyses, clusters, and segments.
        List [ of
            Tuples (
                 "analysis label",
                 List [ of cluster
                    Tuples (
                        "cluster label",
                        List [ of segment
                            Tuples (
                                "segment label (e. g. field type)",
                                MessageSegment object
                            )
                        ]
                    )
                ]
            )
        ]
    """
    from math import log
    from .templates import Template
    print("Clustering segments...")
    noise, *clusters = clusterer.clusterSimilarSegments(False)

    if charSegments is not None and len(charSegments) > 0:
        clusters.append(charSegments)

    # TODO handle separately
    if singularTemplates:
        # extract "large" templates from noise that should rather be its own cluster
        for idx, seg in reversed(list(enumerate(noise.copy()))):  # type: int, MessageSegment
            freqThresh = log(len(clusterer.segments))
            if isinstance(seg, Template):
                if len(seg.baseSegments) > freqThresh:
                    clusters.append([noise.pop(idx)])  # .baseSegments

    print("{} clusters generated from {} segments".format(len(clusters), len(clusterer.segments)))

    segmentClusters = list()
    segLengths = set()
    numNoise = len(noise)
    if numNoise > 0:
        noiseSegLengths = {seg.length for seg in noise}
        outputLengths = [str(slen) for slen in noiseSegLengths]
        if len(outputLengths) > 5:
            outputLengths = outputLengths[:2] + ["..."] + outputLengths[-2:]
        segLengths.update(noiseSegLengths)
        noisetypes = {t: len(s) for t, s in segments2types(noise).items()}
        segmentClusters.append(('{} ({} bytes), Noise: {} Seg.s'.format(
            analysisTitle, " ".join(outputLengths), numNoise),
                                   [("{}: {} Seg.s".format(cseg.fieldtype, noisetypes[cseg.fieldtype]), cseg)
                                    for cseg in noise] )) # ''
    for cnum, segs in enumerate(clusters):
        clusterDists = clusterer.distanceCalculator.distancesSubset(segs)
        typegroups = segments2types(segs)
        clusterSegLengths = {seg.length for seg in segs}
        outputLengths = [str(slen) for slen in clusterSegLengths]
        if len(outputLengths) > 5:
            outputLengths = outputLengths[:2] + ["..."] + outputLengths[-2:]
        segLengths.update(clusterSegLengths)

        mostFrequentTypes = sorted(((ftype, len(tsegs)) for ftype, tsegs in typegroups.items()), key=lambda x: -x[1])

        segmentGroups = ('{} ({} bytes), Cluster #{} ({:.2f} {}): {} Seg.s ($d_{{max}}$={:.3f})'.format(
            analysisTitle, " ".join(outputLengths),
            cnum, mostFrequentTypes[0][1]/sum(s for t, s in mostFrequentTypes), mostFrequentTypes[0][0], len(segs), clusterDists.max()), list())
        for ftype, tsegs in typegroups.items():  # [label, segment]
            segmentGroups[1].extend([("{}: {} Seg.s".format(ftype, len(tsegs)), tseg) for tseg in tsegs])
        segmentClusters.append(segmentGroups)

    segmentClusters = [ ( '{} ({} bytes) {}'.format(analysisTitle,
                                                    next(iter(segLengths)) if len(segLengths) == 1 else 'mixedamount',
                                                    clusterer if clusterer else 'n/a'),
                          segmentClusters) ]
    return segmentClusters


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
    """
    :param values: Byte values to test for being a character sequence.
    :param meanCorridor: Corridor of allowed mean values of non-null values in the bytes
    :param minLen: Minimum length of the byte string to be considered as a character sequence.
    :return: The given bytes string is likely a character sequence.
    """
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

def filterChars(segments: Iterable[AbstractSegment], meanCorridor=(50, 115), minLen=6):
    """
    Filter segments by some hypotheses about what might be a char sequence:
        1. Segment is larger than minLen
        2. Segment has not only 0x00 values
        3. All values are < 127 (0x7f)
        4. The sequence's values have a mean of between n and m, e. g. if 0x20 <= char <= 0x7e (printable chars)
        5. The ratio of non-printables is less than 2/3 of all values

    :param segments: List of segments to be filtered
    :param meanCorridor: Corridor of mean value that denotes a probable char sequence.
        A meanCorridor=(0x20, 0x7e) would ensure to include even segments of repetitions
        of " " (bottommost printable) or "~" (topmost printable)
    :param minLen: Minimum length of a segment to be considered for hypothesis testing
    :return: Filtered segments: segments that hypothetically are chars
    """

    filtered = [seg for seg in segments
                if isExtendedCharSeq(seg.bytes, meanCorridor, minLen)
                ]
    return filtered


def wobbleSegmentInMessage(segment: MessageSegment):
    """
    "Wobbles" segment byte values against its own start offsets.

    TODO Instead of the start offsets, for the end if would be, e. g.:
        if segment.nextOffset < len(segment.message.data):  segment.nextOffset + 1

    :param segment: Segment to wobble
    :return: Wobbled segments
    """
    wobbles = [segment]

    if segment.offset > 0:
        wobbles.append(MessageSegment(segment.analyzer, segment.offset - 1, segment.length + 1))
    if segment.length > 1:
        wobbles.append(MessageSegment(segment.analyzer, segment.offset + 1, segment.length - 1))

    return wobbles


def locateNonPrintable(bstring: bytes) -> List[int]:
    """
    A bit broader definition of printable than python string's isPrintable()

    :param bstring: a string of bytes
    :return: position of bytes not in \t, \n, \r or between >= 0x20 and <= 0x7e
    """
    from nemere.inference.formatRefinement import isPrintableChar

    npr = list()
    for idx, bchar in enumerate(bstring):
        if isPrintableChar(bchar):
            continue
        else:
            npr.append(idx)
    return npr
