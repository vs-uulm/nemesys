"""
segment messages by NEMESYS and cluster
"""

import argparse, IPython
from os.path import isfile, basename, join, splitext, exists
from os import makedirs
from math import log
from typing import Union, Any
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.colors as colors
from collections import Counter

from inference.templates import DBSCANsegmentClusterer, FieldTypeTemplate, Template, TypedTemplate, FieldTypeContext, \
    ClusterAutoconfException
from inference.segmentHandler import symbolsFromSegments, wobbleSegmentInMessage, isExtendedCharSeq
from inference.formatRefinement import RelocatePCA
from validation import reportWriter
from visualization.distancesPlotter import DistancesPlotter
from visualization.simplePrint import *
from utils.evaluationHelpers import *
from validation.dissectorMatcher import FormatMatchScore


debug = False

# fix the analysis method to VALUE
analysisTitle = 'value'
# fix the distance method to canberra
distance_method = 'canberra'
# use NEMESYS segments
tokenizer = 'nemesys'

kneedleSensitivity=12.0



def markSegNearMatch(segment: Union[MessageSegment, Template]):
    """
    Print messages with the given segment in each message marked.
    Supports Templates by resolving them to their base segments.

    :param segment: list of segments that should be printed, i. e.,
        marked within the print of the message it is originated from.
    """
    if isinstance(segment, Template):
        segs = segment.baseSegments
    else:
        segs = [segment]

    print()  # one blank line for visual structure
    for seg in segs:
        inf4seg = inferred4segment(seg)
        comparator.pprint2Interleaved(seg.message, [infs.nextOffset for infs in inf4seg],
                                      (seg.offset, seg.nextOffset))

    # # a simpler approach
    # markSegmentInMessage(segment)

    # # get field number of next true field
    # tsm = trueSegmentedMessages[segment.message]  # type: List[MessageSegment]
    # fsnum, offset = 0, 0
    # while offset < segment.offset:
    #     offset += tsm[fsnum].offset
    #     fsnum += 1
    # markSegmentInMessage(trueSegmentedMessages[segment.message][fsnum])

    # # limit to immediate segment context
    # posSegMatch = None  # first segment that starts at or after the recognized field
    # for sid, seg in enumerate(trueSegmentedMessages[segment.message]):
    #     if seg.offset > segment.offset:
    #         posSegMatch = sid
    #         break
    # posSegEnd = None  # last segment that ends after the recognized field
    # for sid, seg in enumerate(trueSegmentedMessages[segment.message]):
    #     if seg.nextOffset > segment.nextOffset:
    #         posSegEnd = sid
    #         break
    # if posSegMatch is not None:
    #     contextStart = max(posSegMatch - 2, 0)
    #     if posSegEnd is None:
    #         posSegEnd = posSegMatch
    #     contextEnd = min(posSegEnd + 1, len(trueSegmentedMessages))

def inferred4segment(segment: MessageSegment) -> Sequence[MessageSegment]:
    """
    :param segment: The input segment.
    :return: All inferred segments for the message which the input segment is from.
    """
    return next(msegs for msegs in inferredSegmentedMessages if msegs[0].message == segment.message)

def inferredFEs4segment(segment: MessageSegment) -> List[int]:
    return [infs.nextOffset for infs in inferred4segment(segment)]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Cluster NEMESYS segments of messages according to similarity.')
    parser.add_argument('pcapfilename', help='Filename of the PCAP to load.')
    parser.add_argument('-i', '--interactive', help='Show interactive plot instead of writing output to file and '
                                                    'open ipython prompt after finishing the analysis.',
                        action="store_true")
    parser.add_argument('-s', '--sigma', type=float, help='Sigma for noise reduction (gauss filter) in NEMESYS,'
                                                          'default: 0.9')
    parser.add_argument('-p', '--with-plots',
                        help='Generate plots of true field types and their distances.',
                        action="store_true")
    args = parser.parse_args()

    if not isfile(args.pcapfilename):
        print('File not found: ' + args.pcapfilename)
        exit(1)
    pcapbasename = basename(args.pcapfilename)
    withPlots = args.with_plots

    analyzerType = analyses[analysisTitle]
    analysisArgs = None

    # # # # # # # # # # # # # # # # # # # # # # # #
    # cache/load the DistanceCalculator to the filesystem
    #
    # noinspection PyUnboundLocalVariable
    specimens, comparator, inferredSegmentedMessages, dc, segmentationTime, dist_calc_segmentsTime = cacheAndLoadDC(
        args.pcapfilename, analysisTitle, tokenizer, debug, analyzerType, analysisArgs, args.sigma, True#, True
    )  # Note!  When manipulating distances, deactivate caching by adding "True".
    # chainedSegments = dc.rawSegments
    # # # # # # # # # # # # # # # # # # # # # # # #
    trueSegmentedMessages = {msgseg[0].message: msgseg
                         for msgseg in annotateFieldTypes(analyzerType, analysisArgs, comparator)
                         }
    # tabuSeqOfSeg(trueSegmentedMessages)
    # print(trueSegmentedMessages.values())

    reportFolder = join(reportFolder, splitext(pcapbasename)[0])
    makedirs(reportFolder)

    # # # # # # # # # # # # # # # # # # # # # # # #
    # conduct PCA refinement
    try:
        refinedSM = RelocatePCA.refineSegments(inferredSegmentedMessages, dc)
        # TODO needs recalculation of segment distances
    except ClusterAutoconfException as e:
        print("Initial clustering of the segments in the trace failed. The protocol in this trace cannot be inferred. "
              "The original exception message was:\n", e)
        exit(10)
    # # # # # # # # # # # # # # # # # # # # # # # #



    # # # # # # # # # # # # # # # # # # # # # # # #
    # some paramters
    trace = splitext(pcapbasename)[0]
    # # # # # # # # # # # # # # # # # # # # # # # #









    # # # # # # # # # # # # # # # # # # # # # # # #
    # cluster segments to determine field types on commonality
    try:
        clusterer = DBSCANsegmentClusterer(dc, S=kneedleSensitivity)
    except ClusterAutoconfException as e:
        print("Initial clustering of the segments in the trace failed. The protocol in this trace cannot be inferred. "
              "The original exception message was:\n", e)
        exit(10)

    # noinspection PyUnboundLocalVariable
    noise, *clusters = clusterer.clusterSimilarSegments(False)
    # noise: List[MessageSegment]
    # clusters: List[List[MessageSegment]]

    # extract "large" templates from noise that should rather be its own cluster
    for idx, seg in reversed(list(enumerate(noise.copy()))):  # type: int, MessageSegment
        freqThresh = log(len(dc.rawSegments))
        if isinstance(seg, Template):
            if len(seg.baseSegments) > freqThresh:
                clusters.append(noise.pop(idx).baseSegments)

    print("{} clusters generated from {} distinct segments".format(len(clusters), len(dc.segments)))
    # # # # # # # # # # # # # # # # # # # # # # # #




    # # # # # # # # # # # # # # # # # # # # # # # #
    fTypeTemplates = list()
    fTypeContext = list()
    for cLabel, segments in enumerate(clusters):
        # generate FieldTypeTemplates (padded nans)
        ftype = FieldTypeTemplate(segments)
        ftype.fieldtype = "tf{:02d}".format(cLabel)
        fTypeTemplates.append(ftype)
        with open(join(reportFolder, "segmentclusters-" + splitext(pcapbasename)[0] + ".csv"), "a") as segfile:
            segcsv = csv.writer(segfile)
            segcsv.writerows([
                [],
                ["# Cluster", cLabel, "Segments", len(segments)],
                ["-"*10]*4,
            ])
            segcsv.writerows(
                {(seg.bytes.hex(), seg.bytes) for seg in segments}
            )

        # generate FieldTypeContexts (padded values)
        resolvedSegments = list()
        for seg in segments:
            if isinstance(seg, Template):
                resolvedSegments.extend(seg.baseSegments)
            else:
                resolvedSegments.append(seg)
        fcontext = FieldTypeContext(resolvedSegments)
        fcontext.fieldtype = ftype.fieldtype
        fTypeContext.append(fcontext)

        # print("\nCluster", cLabel, "Segments", len(segments))
        # print({seg.bytes for seg in segments})

        # for seg in segments:
        #     # # sometimes raises: ValueError: On entry to DLASCL parameter number 4 had an illegal value
        #     # try:
        #     #     confidence = float(ftype.confidence(numpy.array(seg.values))) if ftype.length == seg.length else 0.0
        #     # except ValueError as e:
        #     #     print(seg.values)
        #     #     raise e
        #     #
        #
        #     confidence = 0.0
        #     if isinstance(seg, Template):
        #         for bs in seg.baseSegments:
        #             recog = RecognizedVariableLengthField(bs.message, ftype, bs.offset, bs.nextOffset, confidence)
        #             printFieldContext(trueSegmentedMessages, recog)
        #     else:
        #         recog = RecognizedVariableLengthField(seg.message, ftype, seg.offset, seg.nextOffset, confidence)
        #         printFieldContext(trueSegmentedMessages, recog)
    # # # # # # # # # # # # # # # # # # # # # # # #






    # # # # # # # # # # # # # # # # # # # # # # # # #
    # # TODO mark (at least) exact segment matches with the true data type
    # # currently marks only very few segments. Improve boundaries first!
    # typedMatchSegs = list()
    # for seg in dc.segments:
    #     if isinstance(seg, MessageSegment):
    #         typedMatchSegs.append(comparator.segment2typed(seg))
    #     elif isinstance(seg, Template):
    #         machingType = None
    #         typedBasesegments = list()
    #         for bs in seg.baseSegments:
    #             tempTyped = comparator.segment2typed(bs)
    #             if not isinstance(tempTyped, TypedSegment):
    #                 # print("At least one segment in template is not a field match.")
    #                 # markSegNearMatch(tempTyped)
    #                 machingType = None
    #                 break
    #             elif machingType == tempTyped.fieldtype or machingType is None:
    #                 machingType = tempTyped.fieldtype
    #                 typedBasesegments.append(tempTyped)
    #             else:
    #                 # print("Segment's matching field types are not the same in template, e. g., {} and {} ({})".format(
    #                 #     machingType, tempTyped.fieldtype, tempTyped.bytes.hex()
    #                 # ))
    #                 machingType = None
    #                 break
    #         if machingType is None:
    #             typedMatchSegs.append(seg)
    #         else:
    #             typedMatchSegs.append(TypedTemplate(seg.values, typedBasesegments, seg._method))
    # # # # # # # # # # # # # # # # # # # # # # # # #





    # # # # # # # # # # # # # # # # # # # # # # # #
    if withPlots:
        print("Plot distances...")
        sdp = DistancesPlotter(specimens,
                               'distances-' + "nemesys-segments_DBSCAN-eps{:0.3f}-ms{:d}".format(
                                   clusterer.eps, int(clusterer.min_samples)), False)

        clustermask = {segid: cluN for cluN, segL in enumerate(clusters) for segid in dc.segments2index(segL)}
        clustermask.update({segid: "Noise" for segid in dc.segments2index(noise)})
        labels = numpy.array([clustermask[segid] for segid in range(len(dc.segments))])
        # plotManifoldDistances(dc.segments, dc.distanceMatrix, labels)
        sdp.plotSegmentDistances(dc, labels)
        sdp.writeOrShowFigure()
        del sdp
    # # # # # # # # # # # # # # # # # # # # # # # #




    # # # # # # # # # # # # # # # # # # # # # # # #
    # The following is strictly speaking not clustering relevant:
    # # # # # # # # # # # # # # # # # # # # # # # #


    # # show position of each segment individually.
    # for seg in dc.segments:
    #     markSegNearMatch(seg)

    # comparator.pprint2Interleaved(dc.segments[6].message, inferredFEs4segment(dc.segments[6]),
    #                               (dc.segments[6].offset-2, dc.segments[6].nextOffset+1))









    if args.interactive:
        from tabulate import tabulate
        # globals().update(locals())
        IPython.embed()










