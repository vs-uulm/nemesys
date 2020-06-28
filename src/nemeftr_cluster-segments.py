"""
NEMEFTR-full mode 1:
Clustering of segments on similarity without ground truth.
Segments are created from messages by NEMESYS and clustered with DBSCANsegmentClusterer
and refined by the method selected at the command line (-r).
Generates segment-dissimilarity topology plots of the clustering result.
"""

import argparse, IPython
from collections import Counter
from math import log
from os.path import isfile, basename, join, splitext, exists
from os import makedirs
import matplotlib.pyplot as plt
import numpy

from nemere.inference.templates import DBSCANsegmentClusterer, FieldTypeTemplate, TypedTemplate, Template,\
    ClusterAutoconfException
from nemere.inference.segments import MessageSegment, TypedSegment
from nemere.inference.segmentHandler import baseRefinements, originalRefinements, pcaMocoRefinements, \
    pcaPcaRefinements, zeroBaseRefinements, isExtendedCharSeq, zeroPCARefinements
from nemere.visualization.distancesPlotter import DistancesPlotter
from nemere.utils.evaluationHelpers import *

debug = False

# fix the analysis method to VALUE
analysisTitle = 'value'
# fix the distance method to canberra
distance_method = 'canberra'
# use NEMESYS segments
tokenizer = 'nemesys'
# refinement methods
refinementMethods = [
    "original", # WOOT2018 paper
    "base",     # moco+splitfirstseg
    "PCA",      # PCA
    "PCAmoco",  # PCA+moco
    "zero",      # zeroslices+base
    "zeroPCA",  # zero+base + 2-pass PCA
    ]


# kneedleSensitivity=4.0
# kneedleSensitivity=6.0
# kneedleSensitivity=9.0
kneedleSensitivity=24.0




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
    parser.add_argument('-r', '--refinement', help='Select segment refinement method.', choices=refinementMethods)
    args = parser.parse_args()

    if not isfile(args.pcapfilename):
        print('File not found: ' + args.pcapfilename)
        exit(1)
    pcapbasename = basename(args.pcapfilename)
    trace = splitext(pcapbasename)[0]
    reportFolder = join(reportFolder, trace)
    if not exists(reportFolder):
        makedirs(reportFolder)
    withPlots = args.with_plots
    analyzerType = analyses[analysisTitle]
    analysisArgs = None

    # # # # # # # # # # # # # # # # # # # # # # # #
    # cache/load the DistanceCalculator to the filesystem
    #
    doCache = False
    if args.refinement == "original":
        specimens, comparator, inferredSegmentedMessages, dc, segmentationTime, dist_calc_segmentsTime = cacheAndLoadDC(
            args.pcapfilename, analysisTitle, tokenizer, debug, analyzerType, analysisArgs, args.sigma, True,
            refinementCallback=originalRefinements
            , disableCache=not doCache
        )
    elif args.refinement == "base":
        specimens, comparator, inferredSegmentedMessages, dc, segmentationTime, dist_calc_segmentsTime = cacheAndLoadDC(
            args.pcapfilename, analysisTitle, tokenizer, debug, analyzerType, analysisArgs, args.sigma, True,
            refinementCallback=baseRefinements
            , disableCache=not doCache
        )
    elif args.refinement == "PCA":
        specimens, comparator, inferredSegmentedMessages, dc, segmentationTime, dist_calc_segmentsTime = cacheAndLoadDC(
            args.pcapfilename, analysisTitle, tokenizer, debug, analyzerType, analysisArgs, args.sigma, True,
            refinementCallback=pcaPcaRefinements
            , disableCache=not doCache
        )
    elif args.refinement == "PCAmoco":
        specimens, comparator, inferredSegmentedMessages, dc, segmentationTime, dist_calc_segmentsTime = cacheAndLoadDC(
            args.pcapfilename, analysisTitle, tokenizer, debug, analyzerType, analysisArgs, args.sigma, True,
            refinementCallback=pcaMocoRefinements
            , disableCache=not doCache
        )
    elif args.refinement == "zero":
        specimens, comparator, inferredSegmentedMessages, dc, segmentationTime, dist_calc_segmentsTime = cacheAndLoadDC(
            args.pcapfilename, analysisTitle, tokenizer, debug, analyzerType, analysisArgs, args.sigma, True,
            refinementCallback=zeroBaseRefinements
            , disableCache=not doCache
        )
    elif args.refinement == "zeroPCA":
        specimens, comparator, inferredSegmentedMessages, dc, segmentationTime, dist_calc_segmentsTime = cacheAndLoadDC(
            args.pcapfilename, analysisTitle, tokenizer, debug, analyzerType, analysisArgs, args.sigma, True,
            refinementCallback=zeroPCARefinements
            , disableCache=not doCache
        )
    else:
        print("Unknown refinement", args.refinement, "\nAborting")
        exit(2)
    trueSegmentedMessages = {msgseg[0].message: msgseg
                             for msgseg in annotateFieldTypes(analyzerType, analysisArgs, comparator)}
    # # # # # # # # # # # # # # # # # # # # # # # #



    separateChars = True

    # extract char sequences
    if separateChars:
        charSegments = list()
        nonCharSegs = list()
        for seg in dc.rawSegments:
            if isExtendedCharSeq(seg.bytes):
                charSegments.append(seg)
            else:
                nonCharSegs.append(seg)

    # # # # # # # # # # # # # # # # # # # # # # # #
    # cluster segments to determine field types on commonality
    try:
        if separateChars:
            k = round(log(len(nonCharSegs)) / 2)
            clusterer = DBSCANsegmentClusterer(dc, nonCharSegs, S=kneedleSensitivity, k=k)
        else:
            k = round(log(len(dc.rawSegments)) / 2)
            clusterer = DBSCANsegmentClusterer(dc, dc.rawSegments, S=kneedleSensitivity, k=k)
        # clusterer = DBSCANsegmentClusterer(dc, S=kneedleSensitivity)

        # reduce k if no realistic eps is detected (TODO move into autoconfigure and use if y below < .5 of samples)
        if clusterer.eps < 0.05:
            clusterer.k //= 2
            clusterer.min_samples, clusterer.eps = clusterer._autoconfigure()

        # adjust epsilon  TODO test
        epsfrac = 3  # done: 5, 4,
        epspivot = 0.15
        autoeps = clusterer.eps
        adjeps = clusterer.eps + clusterer.eps / epsfrac * (1 if clusterer.eps < epspivot else -1)
        clusterer.eps = adjeps

        clusterer.kneelocator.plot_knee() #plot_knee_normalized()
        plt.text(0.5, 0.2, "S = {:.1f}\neps = {:.3f}\nk = {:.0f}".format(clusterer.S, clusterer.eps, clusterer.k))
        plt.savefig(join(reportFolder, "knee-{}-S{:.1f}-eps{:.3f}.pdf".format(trace, kneedleSensitivity, clusterer.eps)))

        # clusterer.eps *= 1.15
        # clusterer.eps *= 0.5 * 1/(1+math.exp(24*clusterer.eps-6)) + 0.8

        # import math
        # clusterer.min_samples = math.sqrt(len(clusterer.segments))

    except ClusterAutoconfException as e:
        print("Initial clustering of the segments in the trace failed. The protocol in this trace cannot be inferred. "
              "The original exception message was:\n", e)
        exit(10)
    # clusterer = HDBSCANsegmentClusterer(dc, dc.rawSegments, min_cluster_size=12)

    # noinspection PyUnboundLocalVariable
    noise, *clusters = clusterer.clusterSimilarSegments(False)
    # noise: List[MessageSegment]
    # clusters: List[List[MessageSegment]]

    # # extract "large" templates from noise that should rather be its own cluster
    # for idx, seg in reversed(list(enumerate(noise.copy()))):  # type: int, MessageSegment
    #     freqThresh = log(len(dc.rawSegments))
    #     if isinstance(seg, Template):
    #         if len(seg.baseSegments) > freqThresh:
    #             clusters.append(noise.pop(idx).baseSegments)

    print("{} clusters generated from {} distinct segments".format(len(clusters), len(dc.segments)))
    # # # # # # # # # # # # # # # # # # # # # # # #

    if separateChars and len(charSegments) > 0:
        clusters.append(charSegments)

    # # # # # # # # # # # # # # # # # # # # # # # # #
    # # make "enum" cluster TODO
    # cluCopy = list(clusters)
    # enumCluster = list()
    # for clu in cluCopy:
    #     if len({seg.bytes for seg in clu}) < clusterer.min_samples:
    #         enumCluster.extend(clu)
    #         clusters.remove(clu)
    # if len(enumCluster) > 0:
    #     clusters.append(enumCluster)
    # # # # # # # # # # # # # # # # # # # # # # # # #

    uniqueClusters = list()
    for elements in clusters + [noise]:
        # same template with different labels
        uniqueSegments = {dc.segments[dc.segments2index([tSegment])[0]] for tSegment in elements}
        uniqueClusters.append(sorted(uniqueSegments, key=lambda x: x.values))
    mixedSegments = [seg for seg, cnt in
                     Counter(tSegment for elements in uniqueClusters for tSegment in elements).items()
                     if cnt > 1]
    for tSegment in mixedSegments:
        mixedClusters = [elements for elements in uniqueClusters
                         if tSegment in elements]
        assert len(mixedClusters) < 2  # that would be strange and we needed to find some solution then
        toReplace = [sIdx for sIdx, mSegment in enumerate(mixedClusters[0]) if mSegment == tSegment]
        for rIdx in reversed(sorted(toReplace)):
            del mixedClusters[0][rIdx]
        mixedClusters[0].append(("[mixed]", tSegment))
    uniqueNoise = uniqueClusters[-1]
    uniqueClusters = uniqueClusters[:-1]


    # # # # # # # # # # # # # # # # # # # # # # # #
    fTypeTemplates = list()
    fTypeContext = list()
    # for cLabel, segments in enumerate(clusters):
    for cLabel, segments in enumerate(uniqueClusters):
        # generate FieldTypeTemplates (padded nans) - Templates as is
        ftype = FieldTypeTemplate(segments)
        ftype.fieldtype = "tf{:02d}".format(cLabel)
        fTypeTemplates.append(ftype)

        # # generate FieldTypeContexts (padded values) - Templates resolved to single Segments
        # resolvedSegments = resolveTemplates2Segments(segments)
        # fcontext = FieldTypeContext(resolvedSegments)
        # fcontext.fieldtype = ftype.fieldtype
        # fTypeContext.append(fcontext)

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







    # # # # # # # # # # # # # # # # # # # # # # # #
    # mark segment matches with > 50% overlap with the prevalent true data type for the nearest boundaries.
    # list of tuples of overlap ratio ("intensity of match") and segment
    typedMatchSegs = dict()  # type: Dict[Union[Template, MessageSegment], Tuple[float, Union[TypedSegment, MessageSegment]]]
    typedMatchTemplates = dict()  # type: Dict[Union[Template, MessageSegment], Tuple[float, Union[TypedSegment, TypedTemplate, Template, MessageSegment]]]
    for seg in dc.segments:
        # create typed segments/templates per cluster to get the inferred assignment
        if isinstance(seg, MessageSegment):
            typedMatchSegs[seg] = comparator.segment2typed(seg)
            typedMatchTemplates[seg] = comparator.segment2typed(seg)
        elif isinstance(seg, Template):
            machingType = None
            typedBaseSegments = [comparator.segment2typed(bs) for bs in seg.baseSegments]
            typedMatchSegs.update({bs: ts for bs, ts in zip(seg.baseSegments, typedBaseSegments)})
            for ratio, baseS in typedBaseSegments:
                # assign true type to inferred segment if overlap is larger than 50%, i. e.,
                # (number of bytes of true segment)/(number of bytes of inferred segment) > 0.5
                if not isinstance(baseS, TypedSegment) or ratio <= 0.5:
                    # print("At least one segment in template is not a field match.")
                    # markSegNearMatch(tempTyped)
                    machingType = None
                    break
                elif machingType == baseS.fieldtype or machingType is None:
                    machingType = baseS.fieldtype
                else:
                    # print("Segment's matching field types are not the same in template, e. g., {} and {} ({})".format(
                    #     machingType, tempTyped.fieldtype, tempTyped.bytes.hex()
                    # ))
                    machingType = None
                    break

            if machingType is None:
                typedMatchTemplates[seg] = (0.0, seg)
            else:
                typedMatchTemplates[seg] = (float(numpy.mean([tr for tr, _ in typedBaseSegments])),
                                           TypedTemplate(seg.values, [ts for _, ts in typedBaseSegments], seg._method))
    # # unknown segments
    # unk = [(o, t) for o, t in typedMatchSegs.values() if o < 1]
    # # print all unidentified segments
    # for ovr, seg in unk:
    #     print("overlap: {:.2f}".format(ovr), seg.fieldtype if isinstance(seg, TypedSegment) else "")
    #     # if isinstance(seg, Template):
    #     #     for bs in seg.baseSegments:
    #     #         comparator.pprint2Interleaved(bs.message, mark=bs)
    #     # else:
    #     comparator.pprint2Interleaved(seg.message, mark=seg)
    # # # # # # # # # # # # # # # # # # # # # # # #

    # # # # # # # # # # # # # # # # # # # # # # # #
    # # allover clustering quality statistics
    # if tokenizer == "nemesys":
    # if args.refinement == "PCAmoco":
    #     sigma = pcamocoSigmapertrace[pcapbasename] if not args.sigma and pcapbasename in pcamocoSigmapertrace else \
    #         0.9 if not args.sigma else args.sigma
    # else:
    sigma = sigmapertrace[pcapbasename] if not args.sigma and pcapbasename in sigmapertrace else \
        1.2 if not args.sigma else args.sigma

    # if dc.segments != clusterer.segments:
    #     # print("resolve Templates")
    #     #
    #     # Templates resolved to single Segments
    #     ftclusters = {ftc.fieldtype : ftc.baseSegments for ftc in fTypeContext}
    #     ftclusters["Noise"] = resolveTemplates2Segments(noise)
    #     groundtruth = {rawSeg: typSeg[1].fieldtype if typSeg[0] > 0.5 else "[unknown]"
    #                    for rawSeg, typSeg in typedMatchSegs.items()}
    # else:
    # print("keep Templates")
    #
    # Templates as is
    ftclusters = {ftc.fieldtype: ftc.baseSegments for ftc in fTypeTemplates}
    ftclusters["Noise"] = uniqueNoise
    # ftclusters["Noise"] = noise
    groundtruth = {rawSeg: typSeg[1].fieldtype if typSeg[0] > 0.5 else "[unknown]"
                   for rawSeg, typSeg in typedMatchTemplates.items()}

    if isinstance(clusterer, DBSCANsegmentClusterer):
        runtitle = "{}-{}-{}-S={:.1f}-eps={:.2f}-min_samples={:.2f}".format(
            tokenizer if tokenizer != "nemesys" else "{}-sigma={:.1f}".format(tokenizer, sigma),
            args.refinement, type(clusterer).__name__, kneedleSensitivity, clusterer.eps, clusterer.min_samples)
    else:
        runtitle = "{}-{}-{}-S={:.1f}-min_samples={:.2f}".format(
            tokenizer if tokenizer != "nemesys" else "{}-sigma={:.1f}".format(tokenizer, sigma),
            args.refinement, type(clusterer).__name__, kneedleSensitivity, clusterer.min_samples)

    writeCollectiveClusteringStaticstics(ftclusters, groundtruth, runtitle, comparator)
    # # field-type-wise cluster quality statistics
    clusterStats, conciseness = writeIndividualClusterStatistics(ftclusters, groundtruth, runtitle, comparator)

    # # # # # # # # # # # # # # # # # # # # # # # #
    with open(join(reportFolder, "segmentclusters-" + splitext(pcapbasename)[0] + ".csv"), "a") as segfile:
        segcsv = csv.writer(segfile)
        for cLabel, segments in ftclusters.items():
            segcsv.writerows([
                [],
                ["# Cluster", cLabel, "Segments", len(segments)],
                ["-" * 10] * 4,
            ])
            # if dc.segments != clusterer.segments:
            #     # Templates resolved to single Segments
            #     segcsv.writerows({(seg.bytes.hex(), seg.bytes, typedMatchSegs[seg][1].fieldtype, typedMatchSegs[seg][0])
            #                       for seg in segments})
            # else:
            # Templates as is
            segcsv.writerows({(seg.bytes.hex(), seg.bytes,
                               typedMatchTemplates[seg][1].fieldtype if isinstance(typedMatchTemplates[seg][1],
                                                                                   (TypedTemplate, TypedSegment))
                                                                     else "[unknown]",
                               typedMatchTemplates[seg][0])
                              for seg in segments})
    # # # # # # # # # # # # # # # # # # # # # # # #

    # # # # # # # # # # # # # # # # # # # # # # # #
    if withPlots:
        # largest clusters
        clusterCutoff = 15

        print("Plot distances...")
        atitle = 'distances-'
        if isinstance(clusterer, DBSCANsegmentClusterer):
             atitle += "nemesys-segments_DBSCAN-eps{:0.3f}-ms{:d}".format(
                clusterer.eps, int(clusterer.min_samples))
        else:
            atitle += "nemesys-segments_HDBSCAN-ms{:d}".format(
                int(clusterer.min_samples))
        if clusterCutoff > 0:
            atitle +="-largest{}clusters".format(clusterCutoff)
        sdp = DistancesPlotter(specimens, atitle, False)

        clusterStatsLookup = {stats[0]: (stats[4], stats[2], stats[1])  # label, mostFreqentType, precision, recall, numSegsinCuster
                              for stats in clusterStats if stats is not None}
        sortedClusters = sorted(fTypeTemplates, key=lambda x: -len(x.baseSegments))
        if clusterCutoff > 0:
            selectedClusters = [ftt for ftt in sortedClusters
                                if clusterStatsLookup[ftt.fieldtype][2] != unknown][:clusterCutoff]
        else:
            selectedClusters = sortedClusters
        omittedClusters = [ftt for ftt in sortedClusters if ftt not in selectedClusters]
        clustermask = {segid: "{}: {} seg.s ({:.2f} {})".format(ftt.fieldtype, *clusterStatsLookup[ftt.fieldtype])
            for ftt in selectedClusters for segid in dc.segments2index(ftt.baseSegments)}
        clustermask.update({segid: "Noise" for segid in dc.segments2index(
            uniqueNoise + [bs for ftt in omittedClusters for bs in ftt.baseSegments]
        )})
        labels = numpy.array([clustermask[segid] for segid in range(len(dc.segments))])
        sdp.plotManifoldDistances(
            [typedMatchTemplates[seg][1] if typedMatchTemplates[seg][0] > 0.5 else seg for seg in dc.segments],
            dc.distanceMatrix, labels)
        # sdp.plotSegmentDistances(dc, labels)
        sdp.writeOrShowFigure()
        del sdp
    # # # # # # # # # # # # # # # # # # # # # # # #




    # # show position of each segment individually.
    # for clu in clusters:
    #     print("# "*20)
    #     for seg in clu:
    #         markSegNearMatch(seg)

    # # # show segmentation of messages.
    # for msgsegs in inferredSegmentedMessages:
    #     comparator.pprint2Interleaved(msgsegs[0].message, [infs.nextOffset for infs in msgsegs])




    if args.interactive:
        # globals().update(locals())
        IPython.embed()










