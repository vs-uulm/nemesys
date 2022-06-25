"""
NEMEFTR-full mode 1:
Clustering of segments on similarity without ground truth. However, ground truth is expected and used for evaluation.
Segments are created from messages by NEMESYS and clustered with DBSCANsegmentClusterer
and refined by the method selected at the command line (-r).
Generates segment-dissimilarity topology plots of the clustering result.
"""

import argparse
import math

from nemere.inference.templates import ClusterAutoconfException, DBSCANadjepsClusterer, \
    DBSCANsegmentClusterer, OPTICSsegmentClusterer
from nemere.inference.segmentHandler import baseRefinements, originalRefinements, \
    isExtendedCharSeq, nemetylRefinements
from nemere.utils.reportWriter import IndividualClusterReport, CombinatorialClustersReport, \
    SegmentClusterGroundtruthReport, writeFieldTypesTikz
from nemere.validation.clusterInspector import SegmentClusterCauldron
from nemere.visualization.distancesPlotter import SegmentTopology
from nemere.utils.evaluationHelpers import *

debug = False

# fix the analysis method to VALUE
analysis_method = 'value'
# fix the distance method to canberra
distance_method = 'canberra'

# Parameter for DBSCAN epsilon autoconfiguration by Kneedle
kneedleSensitivity=24.0
# kneedleSensitivity=4.0
# kneedleSensitivity=6.0
# kneedleSensitivity=9.0



def inferred4segment(segment: MessageSegment) -> Sequence[MessageSegment]:
    """
    :param segment: The input segment.
    :return: All inferred segments for the message which the input segment is from.
    """
    return next(msegs for msegs in segmentedMessages if msegs[0].message == segment.message)

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
    parser.add_argument('-l', '--layer', type=int, default=2,
                        help='Protocol layer relative to IP to consider. Default is 2 layers above IP '
                             '(typically the payload of a transport protocol).')
    parser.add_argument('-r', '--relativeToIP', default=False, action='store_true')
    parser.add_argument('-t', '--tokenizer', help='Select the tokenizer for this analysis run.',
                        choices=CachedDistances.tokenizers, default=CachedDistances.tokenizers[0])
    parser.add_argument('-f', '--refinement', help='Select segment refinement method.',
                        choices=CachedDistances.refinementMethods, default=CachedDistances.refinementMethods[-1])
    parser.add_argument('-e', '--littleendian', help='Toggle presumed endianness to little.', action="store_true")
    args = parser.parse_args()

    filechecker = StartupFilecheck(args.pcapfilename)
    withplots = args.with_plots
    littleendian = args.littleendian == True
    analyzerType = analyses[analysis_method]
    analysisArgs = None
    analysisTitle = analysis_method
    tokenizer = args.tokenizer
    if littleendian:
        tokenizer += "le"

    # # # # # # # # # # # # # # # # # # # # # # # #
    # cache/load the DistanceCalculator to the filesystem
    #
    fromCache = CachedDistances(args.pcapfilename, analysisTitle, args.layer, args.relativeToIP)
    # Note!  When manipulating distances calculation, deactivate caching by uncommenting the following assignment.
    # fromCache.disableCache = True
    fromCache.debug = debug
    if analysisArgs is not None:
        # noinspection PyArgumentList
        fromCache.configureAnalysis(*analysisArgs)
    fromCache.configureTokenizer(tokenizer, args.sigma)
    fromCache.filter = True

    refinement = args.refinement
    if tokenizer[:7] == "nemesys" or tokenizer[:4] == "bide":
        if args.refinement == "original":
            fromCache.configureRefinement(originalRefinements)
        elif args.refinement == "base":
            fromCache.configureRefinement(baseRefinements)
        elif args.refinement == "nemetyl":
            fromCache.configureRefinement(nemetylRefinements)
        elif args.refinement is None or args.refinement == "none":
            print("No refinement selected. Performing raw segmentation.")
        else:
            print(f"The refinement {args.refinement} is not supported with this tokenizer. Abort.")
            exit(2)
    try:
        fromCache.get()
    except ClusterAutoconfException as e:
        print("Initial clustering of the segments in the trace failed. The protocol in this trace cannot be inferred. "
              "The original exception message was:\n", e)
        exit(10)
    segmentedMessages = fromCache.segmentedMessages
    # noinspection PyTypeChecker
    specimens, comparator, dc = fromCache.specimens, fromCache.comparator, fromCache.dc  # type: SpecimenLoader, MessageComparator, DistanceCalculator
    segmentationTime, dist_calc_segmentsTime = fromCache.segmentationTime, fromCache.dist_calc_segmentsTime
    #
    # # gt for manual usage
    # trueSegmentedMessages = {msgseg[0].message: msgseg
    #                          for msgseg in annotateFieldTypes(analyzerType, analysisArgs, comparator)}
    # # # # # # # # # # # # # # # # # # # # # # # #

    # Configure some clustering alternatives during evaluations: (can be removed after final decision on either way.)
    separateChars = True
    singularsFromNoise = False
    # to evaluate clustering of unique-valued segments
    clusterUnique = False
    if clusterUnique:
        # TODO eval clustering of unique-valued segments
        segments2cluster = dc.segments
    else:
        segments2cluster = dc.rawSegments

    # extract char sequences
    if separateChars:
        charSegments = list()
        nonCharSegs = list()
        for seg in segments2cluster:
            if isExtendedCharSeq(seg.bytes):
                charSegments.append(seg)
            else:
                nonCharSegs.append(seg)

    # # # # # # # # # # # # # # # # # # # # # # # #
    # cluster segments to determine field types on commonality
    clusterer = None  # just to prevent PyCharm's warnings
    try:
        if separateChars:
            if tokenizer[:7] == "nemesys":
                # noinspection PyUnboundLocalVariable
                clusterer = DBSCANadjepsClusterer(dc, nonCharSegs, S=kneedleSensitivity)
                if args.refinement in ["emzcPCAmocoSF", "zerocharPCAmocoSF"]:
                    clusterer.eps *= 1.3
            else:
                clusterer = DBSCANsegmentClusterer(dc, nonCharSegs, S=kneedleSensitivity)
            # clusterer = OPTICSsegmentClusterer(dc, nonCharSegs)
            # clusterer = HDBSCANsegmentClusterer(dc, nonCharSegs, min_cluster_size=12)
        else:
            if tokenizer[:7] == "nemesys":
                clusterer = DBSCANadjepsClusterer(dc, segments2cluster, S=kneedleSensitivity)
            else:
                clusterer = DBSCANsegmentClusterer(dc, segments2cluster, S=kneedleSensitivity)
            # clusterer = OPTICSsegmentClusterer(dc, segments2cluster)
            # clusterer = HDBSCANsegmentClusterer(dc, segments2cluster, min_cluster_size=12)
        # # # # # # # # # # # # # # # # # # # # # # # # #
        if isinstance(clusterer, DBSCANsegmentClusterer):
            clusterer.preventLargeCluster()
    except ClusterAutoconfException as e:
        print("Initial clustering of the segments in the trace failed using DBSCAN."
              " The original exception message was:\n ", e, "\nFalling back to OPTICS clusterer.")
        ms = round(math.sqrt(len(nonCharSegs if separateChars else segments2cluster)))
        clusterer = OPTICSsegmentClusterer(dc, nonCharSegs, min_samples=ms)
        # print("Initial clustering of the segments in the trace failed. The protocol in this trace cannot be inferred. "
        #       "The original exception message was:\n", e)
        # exit(10)
    # # # # # # # # # # # # # # # # # # # # # # # # #

    inferenceParams = TitleBuilderSens(tokenizer, refinement, args.sigma, clusterer)

    cauldron = SegmentClusterCauldron(clusterer, analysisTitle)

    if separateChars:
        # noinspection PyUnboundLocalVariable
        cauldron.appendCharSegments(charSegments)
    # # TODO extract "large" templates from noise that should rather be its own cluster
    if singularsFromNoise:
        cauldron.extractSingularFromNoise()
    # # # # # # # # # # # # # # # # # # # # # # # #
    # # # # # # # # # # # # # # # # # # # # # # # # #
    # TODO make one "enum" cluster
    #  see nemere.validation.clusterInspector.SegmentClusterCauldron.extractSingularFromNoise
    # # # # # # # # # # # # # # # # # # # # # # # # #

    cauldron.clustersOfUniqueSegments()

    # # # # # # # # # # # # # # # # # # # # # # # #
    fTypeTemplates = cauldron.exportAsTemplates()

    # # for treating singular clusters individually (in plot and reports)
    # # TODO reinstate non-singular clusters for everything else besides Topo Plots
    # for i in cauldron.unisegClusters.clusterIndices:
    #     # generate FieldTypeTemplates (padded nans) - Templates as is
    #     ftype = FieldTypeTemplate(cauldron.unisegClusters.clusterElements(i))
    #     ftype.fieldtype = cauldron.unisegClusters.clusterLabel(i)
    #     fTypeTemplates.append(ftype)

    # fTypeContext = list()
    # for cLabel, segments in enumerate(clusters):
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
    # Templates resolved to single Segments
    #  see adjustments to make in nemere.utils.reportWriter.SegmentClusterGroundtruthReport._writeCSV
    # if dc.segments != clusterer.segments:
    #     # print("resolve Templates")
    #     ftclusters = {ftc.fieldtype : ftc.baseSegments for ftc in fTypeContext}
    #     ftclusters["Noise"] = resolveTemplates2Segments(noise)
    # else:
    # print("keep Templates")
    # Templates as is
    ftclusters = {ftc.fieldtype: ftc.baseSegments for ftc in fTypeTemplates}
    """ftclusters is a mixed list of MessageSegment and Template"""
    ftclusters["Noise"] = cauldron.noise
    # ftclusters["Noise"] = noise

    # # # # # # # # # # # # # # # # # # # # # # # #
    # Report: write cluster elements to csv
    # # # # # # # # # # # # # # # # # # # # # # # #
    elementsReport = SegmentClusterGroundtruthReport(comparator, dc.segments, filechecker)
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
    elementsReport.write(ftclusters)
    # # # # # # # # # # # # # # # # # # # # # # # #

    # # # # # # # # # # # # # # # # # # # # # # # #
    # # Report: allover clustering quality statistics
    # # # # # # # # # # # # # # # # # # # # # # # #
    report = CombinatorialClustersReport(elementsReport.groundtruth, filechecker)
    report.write(ftclusters, inferenceParams.plotTitle)
    #
    # field-type-wise cluster quality statistics
    report = IndividualClusterReport(elementsReport.groundtruth, filechecker)
    # add column $d_max$ to report
    cluDists = {lab: clusterer.distanceCalculator.distancesSubset(clu) for lab, clu in ftclusters.items()}
    cluDistsMax = {lab: clu.max() for lab, clu in cluDists.items()}
    report.addColumn(cluDistsMax, "$d_max$")
    #
    # # clusters with any internal distance > 0
    # nonZeroDmaxClusters = {lab: clu for lab, clu in ftclusters.items() if cluDistsMax[lab] > 0}
    #
    report.write(ftclusters, inferenceParams.plotTitle)
    clusterStats = report.precisionRecallList

    # # # # # # # # # # # # # # # # # # # # # # # #
    if withplots:
        # distance Topology plot
        topoplot = SegmentTopology(clusterStats, fTypeTemplates, cauldron.noise, dc)
        topoplot.writeFigure(specimens, inferenceParams, elementsReport, filechecker)
    # # # # # # # # # # # # # # # # # # # # # # # #
    writeFieldTypesTikz(comparator, segmentedMessages, fTypeTemplates, filechecker)
    # # # # # # # # # # # # # # # # # # # # # # # #
    filechecker.writeReportMetadata(fromCache.dccachefn if fromCache.isLoaded else None)

    # # show position of each segment individually.
    # for clu in clusters:
    #     print("# "*20)
    #     for seg in clu:
    #         markSegNearMatch(seg)

    # # # show segmentation of messages.
    # for msgsegs in inferredSegmentedMessages:
    #     comparator.pprint2Interleaved(msgsegs[0].message, [infs.nextOffset for infs in msgsegs])

    if args.interactive:
        from collections import Counter
        from nemere.inference.segments import MessageSegment, TypedSegment
        # noinspection PyUnresolvedReferences
        import numpy

        # globals().update(locals())
        IPython.embed()










