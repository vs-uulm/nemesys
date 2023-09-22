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
from nemere.inference.segmentHandler import baseRefinements, originalRefinements, pcaMocoRefinements, \
    isExtendedCharSeq, nemetylRefinements, pcaRefinements, \
    zerocharPCAmocoSFrefinements, pcaMocoSFrefinements, entropymergeZeroCharPCAmocoSFrefinements
from nemere.utils.reportWriter import IndividualClusterReport, CombinatorialClustersReport, \
    SegmentClusterGroundtruthReport,  writeFieldTypesTikz, writeSemanticTypeHypotheses
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
        elif args.refinement == "PCA1":
            fromCache.configureRefinement(pcaRefinements, littleEndian=littleendian)
            if littleendian:
                refinement = args.refinement + "le"
        elif args.refinement == "PCAmoco":
            fromCache.configureRefinement(pcaMocoRefinements, littleEndian=littleendian)
            if littleendian:
                refinement = args.refinement + "le"
        elif args.refinement == "zerocharPCAmocoSF":
            fromCache.configureRefinement(zerocharPCAmocoSFrefinements, littleEndian=littleendian)
            if littleendian:
                refinement = args.refinement + "le"
        elif args.refinement == "emzcPCAmocoSF":
            fromCache.configureRefinement(entropymergeZeroCharPCAmocoSFrefinements, littleEndian=args.littleendian)
        elif args.refinement is None or args.refinement == "none":
            print("No refinement selected. Performing raw segmentation.")
        else:
            print(f"The refinement {args.refinement} is not supported with this tokenizer. Abort.")
            exit(2)
    elif tokenizer[:5] == "zeros":
        if args.refinement == "PCA1":
            fromCache.configureRefinement(pcaRefinements, littleEndian=littleendian)
            if littleendian:
                refinement = args.refinement + "le"
        elif args.refinement == "PCAmocoSF":
            fromCache.configureRefinement(pcaMocoSFrefinements, littleEndian=littleendian)
            if littleendian:
                refinement = args.refinement + "le"
        elif args.refinement is None or args.refinement == "none":
            print("No refinement selected. Performing zeros segmentation with CropChars.")
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
        else:
            if tokenizer[:7] == "nemesys":
                clusterer = DBSCANadjepsClusterer(dc, segments2cluster, S=kneedleSensitivity)
            else:
                clusterer = DBSCANsegmentClusterer(dc, segments2cluster, S=kneedleSensitivity)
        # # # # # # # # # # # # # # # # # # # # # # # # #
        if isinstance(clusterer, DBSCANsegmentClusterer):
            clusterer.preventLargeCluster()
    except ClusterAutoconfException as e:
        print("Initial clustering of the segments in the trace failed using DBSCAN."
              " The original exception message was:\n ", e, "\nFalling back to OPTICS clusterer.")
        ms = round(math.sqrt(len(nonCharSegs if separateChars else segments2cluster)))
        clusterer = OPTICSsegmentClusterer(dc, nonCharSegs, min_samples=ms)
    # # # # # # # # # # # # # # # # # # # # # # # # #

    inferenceParams = TitleBuilderSens(tokenizer, refinement, args.sigma, clusterer)

    cauldron = SegmentClusterCauldron(clusterer, analysisTitle)

    if separateChars:
        # noinspection PyUnboundLocalVariable
        cauldron.appendCharSegments(charSegments)
    # TODO extract "large" templates from noise that should rather be its own cluster
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

    ftclusters = {ftc.fieldtype: ftc.baseSegments for ftc in fTypeTemplates}
    """ftclusters is a mixed list of MessageSegment and Template"""
    ftclusters["Noise"] = cauldron.noise

    # # # # # # # # # # # # # # # # # # # # # # # #
    # Report: write cluster elements to csv
    # # # # # # # # # # # # # # # # # # # # # # # #
    elementsReport = SegmentClusterGroundtruthReport(comparator, dc.segments, filechecker)
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

    # # # # # # # # # # # # # # # # # # # # # # # #
    writeSemanticTypeHypotheses(cauldron, filechecker)
    # # # # # # # # # # # # # # # # # # # # # # # #

    if args.interactive:
        from collections import Counter
        from nemere.inference.segments import MessageSegment, TypedSegment
        # noinspection PyUnresolvedReferences
        import numpy

        IPython.embed()


