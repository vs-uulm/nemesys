"""
NEMEFTR-full mode 1:
Clustering of segments on similarity without ground truth. However, ground truth is expected and used for evaluation.
Segments are created from messages by NEMESYS and clustered with DBSCANsegmentClusterer
and refined by the method selected at the command line (-r).
Generates segment-dissimilarity topology plots of the clustering result.
"""

import argparse
from math import log

from nemere.inference.templates import FieldTypeTemplate, DBSCANsegmentClusterer, ClusterAutoconfException
from nemere.inference.segmentHandler import baseRefinements, originalRefinements, pcaMocoRefinements, \
    isExtendedCharSeq, nemetylRefinements, pcaRefinements, \
    zerocharPCAmocoSFrefinements, pcaMocoSFrefinements
from nemere.utils.reportWriter import IndividualClusterReport, CombinatorialClustersReport, SegmentClusterGroundtruthReport
from nemere.visualization.distancesPlotter import SegmentTopology
from nemere.utils.evaluationHelpers import *

debug = False

# fix the analysis method to VALUE
analysis_method = 'value'
# fix the distance method to canberra
distance_method = 'canberra'
# tokenizers to select from
tokenizers = ('nemesys', 'zeros')     # zeroslices + CropChars
# refinement methods
refinementMethods = [
    "none",
    "original",  # WOOT2018 paper
    "base",  # ConsecutiveChars+moco
    "nemetyl",  # INFOCOM2020 paper: ConsecutiveChars+moco+splitfirstseg
    "PCA1",  # PCA 1-pass | applicable to nemesys and zeros
    "PCAmoco",  # PCA+moco
    "PCAmocoSF",  # PCA+moco+SF (v2) | applicable to zeros
    "zerocharPCAmocoSF",  # with split fixed (v2)
    ]



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
                        choices=tokenizers, default=tokenizers[0])
    parser.add_argument('-f', '--refinement', help='Select segment refinement method.', choices=refinementMethods,
                        default=refinementMethods[-1])
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

    refinement = args.refinement
    if tokenizer[:7] == "nemesys":
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

    trueSegmentedMessages = {msgseg[0].message: msgseg
                             for msgseg in annotateFieldTypes(analyzerType, analysisArgs, comparator)}
    # # # # # # # # # # # # # # # # # # # # # # # #


    # Configure some clustering alternatives during evaluations: (can be removed after final decision on either way.)
    separateChars = True
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

    for epsint in range(5,50):
        eps = epsint*0.01
        # # # # # # # # # # # # # # # # # # # # # # # #
        # cluster segments to determine field types on commonality
        clusterer = None  # just to prevent PyCharm's warnings

        print("Clustering...")
        try:
            if separateChars:
                # noinspection PyUnboundLocalVariable
                min_samples = round(log(len(nonCharSegs)))
                clusterer = DBSCANsegmentClusterer(dc, nonCharSegs, min_samples=min_samples, eps=eps)
            else:
                min_samples = round(log(len(segments2cluster)))
                clusterer = DBSCANsegmentClusterer(dc, segments2cluster, min_samples=min_samples, eps=eps)
            # # # # # # # # # # # # # # # # # # # # # # # # #
        except ClusterAutoconfException as e:
            print("Initial clustering of the segments in the trace failed. The protocol in this trace cannot be inferred. "
                  "The original exception message was:\n", e)
            exit(10)
        # # # # # # # # # # # # # # # # # # # # # # # # #

        inferenceParams = TitleBuilderSens(tokenizer, refinement, args.sigma, clusterer)

        noise, *clusters = clusterer.clusterSimilarSegments(False)
        print("{} clusters generated from {} distinct segments".format(len(clusters), len(dc.segments)))
        # # # # # # # # # # # # # # # # # # # # # # # #

        # noinspection PyUnboundLocalVariable
        if separateChars and len(charSegments) > 0:
            clusters.append(charSegments)

        # The same value in different segments only represented by once
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
        for cLabel, segments in enumerate(uniqueClusters):
            # generate FieldTypeTemplates (padded nans) - Templates as is
            ftype = FieldTypeTemplate(segments)
            ftype.fieldtype = "tf{:02d}".format(cLabel)
            fTypeTemplates.append(ftype)
        # # # # # # # # # # # # # # # # # # # # # # # #


        # # # # # # # # # # # # # # # # # # # # # # # #
        ftclusters = {ftc.fieldtype: ftc.baseSegments for ftc in fTypeTemplates}
        """ftclusters is a mixed list of MessageSegment and Template"""
        ftclusters["Noise"] = uniqueNoise

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
            topoplot = SegmentTopology(clusterStats, fTypeTemplates, uniqueNoise, dc)
            topoplot.writeFigure(specimens, inferenceParams, elementsReport, filechecker)
        # # # # # # # # # # # # # # # # # # # # # # # #

        filechecker.writeReportMetadata(fromCache.dccachefn if fromCache.isLoaded else None)



    if args.interactive:
        from collections import Counter
        from nemere.inference.segments import MessageSegment, TypedSegment
        # noinspection PyUnresolvedReferences
        import numpy

        IPython.embed()










