"""
Reference implementation for calling NEMEFTR-full mode 1, the NEtwork MEssage Field Type Recognition,
classification of data types, with an unknown protocol.

Clustering of segments on similarity without ground truth.
Segments are created from messages by NEMESYS and clustered with DBSCANsegmentClusterer
and refined by the method selected at the command line (-r).
Generates segment-dissimilarity topology plots of the clustering result.
"""
import argparse
import os

import IPython
import matplotlib.pyplot as plt
import numpy as numpy

from nemere.inference.segmentHandler import originalRefinements, nemetylRefinements, zerocharPCAmocoSFrefinements, \
    pcaMocoSFrefinements, isExtendedCharSeq
from nemere.inference.templates import ClusterAutoconfException, FieldTypeTemplate, \
    DBSCANadjepsClusterer
from nemere.utils.evaluationHelpers import StartupFilecheck, CachedDistances, TitleBuilderSens
from nemere.utils.reportWriter import SegmentClusterReport
from nemere.visualization.distancesPlotter import DistancesPlotter
from nemere.visualization.simplePrint import FieldClassesPrinter

debug = False

# fix the analysis method to VALUE
analysis_method = 'value'
# fix the distance method to canberra
distance_method = 'canberra'
# tokenizers to select from
tokenizers = ('nemesys', 'zeros')
# refinement methods
refinementMethods = [
    "none",
    "original", # WOOT2018 paper
    "nemetyl",  # INFOCOM2020 paper: ConsecutiveChars+moco+splitfirstseg
    "PCAmocoSF",  # PCA+moco+SF (v2) | applicable to zeros
    "zerocharPCAmocoSF"  # with split fixed (v2)
    ]
# Parameter for DBSCAN epsilon autoconfiguration by Kneedle
kneedleSensitivity=24.0

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Cluster NEMESYS segments of messages according to similarity.')
    parser.add_argument('pcapfilename', help='Filename of the PCAP to load.')
    parser.add_argument('-l', '--layer', type=int, default=2,
                        help='Protocol layer relative to IP to consider. Default is 2 layers above IP '
                             '(typically the payload of a transport protocol).')
    parser.add_argument('-r', '--relativeToIP', default=False, action='store_true')
    parser.add_argument('-e', '--littleendian', help='Toggle presumed endianness to little.', action="store_true")

    parser.add_argument('-t', '--tokenizer', help='Select the tokenizer for this analysis run.',
                        choices=tokenizers, default=tokenizers[0])
    parser.add_argument('-s', '--sigma', type=float, help='Sigma for noise reduction (gauss filter) in NEMESYS,'
                                                          'default: 0.9')
    parser.add_argument('-f', '--refinement', help='Select segment refinement method.', choices=refinementMethods,
                        default=refinementMethods[-1])

    parser.add_argument('-p', '--with-plots',
                        help='Generate plots of true field types and their distances.',
                        action="store_true")
    parser.add_argument('-i', '--interactive', help='Show interactive plot instead of writing output to file and '
                                                    'open ipython prompt after finishing the analysis.',
                        action="store_true")
    args = parser.parse_args()

    filechecker = StartupFilecheck(args.pcapfilename)
    withplots = args.with_plots
    littleendian = args.littleendian == True
    tokenizer = args.tokenizer
    if littleendian:
        tokenizer += "le"

    # # # # # # # # # # # # # # # # # # # # # # # #
    # cache/load the DistanceCalculator to the filesystem
    #
    fromCache = CachedDistances(args.pcapfilename, analysis_method, args.layer, args.relativeToIP)
    # Note!  When manipulating distances calculation, deactivate caching by uncommenting the following assignment.
    # fromCache.disableCache = True
    fromCache.debug = debug
    # As we analyze a truly unknown protocol, tell CachedDistances that it should not try to use tshark to obtain
    # a dissection. The switch may be set to true for evaluating the approach with a known protocol.
    # see src/nemetyl_align-segments.py
    fromCache.dissectGroundtruth = False
    fromCache.configureTokenizer(tokenizer, args.sigma)
    refinement = args.refinement
    if tokenizer[:7] == "nemesys":
        if args.refinement == "original":
            fromCache.configureRefinement(originalRefinements)
        elif args.refinement == "nemetyl":
            fromCache.configureRefinement(nemetylRefinements)
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
        if args.refinement == "PCAmocoSF":
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
    specimens, comparator, dc = fromCache.specimens, fromCache.comparator, fromCache.dc
    segmentationTime, dist_calc_segmentsTime = fromCache.segmentationTime, fromCache.dist_calc_segmentsTime

    segments2cluster = dc.segments

    # extract char sequences
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
        clusterer = DBSCANadjepsClusterer(dc, nonCharSegs, S=kneedleSensitivity)
        clusterer.preventLargeCluster()
    except ClusterAutoconfException as e:
        print("Initial clustering of the segments in the trace failed. The protocol in this trace cannot be inferred. "
              "The original exception message was:\n", e)
        exit(10)
    # # # # # # # # # # # # # # # # # # # # # # # # #
    if withplots:
        clusterer.kneelocator.plot_knee()  # plot_knee_normalized()
        plt.text(0.5, 0.2, "S = {:.1f}\neps = {:.3f}\nk = {:.0f}".format(clusterer.S, clusterer.eps, clusterer.k))
        plt.savefig(os.path.join(filechecker.reportFullPath, "knee-{}-S{:.1f}-eps{:.3f}.pdf".format(
            filechecker.pcapstrippedname, kneedleSensitivity, clusterer.eps)))

    noise, *clusters = clusterer.clusterSimilarSegments(False)
    # # # # # # # # # # # # # # # # # # # # # # # # #

    inferenceParams = TitleBuilderSens(tokenizer, refinement, args.sigma, clusterer)
    print("{} clusters generated from {} distinct segments".format(len(clusters), len(dc.segments)))

    if len(charSegments) > 0:
        clusters.append(charSegments)

    # generate labels for inferred clusters.
    ftclusters = {"tf{:02d}".format(cLabel): segments for cLabel, segments in enumerate(clusters)}
    ftclusters["Noise"] = noise
    # alternative representation of the same clusters as FieldTypeTemplate
    ftTemplates = list()
    for cLabel, segments in ftclusters.items():
        ftype = FieldTypeTemplate(segments)
        ftype.fieldtype = cLabel
        ftTemplates.append(ftype)

    # # # # # # # # # # # # # # # # # # # # # # # #
    # Report: write cluster elements to csv
    # # # # # # # # # # # # # # # # # # # # # # # #
    elementsReport = SegmentClusterReport(filechecker, filechecker.reportFullPath)
    elementsReport.write(ftclusters)

    # # # # # # # # # # # # # # # # # # # # # # # #
    # distance Topology plot
    # # # # # # # # # # # # # # # # # # # # # # # #
    if withplots:
        # show only largest clusters
        clusterCutoff = 20
        print("Plot distances...")

        # look up cluster sizes, sort them by size, and select the largest clusters (if clusterCutoff > 0)
        clusterStatsLookup = {cLabel: len(segments)  # label, numSegsinCuster
                              for cLabel, segments in ftclusters.items()}
        sortedClusters = sorted([cLabel for cLabel in ftclusters.keys() if cLabel != "Noise"],
                                key=lambda x: -clusterStatsLookup[x])
        if clusterCutoff > 0:
            selectedClusters = [ftt for ftt in sortedClusters][:clusterCutoff]
            inferenceParams.postProcess = "largest{}clusters".format(clusterCutoff)
        else:
            selectedClusters = sortedClusters
        atitle = 'segment-distances_' + inferenceParams.plotTitle

        # Generate the kind of labels suited and needed for the plot
        omittedClusters = [ftt for ftt in sortedClusters if ftt not in selectedClusters] + ["Noise"]
        clustermask = {segid: "{}: {} seg.s".format(ftt, clusterStatsLookup[ftt])
            for ftt in selectedClusters for segid in dc.segments2index(ftclusters[ftt])}
        # In the plot, label everything as noise that is not in the selected clusters (including the actual noise)
        clustermask.update({segid: "Noise" for segid in dc.segments2index(
            [bs for ftt in omittedClusters for bs in ftclusters[ftt]]
        )})
        labels = numpy.array([clustermask[segid] for segid in range(len(dc.segments))])

        sdp = DistancesPlotter(specimens, atitle, False)
        # hand over selected subset of clusters to plot
        sdp.plotManifoldDistances(dc.segments, dc.distanceMatrix, labels)
        # sdp.plotSegmentDistances(dc, labels)
        sdp.writeOrShowFigure(filechecker.reportFullPath)
        del sdp

    # # # # # # # # # # # # # # # # # # # # # # # #
    # visualization of segments from clusters in messages.
    # # # # # # # # # # # # # # # # # # # # # # # #
    cp = FieldClassesPrinter(ftTemplates)
    msgsupto400bytes = [msg for msg in specimens.messagePool.keys() if len(msg.data) <= 400]
    cp.toTikzFile(msgsupto400bytes[:100])



    filechecker.writeReportMetadata(fromCache.dccachefn if fromCache.isLoaded else None)

    if args.interactive:
        # noinspection PyUnresolvedReferences
        from collections import Counter
        # noinspection PyUnresolvedReferences
        from nemere.inference.segments import MessageSegment, TypedSegment
        # noinspection PyUnresolvedReferences
        import numpy

        # globals().update(locals())
        IPython.embed()
