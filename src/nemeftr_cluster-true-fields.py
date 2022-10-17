"""
NEMEFTR: early state and evaluation of epsilon autoconfiguration. Optimal-segmentation baseline.

Plot and print dissimilarities between segments. Clusters on dissimilarities and compares the results to ground truth.
Segmentations are obtained by dissectors and apply field type identification to them.
Output for evaluation are a dissimilarity topology plot and histogram, ECDF plot, clustered vector visualization plots,
and segment cluster statistics.

Takes a PCAP trace of a known protocol, dissects each message into their fields, and yields segments from each of them.
These segments get analyzed by the given analysis method which is used as feature to determine their similarity.
The distance matrix is generated using representatives for field-type-hypothesis specific values modifications (chars).
Similar fields are then clustered by DBSCAN and for comparison plotted in groups of their real field types.
In addition, a MDS projection into a 2D plane for visualization of the relative distances of the features is plotted.
"""

import argparse, IPython
import csv
from os.path import join, exists

from matplotlib import pyplot as plt
import numpy, math


from nemere.utils.evaluationHelpers import analyses, StartupFilecheck, consolidateLabels, CachedDistances
from nemere.utils.reportWriter import plotMultiSegmentLines, CombinatorialClustersReport, reportFolder, \
    SegmentClusterGroundtruthReport, writeSemanticTypeHypotheses
from nemere.inference.templates import DBSCANsegmentClusterer, ClusterAutoconfException
from nemere.inference.segmentHandler import segments2types, isExtendedCharSeq
from nemere.validation.clusterInspector import TypedSegmentClusterCauldron
from nemere.validation.dissectorMatcher import MessageComparator
from nemere.visualization.distancesPlotter import DistancesPlotter
from nemere.visualization.singlePlotter import SingleMessagePlotter

debug = False

# fix the analysis method to VALUE
analysisTitle = 'value'
# fix the distance method to canberra
distance_method = 'canberra'

kneedleSensitivity=24.0

# for evaluation
besteps = {
    "dhcp_SMIA2011101X_deduped-100.pcap": 0.188,
    "dhcp_SMIA2011101X_deduped-1000.pcap": 0.251,
    "dns_ictf2010_deduped-100.pcap": 0.483,
    "dns_ictf2010_deduped-982-1000.pcap": 0.167,
    "nbns_SMIA20111010-one_deduped-100.pcap": 0.346,
    "nbns_SMIA20111010-one_deduped-1000.pcap": 0.400,
    "ntp_SMIA-20111010_deduped-100.pcap": 0.340,
    "ntp_SMIA-20111010_deduped-1000.pcap": 0.351,
    "smb_SMIA20111010-one_deduped-100.pcap": 0.259,
    "smb_SMIA20111010-one_deduped-1000.pcap": 0.242,
}







if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Analyze fields as segments of messages and plot field type identification quality.')
    parser.add_argument('pcapfilename', help='Filename of the PCAP to load.')
    parser.add_argument('-i', '--interactive', help='Show interactive plot instead of writing output to file and '
                                                    'open ipython prompt after finishing the analysis.',
                        action="store_true")
    parser.add_argument('-p', '--with-plots',
                        help='Write plots and statistics, e. g., about the knee detection, Topology Plot.',
                        action="store_true")
    parser.add_argument('-l', '--layer', type=int, default=2,
                        help='Protocol layer relative to IP to consider. Default is 2 layers above IP '
                             '(typically the payload of a transport protocol).')
    parser.add_argument('-r', '--relativeToIP', default=False, action='store_true')
    args = parser.parse_args()

    filechecker = StartupFilecheck(args.pcapfilename)
    withplots = args.with_plots
    analyzerType = analyses[analysisTitle]
    analysisArgs = None
    singularsFromNoise = False
    separateChars = False

    #
    # # # # # # # # # # # # # # # # # # # # # # # #
    # segment messages according to true fields from the labels and filter 1-byte segments before clustering,
    # respectively cache/load the DistanceCalculator to the filesystem
    #
    fromCache = CachedDistances(args.pcapfilename, analysisTitle, args.layer, args.relativeToIP)
    # Note!  When manipulating distances calculation, deactivate caching by uncommenting the following assignment.
    # fromCache.disableCache = True
    fromCache.debug = debug
    fromCache.configureAnalysis(analysisArgs)
    fromCache.configureTokenizer("tshark", filtering=True)
    try:
        fromCache.get()
    except ClusterAutoconfException as e:
        print("Initial clustering of the segments in the trace failed. The protocol in this trace cannot be inferred. "
              "The original exception message was:\n", e)
        exit(10)
    specimens, comparator, dc = fromCache.specimens, fromCache.comparator, fromCache.dc
    assert isinstance(comparator, MessageComparator)

    # extract char sequences
    if separateChars:
        charSegments = list()
        nonCharSegs = list()
        for seg in dc.rawSegments:
            if isExtendedCharSeq(seg.bytes):
                charSegments.append(seg)
            else:
                nonCharSegs.append(seg)

    print("Clustering...")

    # use DBSCAN
    if separateChars:
        # noinspection PyUnboundLocalVariable
        clusterer = DBSCANsegmentClusterer(dc, nonCharSegs, S=kneedleSensitivity)
    else:
        # clustering raw, possibly duplicate segments, not unique values, here
        clusterer = DBSCANsegmentClusterer(dc, dc.rawSegments, S=kneedleSensitivity)

    # only unique
    clusterer.preventLargeCluster()
    clusterer.min_samples = round(math.sqrt(len(clusterer.distanceCalculator.segments)))

    titleFormat = "tshark {} S {:.1f}".format(
        clusterer, kneedleSensitivity)

    if withplots:
        # Statistics about the knee detection
        # # # # # # # # # # # # # # # # # # # # # # # # #
        autoeps = max(clusterer.kneelocator.all_knees)
        adjeps = clusterer.eps
        clusterer.kneelocator.plot_knee()
        plt.text(0.5, 0.2, "S = {:.1f}\nautoeps = {:.3f}\nadjeps = {:.3f}\nk = {:.0f}".format(
            clusterer.S, autoeps, adjeps, clusterer.k))
        plt.axvline(adjeps, linestyle="dotted", color="blue", alpha=.4)
        plt.savefig(join(filechecker.reportFullPath, "knee-{}-S{:.1f}-eps{:.3f}.pdf".format(
            filechecker.pcapstrippedname, kneedleSensitivity, clusterer.eps)))

        clusterer.kneelocator.plot_knee_normalized()
        plt.text(0.7, 0.5, "S = {:.1f}\nautoeps = {:.3f}\nknee_y = {:.3f}\nnorm_dmax = {:.3f}".format(
            clusterer.S, autoeps, clusterer.kneelocator.knee_y, max(clusterer.kneelocator.y_difference_maxima)))
        plt.savefig(join(filechecker.reportFullPath, "knee_normalized-{}-S{:.1f}-eps{:.3f}.pdf".format(
            filechecker.pcapstrippedname, kneedleSensitivity, clusterer.eps)))

        kneePath = join(reportFolder, "knee-statistics.csv")
        writeHeader = not exists(kneePath)
        with open(kneePath, "a") as kneeStats:
            kneecsv = csv.writer(kneeStats)  # type: csv.writer
            if writeHeader:
                kneecsv.writerow([ "run_title", "trace", "autoeps", "adjeps", "k", "polydeg", "knee_y", "y_dmax", "norm_knees" ])

            kneecsv.writerow([titleFormat, filechecker.pcapstrippedname, autoeps, adjeps, clusterer.k,
                              clusterer.kneelocator.polynomial_degree,
                              ";".join(f"{v:.5f}" for v in clusterer.kneelocator.all_knees_y),
                              ";".join(f"{v:.5f}" for v in clusterer.kneelocator.y_difference_maxima),
                              ";".join(f"{v:.5f}" for v in clusterer.kneelocator.all_norm_knees)
                              ])
        # # # # # # # # # # # # # # # # # # # # # # # # #

        # Histogram of all the distances between the segments
        hstplt = SingleMessagePlotter(specimens, 'histo-distance-1nn-' + titleFormat, False)
        knn = [dc.neighbors(seg)[0][1] for seg in dc.segments]
        hstplt.histogram(knn, bins=[x / 50 for x in range(50)])
        if filechecker.pcapbasename in besteps:
            plt.axvline(besteps[filechecker.pcapbasename], label=besteps[filechecker.pcapbasename],
                        color="orchid", linestyle="dotted")
        plt.axvline(clusterer.eps, label="{:.3f}".format(clusterer.eps), color="darkmagenta")
        hstplt.writeOrShowFigure()
        plt.clf()

    # # # # # # # # # # # # # # # # # # # # # # # # #
    cauldron = TypedSegmentClusterCauldron(clusterer, analysisTitle)
    if singularsFromNoise:
        cauldron.extractSingularFromNoise()
    if separateChars:
        # noinspection PyUnboundLocalVariable
        cauldron.appendCharSegments(charSegments)
    uniqueGroups = cauldron.clustersOfUniqueSegments()
    # # # # # # # # # # # # # # # # # # # # # # # # #

    titleFormat = "{} ({}, {}-{})".format(
        cauldron.analysisLabel(),
        distance_method,
        dc.thresholdFunction.__name__,
        "".join([str(k) + str(v) for k, v in dc.thresholdArgs.items()]) if dc.thresholdArgs else '')

    if withplots:
        clusterer.kneelocator.plot_knee()  # plot_knee_normalized()
        plt.text(0.5, 0.2, "S = {:.1f}\neps = {:.3f}\nk = {:.0f}".format(clusterer.S, clusterer.eps, clusterer.k))
        plt.savefig(join(
            reportFolder,
            "knee-{}-S{:.1f}-eps{:.3f}.pdf".format(filechecker.pcapstrippedname, kneedleSensitivity, clusterer.eps)))

        # # # # # # # # # # # # # # # # # # # # # # # # #
        # re-extract cluster labels for segments, templates can only be represented as one label for this distinct position
        print("Plot distances...")
        sdp = DistancesPlotter(specimens, 'distances-' + titleFormat, False)
        labels = numpy.array([cauldron.label4segment(seg) for seg in dc.segments])

        # we need to omit some labels, if the amount amount of unique labels is greater than threshold
        # (to prevent the plot from overflowing)
        uniqueLabelCount = len(set(labels))
        if uniqueLabelCount > 20:
            consolidateLabels(labels)
        sdp.plotManifoldDistances(
            dc.segments, dc.distanceMatrix, labels)
        sdp.writeOrShowFigure()
        del sdp
        # # # # # # # # # # # # # # # # # # # # # # # # #

    print("Prepare output...")
    if withplots:
        analysisLabel = cauldron.analysisLabel()
        paginatedGroups = [
            (analysisLabel + " (regular clusters)", cauldron.regularClusters.clusters),
            (analysisLabel + " (singular clusters)", cauldron.singularClusters.clusters)
        ]
        typeDict = segments2types(dc.rawSegments)
        for pagetitle, segmentClusters in paginatedGroups:
            plotMultiSegmentLines(segmentClusters, specimens, pagetitle,  # titleFormat
                                  True, typeDict, False)

    # unique segments
    ftclusters = {cauldron.unisegClusters.clusterLabel(i): cauldron.unisegClusters.clusterElements(i)
                  for i in range(len(cauldron.unisegClusters))}
    # remove noise if present
    # may not be necessary any more (due to the new SegmentClusterCauldron class),
    #   leave at the moment to be on the safe side
    noisekeys = [ftk for ftk in ftclusters.keys() if ftk.find("Noise") >= 0]
    if len(noisekeys) > 0:
        ftclusters["Noise"] = ftclusters[noisekeys[0]]
        del ftclusters[noisekeys[0]]
    # gt from ft
    groundtruth = {seg: seg.fieldtype
                   for l, segs in ftclusters.items() for seg in segs}
    report = CombinatorialClustersReport(groundtruth, filechecker)
    report.write(ftclusters, titleFormat)

    elementsReport = SegmentClusterGroundtruthReport(comparator, dc.segments, filechecker)
    elementsReport.write(ftclusters)
    # # # # # # # # # # # # # # # # # # # # # # # # #

    filechecker.writeReportMetadata(None)

    # # # # # # # # # # # # # # # # # # # # # # # # #

    for i in range(len(cauldron.regularClusters)): cauldron.regularClusters.plotDistances(i, specimens)
    writeSemanticTypeHypotheses(cauldron, filechecker)

    # # # # # # # # # # # # # # # # # # # # # # # # #
    #
    #
    #
    #
    # # # # # # # # # # # # # # # # # # # # # # # # #

    if args.interactive:
        # noinspection PyUnresolvedReferences
        from tabulate import tabulate

        IPython.embed()




