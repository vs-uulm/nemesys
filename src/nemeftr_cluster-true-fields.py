"""
NEMEFTR: early state and evaluation of epsilon autoconfiguration.
Plot and print dissimilarities between segments. Clusters on dissimilarities and compares the results to ground truth.
Segmentations are obtained by dissectors and apply field type identification to them.
Output for evaluation are a dissimilarity topology plot and histogram, ECDF plot, clustered vector visualization plots,
and segment cluster statistics.

Takes a PCAP trace of a known protocol, dissects each message into their fields, and yields segments from each of them.
These segments get analyzed by the given analysis method which is used as feature to determine their similarity.
The distance matrix is generated using representatives for field-type-hypothesis specific values modifications. TODO check this!
Similar fields are then clustered by DBSCAN and for comparison plotted in groups of their real field types.
In addition, a MDS projection into a 2D plane for visualization of the relative distances of the features is plotted.
"""

import argparse, IPython
from collections import Counter
from math import log
from os.path import isfile, basename, splitext, join
from itertools import chain
from matplotlib import pyplot as plt
import numpy

from utils.evaluationHelpers import analyses, annotateFieldTypes, labelForSegment, \
    plotMultiSegmentLines, writeCollectiveClusteringStaticstics
from inference.templates import DBSCANsegmentClusterer, MemmapDC, DelegatingDC, ClusterAutoconfException

from inference.segmentHandler import segments2types, segments2clusteredTypes, isExtendedCharSeq
from validation.dissectorMatcher import MessageComparator
from utils.loader import SpecimenLoader
from visualization.distancesPlotter import DistancesPlotter
from visualization.singlePlotter import SingleMessagePlotter

debug = False

# fix the analysis method to VALUE
analysisTitle = 'value'

# fix the distance method to canberra
distance_method = 'canberra'

reportFolder = "reports"

# kneedleSensitivity=9.0
# kneedleSensitivity=8.0
# kneedleSensitivity=4.0
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




# epsfactors = (1, 1.4, 1.6, 2)
epsfactors = (0.8, 0.9, 1.1, 1.2)


def lookupTemplates4Segments(dc):
    dc.segments, dc.segments2index()
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Analyze fields as segments of messages and plot field type identification quality.')
    parser.add_argument('pcapfilename', help='Filename of the PCAP to load.')
    parser.add_argument('-i', '--interactive', help='Show interactive plot instead of writing output to file and '
                                                    'open ipython prompt after finishing the analysis.',
                        action="store_true")
    parser.add_argument('-f', '--epsfactor', help='Vary epsilon by factors ' + repr(epsfactors), action="store_true")
    args = parser.parse_args()

    if not isfile(args.pcapfilename):
        print('File not found: ' + args.pcapfilename)
        exit(1)

    analyzerType = analyses[analysisTitle]
    analysisArgs = None

    pcapbasename = basename(args.pcapfilename)
    trace = splitext(pcapbasename)[0]
    # reportFolder = join(reportFolder, trace)
    print("Trace:", pcapbasename)

    # dissect and label messages
    print("Load messages...")
    specimens = SpecimenLoader(args.pcapfilename, 2, True)
    comparator = MessageComparator(specimens, 2, True, debug=debug)

    # segment messages according to true fields from the labels
    print("Segmenting messages...")
    # produce TypedSegments from dissection field information
    segmentedMessages = annotateFieldTypes(analyzerType, analysisArgs, comparator)
    # filter 1-byte segments before clustering
    segments = [seg for seg in chain.from_iterable(segmentedMessages) if seg.length > 1 and set(seg.values) != {0} ]

    print("Calculate distances...")
    if len(segments) ** 2 > MemmapDC.maxMemMatrix:
        dc = MemmapDC(segments)
    else:
        dc = DelegatingDC(segments)
    # dc = DistanceCalculator(segments)

    separateChars = False

    # extract char sequences
    if separateChars:
        charSegments = list()
        nonCharSegs = list()
        for seg in segments:
            if isExtendedCharSeq(seg.bytes):
                charSegments.append(seg)
            else:
                nonCharSegs.append(seg)

    print("Clustering...")
    # # use HDBSCAN
    # segmentGroups = segments2clusteredTypes(tg, analysisTitle, min_cluster_size=15)

    # use DBSCAN
    # kneedleSensitivity += (log(len(dc.segments),10) - 2) * 4  # for larger traces
    # clusterer = DBSCANsegmentClusterer(dc, S=kneedleSensitivity)
    if separateChars:
        clusterer = DBSCANsegmentClusterer(dc, nonCharSegs, S=kneedleSensitivity, k=-1)
    else:
        clusterer = DBSCANsegmentClusterer(dc, dc.rawSegments, S=kneedleSensitivity, k=-1)


    import math  # only unique
    clusterer.min_samples = math.sqrt(len(clusterer.distanceCalculator.segments))

    titleFormat = "{} ({}-{})".format(
        distance_method, dc.thresholdFunction.__name__,
        "".join([str(k) + str(v) for k, v in dc.thresholdArgs.items()]) if dc.thresholdArgs else '')

    # Histogram of all the distances between the segments
    hstplt = SingleMessagePlotter(specimens, 'histo-distance-1nn-' + titleFormat, False)
    # hstplt.histogram(tril(dc.distanceMatrix), bins=[x/50 for x in range(50)])
    knn = [dc.neigbors(seg)[0][1] for seg in dc.segments]
    # print(knn)
    hstplt.histogram(knn, bins=[x / 50 for x in range(50)])
    if pcapbasename in besteps:
        plt.axvline(besteps[pcapbasename], label=besteps[pcapbasename], color="orchid", linestyle="dotted")
    plt.axvline(clusterer.eps, label="{:.3f}".format(clusterer.eps), color="darkmagenta")
    hstplt.writeOrShowFigure()
    plt.clf()

    # adjust epsilon
    epsfraclist = (5,)  # 5
    epspivot = 0.2
    autoeps = clusterer.eps
    for epsfrac in epsfraclist if args.epsfactor else (0,):
        if args.epsfactor:
            adjeps = clusterer.eps + clusterer.eps/epsfrac * (1 if clusterer.eps < epspivot else -1)
            clusterer.eps = adjeps

    # # vary epsilon
    # autoeps = clusterer.eps
    # for epsfactor in epsfactors if args.epsfactor else (1,):
    #     clusterer.eps = epsfactor * autoeps

    # # vary S
    # for kneedS in (1,5,15):
    #     prevEps = round(clusterer.eps, 3)
    #     clusterer.S = kneedS
    #     try:
    #         ms, clusterer.eps = clusterer._autoconfigure()
    #         if prevEps == round(clusterer.eps, 3):
    #             print("  no change to previous S for S =", kneedS)
    #             continue
    #     except ClusterAutoconfException as e:
    #         print(e, "for S =", kneedS)
    #         continue

        segmentGroups = segments2clusteredTypes(clusterer, analysisTitle, False, charSegments if separateChars else None)
        clusterer.kneelocator.plot_knee()  # plot_knee_normalized()
        plt.text(0.5, 0.2, "S = {:.1f}\neps = {:.3f}\nk = {:.0f}".format(clusterer.S, clusterer.eps, clusterer.k))
        plt.savefig(join(reportFolder, "knee-{}-S{:.1f}-eps{:.3f}.pdf".format(trace, kneedleSensitivity, clusterer.eps)))
        titleFormat = "{} ({}, {}-{})".format(
            segmentGroups[0][0], distance_method, dc.thresholdFunction.__name__,
            "".join([str(k) + str(v) for k, v in dc.thresholdArgs.items()]) if dc.thresholdArgs else '')

        uniqueClusters = list()
        for cLabel, elements in segmentGroups[0][1]:
            # same template with different labels
            uniqueSegments = {(sLabel, dc.segments[dc.segments2index([tSegment])[0]]) for sLabel, tSegment in elements}
            uniqueClusters.append((cLabel, sorted(uniqueSegments, key=lambda x: x[1].values)))
        mixedSegments = [seg for seg, cnt in
            Counter(tSegment for cLabel, elements in uniqueClusters for sLabel, tSegment in elements).items() if cnt > 1]
        for tSegment in mixedSegments:
            mixedClusters = [elements for cLabel, elements in uniqueClusters
                             if tSegment in (sElem for sLabel, sElem in elements)]
            assert len(mixedClusters) < 2  # that would be strange and we needed to find some solution then
            toReplace = [sIdx for sIdx,sTuple in enumerate(mixedClusters[0]) if sTuple[1] == tSegment]
            for rIdx in reversed(sorted(toReplace)):
                del mixedClusters[0][rIdx]
            mixedClusters[0].append(("[mixed]", tSegment))
        uniqueGroups = [(segmentGroups[0][0], uniqueClusters)]

        # # # # # # # # # # # # # # # # # # # # # # # # #
        # re-extract cluster labels for segments, templates can only be represented as one label for this distinct position
        print("Plot distances...")
        sdp = DistancesPlotter(specimens, 'distances-' + titleFormat, False)
        # labels = numpy.array([labelForSegment(segmentGroups, seg) for seg in dc.segments])
        labels = numpy.array([labelForSegment(uniqueGroups, seg) for seg in dc.segments])
        sdp.plotManifoldDistances(
            dc.segments, dc.distanceMatrix, labels)
        sdp.writeOrShowFigure()
        del sdp
        # # # # # # # # # # # # # # # # # # # # # # # # #


        print("Prepare output...")
        typeDict = segments2types(segments)
        # for pagetitle, segmentClusters in segmentGroups:
        for pagetitle, segmentClusters in uniqueGroups:
            plotMultiSegmentLines(segmentClusters, specimens, titleFormat,
                                  True, typeDict, False)

        # # total/all segments
        # ftclusters = {label: resolveTemplates2Segments(e for t, e in elements)
        #               for label, elements in segmentGroups[0][1]}
        ftclusters = {label: [e for t, e in elements]
                      for label, elements in uniqueGroups[0][1]}
        noisekeys = [ftk for ftk in ftclusters.keys() if ftk.find("Noise") >= 0]
        if len(noisekeys) > 0:
            ftclusters["Noise"] = ftclusters[noisekeys[0]]
        del ftclusters[noisekeys[0]]
        groundtruth = {seg: seg.fieldtype
                       for l, segs in ftclusters.items() for seg in segs}

        # # unique segments
        # ftclusters = {ftc.fieldtype: ftc.baseSegments for ftc in fTypeContext}
        # ftclusters["Noise"] = resolveTemplates2Segments(noise)
        # groundtruth = {rawSeg: typSeg[1].fieldtype if typSeg[0] > 0.5 else "[unknown]"
        #                for rawSeg, typSeg in typedMatchSegs.items()}
        writeCollectiveClusteringStaticstics(ftclusters, groundtruth, titleFormat, comparator)

    if args.interactive:
        IPython.embed()




