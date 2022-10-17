"""
NEMEFTR: Optimal-segmentation baseline. Determine the best epsilon value by iterating all epsilons between 0.05 and 0.5.

Plot and print dissimilarities between segments. Clusters on dissimilarities and compares the results to ground truth.
Segmentations are obtained by dissectors and apply field type identification to them.
Output for evaluation are a dissimilarity topology plot, histogram and segment cluster statistics.

Takes a PCAP trace of a known protocol, dissects each message into their fields, and yields segments from each of them.
These segments get analyzed by the given analysis method which is used as feature to determine their similarity.
The distance matrix is generated using representatives for field-type-hypothesis specific values modifications (chars).
Similar fields are then clustered by DBSCAN iterating all epsilons between 0.05 and 0.5.
In addition, a MDS projection into a 2D plane for visualization of the relative distances of the features is plotted.
"""

import argparse, IPython
from collections import Counter
from itertools import chain
from matplotlib import pyplot as plt
import math, numpy

from nemere.utils.evaluationHelpers import analyses, annotateFieldTypes, labelForSegment, StartupFilecheck, consolidateLabels
from nemere.utils.reportWriter import CombinatorialClustersReport, reportFolder, \
    SegmentClusterGroundtruthReport
from nemere.inference.templates import DBSCANsegmentClusterer, MemmapDC, DelegatingDC
from nemere.inference.segmentHandler import segments2clusteredTypes, isExtendedCharSeq
from nemere.validation.dissectorMatcher import MessageComparator
from nemere.utils.loader import SpecimenLoader
from nemere.visualization.distancesPlotter import DistancesPlotter
from nemere.visualization.singlePlotter import SingleMessagePlotter

debug = False

# fix the analysis method to VALUE
analysisTitle = 'value'
# fix the distance method to canberra
distance_method = 'canberra'


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Analyze fields as segments of messages and plot field type identification quality.')
    parser.add_argument('pcapfilename', help='Filename of the PCAP to load.')
    parser.add_argument('-i', '--interactive', help='Show interactive plot instead of writing output to file and '
                                                    'open ipython prompt after finishing the analysis.',
                        action="store_true")
    parser.add_argument('-l', '--layer', type=int, default=2,
                        help='Protocol layer relative to IP to consider. Default is 2 layers above IP '
                             '(typically the payload of a transport protocol).')
    parser.add_argument('-r', '--relativeToIP', default=False, action='store_true')
    args = parser.parse_args()

    filechecker = StartupFilecheck(args.pcapfilename)
    analyzerType = analyses[analysisTitle]
    analysisArgs = None

    # dissect and label messages
    print("Load messages...")
    specimens = SpecimenLoader(args.pcapfilename, layer=args.layer, relativeToIP = args.relativeToIP)
    comparator = MessageComparator(specimens, layer=args.layer, relativeToIP=args.relativeToIP, debug=debug)

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
    for epsint in range(5,50):
        eps = epsint*0.01
        min_samples = round(math.log(len(dc.segments)))
        if separateChars:
            # noinspection PyUnboundLocalVariable
            clusterer = DBSCANsegmentClusterer(dc, nonCharSegs, min_samples=min_samples, eps=eps)
        else:
            clusterer = DBSCANsegmentClusterer(dc, dc.rawSegments, min_samples=min_samples, eps=eps)

        # only unique and sqrt instead of ln
        clusterer.min_samples = round(math.sqrt(len(clusterer.distanceCalculator.segments)))

        titleFormat = "{} (eps {:.3f}, ms {})".format(distance_method, eps, clusterer.min_samples)

        # Histogram of all the distances between the segments
        hstplt = SingleMessagePlotter(specimens, 'histo-distance-1nn-' + titleFormat, False)
        knn = [dc.neighbors(seg)[0][1] for seg in dc.segments]
        hstplt.histogram(knn, bins=[x / 50 for x in range(50)])
        plt.axvline(clusterer.eps, label="{:.3f}".format(clusterer.eps), color="darkmagenta")
        hstplt.writeOrShowFigure()
        plt.clf()

        # # # # # # # # # # # # # # # # # # # # # # # # #
        # noinspection PyUnboundLocalVariable
        segmentGroups = segments2clusteredTypes(clusterer, analysisTitle, False, charSegments if separateChars else None)
        titleFormat = "{} ({}, eps {:.3f}, ms {})".format(
            segmentGroups[0][0], distance_method, eps, clusterer.min_samples)

        # The same value in different segments only represented by once
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
        labels = numpy.array([labelForSegment(uniqueGroups, seg) for seg in dc.segments])

        # we need to omit some labels, if the amount amount of unique labels is greater than threshold
        # (to prevent the plot from overflowing)
        uniqueLabelCount = len(set(labels))
        if uniqueLabelCount > 20:
            consolidateLabels(labels)
            uniqueLabelCount = len(set(labels))
            if uniqueLabelCount > 20:
                print("Still too many cluster labels!")

        sdp.plotManifoldDistances(dc.segments, dc.distanceMatrix, labels)
        sdp.writeOrShowFigure()
        del sdp
        # # # # # # # # # # # # # # # # # # # # # # # # #


        print("Prepare output...")
        # unique segments
        ftclusters = {label: [e for t, e in elements]
                      for label, elements in uniqueGroups[0][1]}
        noisekeys = [ftk for ftk in ftclusters.keys() if ftk.find("Noise") >= 0]
        if len(noisekeys) > 0:
            ftclusters["Noise"] = ftclusters[noisekeys[0]]
        del ftclusters[noisekeys[0]]
        groundtruth = {seg: seg.fieldtype
                       for l, segs in ftclusters.items() for seg in segs}

        report = CombinatorialClustersReport(groundtruth, filechecker)
        report.write(ftclusters, titleFormat)

        elementsReport = SegmentClusterGroundtruthReport(comparator, dc.segments, filechecker, reportFolder)
        elementsReport.write(ftclusters)
        # # # # # # # # # # # # # # # # # # # # # # # # #

    filechecker.writeReportMetadata(None)

    if args.interactive:
        IPython.embed()




