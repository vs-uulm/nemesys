"""
Use groundtruth about field segmentation by dissectors and apply field type identification to them.
Then evaluate distance matrix using representatives for field-type-hypothesis specific values modifications.

Takes a PCAP trace of a known protocol, dissects each message into their fields, and yields segments from each of them.
These segments get analyzed by the given analysis method which is used as feature to determine their similarity.
The distance matrix is generated using representatives for field-type-hypothesis specific values modifications.
Similar fields are then clustered by DBSCAN and for comparison plotted in groups of their real field types.
In addition, a MDS projection into a 2D plane for visualization of the relative distances of the features is plotted.
"""

import argparse, IPython
from os.path import isfile, basename, splitext
from itertools import chain
from matplotlib import pyplot as plt
import numpy

from utils.baseAlgorithms import tril
from utils.evaluationHelpers import epspertrace, epsdefault, analyses, annotateFieldTypes, labelForSegment, \
    plotMultiSegmentLines
from inference.templates import TemplateGenerator, DistanceCalculator, DelegatingDC, DBSCANsegmentClusterer
from inference.segments import TypedSegment

from inference.segmentHandler import groupByLength, segments2types, segments2clusteredTypes, \
    filterSegments
from validation.dissectorMatcher import MessageComparator
from utils.loader import SpecimenLoader
from visualization.multiPlotter import MultiMessagePlotter
from visualization.distancesPlotter import DistancesPlotter
from visualization.singlePlotter import SingleMessagePlotter

debug = False

# fix the analysis method to VALUE
analysisTitle = 'value'

# fix the distance method to canberra
distance_method = 'canberra'

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
epsfactors = (1, 0.9, 1.1)


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
    dc = DelegatingDC(segments)
    # dc = DistanceCalculator(segments)

    print("Clustering...")
    # # use HDBSCAN
    # segmentGroups = segments2clusteredTypes(tg, analysisTitle, min_cluster_size=15)
    # use DBSCAN
    clusterer = DBSCANsegmentClusterer(dc)

    titleFormat = "{} ({}-{})".format(
        distance_method, dc.thresholdFunction.__name__,
        "".join([str(k) + str(v) for k, v in dc.thresholdArgs.items()]) if dc.thresholdArgs else '')

    # TODO hier gehts weiter
    clusterer.autoconfigureEvaluation("reports/knn_ecdf_{}_{}.pdf".format(titleFormat, pcapbasename))
                                      # besteps[pcapbasename]) # clusterer.eps)

    # Histogram of all the distances between the segments
    hstplt = SingleMessagePlotter(specimens, 'histo-distance-1nn-' + titleFormat, False)
    # hstplt.histogram(tril(dc.distanceMatrix), bins=[x/50 for x in range(50)])
    knn = [dc.neigbors(seg)[0][1] for seg in dc.segments]
    # print(knn)
    hstplt.histogram(knn, bins=[x / 50 for x in range(50)])
    plt.axvline(besteps[pcapbasename], label=besteps[pcapbasename], color="darkmagenta")
    plt.axvline(besteps[pcapbasename]/2, label=besteps[pcapbasename]/2, color="orchid", linestyle="dotted")
    hstplt.writeOrShowFigure()
    plt.clf()

    # TODO hier gehts weiter
    # exit()

    # vary epsilon
    autoeps = clusterer.eps
    for epsfactor in epsfactors if args.epsfactor else (1,):
        clusterer.eps = epsfactor * autoeps
        segmentGroups = segments2clusteredTypes(clusterer, analysisTitle)

        # re-extract cluster labels for segments, templates can only be represented as one label for this distinct position
        labels = numpy.array([
            labelForSegment(segmentGroups, seg) for seg in dc.segments
        ])

        titleFormat = "{} ({}, {}-{})".format(
            segmentGroups[0][0], distance_method, dc.thresholdFunction.__name__,
            "".join([str(k) + str(v) for k, v in dc.thresholdArgs.items()]) if dc.thresholdArgs else '')

        print("Plot distances...")
        sdp = DistancesPlotter(specimens, 'distances-' + titleFormat, False)
        sdp.plotSegmentDistances(dc, labels)
        sdp.writeOrShowFigure()
        del sdp

        print("Prepare output...")
        typeDict = segments2types(segments)
        for pagetitle, segmentClusters in segmentGroups:
            plotMultiSegmentLines(segmentClusters, specimens, titleFormat,
                                  True, typeDict, False)



    if args.interactive:
        IPython.embed()




