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
from os.path import isfile, basename
from itertools import chain
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




if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Analyze fields as segments of messages and plot field type identification quality.')
    parser.add_argument('pcapfilename', help='Filename of the PCAP to load.')
    parser.add_argument('-i', '--interactive', help='Show interactive plot instead of writing output to file and '
                                                    'open ipython prompt after finishing the analysis.',
                        action="store_true")

    parser.add_argument('--epsilon', '-e', help='Parameter epsilon for the DBSCAN clusterer.', type=float, default=epsdefault)
    args = parser.parse_args()

    if not isfile(args.pcapfilename):
        print('File not found: ' + args.pcapfilename)
        exit(1)
    # if args.isolengths and args.iterate:
    #     print('Iterating clustering parameters over isolated-lengths fields is not implemented.')
    #     exit(2)
    if args.epsilon != parser.get_default('epsilon') and (args.isolengths or args.iterate):
        print('Setting epsilon is not supported for clustering over isolated-lengths fields and parameter iteration.')
        exit(2)

    analyzerType = analyses[analysisTitle]
    analysisArgs = None

    pcapbasename = basename(args.pcapfilename)
    print("Trace:", pcapbasename)
    epsilon = args.epsilon  # TODO make "not set" epsilon and "default" distinguishable
    if args.epsilon == epsdefault and pcapbasename in epspertrace:
        epsilon = epspertrace[pcapbasename]

    # dissect and label messages
    print("Load messages...")
    specimens = SpecimenLoader(args.pcapfilename, 2, True)
    comparator = MessageComparator(specimens, 2, True, debug=debug)

    # segment messages according to true fields from the labels
    print("Segmenting messages...")
    # produce TypedSegments from dissection field information
    segmentedMessages = annotateFieldTypes(analyzerType, analysisArgs, comparator)
    segments = list(chain.from_iterable(segmentedMessages))

    print("Calculate distances...")
    # dc = DelegatingDC(segments)
    dc = DistanceCalculator(segments)

    print("Clustering...")
    # # use HDBSCAN
    # segmentGroups = segments2clusteredTypes(tg, analysisTitle, min_cluster_size=15)
    # use DBSCAN
    clusterer = DBSCANsegmentClusterer(dc, eps=epsilon, min_samples=20)
    segmentGroups = segments2clusteredTypes(clusterer, analysisTitle)
    # re-extract cluster labels for segments
    clusterer.getClusterLabels()
    labels = numpy.array([
        labelForSegment(segmentGroups, seg) for seg in dc.segments
    ])

    titleFormat = "{} ({}, {}-{})".format(
        segmentGroups[0][0], distance_method, dc.thresholdFunction.__name__,
        "".join([str(k) + str(v) for k, v in dc.thresholdArgs.items()]) if dc.thresholdArgs else '')

    print("Plot distances...")
    sdp = DistancesPlotter(specimens, 'distances-' + titleFormat,
                           args.interactive)
    # old: 'mixedlength', analysisTitle, distance_method, tg.clusterer if tg.clusterer else 'n/a'
    # sdp.plotSegmentDistances(tg, numpy.array([seg.fieldtype for seg in tg.segments]))
    sdp.plotSegmentDistances(dc, labels)
    sdp.writeOrShowFigure()
    del sdp

    hstplt = SingleMessagePlotter(specimens, 'histo-distance-' + titleFormat, args.interactive)
    hstplt.histogram(tril(dc.distanceMatrix), bins=[x/50 for x in range(50)])
    hstplt.writeOrShowFigure()
    del hstplt

    print("Prepare output...")
    typeDict = segments2types(segments)
    for pagetitle, segmentClusters in segmentGroups:
        plotMultiSegmentLines(segmentClusters, specimens, titleFormat,
                              True, typeDict, args.interactive)






    if args.interactive:
        IPython.embed()




