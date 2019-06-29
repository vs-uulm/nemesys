"""
segment messages by NEMESYS and cluster
"""

import argparse, IPython
from os.path import isfile, basename

from inference.templates import DBSCANsegmentClusterer
from inference.fieldTypes import BaseTypeMemento, RecognizedField
from inference.segments import TypedSegment, HelperSegment
from inference.analyzers import *
from visualization.distancesPlotter import DistancesPlotter
from visualization.simplePrint import printFieldContext, tabuSeqOfSeg
from visualization.multiPlotter import MultiMessagePlotter, PlotGroups
from utils.evaluationHelpers import *


debug = False

# fix the analysis method to VALUE
analysisTitle = 'value'
# fix the distance method to canberra
distance_method = 'canberra'
# use NEMESYS segments
tokenizer = 'nemesys'




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
    args = parser.parse_args()

    if not isfile(args.pcapfilename):
        print('File not found: ' + args.pcapfilename)
        exit(1)
    pcapbasename = basename(args.pcapfilename)
    withPlots = args.with_plots

    analyzerType = analyses[analysisTitle]
    analysisArgs = None

    # # # # # # # # # # # # # # # # # # # # # # # #
    # cache/load the DistanceCalculator to the filesystem
    #
    # noinspection PyUnboundLocalVariable
    specimens, comparator, inferredSegmentedMessages, dc, segmentationTime, dist_calc_segmentsTime = cacheAndLoadDC(
        args.pcapfilename, analysisTitle, tokenizer, debug, analyzerType, analysisArgs, args.sigma #, True
    )  # Note!  When manipulating distances, deactivate caching by adding "True".
    # chainedSegments = dc.rawSegments
    # # # # # # # # # # # # # # # # # # # # # # # #
    trueSegmentedMessages = {msgseg[0].message: msgseg
                         for msgseg in annotateFieldTypes(analyzerType, analysisArgs, comparator)
                         }
    # tabuSeqOfSeg(trueSegmentedMessages)
    # print(trueSegmentedMessages.values())

    # # # # # # # # # # # # # # # # # # # # # # # #
    clusterer = DBSCANsegmentClusterer(dc)
    noise, *clusters = clusterer.clusterSimilarSegments(False)  # type: Tuple[List[MessageSegment], List[List[MessageSegment]]]
    print("{} clusters generated from {} distinct segments".format(len(clusters), len(dc.segments)))

    for cLabel, segments in enumerate(clusters):
    #     # print({seg.length for seg in segments})  # TODO yea, they have different lengths
        ftype = BaseTypeMemento("tf{:02d}".format(cLabel))  # TODO we need a length lateron!
        for seg in segments:

            # recog = RecognizedField(seg.message, ftype, seg.offset, 0.0)
            # printFieldContext(trueSegmentedMessages, recog)
    # # # # # # # # # # # # # # # # # # # # # # # #





    # # # # # # # # # # # # # # # # # # # # # # # #
    if withPlots:
        print("Plot distances...")
        sdp = DistancesPlotter(specimens,
                               'distances-' + "nemesys-segments_DBSCAN-eps{:0.3f}-ms{}".format(
                                   clusterer.eps, clusterer.min_samples), False)
        clustermask = {segid: cluN for cluN, segL in enumerate(clusters) for segid in dc.segments2index(segL)}
        clustermask.update({segid: "Noise" for segid in dc.segments2index(noise)})
        sdp.plotSegmentDistances(dc, numpy.array([clustermask[segid] for segid in range(len(dc.segments))]))
        sdp.writeOrShowFigure()
        del sdp
    # # # # # # # # # # # # # # # # # # # # # # # #









    if args.interactive:
        from tabulate import tabulate
        # globals().update(locals())
        IPython.embed()










