"""
segment messages by NEMESYS and cluster
"""

import argparse, IPython
from os.path import isfile, basename
from math import log

from inference.templates import DBSCANsegmentClusterer, FieldTypeTemplate, Template
from inference.fieldTypes import BaseTypeMemento, RecognizedField, RecognizedVariableLengthField
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
    # noise: List[MessageSegment]
    # clusters: List[List[MessageSegment]]
    noise, *clusters = clusterer.clusterSimilarSegments(False)
    # extract "large" templates from noise that should rather be its own cluster
    for idx, seg in reversed(list(enumerate(noise.copy()))):  # type: int, MessageSegment
        freqThresh = log(len(dc.rawSegments))
        if isinstance(seg, Template):
            if len(seg.baseSegments) > freqThresh:
                clusters.append(noise.pop(idx).baseSegments)

    print("{} clusters generated from {} distinct segments".format(len(clusters), len(dc.segments)))

    # TODO filter 1-byte segments before clustering see src/fieldtype-aware_distances.py
    filteredClusters = list()
    for segments in clusters:
        # omit one-byte segments. Such clusters are not meaningful.
        if max({seg.length for seg in segments}) < 2:
            print("Skip cluster due to length < 2. It has", len(segments), "Segments")
        else:
            filteredClusters.append(segments)


    fTypeTemplates = list()
    for cLabel, segments in enumerate(filteredClusters):
        ftype = FieldTypeTemplate(segments)
        ftype.fieldtype = "tf{:02d}".format(cLabel)
        fTypeTemplates.append(ftype)
        print("\nCluster", cLabel, "Segments", len(segments))
        print({seg.bytes for seg in segments})
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
    if withPlots:
        print("Plot distances...")
        sdp = DistancesPlotter(specimens,
                               'distances-' + "nemesys-segments_DBSCAN-eps{:0.3f}-ms{}".format(
                                   clusterer.eps, clusterer.min_samples), False)
        clustermask = {segid: cluN for cluN, segL in enumerate(filteredClusters) for segid in dc.segments2index(segL)}
        clustermask.update({segid: "Noise" for segid in dc.segments2index(noise)})
        sdp.plotSegmentDistances(dc, numpy.array([clustermask[segid] for segid in range(len(dc.segments))]))
        sdp.writeOrShowFigure()
        del sdp
    # # # # # # # # # # # # # # # # # # # # # # # #









    if args.interactive:
        from tabulate import tabulate
        # globals().update(locals())
        IPython.embed()










