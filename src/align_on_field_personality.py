"""
Use groundtruth about field segmentation by dissectors and align segments
on the similarity of their feature "personality".

Takes a PCAP trace of a known protocol, dissects each message into their fields, and yields segments from each of them.
These segments get analyzed by the given analysis method which is used as feature to determine their similarity.
Similar fields are then aligned.
"""

import argparse, IPython
from os.path import isfile
import itertools

from inference.templates import TemplateGenerator, DistanceCalculator, InterSegment
from alignment.hirschbergAlignSegments import Alignment, HirschbergOnSegmentSimilarity
from inference.analyzers import *
from inference.segmentHandler import annotateFieldTypes, groupByLength, segments2types, segmentsFixed, matrixFromTpairs
from validation.dissectorMatcher import MessageComparator
from utils.loader import SpecimenLoader
from characterize_fieldtypes import analyses, segments2clusteredTypes, filterSegments, labelForSegment

debug = False





analysis_method = 'progcumudelta'
distance_method = 'canberra'





if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Analyze fields as segments of messages and align on the similarity of their feature "personality".')
    parser.add_argument('pcapfilename', help='Filename of the PCAP to load.')
    parser.add_argument('-i', '--interactive', help='Open ipython prompt after finishing the analysis.',
                        action="store_true")
    # parser.add_argument('analysis', type=str,
    #                     help='The kind of analysis to apply on the messages. Available methods are: '
    #                     + ', '.join(analyses.keys()) + '.')
    # parser.add_argument('--parameters', '-p', help='Parameters for the analysis.')
    args = parser.parse_args()

    if not isfile(args.pcapfilename):
        print('File not found: ' + args.pcapfilename)
        exit(1)
    # if args.analysis not in analyses:
    #     print('Analysis {} unknown. Available methods are:\n' + ', '.join(analyses.keys()) + '.')
    #     exit(2)
    # analyzerType = analyses[args.analysis]
    # analysisArgs = args.parameters
    # analysisTitle = "{}{}".format(args.analysis, "" if not analysisArgs else " ({})".format(analysisArgs))
    analyzerType = analyses[analysis_method]
    analysisArgs = None
    analysisTitle = analysis_method


    # dissect and label messages
    print("Load messages...")
    specimens = SpecimenLoader(args.pcapfilename, 2, True)
    comparator = MessageComparator(specimens, 2, True, debug=debug)

    print("Segmenting messages...", end=' ')
    # # segment messages according to true fields from the labels
    segmentedMessages = annotateFieldTypes(analyzerType, analysisArgs, comparator)
    # segsByLen = groupByLength(segmentedMessages)

    # segment messages into fixed size chunks for testing
    # segmentedMessages = segmentsFixed(analyzerType,analysisArgs,comparator,4)
    print("done.")


    # for length, segments in segsByLen.items():  # type: int, List[MessageSegment]
    #     filteredSegments = filterSegments(segments)
    #
    #     # if length < 3:
    #     #     continue
    #     # if len(filteredSegments) < 16:
    #     #     print("Too few relevant segments for length {} after Filtering. {} segments remaining:".format(
    #     #         length, len(filteredSegments)
    #     #     ))
    #     #     for each in filteredSegments:
    #     #         print("   ", each)
    #     #     print()
    #     #     continue
    #
    #     typeDict = segments2types(filteredSegments)
    #
    #     print("Calculate distances...")
    #     tg = TemplateGenerator(filteredSegments, distance_method)
    #
    #     segmentGroups = segments2clusteredTypes(tg, analysisTitle, min_cluster_size=5)
    #     # re-extract cluster labels for segments
    #     labels = numpy.array([
    #         labelForSegment(segmentGroups, seg) for seg in tg.segments
    #     ])
    #
    #     # print("Prepare output...")
    #
    #
    #
    chainedSegments = list(itertools.chain.from_iterable(segmentedMessages))
    print("Calculate distance for {} segments...".format(len(chainedSegments)))
    dc = DistanceCalculator(chainedSegments)  # Pairwise similarity of segments: dc.distanceMatrix
    IPython.embed()
    # convert dc.distanceMatrix from being a distance to a similarity measure
    print("Convert to segment similarity...")
    similarityMatrix = dc.similarityMatrix()

    print("Aligning", end='')
    hirsch = HirschbergOnSegmentSimilarity(similarityMatrix)
    nwscores = list()
    seg2idx = {seg: idx for idx, seg in enumerate(dc.segments)}  # prepare lookup for matrix indices
    for msg0, msg1 in itertools.combinations(segmentedMessages, 2):  # type: Tuple[MessageSegment], Tuple[MessageSegment]
        segseq0 = [seg2idx[seg] for seg in msg0]
        segseq1 = [seg2idx[seg] for seg in msg1]

        # Needleman-Wunsch alignment score of the two messages:
        nwscores.append((msg0, msg1, hirsch.nwScore(segseq0, segseq1)[-1]))
        print('.', end='', flush=True)

    print("\nCalculate message similarity from alignment scores...")
    # convert nwscores from being a similarity to a distance measure
    messageSimilarityMatrix = matrixFromTpairs(nwscores, segmentedMessages)
    minDim = numpy.empty(messageSimilarityMatrix.shape)
    for i in range(messageSimilarityMatrix.shape[0]):
        for j in range(messageSimilarityMatrix.shape[1]):
            minDim[i, j] = min(len(segmentedMessages[i]), len(segmentedMessages[j])) * Alignment.SCORE_MATCH
    messageDistanceMatrix = 1 - (messageSimilarityMatrix / minDim)

    print('Clustering...')
    from inference.templates import TemplateGenerator
    hdbscan = TemplateGenerator.HDBSCAN(messageDistanceMatrix, None)
    hdbscan.min_cluster_size = 3
    labels = hdbscan.getClusterLabels()
    ulab = set(labels)
    segmentClusters = dict()
    for l in ulab:
        class_member_mask = (labels == l)
        segmentClusters[l] = [seg for seg in itertools.compress(segmentedMessages, class_member_mask)]

    print('Prepare output...')
    from visualization.distancesPlotter import DistancesPlotter
    dp = DistancesPlotter(specimens, 'message-alignment-distances-4bytesfixed', not args.interactive)
    dp._plotManifoldDistances(segmentedMessages, messageDistanceMatrix, labels)
    dp.writeOrShowFigure()

    from tabulate import tabulate
    for clunu, seclu in segmentClusters.items():
        print('Cluster', clunu)
        print(tabulate([[s.bytes.hex() for s in m] for m in seclu], disable_numparse=True))




    # # TODO: these are test calls for validating embedSegment -> doctest there?!
    # m, s, inters = DistanceCalculator.embedSegment(segsByLen[4][50], segsByLen[8][50])
    #
    # overlay = ([None] * s + inters.segA.values, inters.segB.values)
    # from visualization.singlePlotter import SingleMessagePlotter
    # smp = SingleMessagePlotter(specimens, "test feature embedding", True)
    # smp.plotAnalysis(overlay)
    # smp.writeOrShowFigure()

    if args.interactive:
        from tabulate import tabulate
        IPython.embed()




