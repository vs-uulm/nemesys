"""
Use groundtruth about field segmentation by dissectors and align segments
on the similarity of their feature "personality".

Takes a PCAP trace of a known protocol, dissects each message into their fields, and yields segments from each of them.
These segments get analyzed by the given analysis method which is used as feature to determine their similarity.
Similar fields are then aligned.
"""

import argparse, IPython
from os.path import isfile, splitext, basename, exists
from typing import Sequence
import itertools, pickle

from inference.templates import TemplateGenerator, DistanceCalculator
from alignment.hirschbergAlignSegments import Alignment, HirschbergOnSegmentSimilarity
from inference.analyzers import *
from inference.segmentHandler import annotateFieldTypes, groupByLength, segments2types, segmentsFixed, matrixFromTpairs, \
    segments2clusteredTypes, filterSegments
from validation.dissectorMatcher import MessageComparator
from utils.loader import SpecimenLoader
from characterize_fieldtypes import analyses, labelForSegment

debug = False

analysis_method = 'value'
distance_method = 'canberra'
tokenizer = 'tshark'  # (, '4bytesfixed', 'nemesys')




class SegmentedMessages(object):
    def __init__(self, dc: DistanceCalculator, segmentedMessages: Sequence[Tuple[MessageSegment]]):
        self._dc = dc
        self._segmentedMessages = segmentedMessages  # type: Sequence[Tuple[MessageSegment]]
        self._similarities = self._calcSimilarityMatrix()
        self._distances = self._calcDistanceMatrix(self._similarities)

    @property
    def messages(self):
        return self._segmentedMessages

    @property
    def similarities(self):
        return self._similarities

    @property
    def distances(self):
        return self._distances

    def _nwScores(self):
        """
        Calculate nwscores for each message pair by Hirschberg algorithm.

        :return:
        """
        # convert dc.distanceMatrix from being a distance to a similarity measure
        segmentSimilarities = self._dc.similarityMatrix()

        print("Calculate message alignment scores", end=' ')
        hirsch = HirschbergOnSegmentSimilarity(segmentSimilarities)
        nwscores = list()
        for msg0, msg1 in itertools.combinations(self._segmentedMessages, 2):  # type: Tuple[MessageSegment], Tuple[MessageSegment]
            segseq0 = self._dc.segments2index(msg0)
            segseq1 = self._dc.segments2index(msg1)

            # Needleman-Wunsch alignment score of the two messages:
            nwscores.append((msg0, msg1, hirsch.nwScore(segseq0, segseq1)[-1]))
            print('.', end='', flush=True)
        print()
        return nwscores

    def _calcSimilarityMatrix(self):
        """
        Calculate a similarity matrix of messages from nwscores yielded by HirschbergOnSegmentSimilarity.

        >>> (numpy.diag(sm.distanceMatrix()) > 0).all()

        :return: Similarity matrix for messages
        """
        nwscores = self._nwScores()
        print("Calculate message similarity from alignment scores...")
        messageSimilarityMatrix = matrixFromTpairs(nwscores, self._segmentedMessages)

        # fill diagonal with max similarity per pair
        for ij in range(messageSimilarityMatrix.shape[0]):
            # The max similarity for a pair is len(shorter) * SCORE_MATCH
            messageSimilarityMatrix[ij,ij] = len(self._segmentedMessages[ij]) * Alignment.SCORE_MATCH

        return messageSimilarityMatrix

    def _calcDistanceMatrix(self, similarityMatrix: numpy.ndarray):
        """
        For clustering, convert the nwscores-based similarity matrix to a distance measure.

        >>> (numpy.diag(sm.distanceMatrix()) == 0).all()

        :param similarityMatrix: Similarity matrix for messages
        :return: Distance matrix for messages
        """
        minDim = numpy.empty(similarityMatrix.shape)
        for i in range(similarityMatrix.shape[0]):
            for j in range(similarityMatrix.shape[1]):
                minDim[i, j] = min(  # The max similarity for a pair is len(shorter) * SCORE_MATCH
                    # the diagonals contain the max score match for the pair, calculated in _calcSimilarityMatrix
                    similarityMatrix[i, i], similarityMatrix[j, j]
                    # len(self._segmentedMessages[i]), len(self._segmentedMessages[j])
                ) # * Alignment.SCORE_MATCH

        distanceMatrix = 1 - (similarityMatrix / minDim)
        # TODO
        assert distanceMatrix.min() >= 0, "prevent negative values for highly mismatching messages"
        return distanceMatrix

    def clusterMessageTypes(self) -> Tuple[Dict[int, List[MessageSegment]], numpy.ndarray]:
        from inference.templates import TemplateGenerator
        hdbscan = TemplateGenerator.HDBSCAN(self._distances, None)
        hdbscan.min_cluster_size = 3
        labels = hdbscan.getClusterLabels()
        ulab = set(labels)
        segmentClusters = dict()
        for l in ulab:
            class_member_mask = (labels == l)
            segmentClusters[l] = [seg for seg in itertools.compress(self._segmentedMessages, class_member_mask)]
        return segmentClusters, labels

    def similaritiesSubset(self, As: Sequence[Tuple[MessageSegment]], Bs: Sequence[Tuple[MessageSegment]] = None) \
                -> numpy.ndarray:
        """
        Retrieve a matrix of pairwise distances for two lists of messages (tuples of segments).

        :param As: List of messages
        :param Bs: List of messages
        :return: Matrix of distances: As are rows, Bs are columns.
        """
        clusterI = As
        clusterJ = As if Bs is None else Bs
        simtrx = numpy.ones((len(clusterI), len(clusterJ)))

        transformatorK = dict()  # maps indices i from clusterI to matrix rows k
        for i, seg in enumerate(clusterI):
            transformatorK[i] = self._segmentedMessages.index(seg)
        if Bs is not None:
            transformatorL = dict()  # maps indices j from clusterJ to matrix cols l
            for j, seg in enumerate(clusterJ):
                transformatorL[j] = self._segmentedMessages.index(seg)
        else:
            transformatorL = transformatorK

        for i, k in transformatorK.items():
            for j, l in transformatorL.items():
                simtrx[i, j] = self._distances[k, l]
        return simtrx

















if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Analyze fields as segments of messages and align on the similarity of their feature "personality".')
    parser.add_argument('pcapfilename', help='Filename of the PCAP to load.')
    parser.add_argument('-i', '--interactive', help='Open ipython prompt after finishing the analysis.',
                        action="store_true")
    # TODO select tokenizer by command line parameter
    # TODO toggle filtering on/off
    args = parser.parse_args()

    if not isfile(args.pcapfilename):
        print('File not found: ' + args.pcapfilename)
        exit(1)
    analyzerType = analyses[analysis_method]
    analysisArgs = None
    analysisTitle = analysis_method

    # cache the DistanceCalculator to the filesystem
    pcapName = splitext(basename(args.pcapfilename))[0]
    dccachefn = 'cache-dc-{}-{}-{}.{}'.format(analysisTitle, tokenizer, pcapName, 'dc')
    if not exists(dccachefn):
        # dissect and label messages
        print("Load messages...")
        specimens = SpecimenLoader(args.pcapfilename, 2, True)
        comparator = MessageComparator(specimens, 2, True, debug=debug)

        # TODO select tokenizer by command line parameter
        print("Segmenting messages...", end=' ')
        # 1. segment messages according to true fields from the labels
        segmentedMessages = annotateFieldTypes(analyzerType, analysisArgs, comparator)
        # # 2. segment messages into fixed size chunks for testing
        # segmentedMessages = segmentsFixed(4, comparator, analyzerType, analysisArgs)
        # # 3. segment messages by NEMESYS
        print("done.")

        # TODO filter segments to reduce similarity calculation load - use command line parameter to toggle filtering on/off
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

        chainedSegments = list(itertools.chain.from_iterable(segmentedMessages))

        print("Calculate distance for {} segments...".format(len(chainedSegments)))
        dc = DistanceCalculator(chainedSegments)  # Pairwise similarity of segments: dc.distanceMatrix
        with open(dccachefn, 'wb') as f:
            pickle.dump((segmentedMessages, comparator, dc), f, pickle.HIGHEST_PROTOCOL)
    else:
        print("Load distances from cache file {}".format(dccachefn))
        segmentedMessages, comparator, dc = pickle.load(open(dccachefn, 'rb'))
        if not (isinstance(comparator, MessageComparator)
                and isinstance(dc, DistanceCalculator)):
            print('Loading of cached distances failed.')
            exit(10)
        specimens = comparator.specimens
        chainedSegments = list(itertools.chain.from_iterable(segmentedMessages))

    # IPython.embed()

    sm = SegmentedMessages(dc, segmentedMessages)
    print('Clustering messages...')
    messageClusters, labels = sm.clusterMessageTypes()

    # print('Prepare output...')
    # from visualization.distancesPlotter import DistancesPlotter
    # dp = DistancesPlotter(specimens, 'message-alignment-distances-{}'.format(tokenizer), args.interactive)
    # dp.plotManifoldDistances(segmentedMessages, sm.distances, labels)
    # dp.writeOrShowFigure()

    from tabulate import tabulate
    for clunu, msgcluster in messageClusters.items():  # type: int, List[Tuple[MessageSegment]]
        # distances within this cluster
        distSubMatrix = sm.similaritiesSubset(msgcluster)
        # determine the medoid
        mid = distSubMatrix.sum(axis=1).argmin()
        assert msgcluster[mid] in msgcluster, "Medoid message was not found in cluster"
        commonmsg = dc.segments2index(msgcluster[mid])  # medoid message for this cluster (message type candidate)

        # message tuples from cluster sorted distances to medoid
        distToCommon = sorted([(idx, distSubMatrix[mid, idx])
                               for idx, msg in enumerate(msgcluster) if idx != mid], key=lambda x: x[1])

        # calculate alignments to commonmsg
        subhirsch = HirschbergOnSegmentSimilarity(dc.similarityMatrix())
        clusteralignment = numpy.array([commonmsg], dtype=numpy.int64)
        # TODO Simple progressive alignment
        for idx, dst in distToCommon:
            gappedcommon = clusteralignment[0].tolist()  # make a copy of the line (otherwise we modify the referred line afterwards!)
            # replace all gaps introduced in commonmsg by segment from the next closest aligned at this position
            for alpos, alseg in enumerate(clusteralignment[0]):
                if alseg == -1:
                    for gapaligned in clusteralignment[:,alpos]:
                        if gapaligned > -1:
                            gappedcommon[alpos] = gapaligned
                            break
            # print("gappedcommon", gappedcommon)
            # print("clusteralignment[0]", clusteralignment[0])
            try:
                commonaligned, currentaligned = subhirsch.align(gappedcommon, dc.segments2index(msgcluster[idx]))

                # add gap in already aligned messages
                for gappos, seg in reversed(list(enumerate(commonaligned))):
                    if seg == -1:
                        # print("commonaligned, currentaligned\n", tabulate([commonaligned, currentaligned]))
                        # IPython.embed()
                        clusteralignment = numpy.insert(clusteralignment, gappos, -1, axis=1)

                # add the new message, aligned to all others via the common one, to the end of the matrix
                #
                # print("before")
                # print(tabulate(clusteralignment))
                try:
                    clusteralignment = numpy.append(clusteralignment, [currentaligned], axis=0)
                    print("after")
                    print(tabulate(clusteralignment))
                except ValueError as e:
                    print(e)
                    IPython.embed()
            except AssertionError as e:
                print("Assertion failed")
                IPython.embed()

        # print gaps at the corresponding positions
        print('Cluster', clunu)
        print(tabulate([[dc.segments[s].bytes.hex() if s != -1 else None for s in m]
                        for m in clusteralignment], disable_numparse=True))

        """
        >>> print("aligned")
        >>> print(tabulate([[dc.segments[s].bytes.hex() for s in m] for m in clusteralignment], disable_numparse=True))
        >>> print("raw")
        >>> print(tabulate([[s.bytes.hex() for s in m] for m in msgcluster], disable_numparse=True))
        """

        # IPython.embed()




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




