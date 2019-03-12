"""
Use groundtruth about field segmentation by dissectors and align segments
on the similarity of their feature "personality".

Takes a PCAP trace of a known protocol, dissects each message into their fields, and yields segments from each of them.
These segments get analyzed by the given analysis method which is used as feature to determine their similarity.
Similar fields are then aligned.
"""

import argparse, IPython, itertools, pickle, csv, numpy
import time
from os.path import isfile, splitext, basename, exists, join
from typing import Sequence
from hdbscan import HDBSCAN
from sklearn.cluster import DBSCAN
from tabulate import tabulate
from netzob.all import RawMessage
import matplotlib.pyplot as plt

from inference.templates import DistanceCalculator, DelegatingDC
from alignment.hirschbergAlignSegments import Alignment, HirschbergOnSegmentSimilarity
from inference.analyzers import *
from inference.segmentHandler import matrixFromTpairs
from utils.baseAlgorithms import tril
from utils.evaluationHelpers import annotateFieldTypes, writeMessageClusteringStaticstics, writePerformanceStatistics, \
    message_epspertrace
from validation.dissectorMatcher import MessageComparator
from utils.loader import SpecimenLoader
from characterize_fieldtypes import analyses
from visualization.singlePlotter import SingleMessagePlotter
from visualization.multiPlotter import MultiMessagePlotter

debug = False

analysis_method = 'value'
distance_method = 'canberra'
tokenizer = 'tshark'  # (, '4bytesfixed', 'nemesys')




class SegmentedMessages(object):
    def __init__(self, dc, segmentedMessages):
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
        from scipy.special import comb
        combcount = int(comb(len(self._segmentedMessages), 2))
        combcstep = combcount/100

        # convert dc.distanceMatrix from being a distance to a similarity measure
        segmentSimilarities = self._dc.similarityMatrix()

        print("Calculate message alignment scores for {} messages and {} pairs".format(
            len(self._segmentedMessages), combcount), end=' ')
        import time; timeBegin = time.time()
        hirsch = HirschbergOnSegmentSimilarity(segmentSimilarities)
        nwscores = list()
        for c, (msg0, msg1) in enumerate(itertools.combinations(self._segmentedMessages, 2)):  # type: Tuple[MessageSegment], Tuple[MessageSegment]
            segseq0 = self._dc.segments2index(msg0)
            segseq1 = self._dc.segments2index(msg1)

            # Needleman-Wunsch alignment score of the two messages:
            nwscores.append((msg0, msg1, hirsch.nwScore(segseq0, segseq1)[-1]))
            if c % combcstep == 0:
                print(" .", end="", flush=True)
        print()
        print("finished in {:.2f} seconds.".format(time.time() - timeBegin))
        return nwscores

    def _calcSimilarityMatrix(self):
        """
        Calculate a similarity matrix of messages from nwscores yielded by HirschbergOnSegmentSimilarity.

        >>> from utils.baseAlgorithms import generateTestSegments
        >>> segments = generateTestSegments()
        >>> DistanceCalculator.debug = False
        >>> dc = DistanceCalculator(segments)
        Calculated distances for 37 segment pairs.
        >>> m0 = segments[:3]
        >>> m1 = segments[3:5]
        >>> m2 = segments[5:9]
        >>> sm = SegmentedMessages(dc, [m0, m1, m2])
        Calculate message alignment scores ...
        Calculate message similarity from alignment scores...
        >>> (numpy.diag(sm.similarities) > 0).all()
        True

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

        >>> from utils.baseAlgorithms import generateTestSegments
        >>> segments = generateTestSegments()
        >>> DistanceCalculator.debug = False
        >>> dc = DistanceCalculator(segments)
        Calculated distances for 37 segment pairs.
        >>> m0 = segments[:3]
        >>> m1 = segments[3:5]
        >>> m2 = segments[5:9]
        >>> sm = SegmentedMessages(dc, [m0, m1, m2])
        Calculate message alignment scores ...
        Calculate message similarity from alignment scores...
        >>> (numpy.diag(sm.distances) == 0).all()
        True

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

    # 10 2
    def clusterMessageTypesHDBSCAN(self, min_cluster_size = 10, min_samples = 2) \
            -> Tuple[Dict[int, List[MessageSegment]], numpy.ndarray, HDBSCAN]:
        clusterer = HDBSCAN(metric='precomputed', allow_single_cluster=True, cluster_selection_method='leaf',
                         min_cluster_size=min_cluster_size,
                         min_samples=min_samples)

        print("Messages: HDBSCAN min cluster size:", min_cluster_size, "min samples:", min_samples)
        return self._postprocessClustering(clusterer)


    def clusterMessageTypesDBSCAN(self, eps = 1.5, min_samples = 3) \
            -> Tuple[Dict[int, List[MessageSegment]], numpy.ndarray, DBSCAN]:
        clusterer = DBSCAN(metric='precomputed', eps=eps,
                         min_samples=min_samples)

        print("Messages: DBSCAN epsilon:", eps, "min samples:", min_samples)
        return self._postprocessClustering(clusterer)


    def _postprocessClustering(self, clusterer):
        clusterer.fit(self._distances)

        labels = clusterer.labels_
        ulab = set(labels)
        segmentClusters = dict()
        for l in ulab:
            class_member_mask = (labels == l)
            segmentClusters[l] = [seg for seg in itertools.compress(self._segmentedMessages, class_member_mask)]

        print(len([ul for ul in ulab if ul >= 0]), "Clusters found",
              "(with noise {})".format(len(segmentClusters[-1]) if -1 in segmentClusters else 0))

        return segmentClusters, labels, clusterer


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

    def alignMessageType(self, msgcluster: List[Tuple[MessageSegment]]):
        """
        Messages segments of one cluster aligned to the medoid ("segments that is most similar too all segments")
        of the cluster.

        >>> indicesalignment, alignedsegments = sm.alignMessageType(msgcluster)
        >>> hexclualn = [[dc.segments[s].bytes.hex() if s != -1 else None for s in m] for m in indicesalignment]
        >>> hexalnseg = [[s.bytes.hex() if s is not None else None for s in m] for m in alignedsegments]
        >>> hexalnseg == hexclualn
        True

        :param msgcluster: List of messages in the form of one tuple of segments per message.
        :return: A numpy array containing the indices of all segments or their representatives.
        """
        # distances within this cluster
        distSubMatrix = self.similaritiesSubset(msgcluster)
        # determine the medoid
        mid = distSubMatrix.sum(axis=1).argmin()
        assert msgcluster[mid] in msgcluster, "Medoid message was not found in cluster"
        commonmsg = self._dc.segments2index(msgcluster[mid])  # medoid message for this cluster (message type candidate)

        # message tuples from cluster sorted distances to medoid
        distToCommon = sorted([(idx, distSubMatrix[mid, idx])
                               for idx, msg in enumerate(msgcluster) if idx != mid], key=lambda x: x[1])

        # calculate alignments to commonmsg
        subhirsch = HirschbergOnSegmentSimilarity(self._dc.similarityMatrix())
        clusteralignment = numpy.array([commonmsg], dtype=numpy.int64)
        # simple progressive alignment
        for idx, dst in distToCommon:
            gappedcommon = clusteralignment[0].tolist()  # make a copy of the line (otherwise we modify the referred line afterwards!)
            # replace all gaps introduced in commonmsg by segment from the next closest aligned at this position
            for alpos, alseg in enumerate(clusteralignment[0]):
                if alseg == -1:
                    for gapaligned in clusteralignment[:,alpos]:
                        if gapaligned > -1:
                            gappedcommon[alpos] = gapaligned
                            break
            commonaligned, currentaligned = subhirsch.align(gappedcommon, self._dc.segments2index(msgcluster[idx]))

            # add gap in already aligned messages
            for gappos, seg in enumerate(commonaligned):
                if seg == -1:
                    clusteralignment = numpy.insert(clusteralignment, gappos, -1, axis=1)

            # add the new message, aligned to all others via the common one, to the end of the matrix
            clusteralignment = numpy.append(clusteralignment, [currentaligned], axis=0)

        alignedsegments = list()
        for (msgidx, dst), aln in zip([(mid, 0)] + distToCommon, clusteralignment):
            segiter = iter(msgcluster[msgidx])
            segalnd = list()
            for segidx in aln:
                if segidx > -1:
                    segalnd.append(next(segiter))
                else:
                    segalnd.append(None)
            alignedsegments.append(tuple(segalnd))

        return clusteralignment, alignedsegments

    def neighbors(self):
        neighbors = list()
        for idx, dists in enumerate(self._distances):  # type: int, numpy.ndarray
            neighbors.append(sorted([(i, d) for i, d in enumerate(dists) if i != idx], key=lambda x: x[1]))
        return neighbors

    def autoconfigureDBSCAN(self):
        """
        Auto configure the clustering parameters epsilon and minPts regarding the input data

        :return: minpts, epsilon
        """
        from scipy.ndimage.filters import gaussian_filter1d
        from math import log

        neighbors = self.neighbors()
        sigma = log(len(neighbors))
        knearest = dict()
        smoothknearest = dict()
        seconddiff = dict()
        seconddiffMax = (0, 0, 0)
        for k in range(0, len(neighbors) // 10):  # first 10% of k-neigbors
            knearest[k] = sorted([nfori[k][1] for nfori in neighbors])
            smoothknearest[k] = gaussian_filter1d(knearest[k], sigma)
            # max of second difference (maximum upwards curvature) as knee
            seconddiff[k] = numpy.diff(smoothknearest[k], 2)
            seconddiffargmax = seconddiff[k].argmax()
            diffrelmax = seconddiff[k].max() / smoothknearest[k][seconddiffargmax]
            if diffrelmax > seconddiffMax[2]:
                seconddiffMax = (k, seconddiffargmax, diffrelmax)

        k = seconddiffMax[0]
        x = seconddiffMax[1] + 1

        epsilon = smoothknearest[k][x]
        min_samples = round(sigma)
        return epsilon, min_samples









# # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # Evaluation helpers  # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


def delign(segseqA):
    return [s for s in segseqA if s >= -1]

# realign
def relign(segseqA, segseqB):
    hirsch = HirschbergOnSegmentSimilarity(dc.similarityMatrix())
    return hirsch.align(dc.segments2index([s for s in segseqA if isinstance(s, MessageSegment)]),
                        dc.segments2index([s for s in segseqB if isinstance(s, MessageSegment)]))

def resolveIdx2Seg(segseq):
    """

    :param segseq: list of segment indices (from raw segment list) per message
    :return:
    """
    print(tabulate([[dc.segments[s].bytes.hex() if s != -1 else None for s in m]
                        for m in segseq], disable_numparse=True, headers=range(len(segseq[0]))))

def columnOfAlignment(alignedSegments: List[List[MessageSegment]], colnum: int):
    return [msg[colnum] for msg in alignedSegments]

def column2first(dc: DistanceCalculator, alignedSegments: List[List[MessageSegment]], colnum: int):
    """
    Similarities of entries 1 to n of one column to its first (not None) entry.

    :param dc:
    :param alignedSegments:
    :param colnum:
    :return:
    """
    column = [msg[colnum] for msg in alignedSegments] #columnOfAlignment(alignedSegments, colnum)

    # strip Nones
    nonepos = [idx for idx, seg in enumerate(column) if seg is None]
    stripedcol = [seg for seg in column if seg is not None]

    dists2first = ["- (reference)"] + dc.distancesSubset(stripedcol[0:1], stripedcol[1:]).tolist()[0]

    # re-insert Nones
    for idx in nonepos:
        dists2first.insert(idx, None)

    # transpose
    d2ft = list(map(list, zip(column, dists2first)))
    return d2ft

def printSegDist(d2ft: List[Tuple[MessageSegment, float]]):
    print(tabulate([(s.bytes.hex() if isinstance(s, MessageSegment) else "-", d) for s, d in d2ft],
                   headers=['Seg (hex)', 'Distance'], floatfmt=".4f"))

def seg2seg(dc: DistanceCalculator, alignedSegments: List[List[MessageSegment]],
            coordA: Tuple[int, int], coordB: Tuple[int, int]):
    """
    Distance between segments that are selected by coordinates in an alignment

    :param dc: DistanceCalculator holding the segment distances.
    :param alignedSegments: 2-dimensional list holding the alignment
    :param coordA: Coordinates of segment A within the alignment
    :param coordB: Coordinates of segment B within the alignment
    :return:
    """
    segA = alignedSegments[coordA[0]][coordA[1]]
    print(segA)
    segB = alignedSegments[coordB[0]][coordB[1]]
    print(segB)
    return dc.pairDistance(segA, segB)

def quicksegmentTuple(dc: DistanceCalculator, segment: MessageSegment):
    return dc.segments2index([segment])[0], segment.length, tuple(segment.values)


def epsautoconfeval(epsilon):
    """
    # investigate distance properties for clustering autoconfiguration
    # plots of k-nearest-neighbor distance histogram and "knee"

    :param epsilon The manually determined "best" epsilon for comparison
    :return:
    """

    # # distribution of all distances in matrix
    # hstplt = SingleMessagePlotter(specimens, tokenizer+'-distance-distribution-histo', args.interactive)
    # hstplt.histogram(tril(sm.distances), bins=[x / 50 for x in range(50)])
    # plt.axvline(epsilon, label="manually determined eps={:0.2f}".format(epsilon), c="red")
    # hstplt.text('max {:.3f}, mean {:.3f}'.format(sm.distances.max(), sm.distances.mean()))
    # hstplt.writeOrShowFigure()
    # del hstplt

    neighbors = sm.neighbors()  # list of tuples: (index from sm.distances, distance) sorted by distance

    mmp = MultiMessagePlotter(specimens, tokenizer + "-k-nearest-neighbor-distance-distribution", 1, 2,
                              isInteractive=args.interactive)
    mmp.axes[0].axhline(epsilon, label="manually determined eps={:0.2f}".format(epsilon), c="red")
    mmp.axes[1].axhline(epsilon, label="manually determined eps={:0.2f}".format(epsilon), c="red")

    krange = (0, 16, 1)

    for k in range(*krange):
        knearest = sorted([nfori[k][1] for nfori in neighbors])
        mmp.plotToSubfig(1, knearest, alpha=.4, label="k={}".format(k))

    # # kneedle approach
    # # unusable results. does not find a knee!
    # from kneed import KneeLocator
    # kneeK = dict()
    # for k in range(*krange):
    #     knearest = sorted([nfori[k][1] for nfori in neighbors])
    #     mmp.plotToSubfig(0, knearest, alpha=.1)  # , label="k={}".format(k)
    #
    #     kneel = KneeLocator(range(len(knearest)), knearest, curve='convex', direction='increasing')
    #     kneeX = kneel.knee
    #     kneeK[knearest[kneeX]] = k
    #
    # kneeDists = list(kneeK.keys())
    # bestEpsMatchIdx = numpy.argmin([abs(kneeDist-epsilon) for kneeDist in kneeDists])
    # bestEpsMatchDist = kneeDists[bestEpsMatchIdx]

    # # range of distances for a k-neighborhood, determines the maximum range of distances
    # # results do not correlate with a suitable choice of k compared to manual introspection.
    # distrange4k = list()
    # for k in range(*krange):
    #     knearest = sorted([nfori[k][1] for nfori in neighbors])
    #     distrange4k.append((k, max(knearest) - min(knearest)))
    # maxdran = max([dran[1] for dran in distrange4k])
    # maxkran = [dran[0] for dran in distrange4k if dran[1] == maxdran][0]
    # mmp.textInEachAx(["max range {:.2f} at k={}".format(maxdran, maxkran)])
    # mmp.axes[0].axhline(bestEpsMatchDist, linestyle='dashed', color='blue', alpha=.4,
    #                     label="kneedle {:.2f} of k={}".format(bestEpsMatchDist, kneeK[bestEpsMatchDist]))

    # smoothing approach
    from scipy.ndimage.filters import gaussian_filter1d
    from math import log

    sigma = log(len(neighbors))
    knearest = dict()
    smoothknearest = dict()
    seconddiff = dict()
    seconddiffMax = (0, 0, 0)
    for k in range(0, len(neighbors) // 10):  # round(2*log(len(neighbors)))
        knearest[k] = sorted([nfori[k][1] for nfori in neighbors])
        smoothknearest[k] = gaussian_filter1d(knearest[k], sigma)
        # max of second difference (maximum upwards curvature) as knee
        seconddiff[k] = numpy.diff(smoothknearest[k], 2)
        seconddiffargmax = seconddiff[k].argmax()
        diffrelmax = seconddiff[k].max() / smoothknearest[k][seconddiffargmax]
        if diffrelmax > seconddiffMax[2]:
            seconddiffMax = (k, seconddiffargmax, diffrelmax)

    # prepare to plot the smoothed nearest neigbor distribution and its second derivative
    k = seconddiffMax[0]
    x = seconddiffMax[1] + 1

    mmp.plotToSubfig(0, smoothknearest[k], label="smooth k={}, sigma={:.2f}".format(k, sigma), alpha=.4)
    mmp.plotToSubfig(1, smoothknearest[k], label="smooth k={}, sigma={:.2f}".format(k, sigma), alpha=1, color='blue')
    mmp.plotToSubfig(0, knearest[k], alpha=.4)

    ax0twin = mmp.axes[0].twinx()
    # mmp.plotToSubfig(ax0twin, seconddiff[k], linestyle='dotted', color='cyan', alpha=.4)
    mmp.plotToSubfig(ax0twin, [None] + seconddiff[k].tolist(), linestyle='dotted',
                     color='magenta', alpha=.4)

    # epsilon = knearest[k][x]
    epsilon = smoothknearest[k][x]

    mmp.axes[0].axhline(epsilon, linestyle='dashed', color='blue', alpha=.4,
                        label="curvature max {:.2f} of k={}".format(
                            epsilon, k))
    mmp.axes[0].axvline(x, linestyle='dashed', color='blue', alpha=.4)


    mmp.writeOrShowFigure()
    del mmp

    # if args.interactive:
    #     from tabulate import tabulate
    #     IPython.embed()
    # exit(0)

    return epsilon



# # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # END : Evaluation helpers  # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # #














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
    dccachefn = 'cache-dc-{}-{}-{}.{}'.format(analysisTitle, tokenizer, pcapName, 'ddc')
    # dccachefn = 'cache-dc-{}-{}-{}.{}'.format(analysisTitle, tokenizer, pcapName, 'dc')
    if not exists(dccachefn):
        # dissect and label messages
        print("Load messages...")
        specimens = SpecimenLoader(args.pcapfilename, 2, True)
        comparator = MessageComparator(specimens, 2, True, debug=debug)

        # TODO select tokenizer by command line parameter
        print("Segmenting messages...", end=' ')
        segmentationTime = time.time()
        # 1. segment messages according to true fields from the labels
        segmentedMessages = annotateFieldTypes(analyzerType, analysisArgs, comparator)
        # # 2. segment messages into fixed size chunks for testing
        # segmentedMessages = segmentsFixed(4, comparator, analyzerType, analysisArgs)
        # # 3. segment messages by NEMESYS
        # segmentedMessages = ...
        segmentationTime = time.time() - segmentationTime
        print("done.")

        chainedSegments = list(itertools.chain.from_iterable(segmentedMessages))

        # TODO filter segments to reduce similarity calculation load - use command line parameter to toggle filtering on/off
        # filteredSegments = filterSegments(chainedSegments)
        # if length < 3:
        #     pass  # Handle short segments

        print("Calculate distance for {} segments...".format(len(chainedSegments)))
        # dc = DistanceCalculator(chainedSegments, reliefFactor=0.33)  # Pairwise similarity of segments: dc.distanceMatrix
        dist_calc_segmentsTime = time.time()
        dc = DelegatingDC(chainedSegments)
        dist_calc_segmentsTime = time.time() - dist_calc_segmentsTime
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
        segmentationTime, dist_calc_segmentsTime = None, None

    print("Calculate distance for {} messages...".format(len(segmentedMessages)))
    dist_calc_messagesTime = time.time()
    sm = SegmentedMessages(dc, segmentedMessages)
    dist_calc_messagesTime = time.time() - dist_calc_messagesTime

    cluster_params_autoconfTime = time.time()
    eps, min_samples = sm.autoconfigureDBSCAN()
    cluster_params_autoconfTime = time.time() - cluster_params_autoconfTime


    # # DEBUG and TESTING
    # #
    # # retrieve manually determined epsilon value
    # pcapbasename = basename(args.pcapfilename)
    # epsilon = message_epspertrace[pcapbasename] if pcapbasename in message_epspertrace else 0.15
    # eps = epsautoconfeval(epsilon)
    # #
    # # DEBUG and TESTING


    # cluster and align messages and calculate statistics of it
    print('Clustering messages...')
    cluster_messagesTime = time.time()
    messageClusters, labels, clusterer = sm.clusterMessageTypesDBSCAN(eps=eps, min_samples=3)
    cluster_messagesTime = time.time() - cluster_messagesTime
    plotTitle = "{}-{} eps {:.3f} ms {}".format(
        tokenizer, type(clusterer).__name__, clusterer.eps, clusterer.min_samples)
    # messageClusters, labels, clusterer = sm.clusterMessageTypesHDBSCAN()
    # plotTitle = "{}-{} mcs {} ms {}".format(
    #     tokenizer, type(clusterer).__name__, clusterer.min_cluster_size, clusterer.min_samples)


    # write message clustering statistics to csv
    groundtruth = {msg: pm.messagetype for msg, pm in comparator.parsedMessages.items()}
    writeMessageClusteringStaticstics(messageClusters, groundtruth, "{}-{}-eps={:.2f}-min_samples={}".format(
        tokenizer, type(clusterer).__name__, clusterer.eps, clusterer.min_samples), comparator)
    for msg, mtype in groundtruth.items():
        msg.messageType = mtype

    # plot distances and message clusters
    print("Plot distances...")
    from visualization.distancesPlotter import DistancesPlotter
    dp = DistancesPlotter(specimens, 'message-distances-' + plotTitle, False)
    dp.plotManifoldDistances(
        [specimens.messagePool[seglist[0].message] for seglist in segmentedMessages],
        sm.distances, labels)  # segmentedMessages
    dp.writeOrShowFigure()

    # align cluster members
    align_messagesTime = time.time()
    alignedClusters = dict()
    for clunu, msgcluster in messageClusters.items():  # type: int, List[Tuple[MessageSegment]]
        clusteralignment, alignedsegments = sm.alignMessageType(msgcluster)
        alignedClusters[clunu] = alignedsegments

        # get gaps at the corresponding positions
        print('Cluster', clunu)
        hexalnseg = [[s.bytes.hex() if s is not None else None for s in m] for m in alignedsegments]
    align_messagesTime = time.time() - align_messagesTime



    alignedFields = {clunu: [field for field in zip(*cluelms)] for clunu, cluelms in alignedClusters.items()}
    statDynFields = dict()
    for clunu, alfi in alignedFields.items():
        statDynFields[clunu] = list()
        for fvals in alfi:
            if fvals[0] is not None:
                if all([val is not None for val in fvals]):
                    if all([val.bytes == fvals[0].bytes for val in fvals]):
                        statDynFields[clunu].append(fvals[0].bytes)
                    else:
                        statDynFields[clunu].append("DYNAMIC")
                else:
                    statDynFields[clunu].append("DYNAMIC")  # /GAP
            else:
                statDynFields[clunu].append("DYNAMIC")  # /GAP
    # statDynFields = {clunu: [fvals[0].bytes if all([val.bytes == fvals[0].bytes for val in fvals]) else "DYNAMIC" for fvals in alfi]
    #          for clunu, alfi in alignedFields.items()}
    clusterpairs = list(itertools.combinations(statDynFields.keys(), 2))
    matchingClusters = [(clunuA, clunuB) for clunuA, clunuB in clusterpairs if (statDynFields[clunuA] == statDynFields[clunuB])]
    # TODO for non-tshark tokenizers: determine as mergeable by allowing an arbitrary number of DYNAMIC/GAP fields
    #  between identical STATIC fields in the different clusters
    if len(matchingClusters) > 0:
        print("Clusters could be merged:")
        for clunuAB in matchingClusters:
            print(tabulate([(clunu, *[fv.hex() if isinstance(fv, bytes) else fv for fv in fvals])
                            for clunu, fvals in statDynFields.items() if clunu in clunuAB]))

    # write alignments to csv
    reportFolder = "reports"
    fileNameS = "NEMETYL-symbols-" + plotTitle + "-" + pcapName
    csvpath = join(reportFolder, fileNameS + '.csv')
    if not exists(csvpath):
        print('Write alignments to {}...'.format(csvpath))
        with open(csvpath, 'w') as csvfile:
            symbolcsv = csv.writer(csvfile)
            for clunu, clusg in alignedClusters.items():
                symbolcsv.writerow(["# Cluster", clunu, "- Fields -", "- Alignment -"])
                symbolcsv.writerows([sg.bytes.hex() if sg is not None else '' for sg in msg] for msg in clusg)
                symbolcsv.writerow(["---"] * 5)
    else:
        print("Symbols not saved. File {} already exists.".format(csvpath))
        if not args.interactive:
            IPython.embed()

    writePerformanceStatistics(
        specimens, clusterer,
        "{} {} {}".format(tokenizer, analysis_method, distance_method),
        segmentationTime, dist_calc_segmentsTime, dist_calc_messagesTime,
        cluster_params_autoconfTime, cluster_messagesTime, align_messagesTime
    )


    if args.interactive:
        from tabulate import tabulate
        IPython.embed()




