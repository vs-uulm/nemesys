import itertools
from typing import Tuple, Dict, List, Sequence, Union

import numpy
from hdbscan import HDBSCAN
from sklearn.cluster import DBSCAN
from scipy.special import comb

from alignment.hirschbergAlignSegments import HirschbergOnSegmentSimilarity, Alignment
from inference.segmentHandler import matrixFromTpairs
from inference.segments import MessageSegment
from inference.templates import DistanceCalculator


class SegmentedMessages(object):
    def __init__(self, dc: DistanceCalculator, segmentedMessages: Sequence[Tuple[MessageSegment]]):
        self._score_match = None
        self._score_mismatch = None
        self._score_gap = None

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
        combcount = int(comb(len(self._segmentedMessages), 2))
        combcstep = combcount/100

        # convert dc.distanceMatrix from being a distance to a similarity measure
        segmentSimilarities = self._dc.similarityMatrix()

        print("Calculate message alignment scores for {} messages and {} pairs".format(
            len(self._segmentedMessages), combcount), end=' ')
        import time; timeBegin = time.time()
        hirsch = HirschbergOnSegmentSimilarity(segmentSimilarities)
        self._score_gap = hirsch.score_gap
        self._score_match = hirsch.score_match
        self._score_mismatch = hirsch.score_mismatch
        nwscores = list()
        for c, (msg0, msg1) in enumerate(itertools.combinations(self._segmentedMessages, 2)):  # type: Tuple[MessageSegment], Tuple[MessageSegment]
            segseq0 = self._dc.segments2index(msg0)
            segseq1 = self._dc.segments2index(msg1)

            # Needleman-Wunsch alignment score of the two messages: based on the last entry.
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
        >>> from inference.templates import DistanceCalculator
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
            # The max similarity for a pair is len(shorter) * self._score_match
            # see Netzob for reference
            messageSimilarityMatrix[ij,ij] = len(self._segmentedMessages[ij]) * self._score_match

        return messageSimilarityMatrix

    def _calcDistanceMatrix(self, similarityMatrix: numpy.ndarray):
        """
        For clustering, convert the nwscores-based similarity matrix to a distance measure.

        >>> from utils.baseAlgorithms import generateTestSegments
        >>> from inference.templates import DistanceCalculator
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
        minScore = min(self._score_gap, self._score_match, self._score_mismatch)
        base = numpy.empty(similarityMatrix.shape)
        maxScore = numpy.empty(similarityMatrix.shape)
        for i in range(similarityMatrix.shape[0]):
            for j in range(similarityMatrix.shape[1]):
                maxScore[i, j] = min(  # The max similarity for a pair is len(shorter) * self._score_match
                    # the diagonals contain the max score match for the pair, calculated in _calcSimilarityMatrix
                    similarityMatrix[i, i], similarityMatrix[j, j]
                )  # == nu in paper
                minDim = min(len(self._segmentedMessages[i]), len(self._segmentedMessages[j]))
                base[i, j] = minScore * minDim  # == mu in paper

        distanceMatrix = 100 - 100*((similarityMatrix-base) / (maxScore-base))
        # distanceMatrix = 100 - 100 * ((similarityMatrix + base) / (maxScore - base))
        assert distanceMatrix.min() >= 0, "prevent negative values for highly mismatching messages"
        return distanceMatrix


    def clusterMessageTypesHDBSCAN(self, min_cluster_size = 10, min_samples = 2) \
            -> Tuple[Dict[int, List[Tuple[MessageSegment]]], numpy.ndarray, HDBSCAN]:
        clusterer = HDBSCAN(metric='precomputed', allow_single_cluster=True, cluster_selection_method='leaf',
                         min_cluster_size=min_cluster_size,
                         min_samples=min_samples)

        print("Messages: HDBSCAN min cluster size:", min_cluster_size, "min samples:", min_samples)
        segmentClusters, labels = self._postprocessClustering(clusterer)
        return segmentClusters, labels, clusterer


    def clusterMessageTypesDBSCAN(self, eps = 1.5, min_samples = 3) \
            -> Tuple[Dict[int, List[Tuple[MessageSegment]]], numpy.ndarray, DBSCAN]:
        clusterer = DBSCAN(metric='precomputed', eps=eps,
                         min_samples=min_samples)

        print("Messages: DBSCAN epsilon:", eps, "min samples:", min_samples)
        segmentClusters, labels = self._postprocessClustering(clusterer)
        return segmentClusters, labels, clusterer


    def _postprocessClustering(self, clusterer: Union[HDBSCAN, DBSCAN]) -> Tuple[Dict[int, List[Tuple[MessageSegment]]],
                                                         numpy.ndarray]:
        clusterer.fit(self._distances)

        labels = clusterer.labels_  # type: numpy.ndarray
        assert isinstance(labels, numpy.ndarray)
        ulab = set(labels)
        segmentClusters = dict()
        for l in ulab:  # type: int
            class_member_mask = (labels == l)
            segmentClusters[l] = [seg for seg in itertools.compress(self._segmentedMessages, class_member_mask)]

        print(len([ul for ul in ulab if ul >= 0]), "Clusters found",
              "(with noise {})".format(len(segmentClusters[-1]) if -1 in segmentClusters else 0))

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

    def alignMessageType(self, msgcluster: List[Tuple[MessageSegment]]):
        """
        Messages segments of one cluster aligned to the medoid ("segments that is most similar too all segments")
        of the cluster.

        >>> from utils.baseAlgorithms import generateTestSegments
        >>> from inference.templates import DistanceCalculator
        >>> segments = generateTestSegments()
        >>> DistanceCalculator.debug = False
        >>> dc = DistanceCalculator(segments)
        Calculated distances for 37 segment pairs.
        >>> sm = SegmentedMessages(dc, segments)
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
        from utils.baseAlgorithms import autoconfigureDBSCAN
        # can we omit k = 0 ?
        # No - recall and even more so precision deteriorates for dns and dhcp (1000s)
        epsilon, min_samples, k = autoconfigureDBSCAN(self.neighbors())
        print("eps {:0.3f} autoconfigured from k {}".format(epsilon, k))
        return epsilon, min_samples


















