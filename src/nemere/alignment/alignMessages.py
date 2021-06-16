import itertools
from typing import Tuple, Dict, List, Sequence, Union, OrderedDict
import time

import numpy
from hdbscan import HDBSCAN
from sklearn.cluster import DBSCAN
from scipy.special import comb

from netzob.Model.Vocabulary.Messages.AbstractMessage import AbstractMessage
from netzob.Model.Vocabulary.Messages.RawMessage import RawMessage

from nemere.alignment.hirschbergAlignSegments import HirschbergOnSegmentSimilarity
from nemere.inference.segmentHandler import matrixFromTpairs
from nemere.inference.segments import MessageSegment
from nemere.inference.templates import DistanceCalculator, MemmapDC



class SegmentedMessages(object):
    def __init__(self, dc: DistanceCalculator, segmentedMessages: Sequence[Tuple[MessageSegment]]):
        self._score_match = None
        self._score_mismatch = None
        self._score_gap = None

        self._dc = dc
        # For performance reasons, adjust the value domain of the segment similarity matrix to the score domain here
        # for use in the methods (i. e., _nwScores)
        self._segmentSimilaritiesScoreDomain = HirschbergOnSegmentSimilarity.scoreDomainSimilarityMatrix(
            # convert dc.distanceMatrix from being a distance to a similarity measure
            self._dc.similarityMatrix())
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

        print("Calculate message alignment scores for {} messages and {} pairs".format(
            len(self._segmentedMessages), combcount), end=' ')
        import time; timeBegin = time.time()
        # If gaps should be adjusted from the default in the class, this needs to be done in __init__,
        # when calling HirschbergOnSegmentSimilarity.scoreDomainSimilarityMatrix, as long as the matrix is precomputed
        # there
        hirsch = HirschbergOnSegmentSimilarity(self._segmentSimilaritiesScoreDomain, similaritiesScoreDomain=True)
        self._score_gap = hirsch.score_gap
        self._score_match = hirsch.score_match
        self._score_mismatch = hirsch.score_mismatch
        nwscores = list()
        for c, (msg0, msg1) in enumerate(itertools.combinations(self._segmentedMessages, 2)):  # type: Tuple[MessageSegment], Tuple[MessageSegment]
            segseq0 = self._dc.segments2index(msg0)
            segseq1 = self._dc.segments2index(msg1)

            # Needleman-Wunsch alignment score of the two messages: based on the last entry.
            # TODO this could be parallelized to improve performance, but the contained read from the similarity matrix
            #   must not lead to copying of the matrix for each processs! see https://research.wmz.ninja/articles/2018/03/on-sharing-large-arrays-when-using-pythons-multiprocessing.html
            nwscores.append((msg0, msg1, hirsch.nwScore(segseq0, segseq1)[-1]))
            if c % combcstep == 0:
                print(" .", end="", flush=True)
        print()
        print("finished in {:.2f} seconds.".format(time.time() - timeBegin))
        return nwscores

    def _calcSimilarityMatrix(self):
        """
        Calculate a similarity matrix of messages from nwscores yielded by HirschbergOnSegmentSimilarity.

        >>> from nemere.utils.baseAlgorithms import generateTestSegments
        >>> from nemere.inference.templates import DistanceCalculator
        >>> segments = generateTestSegments()
        >>> DistanceCalculator.debug = False
        >>> dc = DistanceCalculator(segments)
        Calculated distances for 37 segment pairs in ... seconds.
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

        >>> from nemere.utils.baseAlgorithms import generateTestSegments
        >>> from nemere.inference.templates import DistanceCalculator
        >>> segments = generateTestSegments()
        >>> DistanceCalculator.debug = False
        >>> dc = DistanceCalculator(segments)
        Calculated distances for 37 segment pairs in ... seconds.
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
        # simtrx = numpy.ones((len(clusterI), len(clusterJ)))
        simtrx = MemmapDC.largeFilled((len(clusterI), len(clusterJ)))

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

        >>> from nemere.utils.baseAlgorithms import generateTestSegments
        >>> from nemere.inference.templates import DistanceCalculator
        >>> segments = generateTestSegments()
        >>> DistanceCalculator.debug = False
        >>> dc = DistanceCalculator(segments)
        Calculated distances for 37 segment pairs in ... seconds.
        >>> msgcluster = [segments[:3], segments[3:5], segments[5:9]]
        >>> sm = SegmentedMessages(dc, msgcluster)
        Calculate message alignment scores for 3 messages and 3 pairs  .
        finished in 0.00 seconds.
        Calculate message similarity from alignment scores...
        >>> indicesalignment, alignedsegments = sm.alignMessageType(msgcluster)
        >>> # noinspection PyUnresolvedReferences
        >>> hexclualn = [[dc.segments[s].bytes.hex() if s != -1 else None for s in m] for m in indicesalignment]
        >>> # noinspection PyUnresolvedReferences
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
        from nemere.utils.baseAlgorithms import autoconfigureDBSCAN
        # can we omit k = 0 ?
        # No - recall and even more so precision deteriorates for dns and dhcp (1000s)
        epsilon, min_samples, k = autoconfigureDBSCAN(self.neighbors())
        print("eps {:0.3f} autoconfigured from k {}".format(epsilon, k))
        return epsilon, min_samples




class TypeIdentificationByAlignment(object):
    """
    Message Type Idenfication as described in NEMETYL, the INFOCOM 2020 paper
      NEMETYL: NEtwork MEssage TYpe identification by aLignment.

    Similar fields are then aligned to determine a score that is used as affinity value (dissimilarities) of messages
    for clustering. The clusters are refined by splitting and merging on heuristics.
    """

    def __init__(self, dc: DistanceCalculator, segmentedMessages: Sequence[Tuple[MessageSegment]],
                 tokenizer: str, messagePool: OrderedDict[AbstractMessage, RawMessage]):
        """
        Initialize the TYL instance to hold all results in its attributes.
        clusterAlignSplitMerge() must be called to perform the inference.

        :param dc:
        :param segmentedMessages:
        :param tokenizer:
        :param messagePool:
        """
        self._dc, self._segmentedMessages = dc, segmentedMessages
        self._tokenizer, self._messagePool = tokenizer, messagePool
        self.sm = None  # type: Union[None, SegmentedMessages]
        self.eps = None  # type: Union[None, float]
        self.messageTupClusters = None  # type: Union[None, Dict[int, List[Tuple[MessageSegment]]]]
        self.messageObjClusters = None  # type: Union[None, Dict[int, List[RawMessage]]]
        self.labels = None  # type: Union[None, numpy.ndarray]
        self.clusterer = None  # type: Union[None, DBSCAN]
        self.alignedClusters = None  # type: Union[None, Dict[int, List[Tuple[MessageSegment]]]]


        self.dist_calc_messagesTime = None  # type: Union[None, float]
        self.cluster_params_autoconfTime = None  # type: Union[None, float]
        self.cluster_messagesTime = None  # type: Union[None, float]
        self.align_messagesTime = None  # type: Union[None, float]

    @property
    def _isNemesys(self):
        return self._tokenizer[:7] == "nemesys"

    @property
    def _isTshark(self):
        return self._tokenizer == "tshark"

    def clusterMessages(self):
        """
        Calculate Alignment-Score and CLUSTER messages
        """
        print("Calculate distance for {} messages...".format(len(self._segmentedMessages)))

        self.dist_calc_messagesTime = time.time()
        self.sm = SegmentedMessages(self._dc, self._segmentedMessages)
        self.dist_calc_messagesTime = time.time() - self.dist_calc_messagesTime

        print('Clustering messages...')
        self.cluster_params_autoconfTime = time.time()
        self.eps, min_samples = self.sm.autoconfigureDBSCAN()
        self.cluster_params_autoconfTime = time.time() - self.cluster_params_autoconfTime
        if self._isNemesys:
            self.eps *= .8
        self.cluster_messagesTime = time.time()
        self.messageTupClusters, self.labels, self.clusterer = \
            self.sm.clusterMessageTypesDBSCAN(eps=self.eps, min_samples=3)
        # messageClusters, labels, clusterer = sm.clusterMessageTypesHDBSCAN()
        self.cluster_messagesTime = time.time() - self.cluster_messagesTime

        # clusters as label to message object mapping
        self.messageObjClusters = {lab : [self._messagePool[element[0].message] for element in segseq]
                            for lab, segseq in self.messageTupClusters.items()}
        # # # # # # # # # # # # # # # # # # # # # # # #

    def alignClusterMembers(self):
        """
        ALIGN cluster members
        """
        assert isinstance(self.sm, SegmentedMessages) and isinstance(self.messageTupClusters, dict), \
            "clusterMessages() must have run before alignClusterMembers()"

        self.align_messagesTime = time.time()
        self.alignedClusters = dict()
        # alignedClustersHex = dict()
        print("Align each cluster...")
        for clunu, msgcluster in self.messageTupClusters.items():  # type: int, List[Tuple[MessageSegment]]
            # TODO perform this in parallel (per future)
            clusteralignment, alignedsegments = self.sm.alignMessageType(msgcluster)
            self.alignedClusters[clunu] = alignedsegments
            # alignedClustersHex[clunu] = [[s.bytes.hex() if s is not None else None for s in m] for m in alignedsegments]
        print()
        self.align_messagesTime = time.time() - self.align_messagesTime
        # # # # # # # # # # # # # # # # # # # # # # # #

    def splitClusters(self, **kwargs):
        """
        SPLIT clusters based on fields without rare values

        :param kwargs: if kwargs are set, they are used to activate CVS output at RelaxedExoticClusterSplitter.
            see nemere.alignment.clusterSplitting.ClusterSplitter.activateCVSout()
        """
        assert isinstance(self.alignedClusters, dict) and isinstance(self.messageTupClusters, dict) \
               and isinstance(self.sm, SegmentedMessages), "alignClusterMembers() must have run before splitClusters()"

        from nemere.alignment.clusterSplitting import RelaxedExoticClusterSplitter
        cSplitter = RelaxedExoticClusterSplitter(6 if not self._isTshark else 3,
                                    self.alignedClusters, self.messageTupClusters, self.sm)
        if kwargs:
            cSplitter.activateCVSout(**kwargs)
        # IN-PLACE split of clusters in alignedClusters and messageClusters
        cSplitter.split()
        # update dependent vars
        self.labels = cSplitter.labels
        self.messageObjClusters = {lab: [self._messagePool[element[0].message] for element in segseq]
                              for lab, segseq in self.messageTupClusters.items()}
        # # # # # # # # # # # # # # # # # # # # # # # #
        return "split"

    def mergeClusters(self):
        """
        Check for cluster MERGE candidates
        """
        assert isinstance(self.alignedClusters, dict) and isinstance(self.messageTupClusters, dict) \
               and isinstance(self._dc, DistanceCalculator), "splitClusters() must have run before mergeClusters()"

        from nemere.alignment.clusterMerging import ClusterMerger
        print("Check for cluster merge candidates...")
        # ClusterMerger
        clustermerger = ClusterMerger(self.alignedClusters, self._dc, self.messageTupClusters)
        self.messageTupClusters = clustermerger.merge(self._isNemesys)
        self.messageObjClusters = {lab: [self._messagePool[element[0].message] for element in segseq]
                              for lab, segseq in self.messageTupClusters.items()}
        map2label = {msg: lab for lab, msglist in self.messageObjClusters.items() for msg in msglist}
        self.labels = numpy.array([map2label[self._messagePool[msg[0].message]] for msg in self._segmentedMessages])
        # # # # # # # # # # # # # # # # # # # # # # # #
        return "split+merged"

    def clusterAlignSplitMerge(self):
        """
        This is the main function that implements all steps of the NEMETYL method.
        Find all results in this obejct instances attributes.
        If intermediate results are required, run clusterMessages(), alignClusterMembers(), splitClusters()
        and mergeClusters() yourself and capture self.messageTupClusters, self.messageObjClusters, and self.labels that
        are updated after each step..
        """
        # Calculate Alignment-Score and CLUSTER messages
        self.clusterMessages()
        # ALIGN cluster members
        self.alignClusterMembers()
        # SPLIT clusters based on fields without rare values
        self.splitClusters()
        # Check for cluster MERGE candidates
        self.mergeClusters()
        # TODO split clusters are internally re-aligned, but NOT merged clusters. Can this lead to an inconsistency?

















