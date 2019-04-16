import itertools
from typing import Tuple, Dict, List, Sequence, Union

import numpy
from hdbscan import HDBSCAN
from sklearn.cluster import DBSCAN
from scipy.special import comb

from alignment.hirschbergAlignSegments import HirschbergOnSegmentSimilarity, Alignment
from inference.segmentHandler import matrixFromTpairs
from inference.segments import MessageSegment


class SegmentedMessages(object):
    def __init__(self, dc, segmentedMessages):
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

            # Needleman-Wunsch alignment score of the two messages:
            # TODO validate using the max, not the last entry. Any difference?
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
                )
                minDim = min(len(self._segmentedMessages[i]), len(self._segmentedMessages[j]))
                base[i, j] = minScore * minDim

        distanceMatrix = 100 - 100*((similarityMatrix-base) / (maxScore-base))
        # TODO prevent negative values for highly mismatching messages
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
        from scipy.ndimage.filters import gaussian_filter1d
        from math import log

        neighbors = self.neighbors()
        sigma = log(len(neighbors))
        knearest = dict()
        smoothknearest = dict()
        seconddiff = dict()
        seconddiffMax = (0, 0, 0)
        # can we omit k = 0 ?
        # No - recall and even more so precision deteriorates for dns and dhcp (1000s)
        for k in range(0, len(neighbors) // 10):  # first 10% of k-neigbors
            knearest[k] = sorted([nfori[k][1] for nfori in neighbors])
            smoothknearest[k] = gaussian_filter1d(knearest[k], sigma)
            # max of second difference (maximum upwards curvature) as knee (this not actually the knee!)
            seconddiff[k] = numpy.diff(smoothknearest[k], 2)
            seconddiffargmax = seconddiff[k].argmax()
            diffrelmax = seconddiff[k].max() / smoothknearest[k][seconddiffargmax]
            if sigma < seconddiffargmax < len(neighbors) - sigma and diffrelmax > seconddiffMax[2]:
                seconddiffMax = (k, seconddiffargmax, diffrelmax)

        k = seconddiffMax[0]
        x = seconddiffMax[1] + 1

        epsilon = smoothknearest[k][x]
        min_samples = round(sigma)
        return epsilon, min_samples






FC_DYN = b"DYNAMIC"
FC_GAP = "GAP"

def alignFieldClasses(alignedClusters: Dict[int, List], dc, mmg = (0,-1,5)):
    from inference.templates import Template, DistanceCalculator
    from alignment.hirschbergAlignSegments import NWonSegmentSimilarity

    alignedFields = {clunu: [field for field in zip(*cluelms)] for clunu, cluelms in alignedClusters.items() if
                     clunu > -1}
    statDynFields = dict()
    for clunu, alfi in alignedFields.items():
        statDynFields[clunu] = list()
        for fvals in alfi:
            fvalsGapless = [val for val in fvals if val is not None]
            # if len(fvalsGapless) < len(fvals): there are GAPs in there
            if all([val.bytes == fvalsGapless[0].bytes for val in fvalsGapless]):
                # This leaves only an example in the field list:
                # The association to the original message is lost for the other segments in this field!
                statDynFields[clunu].append(fvalsGapless[0])
            else:
                dynTemp = Template(list(FC_DYN), fvalsGapless)
                dynTemp.medoid = dc.findMedoid(dynTemp.baseSegments)
                statDynFields[clunu].append(dynTemp)
    # generate a similarity matrix for field-classes (static-dynamic)
    statDynValues = list(set(itertools.chain.from_iterable(statDynFields.values())))
    statDynValuesMap = {sdv: idx for idx, sdv in enumerate(statDynValues)}
    statDynIndices = {clunu: [statDynValuesMap[fc] for fc in sdf] for clunu, sdf in statDynFields.items()}

    # fcSimMatrix = numpy.array([[0 if fcL.bytes != fcK.bytes else 0.5 if isinstance(fcL, Template) else 1
    #               for fcL in statDynValues] for fcK in statDynValues])

    # use medoid distance in fcSimMatrix instead of fixed value (0.5)
    fcSimMatrix = numpy.array([[
        # 1.0 if fcL.bytes == fcK.bytes else
        max(0.85, 1.0 - dc.pairDistance(fcL.medoid, fcK.medoid))  # DYN-DYN similarity
            if isinstance(fcL, Template) and isinstance(fcK, Template)
        else 1.0 - dc.pairDistance(fcL.medoid, fcK)  # DYN-STA similarity, modified 0 field
                 * (1 if {fcK.bytes} != {0} else 0.05)
            if isinstance(fcL, Template) and isinstance(fcK, MessageSegment)
        else 1.0 - dc.pairDistance(fcK.medoid, fcL)  # STA-DYN similarity, modified 0 field  min(0.2,
                 * (1 if {fcK.bytes} != {0} else 0.05)
            if isinstance(fcK, Template) and isinstance(fcL, MessageSegment)
        else max(0.66, 1.0 - dc.pairDistance(fcK, fcL)  # STA-STA similarity, modified 0 field
                 * 1 if {fcK.bytes} != {0} or {fcL.bytes} != {0} else 0.1)
            if isinstance(fcK, MessageSegment) and isinstance(fcL,MessageSegment)
        else 0.0
        for fcL in statDynValues] for fcK in statDynValues])

    # fcdc = DistanceCalculator(statDynValues)
    # fcSimMatrix = fcdc.similarityMatrix()


    clusterpairs = list(itertools.combinations(statDynFields.keys(), 2))
    fclassHirsch = HirschbergOnSegmentSimilarity(fcSimMatrix, *mmg)
    # fclassHirsch = NWonSegmentSimilarity(fcSimMatrix, *mmg)
    alignedFCIndices = {(clunuA, clunuB): fclassHirsch.align(statDynIndices[clunuA], statDynIndices[clunuB])
                        for clunuA, clunuB in clusterpairs}
    alignedFieldClasses = {clupa: ([statDynValues[a] if a > -1 else FC_GAP for a in afciA],
                                   [statDynValues[b] if b > -1 else FC_GAP for b in afciB])
                           for clupa, (afciA, afciB) in alignedFCIndices.items()}
    # from alignment.hirschbergAlignSegments import NWonSegmentSimilarity
    # IPython.embed()
    return clusterpairs, alignedFieldClasses





def mergeClusters(messageClusters, clusterStats, alignedClusters, alignedFieldClasses,
                  clusterpairs, matchingClusters, matchingConditions, dc):
    import IPython
    from utils.evaluationHelpers import printClusterMergeConditions

    # search in filteredMatches for STATIC - DYNAMIC - STATIC with different static values and remove from matchingClusters
    # : the matches on grounds of the STA value in DYN condition, with the DYN role(s) in a set in the first element of each tuple
    dynStaPairs = list()
    for clunuPair in matchingClusters:
        dynRole = [ clunuPair[0] if fieldCond[5] else clunuPair[1] for fieldCond in matchingConditions[clunuPair][1:]
            if not any(fieldCond[:5]) and (fieldCond[5] or fieldCond[6]) ]
        if dynRole:
            dynStaPairs.append((set(dynRole), clunuPair))
    dynRoles = set(itertools.chain.from_iterable([dynRole for dynRole, clunuPair in dynStaPairs]))
    # List of STA roles for each DYN role
    staRoles = {dynRole: [clunuPair[0] if clunuPair[1] in dr else clunuPair[1] for dr, clunuPair in dynStaPairs
                 if dynRole in dr] for dynRole in dynRoles}
    removeFromMatchingClusters = list()
    for dynRole, staRoleList in staRoles.items():
        try:
            staMismatch = False
            staValues = None
            clunuPairs = list()
            for staRole in staRoleList:
                clunuPair = (dynRole, staRole) if (dynRole, staRole) in matchingConditions else (staRole, dynRole) \
                    if (staRole, dynRole) in matchingConditions else None
                if clunuPair is None:
                    print("Skipping ({}, {})".format(staRole, dynRole))
                    continue
                clunuPairs.append(clunuPair)
                cluPairCond = matchingConditions[clunuPair]
                fieldMatches = [fieldNum for fieldNum, fieldCond in enumerate(cluPairCond[1:])
                              if not any(fieldCond[:5]) and (fieldCond[5] or fieldCond[6])]
                curStaValues = [alignedFieldClasses[clunuPair][0][fieldMatch]
                                if clunuPair[0] == staRole else alignedFieldClasses[clunuPair][1][fieldMatch]
                            for fieldMatch in fieldMatches]
                if staValues is None:  # set the static values list to the currently determined STA field values for the DYN-STA pair clunuPair
                    staValues = curStaValues
                elif sorted([sv.bytes for sv in staValues]) != sorted([csv.bytes for csv in curStaValues]):
                    staMismatch = True
                    break
            if staMismatch:
                # mask to remove the clunuPairs of all combinations with this dynRole
                removeFromMatchingClusters.extend([
                    (dynRole, staRole) if (dynRole, staRole) in matchingConditions else (staRole, dynRole)
                    for staRole in staRoleList
                ])
        except KeyError as e:
            print(e)
            IPython.embed()
            raise e


    # if a chain of matches would merge more than .5 of all clusters, remove that chain
    from networkx import Graph
    from networkx.algorithms.components.connected import connected_components
    dracula = Graph()
    dracula.add_edges_from(set(matchingClusters) - set(removeFromMatchingClusters))
    connectedDracula = list(connected_components(dracula))
    for clusterChain in connectedDracula:
        if len(clusterChain) > .5 * len(alignedClusters):
            for clunu in clusterChain:
                for remainingPair in set(matchingClusters) - set(removeFromMatchingClusters):
                    if clunu in remainingPair:
                        removeFromMatchingClusters.append(remainingPair)

    remainingClusters = set(matchingClusters) - set(removeFromMatchingClusters)
    if len(remainingClusters) > 0:
        print("Clusters could be merged:")
        for clunuAB in remainingClusters:
            printClusterMergeConditions(clunuAB, alignedFieldClasses, matchingConditions, dc)

    print("remove:", removeFromMatchingClusters)

    print("remain:", remainingClusters)
    chainedRemains = Graph()
    chainedRemains.add_edges_from(remainingClusters)
    connectedClusters = list(connected_components(chainedRemains))
    print("connected:", connectedClusters)

    print("should merge:")
    # identify over-specified clusters and list which ones ideally would be merged:
    clusterMostFreqStats = {stats[1]: [] for stats in clusterStats if stats is not None}
    # 'cluster_label', 'most_freq_type', 'precision', 'recall', 'cluster_size'
    for stats in clusterStats:
        if stats is None:
            continue
        cluster_label, most_freq_type, precision, recall, cluster_size = stats
        clusterMostFreqStats[most_freq_type].append((cluster_label, precision))
    overspecificClusters = {mft: stat for mft, stat in clusterMostFreqStats.items() if len(stat) > 1}
    superClusters = list()
    for mft, stat in overspecificClusters.items():
        superCluster = [str(label) if precision > 0.5 else "[{}]".format(label) for label, precision in stat]
        superClusters.append(superCluster)
        print("    \"{}\"  ({})".format(mft, ", ".join(superCluster)))

    print("missed clusters for merge:")
    missedmerges = [[ocel[0] for ocel in oc] for oc in overspecificClusters.values()]
    mergeCandidates = [el for cchain in connectedClusters for el in cchain]
    for sc in missedmerges:
        for sidx, selement in reversed(list(enumerate(sc))):
            if selement in mergeCandidates:
                del sc[sidx]
    print(missedmerges)
    print()

    missedmergepairs = [k for k in clusterpairs if any(
        [k[0] in mc and k[1] in mc or
         k[0] in mc and k[1] in itertools.chain.from_iterable([cc for cc in connectedClusters if k[0] in cc]) or
         k[0] in itertools.chain.from_iterable([cc for cc in connectedClusters if k[1] in cc]) and k[1] in mc
         for mc in missedmerges]
    )]

    #     # alignedFieldClasses to look up aligned field candidates from cluster pairs
    #     # in dhcp-1000, cluster pair 0,7
    #     from inference.templates import TemplateGenerator
    #     t0019 = alignedFieldClasses[(0, 7)][0][19]
    #     t7119 = alignedFieldClasses[(0, 7)][1][19]
    #     tg0019 = TemplateGenerator.generateTemplatesForClusters(dc, [t0019.baseSegments])[0]
    #     dc.findMedoid(t0019.baseSegments)
    #     dc.pairDistance(tg0019.medoid, t7119)
    #
    #     # 1:19 should be aligned to 0:21
    #     tg021 = TemplateGenerator.generateTemplatesForClusters(dc, [alignedFieldClasses[(0, 7)][0][21].baseSegments])[0]
    #
    #     dc.pairDistance(tg021.medoid, t7119)
    #     # 0.37706280402265446

    singleClusters = {ck: ml for ck, ml in messageClusters.items() if not chainedRemains.has_node(ck)}
    mergedClusters = {str(mergelist):
                          list(itertools.chain.from_iterable([messageClusters[clunu] for clunu in mergelist]))
                      for mergelist in connectedClusters}
    mergedClusters.update(singleClusters)

    return mergedClusters, missedmergepairs






