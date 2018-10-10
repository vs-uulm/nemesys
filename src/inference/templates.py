from typing import List, Dict, Union, Iterable, Sequence
import numpy, scipy.spatial, itertools

from netzob.Model.Vocabulary.Messages.AbstractMessage import AbstractMessage

from inference.analyzers import MessageAnalyzer
from inference.segments import MessageSegment, AbstractSegment, CorrelatedSegment


# TODO


class InterSegment(object):
    """
    Two segments and their similarity resp. distance.
    """
    def __init__(self, segA: MessageSegment, segB: MessageSegment, distance: float):
        """
        :param segA: MessageSegment A
        :param segB: MessageSegment B
        :param distance: the similarity/distance between A and B
        """
        self.segA = segA
        self.segB = segB
        self.distance = distance


    def wobbleLeft(self, AorB):
        """
        Shift segments segA and segB against each other by one byte to the left.

        :param AorB: Shift A against B or vice versa. True for B.
        :return:
        """
        if AorB:
            seg = self.segB
        else:
            seg = self.segA
        if not seg.offset > 0:
            raise IndexError('There is nothing left of {}.'.format('B' if AorB else 'A'))
        leftOf = MessageSegment(seg.analyzer, seg.offset-1, seg.length)
        if AorB:
            return self.segA.values, leftOf.values
        else:
            return leftOf.values, self.segB.values


    def wobbleRight(self, AorB):
        """
        Shift segments segA and segB against each other by one byte to the right.

        :param AorB: Shift A against B or vice versa. True for B.
        :return:
        """
        if AorB:
            seg = self.segB
        else:
            seg = self.segA
        if seg.offset+seg.length > len(seg.message.data):
            raise IndexError('There is nothing right of {}.'.format('B' if AorB else 'A'))
        rightOf = MessageSegment(seg.analyzer, seg.offset+1, seg.length)
        if AorB:
            return self.segA.values, rightOf.values
        else:
            return rightOf.values, self.segB.values


    def wobble(self):
        """
        Shift segments segA and segB against each other by one byte and test whether their cosine coefficient
        improves (lower value) for any of the four new combinations. If so, returns the minimum value and the
        analysis values for each byte corresponding to each other.

        Example: Original segments BCD and bcd
        wobble to:

        BCD | BCD | ABC | CDE
        abc | cde | bcd | bcd


        :return: the minimum cosine coefficient if better than the one of the original alignment
            and the values of the new match. None if there is no better match.

        """
        wobbls = list()
        try:
            wobbls.append(self.wobbleLeft(True))
        except IndexError:
            pass
        try:
            wobbls.append(self.wobbleRight(True))
        except IndexError:
            pass
        try:
            wobbls.append(self.wobbleLeft(False))
        except IndexError:
            pass
        try:
            wobbls.append(self.wobbleRight(False))
        except IndexError:
            pass
        cosCoeffs = list()
        for wob in wobbls:
            cosCoeffs.append(scipy.spatial.distance.cosine(*wob))
        if min(cosCoeffs, default=1) < self.distance:
            amin = int(numpy.argmin(cosCoeffs))
            return cosCoeffs[amin], wobbls[amin]
        else:
            return None


class Template(AbstractSegment):
    baseSegments = list()  # list/cluster of MessageSegments this template was generated from.


    def __init__(self, values: Union[List[Union[float, int]], numpy.ndarray],
                 baseSegments: List[MessageSegment]):
        self.values = values
        self.baseSegments = baseSegments
        self.checkSegmentsAnalysis()


    def checkSegmentsAnalysis(self):
        for bs in self.baseSegments[1:]:
            if type(self.baseSegments[0].analyzer) != type(bs.analyzer) \
                    or self.baseSegments[0].analyzer.analysisParams != bs.analyzer.analysisParams:
                errordetail = '\n{} mismatches {}'.format(type(self.baseSegments[0].analyzer), type(bs)) \
                    if type(self.baseSegments[0].analyzer) != type(bs.analyzer) \
                    else '\nArguments {} mismatch {}'.format(self.baseSegments[0].analyzer.analysisParams,
                                                             bs.analyzer.analysisParams)
                raise ValueError('All segments a template is based on need to have the same analyzer type.'
                                 + errordetail)


    def correlate(self,
                  haystack: Iterable[Union[MessageSegment, AbstractMessage]],
                  method=AbstractSegment.CORR_COSINE
                  ) -> List[CorrelatedSegment]:
        if all(isinstance(hay, AbstractMessage) for hay in haystack):
            haystack = [ MessageSegment(
                MessageAnalyzer.findExistingAnalysis(
                    type(self.baseSegments[0].analyzer), MessageAnalyzer.U_BYTE,
                        hayhay, self.baseSegments[0].analyzer.analysisParams),
                    0, len(hayhay.data)
            ) for hayhay in haystack ]
        return super().correlate(haystack, method)


    def __hash__(self):
        return hash(tuple(self.values))


    def __repr__(self):
        oid = hash(self)
        # return '{:02x}'.format(oid % 0xffff)
        import visualization.bcolors as bcolors
        # Template
        return bcolors.colorizeStr('{:02x}'.format(oid % 0xffff), oid % 0xff)  # TODO caveat collisions


class TemplateGenerator(object):
    """
    Generate templates for a list of segments according to their distance.
    """

    def __init__(self, segments: List[MessageSegment]):
        """
        Find similar segments, group them, and return a template for each group.

        :param segments:
        """
        self._segments = list()  # type: List[MessageSegment]
        # ensure that all segments have analysis values
        for seg in segments:
            self._segments.append(segments[0].fillCandidate(seg))
        # distance matrix for all rows and columns in order of self._segments
        self._distances = self._getDistanceMatrix(
            TemplateGenerator._calcDistancesPerLen(
                self._groupByLength()))  # type: numpy.array()
        self.clusterer = None  # type: TemplateGenerator.DBSCAN

    @property
    def distanceMatrix(self) -> numpy.ndarray:
        """
        The order of the matrix elements in each row and column is the same as in self.segments.

        :return: The pairwise similarities of all segments in this object represented as an symmetric array.
        """
        return self._distances

    @property
    def segments(self) -> List[MessageSegment]:
        """
        :return: All segments in this object.
        """
        return self._segments


    def _groupByLength(self) -> Dict[int, List[MessageSegment]]:
        """
        Groups segments by value length.

        Used in constructor.

        :return: a dict mapping the length to the list of MessageSegments of that length.
        """
        segsByLen = dict()
        for seg in self._segments:
            seglen = len(seg.values)
            if seglen not in segsByLen:
                segsByLen[seglen] = list()
            segsByLen[seglen].append(seg)
        return segsByLen


    @staticmethod
    def _calcDistancesPerLen(segLenGroups: Dict[int, List[MessageSegment]]) -> List[InterSegment]:
        """
        Calculates distances within groups of equally lengthy segments.

        Used in constructor.

        :param segLenGroups: a dict mapping the length to the list of MessageSegments of that length.
        :return: flat list of pairwise distances for all length groups.
        """
        distance = list()
        for l, segGroup in segLenGroups.items():  # type: int, List[MessageSegment]
            distance.extend(TemplateGenerator._calcDistances(segGroup))
        return distance


    @staticmethod
    def _calcDistances(segments: List[MessageSegment], method='cosine') -> List[InterSegment]:
        """
        Calculates pairwise distances for all input segments.

        :param segments: list of segments to calculate their similarity/distance for.
        :param method: The method to use for distance calculation. See scipy.spatial.distance.pdist.
            defaults to 'cosine'.
        :return: List of all pairwise distances between segements encapsulated in InterSegment-objects.
        """
        # print("Calculating Distances...")
        if method == 'cosine':
            # comparing to zero vectors is undefined in cosine.
            # Its semantically equivalent to a (small) horizontal vector
            segmentValuesMatrix = numpy.array(
                [seg.values if (numpy.array(seg.values) != 0).any() else [1e-16]*len(seg.values) for seg in segments])
        else:
            segmentValuesMatrix = numpy.array([seg.values for seg in segments])
        segPairSimi = scipy.spatial.distance.pdist(segmentValuesMatrix, method)
        segPairs = list()
        for (segA, segB), simi in zip(itertools.combinations(segments, 2), segPairSimi):
            if numpy.isnan(simi):
                if method == 'cosine':
                    if segA.values == segB.values \
                            or numpy.isnan(segA.values).all() and numpy.isnan(segB.values).all():
                        segSimi = 0
                    elif numpy.isnan(segA.values).any() or numpy.isnan(segB.values).any():
                        segSimi = 1
                        # TODO better handling of segments with NaN parts.
                        # print('Set max distance for NaN segments with values: {} and {}'.format(
                        #     segA.values, segB.values))
                    else:
                        raise ValueError('A unresolved zero-values vector could not be handled by method ' + method +
                                         'the segment values are: {}\nand {}'.format(segA.values, segB.values))
                else:
                    raise NotImplementedError('Handling of NaN distances need to be defined for method ' + method)
            else:
                segSimi = simi
            segPairs.append(InterSegment(segA, segB, segSimi))
        # print("    finished.")
        return segPairs


    @staticmethod
    def __altSimilarities(segmentGroup: List[MessageSegment]):
        """
        Calculates pairwise similarities for all segments.
        Less efficient than calcSimilarities()
        This method is to validate the correct function of calcSimilarities()

        :param segmentGroup: list of segments to calculate their similarity/distance for.
        :return: List of all pairwise similarities between segements encapsulated in InterSegment-objects.
        """

        combinations = [(segmentGroup[p1], segmentGroup[p2])
                     for p1 in range(len(segmentGroup))
                     for p2 in range(p1 + 1, len(segmentGroup))]
        segPairs = list()
        for (segA, segB) in combinations:
            segPairs.append(InterSegment(segA, segB,
                                         scipy.spatial.distance.cosine(segA.values, segB.values)))
        return segPairs


    def _getDistanceMatrix(self, distances: List[InterSegment]) -> numpy.ndarray:
        """
        Arrange the representation of the pairwise similarities of the input parameter in an symmetric array.
        The order of the matrix elements in each row and column is the same as in self._segments.

        Used in constructor.

        :param distances: The pairwise similarities to arrange.
        :return: The distance matrix for the given similarities.
            1 for each undefined element, 0 in the diagonal, even if not given in the input.
        """
        numsegs = len(self._segments)
        simtrx = numpy.ones((numsegs, numsegs))
        numpy.fill_diagonal(simtrx, 0)
        # fill matrix with pairwise distances
        for intseg in distances:
            row = self._segments.index(intseg.segA)
            col = self._segments.index(intseg.segB)
            simtrx[row, col] = intseg.distance
            simtrx[col, row] = intseg.distance
        return simtrx


    def clusterSimilarSegments(self, filterNoise=True) -> List[List[MessageSegment]]:
        """
        Find suitable discrimination between dissimilar segments.

        Currently only returning groups of messages with available pairwise distance for all members.

        :param: if filterNoise is False, the first element in the returned list of clusters
            always is the (possibly empty) noise.
        :return: clusters of similar segments
        """
        clusters = [[]]
        for eqLenSegs in self._groupByLength().values():
            clusterInGroup = self.getClusters(eqLenSegs)
            if not filterNoise:
                if -1 in clusterInGroup:
                    clusters[0].extend(clusterInGroup[-1])
            noiseFiltered = [v for k, v in clusterInGroup.items() if k > -1]
            clusters.extend(noiseFiltered)  # omit noise
        return clusters


    def _similaritiesInLengthGroup(self, cluster: Sequence[MessageSegment]) -> numpy.ndarray:
        """
        Get the submatrix of the distances in self._distances for the given list of
        message segments of equal length and their order.

        :param cluster:
        :return:
        """
        numsegs = len(cluster)
        simtrx = numpy.ones((numsegs, numsegs))
        transformator = dict()
        for i, seg in enumerate(cluster):
            transformator[i] = self._segments.index(seg)
        for i,k in transformator.items():
            for j,l in transformator.items():
                simtrx[i,j] = self._distances[k,l]
        return simtrx


    class DBSCAN(object):
        """
        Wrapper for DBSCAN from the sklearn.cluster module including autoconfiguration of the parameters.
        """

        def __init__(self, distances: numpy.ndarray, autoconfigure=True):
            self._distances = distances
            if autoconfigure:
                self.minpts, self.epsilon = self._autoconfigure()
            else:
                self.minpts, self.epsilon = None, None
                # epsilon = numpy.mean(similarities[mask]) + numpy.std(similarities[mask]) * 2 # 2.5? 3?


        def lowertriangle(self):
            """
            distances is a symmetric matrix, so we often only need one triangle:
            :return: mask of the lower triangle of the matrix
            """
            mask = numpy.tril(numpy.ones(self._distances.shape)) != 0
            return mask


        def _autoconfigure(self):
            """
            Auto configure the clustering parameters epsilon and minPts regarding the input data
            according to the paper https://www.sciencedirect.com/science/article/pii/S0169023X06000218

            :return: minpts, epsilon
            """
            import math
            from scipy.ndimage.filters import gaussian_filter1d

            # MinPts ≈ ln(n)
            minpts = round(math.log(self._distances.shape[0]))
            # get minpts-nearest-neighbors:
            neighdistmeans = list()
            for neighid in range(self._distances.shape[0]):
                ndmean = numpy.mean(
                    sorted(self._distances[:, neighid])[1:minpts + 1])  # shift by one: ignore self identity
                neighdistmeans.append((neighid, ndmean))
            neighdistmeans = sorted(neighdistmeans, key=lambda x: -x[1])
            neighdists = [e[1] for e in neighdistmeans]
            # smooth distances to prevent ambiguities about what a "knee" in the L-curve is
            smoothdists = gaussian_filter1d(neighdists, numpy.log(len(neighdists)))
            # approximate 2nd derivative and get its max
            kneeX = numpy.ediff1d(numpy.ediff1d(smoothdists)).argmax()

            # print(kneeX, smoothdists[kneeX], neighdists[kneeX])

            epsilon = neighdists[kneeX]
            return minpts, epsilon


        def getClusterLabels(self) -> Union[List[int], Iterable]:
            """
            Cluster the entries in the similarities parameter by DBSCAN
            and return the resulting labels.

            :return:
            """
            import sklearn.cluster

            if numpy.count_nonzero(self._distances) == 0:  # only identical segments
                return numpy.zeros_like(self._distances[0], int).tolist()


            dbscan = sklearn.cluster.DBSCAN(eps=self.epsilon, min_samples=self.minpts)
            print("DBSCAN epsilon:", self.epsilon, "minpts:", self.minpts)
            dbscan.fit(self._distances)
            return dbscan.labels_


    def getClusters(self, lengthGroup: Sequence[MessageSegment]) -> Dict[int, List[MessageSegment]]:
        similarities = self._similaritiesInLengthGroup(lengthGroup)
        try:
            self.clusterer = TemplateGenerator.DBSCAN(similarities)
            labels = self.clusterer.getClusterLabels()
        except ValueError as e:
            print(lengthGroup)
            # import tabulate
            # print(tabulate.tabulate(similarities))
            raise e
        ulab = set(labels)

        segmentClusters = dict()
        for l in ulab:
            class_member_mask = (labels == l)
            segmentClusters[l] = [ seg for seg in itertools.compress(lengthGroup, class_member_mask) ]
        return segmentClusters


    @staticmethod
    def generateTemplatesForClusters(segmentClusters: Iterable[List[MessageSegment]]) -> List[Template]:
        """
        Find templates representing the message segments in the input clusters.

        :param segmentClusters: list of input clusters
        :return: list of templates for input clusters
        """
        templates = list()
        for segs in segmentClusters:
            tmpl = numpy.array([ seg.values for seg in segs ])
            templates.append(
                Template(numpy.mean(tmpl, 0), segs)
            )

        return templates


    def generateTemplates(self) -> List[Template]:
        allClusters = itertools.chain.from_iterable(
            [ self.getClusters(group).values() for group in self._groupByLength().values() ])

        return TemplateGenerator.generateTemplatesForClusters( allClusters )