from typing import List, Dict, Union, Iterable, Sequence, Tuple
import numpy, scipy.spatial, itertools
from abc import ABC, abstractmethod

from netzob.Model.Vocabulary.Messages.AbstractMessage import AbstractMessage

from inference.analyzers import MessageAnalyzer
from inference.segments import MessageSegment, AbstractSegment, CorrelatedSegment




class InterSegment(object):
    """
    Two segments and their similarity resp. distance.
    """
    def __init__(self, segA: MessageSegment, segB: MessageSegment, distance: float):
        """
        Constructing a new object for each pair of segments is a HUGE performance overhead.
        Therefore it is not practicable to use this class when creating (large) distance matrices!

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




class DistanceCalculator(object):
    """
    Wrapper to calculate and look up pairwise distances between segments.
    """

    debug = True

    def __init__(self, segments: Iterable[MessageSegment], method='canberra',
                 thresholdFunction = None, thresholdArgs = None
                 ):
        """
        Determine the distance between the given segments.

        >>> from inference.templates import *
        >>> from inference.analyzers import Value
        >>> from netzob.Model.Vocabulary.Messages.RawMessage import RawMessage
        >>>
        >>> bytedata = bytes([1,2,3,4])
        >>> message = RawMessage(bytedata)
        >>> analyzer = Value(message)
        >>> segments = [MessageSegment(analyzer, 0, 4)]
        >>> dc = DistanceCalculator(segments)
       	outersegs, length 4, segments 1
        Prepare values. 0.000s
        call pdist from scipy.Calculated distances for 1 segment pairs.

        :param segments: The segments to calculate pairwise distances from
        :param method: The distance method to use for calculating pairwise distances. Supported methods are
            * 'canberra' (default)
            * 'euclidean'
            * 'cosine'
            * 'correlation'
            The calculation finally is performed by scipy.spatial.distance.pdist
        :param thresholdFunction: sets a function to transform each distance by skewing the distribution of distances.
            By default (preset with None), does a neutral transform that does not apply any changes.
            One available alternative implemented in this class is DistanceCalculator.sigmoidThreshold()
        :param thresholdArgs: dict of kwargs for the set thresholdFunction. Empty by default.
        :return: A normalized distance matrix for all input segments.
            If necessary, performs an embedding of mixed-length segments to determine cross-length distances.
        """
        self._method = method
        self.thresholdFunction = thresholdFunction if thresholdFunction else DistanceCalculator.neutralThreshold
        self.thresholdArgs = thresholdArgs if thresholdArgs else {}
        self._segments = list()  # type: List[MessageSegment]
        self._quicksegments = list()  # type: List[Tuple[int, int, Tuple[float]]]
        """List of Tuples: (index of segment in self._segments), (segment length), (Tuple of segment analyzer values)"""
        # ensure that all segments have analysis values
        firstSegment = next(iter(segments))
        for idx, seg in enumerate(segments):
            self._segments.append(firstSegment.fillCandidate(seg))
            self._quicksegments.append((idx, seg.length, tuple(seg.values)))
        # distance matrix for all rows and columns in order of self._segments
        self._distances = self._getDistanceMatrix(self._embdedAndCalcDistances())

    @property
    def distanceMatrix(self) -> numpy.ndarray:
        """
        The order of the matrix elements in each row and column is the same as in self.segments.

        :return: The normalized pairwise distances of all segments in this object represented as an symmetric array.
        """
        return self._distances

    def similarityMatrix(self) -> numpy.ndarray:
        """
        Converts the distances into similarities using the knowledge about distance method and analysis type.

        The order of the matrix elements in each row and column is the same as in self.segments.

        :return: The pairwise similarities of all segments in this object represented as an symmetric array.
        """
        similarityMatrix = 1 - self._distances
        return similarityMatrix

    def _maxDimension(self, segmentX: int, segmentY: int):
        """
        :return: Minimum of the value dimensions of the measured pair of segments
        """
        return min(len(self._segments[segmentX].values), len(self._segments[segmentY].values))

    @property
    def segments(self) -> List[MessageSegment]:
        """
        :return: All segments in this object.
        """
        return self._segments

    def pairDistance(self, A: MessageSegment, B: MessageSegment) -> numpy.float64:
        """
        Retrieve the distance of two segments.

        :param A:
        :param B:
        :return:
        """
        a = self._segments.index(A)
        b = self._segments.index(B)
        return self._distances[a,b]

    def pairwiseDistance(self, As: List[MessageSegment], Bs: List[MessageSegment]) -> numpy.ndarray:
        """
        Retrieve the matrix of pairwise distances for two lists of segments.

        :param As: List of segments
        :param Bs: List of segments
        :return: Matrix of distances: As are rows, Bs are columns.
        """
        distances = list()
        for A in As:
            Alist = list()
            distances.append(Alist)
            for B in Bs:
                Alist.append(self.pairDistance(A, B))
        return numpy.array(distances)

    def _groupByLength(self) -> Dict[int, List[Tuple[int, int, Tuple[float]]]]:
        """
        Groups segments by value length.

        Used in constructor.

        :return: a dict mapping the length to the list of MessageSegments of that length.
        """
        segsByLen = dict()
        for seg in self._quicksegments:
            seglen = seg[1]
            if seglen not in segsByLen:
                segsByLen[seglen] = list()
            segsByLen[seglen].append(seg)
        return segsByLen

    def _calcDistancesPerLen(self, segLenGroups: Dict[int, List[MessageSegment]]) -> List[InterSegment]:
        """
        Calculates distances within groups of equally lengthy segments.

        Used in constructor.

        :param segLenGroups: a dict mapping the length to the list of MessageSegments of that length.
        :return: flat list of pairwise distances for all length groups.
        """
        distance = list()
        for l, segGroup in segLenGroups.items():  # type: int, List[MessageSegment]
            segindices = [self._segments.index(seg) for seg in segGroup]
            qsegs = [self._quicksegments[idx] for idx in segindices]
            distance.extend(DistanceCalculator._calcDistances(qsegs, method=self._method))
        # TODO perf
        return distance


    @staticmethod
    def __prepareValuesMatrix(segments: List[Tuple[int, int, Tuple[float]]], method):
        # TODO if this should become is a performance killer, drop support for cosine and correlation of unsuitable segments altogether

        # fallback for vectors of length 1
        if segments[0][1] == 1:
            # there is no simple fallback solution for 'correlate'
            # factor = 2 if method == 'correlate' else 1
            if method == 'cosine':
                raise NotImplementedError("There is no simple fallback solution to correlate segments of length 1!")
            if method == 'cosine':
                method = 'canberra'

        if method == 'cosine':
            # comparing to zero vectors is undefined in cosine.
            # Its semantically equivalent to a (small) horizontal vector
            segmentValuesMatrix = numpy.array(
                [seg[2] if (numpy.array(seg[2]) != 0).any() else [1e-16]*len(seg[2]) for seg in segments])
        else:
            segmentValuesMatrix = numpy.array([seg[2] for seg in segments])

        return method, segmentValuesMatrix

    @staticmethod
    def _calcDistances(segments: List[Tuple[int, int, Tuple[float]]], method='canberra') -> List[
        Tuple[int, int, float]
    ]:
        """
        Calculates pairwise distances for all input segments.
        The values of all segments have to be of the same length!

        :param segments: list of segments to calculate their similarity/distance for.
        :param method: The method to use for distance calculation. See scipy.spatial.distance.pdist.
            defaults to 'cosine'.
        :return: List of all pairwise distances between segements.
        """
        if DistanceCalculator.debug:
            import time
            tPrep = time.time()
            print('Prepare values.', end='')
        method, segmentValuesMatrix = DistanceCalculator.__prepareValuesMatrix(segments, method)

        if DistanceCalculator.debug:
            tPdist = time.time()
            print(' {:.3f}s\ncall pdist from scipy.'.format(tPdist-tPrep), end='')
        if len(segments) == 1:
            return [(segments[0][0], segments[0][0], 0)]
        # This is the poodle's core
        segPairSimi = scipy.spatial.distance.pdist(segmentValuesMatrix, method)

        if DistanceCalculator.debug:
            tExpand = time.time()
            print(' {:.3f}s\nExpand compressed pairs.'.format(tExpand-tPdist), end='')
        segPairs = list()
        for (segA, segB), simi in zip(itertools.combinations(segments, 2), segPairSimi):
            if numpy.isnan(simi):
                if method == 'cosine':
                    if segA[2] == segB[2] \
                            or numpy.isnan(segA[2]).all() and numpy.isnan(segB[2]).all():
                        segSimi = 0
                    elif numpy.isnan(segA[2]).any() or numpy.isnan(segB[2]).any():
                        segSimi = 1
                        # TODO better handling of segments with NaN parts.
                        # print('Set max distance for NaN segments with values: {} and {}'.format(
                        #     segA[2], segB[2]))
                    else:
                        raise ValueError('An unresolved zero-values vector could not be handled by method ' + method +
                                         ' the segment values are: {}\nand {}'.format(segA[2], segB[2]))
                elif method == 'correlation':
                    # TODO validate this assumption about the interpretation of uncorrelatable segments.
                    if segA[2] == segB[2]:
                        segSimi = 0.0
                    else:
                        segSimi = 9.9
                else:
                    raise NotImplementedError('Handling of NaN distances needs to be defined for method ' + method)
            else:
                segSimi = simi
            segPairs.append((segA[0], segB[0], segSimi))
        if DistanceCalculator.debug:
            tFinal = time.time()
            print(" {:.3f}s".format(tFinal-tExpand))
        return segPairs

    @staticmethod
    def __calcDistances(segments: List[MessageSegment], method='cosine') -> List[InterSegment]:
        """
        LEGACY/FALLBACK

        Calculates pairwise distances for all input segments.
        The values of all segments have to be of the same length!

        :param segments: list of segments to calculate their similarity/distance for.
        :param method: The method to use for distance calculation. See scipy.spatial.distance.pdist.
            defaults to 'cosine'.
        :return: List of all pairwise distances between segements encapsulated in InterSegment-objects.
        """
        if DistanceCalculator.debug:
            import time
            tPrep = time.time()
            print('Prepare values.', end='')
        # TODO perf
        method, segmentValuesMatrix = DistanceCalculator.__prepareValuesMatrix(segments, method)

        if DistanceCalculator.debug:
            tPdist = time.time()
            print(' {:.3f}s\ncall pdist from scipy.'.format(tPdist-tPrep), end='')
        # This is the poodle's core
        segPairSimi = scipy.spatial.distance.pdist(segmentValuesMatrix, method)

        if DistanceCalculator.debug:
            tExpand = time.time()
            print(' {:.3f}s\nExpand compressed pairs.'.format(tExpand-tPdist), end='')
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
                        raise ValueError('An unresolved zero-values vector could not be handled by method ' + method +
                                         ' the segment values are: {}\nand {}'.format(segA.values, segB.values))
                elif method == 'correlation':
                    # TODO validate this assumption about the interpretation of uncorrelatable segments.
                    if segA.values == segB.values:
                        segSimi = 0.0
                    else:
                        segSimi = 9.9
                else:
                    raise NotImplementedError('Handling of NaN distances needs to be defined for method ' + method)
            else:
                segSimi = simi
            segPairs.append(InterSegment(segA, segB, segSimi))
        if DistanceCalculator.debug:
            tFinal = time.time()
            print(" {:.3f}s".format(tFinal-tExpand))
        return segPairs

    def _getDistanceMatrix(self, distances: List[
        Tuple[int, int, float]
    ]) -> numpy.ndarray:
        """
        Arrange the representation of the pairwise similarities of the input parameter in an symmetric array.
        The order of the matrix elements in each row and column is the same as in self._segments.

        Used in constructor.

        :param distances: The pairwise similarities to arrange.
        :return: The distance matrix for the given similarities.
            1 for each undefined element, 0 in the diagonal, even if not given in the input.
        """
        from inference.segmentHandler import matrixFromTpairs
        simtrx = matrixFromTpairs([(ise[0], ise[1], ise[2]) for ise in distances], range(len(self._quicksegments)),
                                  incomparable=-1)  # TODO check for incomparable values (resolve and replace the negative value)
        return simtrx

    @staticmethod
    def embedSegment(shortSegment: Tuple[int, int, Tuple[float]], longSegment: Tuple[int, int, Tuple[float]],
                     method='canberra'):
        """
        Embed shorter segment in longer one to determine a "partial" similarity-based distance between the segments.
        Enumerates all possible shifts of overlaying the short above of the long segment and returns the minimum
        distance of any of the distance calculations performed for each of these overlays.

        The distance is normalized to the length of the shorter tuple

        :param shortSegment:
        :param longSegment:
        :param method: distance method to use
        :return: The minimal partial distance between the segments, wrapped in an "InterSegment-Tuple":
            (method, offset, (
                index of shortSegment, index of longSegment, distance betweent the two
            ))
        """
        assert longSegment[1] > shortSegment[1]

        maxOffset = longSegment[1] - shortSegment[1]

        subsets = list()
        for offset in range(0, maxOffset + 1):
            offSegment = longSegment[2][offset:offset + shortSegment[1]]
            subsets.append((-1, shortSegment[1], offSegment))
        method, segmentValuesMatrix = DistanceCalculator.__prepareValuesMatrix(subsets, method)

        subsetsSimi = scipy.spatial.distance.cdist(segmentValuesMatrix, numpy.array([shortSegment[2]]), method)
        shift = subsetsSimi.argmin() # for debugging and evaluation
        distance = subsetsSimi.min()

        return method, shift, (shortSegment[0], longSegment[0], distance)

    @staticmethod
    def sigmoidThreshold(x, shift=0.5):
        return 1 / (1 + numpy.exp(-numpy.pi ** 2 * (x - shift)))

    @staticmethod
    def neutralThreshold(x):
        return x

    def _embdedAndCalcDistances(self) -> \
            List[Tuple[int, int, float]]:
        """
        Embed all shorter Segments into all larger ones and use the resulting pairwise distances to generate a
        complete distance list of all combinations of the into segment list regardless of their length.

        >>> from tabulate import tabulate
        >>> from netzob.Model.Vocabulary.Messages.RawMessage import RawMessage
        >>> from inference.templates import *
        >>> from inference.analyzers import Value
        >>>
        >>> bytedata = [
        >>>     bytes([1, 2, 3, 4]),
        >>>     bytes([   2, 3, 4]),
        >>>     bytes([   1, 3, 4]),
        >>>     bytes([   2, 4   ]),
        >>>     bytes([   2, 3   ]),
        >>>     bytes([20, 30, 37, 50, 69, 2, 30]),
        >>>     bytes([        37,  5, 69       ]),
        >>>     bytes([70, 2, 3, 4]),
        >>>     bytes([3, 2, 3, 4])
        >>>     ]
        >>> messages  = [RawMessage(bd) for bd in bytedata]
        >>> analyzers = [Value(message) for message in messages]
        >>> segments  = [MessageSegment(analyzer, 0, len(analyzer.message.data)) for analyzer in analyzers]
        >>> dc = DistanceCalculator(segments)
        >>> print(tabulate([
        >>>     [ segments[0].values, segments[1].values, dc.pairDistance(segments[0], segments[1]) ],
        >>>     [ segments[0].values, segments[2].values, dc.pairDistance(segments[0], segments[2]) ],
        >>>     [ segments[0].values, segments[3].values, dc.pairDistance(segments[0], segments[3]) ],
        >>>     [ segments[0].values, segments[4].values, dc.pairDistance(segments[0], segments[4]) ],
        >>>     [ segments[5].values, segments[6].values, dc.pairDistance(segments[5], segments[6]) ],
        >>>     [ segments[0].values, segments[7].values, dc.pairDistance(segments[0], segments[7]) ],
        >>>     [ segments[3].values, segments[4].values, dc.pairDistance(segments[3], segments[4]) ],
        >>>     [ segments[0].values, segments[8].values, dc.pairDistance(segments[0], segments[8]) ],
        >>> ]))

        Calculated distances for 37 segment pairs.
        ---------------------------  -------------  ---------
        [1, 2, 3, 4]                 [2, 3, 4]      0.25
        [1, 2, 3, 4]                 [1, 3, 4]      0.364286
        [1, 2, 3, 4]                 [2, 4]         0.571429
        [1, 2, 3, 4]                 [2, 3]         0.5
        [20, 30, 37, 50, 69, 2, 30]  [37, 5, 69]    0.844156
        [1, 2, 3, 4]                 [70, 2, 3, 4]  0.242958
        [2, 4]                       [2, 3]         0.0714286
        [1, 2, 3, 4]                 [3, 2, 3, 4]   0.125
        ---------------------------  -------------  ---------

        :return: List of Tuples
            (index of segment in self._segments), (segment length), (Tuple of segment analyzer values)
        """
        lenGrps = self._groupByLength()

        distance = list()  # type: List[Tuple[int, int, float]]
        rslens = list(reversed(sorted(lenGrps.keys())))  # lengths, sorted by decreasing length
        for outerlen in rslens:
            outersegs = lenGrps[outerlen]
            if DistanceCalculator.debug:
                print("\toutersegs, length {}, segments {}".format(outerlen, len(outersegs)))
            # a) for segments of identical length: call _calcDistancesPerLen()
            ilDist = DistanceCalculator._calcDistances(outersegs, method=self._method)
            # # # # # # # # # # # # # # # # # # # # # # # #
            distance.extend([(i,l,
                              self.thresholdFunction(
                                  d * self._normFactor(outerlen),
                                  **self.thresholdArgs))
                             for i,l,d in ilDist])
            # # # # # # # # # # # # # # # # # # # # # # # #
            # b) on segments with mismatching length: embedSegment:
            #       for all length groups with length < current length
            for innerlen in rslens[rslens.index(outerlen)+1:]:
                innersegs = lenGrps[innerlen]
                if DistanceCalculator.debug:
                    print("\t\tinnersegs, length {}, segments {}".format(innerlen, len(innersegs)))
                # for all segments in "shorter length" group
                #     for all segments in current length group
                for iseg in innersegs:
                    for oseg in outersegs:
                        # embedSegment(shorter in longer)
                        embedded = DistanceCalculator.embedSegment(iseg, oseg, self._method)
                        # add embedding similarity to list of InterSegments
                        interseg = embedded[2]
                        # # # # # # # # # # # # # # # # # # # # # # # #
                        dlDist = (interseg[0], interseg[1], (
                                    self.thresholdFunction(
                                        interseg[2] * self._normFactor(innerlen) +
                                        .33 * (outerlen - innerlen) * self._normFactor(outerlen),
                                        **self.thresholdArgs)
                                    )
                                )  # minimum of dimensions
                        # # # # # # # # # # # # # # # # # # # # # # # #
                        distance.append(dlDist)
        print("Calculated distances for {} segment pairs.".format(len(distance)))
        return distance


    def _normFactor(self, dimensions):
        return DistanceCalculator.normFactor(self._method, dimensions, self._segments[0].analyzer.domain)


    @staticmethod
    def normFactor(method: str, dimensions: int, analyzerDomain: Tuple[float, float]):
        """
        For a distance value with given parameters, calculate the factor to normalize this distance.
        This is static for cosine and correlation, but dynamic for canberra and euclidean measures.

        :param method: distance calculation method. One of cosine, correlation, canberra, euclidean, sqeuclidean
        :param dimensions: dimensions of the
        :param analyzerDomain: the value domain of the used analyzer
        :return:
        """
        distanceMax = {
            'cosine': 1,
            'correlation': 2,
            'canberra': None,
            'euclidean': None,
            'sqeuclidean': None
        }
        assert method in distanceMax
        if method == 'canberra':
            distanceMax['canberra'] = dimensions  # max number of dimensions
        elif method == 'euclidean':
            domainSize = analyzerDomain[1] - analyzerDomain[0]
            distanceMax['euclidean'] = dimensions * domainSize
        elif method == 'sqeuclidean':
            domainSize = analyzerDomain[1] - analyzerDomain[0]
            distanceMax['sqeuclidean'] = dimensions * domainSize**2
        return 1 / distanceMax[method]


    def neigbors(self, segment: MessageSegment, subset: List[MessageSegment]=None) -> List[Tuple[int, float]]:
        """

        :param segment:
        :param subset:
        :return: a ascendingly sorted list of neighbors of segment
            from all the segments in this object (if subset is None)
            or from the segments in subset.
            The result is a list of tuples with
                * the index of the neigbor (from self.segments or the subset list, respectively) and
                * the distance to this neighbor
        """
        if subset:
            subsetMap = [None] * len(subset)
            for idx, alls in enumerate(self._segments):
                try:
                    subsetMap[subset.index(alls)] = idx
                except ValueError:
                    continue
            assert None not in subsetMap

        home = self._segments.index(segment)
        mask = list(range(home)) + [None] + list(range(home + 1, self._distances.shape[0])) \
            if not subset else subsetMap   # mask self identity by "None"-value if applicable

        candNeigbors = self._distances[:, home]
        neighbors = sorted([(nidx, candNeigbors[n]) for nidx, n in enumerate(mask) if n is not None],
                           key=lambda x: x[1])
        return neighbors



class Template(AbstractSegment):
    """
    Calculates a template (mean values per of vector component).
    """
    def __init__(self, values: Union[List[Union[float, int]], numpy.ndarray, MessageSegment],
                 baseSegments: List[MessageSegment],
                 method='canberra'):
        if isinstance(values, MessageSegment):
            self.values = values.values
            self.medoid = values
        else:
            self.values = values
            self.medoid = None
        self.baseSegments = baseSegments  # list/cluster of MessageSegments this template was generated from.
        self.checkSegmentsAnalysis()
        self._method = method


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


    def distancesTo(self):
        """
        Calculate the normalized distances to the template (cluster center).

        Currently only supports single length segments.

        :return: array of distances
        """
        baseValues = numpy.array([seg.values for seg in self.baseSegments])
        assert baseValues.shape[1] == self.values.shape[0], "cannot calc distances to mixed lengths segments"
        cenDist = scipy.spatial.distance.cdist(self.values.reshape((1, self.values.shape[0])),
                                               baseValues, self._method)
        normf = DistanceCalculator.normFactor(self._method, len(self.values), self.baseSegments[0].analyzer.domain)
        return normf * cenDist


    def distancesToMixedLength(self):
        """

        :return: List of Tuples of distances and offsets: -n for center, +n for segment
        """
        distances = [None] * len(self.baseSegments)
        lengthGroups = dict()
        for idx, bs in enumerate(self.baseSegments):
            if bs.length not in lengthGroups:
                lengthGroups[bs.length] = list()
            lengthGroups[bs.length].append(idx)
        for length, group in lengthGroups.items():
            if length == len(self.values):
                # TODO fix this, according to embedAndCalc...
                normf = DistanceCalculator.normFactor(self._method, len(self.values),
                                                      self.baseSegments[0].analyzer.domain)
                dist = normf * scipy.spatial.distance.cdist(self.values.reshape((1, self.values.shape[0])),
                                                    numpy.array([self.baseSegments[i].values for i in group]),
                                                    self._method)
                for i, d in zip(group, dist):
                    distances[i] = (d, 0)  # 0-offset
                continue

            if length < len(self.values):
                longList = [(None, len(self.values), self.values)]
                shortList = [(i, length, tuple(self.baseSegments[i].values)) for i in group]
            else:
                longList = [(i, length, tuple(self.baseSegments[i].values)) for i in group]
                shortList = [(None, len(self.values), self.values)]

            for shortS in shortList:
                # TODO fix this, according to embedAndCalc...
                normf = DistanceCalculator.normFactor(self._method, shortS[1],
                                                      self.baseSegments[0].analyzer.domain)
                for longS in longList:
                    embedding = DistanceCalculator.embedSegment(shortS, longS, self._method)
                    idxShort = embedding[2][0]
                    idxLong = embedding[2][1]
                    dist = normf * embedding[2][2]
                    idx = idxShort if idxLong is None else idxLong
                    offset = embedding[1]
                    if idxShort is None: offset = -offset
                    distances[idx] = (dist, offset)
        return distances


    def __hash__(self):
        return hash(tuple(self.values))


    def __repr__(self):
        oid = hash(self)
        # return '{:02x}'.format(oid % 0xffff)
        import visualization.bcolors as bcolors
        # Template
        return bcolors.colorizeStr('{:02x}'.format(oid % 0xffff), oid % 0xff)  # TODO caveat collisions


class TemplateGenerator(DistanceCalculator):
    """
    Generate templates for a list of segments according to their distance.
    """
    def __init__(self, segments: List[MessageSegment], method='canberra',
                 thresholdFunction = None, thresholdArgs = {}):
        """
        Find similar segments, group them, and return a template for each group.

        :param segments:
        """
        super().__init__(segments, method, thresholdFunction, thresholdArgs)
        self.clusterer = None  # type: TemplateGenerator.Clusterer

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

    def clusterSimilarSegments(self, filterNoise=True, **kwargs) -> List[List[MessageSegment]]:
        """
        Find suitable discrimination between dissimilar segments.

        Currently only returning groups of messages with available pairwise distance for all members.

        :param: if filterNoise is False, the first element in the returned list of clusters
            always is the (possibly empty) noise.
        :return: clusters of similar segments
        """
        clusters = self.getClusters(self._segments, **kwargs)
        if filterNoise and -1 in clusters: # omit noise
            del clusters[-1]
        clusterlist = [clusters[l] for l in sorted(clusters.keys())]
        return clusterlist


    def _similaritiesSubset(self, cluster: Sequence[MessageSegment]) -> numpy.ndarray:
        """
        From self._distances, extract a submatrix of distances for the given list of message segments.

        :param cluster: The segments to get the distance matrix for. The segments need to be of equal length.
        :return: Matrix of pairwise distances between the given segments with rows and cols in the order of this list.
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


    class Clusterer(ABC):
        """
        Wrapper for any clustering implementation to select and adapt the autoconfiguration of the parameters.
        """
        def __init__(self, distances: numpy.ndarray):
            self._distances = distances


        @abstractmethod
        def getClusterLabels(self) -> numpy.ndarray:
            """
            Cluster the entries in the similarities parameter
            and return the resulting labels.

            :return: (numbered) cluster labels for each segment in the order given in the (symmetric) distance matrix
            """
            raise NotImplementedError("This method needs to be implemented using a cluster algorithm.")


        def lowertriangle(self):
            """
            Distances is a symmetric matrix, and we often only need one triangle:
            :return: the lower triangle of the matrix, all other elements of the matrix are set to nan
            """
            mask = numpy.tril(numpy.ones(self._distances.shape)) != 0
            dist = self._distances.copy()
            dist[~mask] = numpy.nan
            return dist


        def _nearestPerNeigbor(self) -> List[Tuple[int, float]]:
            """
            see also DistanceCalculator.neighbors()
            In comparison to the general implementation in DistanceCalculator, this one does not return a sorted list,
            but just the closest neighbor and its index for all segments.

            This test uses a non-symmetric matrix to detect bugs if any. This is NOT a use case example!
            >>> from inference.templates import *
            >>> clusterer = TemplateGenerator.DBSCAN(numpy.array([[0.0, 0.5, 0.8],[0.1, 0.0, 0.9],[0.7,0.3,0.0]]),
            >>>     autoconfigure=False)
            >>> print(clusterer._distances)
            array([[ 0. ,  0.5,  0.8],
                   [ 0.1,  0. ,  0.9],
                   [ 0.7,  0.3,  0. ]])
            >>> clusterer._nearestPerNeigbor()
            [(1, 0.5), (0, 0.10000000000000001), (1, 0.29999999999999999)]

            :return: a list of the nearest neighbors of each segment in this clusterer object, omitting self identity.
            The position in the list is the index of the segment in the distance matrix.
            The result is a list of tuples with
                * the index of the neigbor (from the distance matrix) and
                * the distance to this neighbor
            """
            neibrNearest = list()
            for homeidx in range(self._distances.shape[0]):
                # mask self identity by "None"-value
                mask = list(range(homeidx)) + [None] + list(range(homeidx + 1, self._distances.shape[0]))
                candNeigbors = self._distances[homeidx]
                minNidx = mask[0]
                for nidx in mask[1:]:
                    if nidx is not None and (minNidx is None or candNeigbors[nidx] < candNeigbors[minNidx]):
                        minNidx = nidx
                minNdst = candNeigbors[minNidx]
                neibrNearest.append((minNidx, minNdst))
            return neibrNearest


        def steepestSlope(self):
            from math import log

            lnN = round(log(self._distances.shape[0]))

            # simple and (too) generic heuristic: MinPts ≈ ln(n)
            minpts = lnN

            # find the first increase, in the mean of the first 2*lnN nearest neighbor distances for all ks,
            # which is larger than the mean of those increases
            # Inspired by Fatma Ozge Ozkok, Mete Celik: "A New Approach to Determine Eps Parameter of DBSCAN Algorithm"
            npn = self._nearestPerNeigbor()
            dpnmln = [numpy.mean([dpn[k] for nid, dpn in npn if dpn[k] > 0][:2 * lnN]) for k in range(0, len(npn) - 1)]
            deltamln = numpy.ediff1d(dpnmln)
            deltamlnmean = deltamln.mean() # + deltamln.std()
            for k, a in enumerate(deltamln[lnN:], lnN):
                if a > deltamlnmean:
                    minpts = k + 1
                    break

            steepslopeK = minpts
            # add standard deviation to mean-threshold for eps (see authors above)
            deltamlnmean = deltamln.mean() + 2 * deltamln.std()
            for k, a in enumerate(deltamln[minpts-1:], minpts-1):
                if a > deltamlnmean:
                    steepslopeK = k + 1
                    break

            return minpts, steepslopeK


        @abstractmethod
        def __repr__(self):
            raise NotImplementedError("This method needs to be implemented giving the configuration of this clusterer.")


    class HDBSCAN(Clusterer):
        """
        Hierarchical Density-Based Spatial Clustering of Applications with Noise

        https://github.com/scikit-learn-contrib/hdbscan
        """

        def __init__(self, distances: numpy.ndarray, autoconfigure=True):
            super().__init__(distances)

            if autoconfigure:
                from math import log

                lnN = round(log(self._distances.shape[0]))
                self.min_cluster_size = self.steepestSlope()[0] # round(lnN * 1.5)
            else:
                self.min_cluster_size = None

            self.min_samples = 2


        def getClusterLabels(self) -> numpy.ndarray:
            """
            Cluster the entries in the similarities parameter by DBSCAN
            and return the resulting labels.

            :return: (numbered) cluster labels for each segment in the order given in the (symmetric) distance matrix
            """
            from hdbscan import HDBSCAN

            if numpy.count_nonzero(self._distances) == 0:  # the distance matrix contains only identical segments
                return numpy.zeros_like(self._distances[0], int)

            dbscan = HDBSCAN(metric='precomputed', allow_single_cluster=True, cluster_selection_method='leaf',
                             min_cluster_size=self.min_cluster_size,
                             min_samples=self.min_samples
                             )
            print("HDBSCAN min cluster size:", self.min_cluster_size, "min samples:", self.min_samples)
            dbscan.fit(self._distances)
            # TODO import IPython; IPython.embed()
            return dbscan.labels_


        def __repr__(self):
            return 'HDBSCAN mcs {} ms {}'.format(self.min_cluster_size, self.min_samples)


    class DBSCAN(Clusterer):
        """
        Wrapper for DBSCAN from the sklearn.cluster module including autoconfiguration of the parameters.
        """

        def __init__(self, distances: numpy.ndarray, autoconfigure=True):
            super().__init__(distances)
            if autoconfigure:
                self.minpts, self.epsilon = self._autoconfigure()
            else:
                self.minpts, self.epsilon = None, None


        def _autoconfigure(self):
            """
            Auto configure the clustering parameters epsilon and minPts regarding the input data

            :return: minpts, epsilon
            """
            from math import log

            minpts, steepslopeK = self.steepestSlope()

            # get minpts-nearest-neighbor distance:
            neighdists = self._knearestdistance(round(steepslopeK + (self._distances.shape[0] - steepslopeK) * .2))
                    # round(minpts + 0.5 * (self._distances.shape[0] - minpts))
            # # lower factors resulted in only few small clusters, since the density distribution is to uneven
            # # (for DBSCAN?)

            # # get mean of minpts-nearest-neighbors:
            # neighdists = self._knearestmean(minpts)
            # # it gives no significantly better results than the direct k-nearest distance,
            # # but requires more computation.


            # # knee calculation by rule of thumb
            # kneeX = self._kneebyruleofthumb(neighdists)
            # # result is far (too far) left of the actual knee

            # # knee by Kneedle alogithm: https://ieeexplore.ieee.org/document/5961514
            from kneed import KneeLocator
            kneel = KneeLocator(range(len(neighdists)), neighdists, curve='convex', direction='increasing')
            kneeX = kneel.knee
            # # knee is too far right/value too small to be useful: the clusters are small/zero size and few,
            # # perhaps density distribution too uneven in this use case?
            # kneel.plot_knee_normalized()

            # steepest-drop position:
            # kneeX = numpy.ediff1d(neighdists).argmax()
            # # better results than "any of the" knee values

            if isinstance(kneeX, int):
                epsilon = neighdists[kneeX]
            else:
                print("Warning: Kneedle could not find a knee in {}-nearest distribution.".format(minpts))
                epsilon = 0.0

            if not epsilon > 0.0:  # fallback if epsilon becomes zero
                lt = self.lowertriangle()
                epsilon = numpy.nanmean(lt) + numpy.nanstd(lt)


            # DEBUG and TESTING
            #
            # plots of k-nearest-neighbor distance histogram and "knee"
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(1, 2)

            axl, axr = ax.flat

            # for k in range(0, 100, 10):
            #     alpha = .4
            #     if k == minpts:
            #         alpha = 1
            #     plt.plot(sorted([dpn[k] for nid, dpn in npn]), alpha=alpha, label=k)

            # farthest
            # plt.plot([max([dpn[k] for nid, dpn in npn]) for k in range(0, len(npn)-1)], alpha=.4)
            # axl.plot(dpnmln, alpha=.4)
            # plt.plot([self._knearestdistance(k) for k in range( round(0.5 * (self._distances.shape[0]-1)) )])
            disttril = numpy.tril(self._distances)
            alldist = [e for e in disttril.flat if e > 0]
            axr.hist(alldist, 50)

            # plt.plot(smoothdists, alpha=.8)
            # axl.axvline(minpts, linestyle='dashed', color='red', alpha=.4)
            axl.axvline(steepslopeK, linestyle='dotted', color='blue', alpha=.4)
            left = axl.get_xlim()[0]
            bottom = axl.get_ylim()[1]
            # axl.text(left, bottom,"mpt={}, eps={:0.3f}".format(minpts, epsilon))
            # plt.axhline(neighdists[int(round(kneeX))], alpha=.4)
            # plt.plot(range(len(numpy.ediff1d(smoothdists))), numpy.ediff1d(smoothdists), linestyle='dotted')
            # plt.plot(range(len(numpy.ediff1d(neighdists))), numpy.ediff1d(neighdists), linestyle='dotted')
            axl.legend()
            # plt.text(0,0,'max {:.3f}, mean {:.3f}'.format(self._distances.max(), self._distances.mean()))
            import time
            # plt.show()
            plt.savefig("reports/k-nearest_distance_{:0.0f}.pdf".format(time.time()))
            plt.close('all')
            plt.clf()

            # print(kneeX, smoothdists[kneeX], neighdists[kneeX])
            # print(tabulate([neighdists[:10]], headers=[i for i in range(10)]))
            # print(tabulate([dpn[:10] for nid, dpn in npn], headers=[i for i in range(10)]))
            # import IPython; IPython.embed()
            #
            # DEBUG and TESTING

            return minpts, epsilon


        def _knearestmean(self, k: int):
            """
            :param k: range of neighbors to be selected
            :return: The mean of the k-nearest neighbors for all the distances of this clusterer instance.
            """
            neighdistmeans = list()
            for neighid in range(self._distances.shape[0]):
                ndmean = numpy.mean(
                    sorted(self._distances[:, neighid])[1:k + 1])  # shift by one: ignore self identity
                neighdistmeans.append((neighid, ndmean))
            neighdistmeans = sorted(neighdistmeans, key=lambda x: x[1])
            return [e[1] for e in neighdistmeans]


        def _knearestdistance(self, k: int):
            """
            :param k: neighbor to be selected
            :return: The distances of the k-nearest neighbors for all distances of this clusterer instance.
            """
            if not k < self._distances.shape[0] - 1:
                raise IndexError("k={} exeeds the number of neighbors.".format(k))
            return sorted([dpn[k] for nid, dpn in self._nearestPerNeigbor()])


        @staticmethod
        def _kneebyruleofthumb(neighdists):
            """
            according to the paper
            (!? this is the wrong reference ?!) https://www.sciencedirect.com/science/article/pii/S0169023X06000218

            :param neighdists: k-nearest-neighbor distances
            :return: x coordinate of knee point on the distance distribution of the parameter neighdists
            """
            # smooth distances to prevent ambiguities about what a "knee" in the L-curve is
            from scipy.ndimage.filters import gaussian_filter1d
            smoothdists = gaussian_filter1d(neighdists, numpy.log(len([n for n in neighdists if n > 0.001 * max(neighdists)])))

            # approximate 2nd derivative and get its max
            # kneeX = numpy.ediff1d(numpy.ediff1d(smoothdists)).argmax()  # alternative for 2nd derivative
            kneeX = numpy.array(
                    [smoothdists[i+1] + smoothdists[i-1] - 2 * smoothdists[i] for i in range(1, len(smoothdists)-1)]
                ).argmax()
            return int(round(kneeX))


        def getClusterLabels(self) -> numpy.ndarray:
            """
            Cluster the entries in the similarities parameter by DBSCAN
            and return the resulting labels.

            :return: (numbered) cluster labels for each segment in the order given in the (symmetric) distance matrix
            """
            import sklearn.cluster

            if numpy.count_nonzero(self._distances) == 0:  # the distance matrix contains only identical segments
                return numpy.zeros_like(self._distances[0], int)

            dbscan = sklearn.cluster.DBSCAN(eps=self.epsilon, min_samples=self.minpts)
            print("DBSCAN epsilon:", self.epsilon, "minpts:", self.minpts)
            dbscan.fit(self._distances)
            return dbscan.labels_


        def __repr__(self):
            return 'DBSCAN eps {:0.3f} mpt'.format(self.epsilon, self.minpts) \
                if self.epsilon and self.minpts \
                else 'DBSCAN unconfigured (need to set epsilon and minpts)'



    def getClusters(self, segments: Sequence[MessageSegment], **kwargs) -> Dict[int, List[MessageSegment]]:
        """
        Do the initialization of the clusterer and performe the clustering.

        :param segments: List of equally lengthy segments.
        :param kwargs: e. g. epsilon: The DBSCAN epsilon value, if it should be fixed.
            If not given (None), it is autoconfigured.
        :return: A dict of labels to lists of segments with that label.
        """
        clustererClass = kwargs['clustererClass'] if 'clustererClass' in kwargs else TemplateGenerator.HDBSCAN

        similarities = self._similaritiesSubset(segments)
        try:
            if not 'epsilon' in kwargs and not 'min_cluster_size' in kwargs:
                self.clusterer = clustererClass(similarities, True)
            else:
                self.clusterer = clustererClass(similarities, False)
                if 'epsilon' in kwargs:
                    # fixed epsilon
                    import math
                    self.clusterer.minpts = kwargs['minpts'] if 'minpts' in kwargs else \
                        round(math.log(similarities.shape[0]))
                    self.clusterer.epsilon = kwargs['epsilon']  # 1.0
                elif 'min_cluster_size' in kwargs:
                    self.clusterer.min_cluster_size = kwargs['min_cluster_size']
            labels = self.clusterer.getClusterLabels()
        except ValueError as e:
            print(segments)
            # import tabulate
            # print(tabulate.tabulate(similarities))
            raise e
        assert isinstance(labels, numpy.ndarray)
        ulab = set(labels)

        segmentClusters = dict()
        for l in ulab:
            class_member_mask = (labels == l)
            segmentClusters[l] = [seg for seg in itertools.compress(segments, class_member_mask)]
        return segmentClusters


    def findMedoid(self, segments: List[MessageSegment]) -> MessageSegment:
        """
        Find the medoid closest describing the given list of segments.
        :param segments: a list of segments that must be known to this template generator.
        :return: The MessageSegment from segments that is the list's medoid.
        """
        distSubMatrix = self.pairwiseDistance(segments, segments)
        mid = distSubMatrix.sum(axis=1).argmin()
        return segments[mid]


    def generateTemplatesForClusters(self, segmentClusters: Iterable[List[MessageSegment]], medoid=True) \
            -> List[Template]:
        """
        Find templates representing the message segments in the input clusters.

        :param medoid: Use medoid as template (supports mixed-length clusters) if true (default),
            use mean of segment values if false (supports only single-length clusters)
        :param segmentClusters: list of input clusters
        :return: list of templates for input clusters
        """
        templates = list()
        for cluster in segmentClusters:
            if not medoid:
                segValues = numpy.array([ seg.values for seg in cluster ])
                center = numpy.mean(segValues, 0)
            else:
                center = self.findMedoid(cluster)
            templates.append(Template(center, cluster))
        return templates


    def generateTemplates(self, **kwargs) -> List[Template]:
        """
        Generate templates for all clusters. Triggers a new clustering run.

        >>> from pprint import pprint
        >>> from netzob.Model.Vocabulary.Messages.RawMessage import RawMessage
        >>> from utils.loader import BaseLoader
        >>> from inference.templates import *
        >>> from inference.analyzers import Value
        >>>
        >>> bytedata = [
        >>>     bytes([1, 2, 3, 4]),
        >>>     bytes([   2, 3, 4]),
        >>>     bytes([   1, 3, 4]),
        >>>     bytes([   2, 4   ]),
        >>>     bytes([   2, 3   ]),
        >>>     bytes([20, 30, 37, 50, 69, 2, 30]),
        >>>     bytes([        37,  5, 69       ]),
        >>>     bytes([70, 2, 3, 4]),
        >>>     bytes([3, 2, 3, 4])
        >>>     ]
        >>> messages  = [RawMessage(bd) for bd in bytedata]
        >>> analyzers = [Value(message) for message in messages]
        >>> segments  = [MessageSegment(analyzer, 0, len(analyzer.message.data)) for analyzer in analyzers]
        >>> specimens = BaseLoader(messages)
        >>> tg = TemplateGenerator(segments)
        >>> clusters = tg.getClusters(segments, clustererClass=TemplateGenerator.DBSCAN,
        ...         epsilon=1.0, minpts=3,
        ...         thresholdFunction=TemplateGenerator.neutralThreshold, thresholdArgs=None)
        >>> templates = tg.generateTemplates(clustererClass=TemplateGenerator.DBSCAN,
        ...     epsilon=1.0, minpts=3,
        ...     thresholdFunction=TemplateGenerator.neutralThreshold, thresholdArgs=None)
        >>> pprint([t.baseSegments for t in templates])
        [[MessageSegment 4 bytes: 01020304... | values: [1, 2, 3...,
          MessageSegment 3 bytes: 020304 | values: [2, 3, 4],
          MessageSegment 3 bytes: 010304 | values: [1, 3, 4],
          MessageSegment 2 bytes: 0204 | values: [2, 4],
          MessageSegment 2 bytes: 0203 | values: [2, 3],
          MessageSegment 4 bytes: 46020304... | values: [70, 2, 3...,
          MessageSegment 4 bytes: 03020304... | values: [3, 2, 3...]]
        >>> pprint(clusters)
        {-1: [MessageSegment 7 bytes: 141e253245021e... | values: [20, 30, 37...,
              MessageSegment 3 bytes: 250545 | values: [37, 5, 69]],
         0: [MessageSegment 4 bytes: 01020304... | values: [1, 2, 3...,
             MessageSegment 3 bytes: 020304 | values: [2, 3, 4],
             MessageSegment 3 bytes: 010304 | values: [1, 3, 4],
             MessageSegment 2 bytes: 0204 | values: [2, 4],
             MessageSegment 2 bytes: 0203 | values: [2, 3],
             MessageSegment 4 bytes: 46020304... | values: [70, 2, 3...,
             MessageSegment 4 bytes: 03020304... | values: [3, 2, 3...]}

        >>> labels = [-1]*len(segments)
        >>> for i, t in enumerate(templates):
        ...     for s in t.baseSegments:
        ...         labels[segments.index(s)] = i
        ...     labels[segments.index(t.medoid)] = "({})".format(i)
        >>> from visualization.distancesPlotter import DistancesPlotter
        >>> sdp = DistancesPlotter(specimens, 'distances-testcase', True)
        >>> sdp.plotDistances(tg, numpy.array(labels))
        >>> sdp.writeOrShowFigure()


        :param kwargs: call the clusterer with these parameters. See getClusters().
        :return: A list of Templates for all clusters.
        """
        # retrieve all clusters and omit noise for template generation.
        allClusters = [cluster for label, cluster in self.getClusters(self.segments, **kwargs).items() if label > -1]
        return self.generateTemplatesForClusters(allClusters)





