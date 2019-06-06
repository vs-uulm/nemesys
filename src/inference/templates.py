from typing import List, Dict, Union, Iterable, Sequence, Tuple
import numpy, scipy.spatial, itertools
from pandas import DataFrame
from collections import Counter
from abc import ABC, abstractmethod

from inference.fieldTypes import FieldTypeMemento
from netzob.Model.Vocabulary.Messages.AbstractMessage import AbstractMessage

from inference.analyzers import MessageAnalyzer, Value
from inference.segments import MessageSegment, AbstractSegment, CorrelatedSegment, HelperSegment, TypedSegment


debug = False


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

    debug = False
    offsetCutoff = 6

    def __init__(self, segments: Iterable[AbstractSegment], method='canberra',
                 thresholdFunction = None, thresholdArgs = None,
                 reliefFactor=.33,
                 manipulateChars=True
                 ):
        """
        Determine the distance between the given segments.

        >>> from inference.analyzers import Value
        >>> from netzob.Model.Vocabulary.Messages.RawMessage import RawMessage
        >>>
        >>> bytedata = bytes([1,2,3,4])
        >>> message = RawMessage(bytedata)
        >>> analyzer = Value(message)
        >>> segments = [MessageSegment(analyzer, 0, 4)]
        >>> DistanceCalculator.debug = False
        >>> dc = DistanceCalculator(segments)
        <BLANKLINE>
        Calculated distances for 1 segment pairs in ... seconds.



        :param segments: The segments to calculate pairwise distances from.
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
        self._reliefFactor = reliefFactor
        self._offsetCutoff = DistanceCalculator.offsetCutoff
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
        # offset of inner segment to outer segment (key: tuple(i, o))
        self._offsets = dict()
        # distance matrix for all rows and columns in order of self._segments
        self._distances = DistanceCalculator._getDistanceMatrix(self._embdedAndCalcDistances(), len(self._quicksegments))

        # prepare lookup for matrix indices
        self._seg2idx = {seg: idx for idx, seg in enumerate(self._segments)}

        if manipulateChars:
            # Manipulate calculated distances for all char/char pairs.
            self._manipulateChars()


    @property
    def distanceMatrix(self) -> numpy.ndarray:
        """
        The order of the matrix elements in each row and column is the same as in self.segments.

        >>> from tabulate import tabulate
        >>> from utils.baseAlgorithms import generateTestSegments
        >>> segments = generateTestSegments()
        >>> DistanceCalculator.debug = False
        >>> dc = DistanceCalculator(segments)
         . . .
         . .
         .
        <BLANKLINE>
        Calculated distances for 37 segment pairs in ... seconds.
        >>> print((numpy.diag(dc.distanceMatrix) == 0).all())
        True
        >>> print(tabulate(dc.distanceMatrix))
        --------  --------  --------  ---------  ---------  --------  --------  -  --------
        0         0.214375  0.301667  0.440536   0.3975     0.808358  0.748036  1  0.125
        0.214375  0         0.111111  0.347593   0.297407   0.788166  0.679282  1  0.214375
        0.301667  0.111111  0         0.367667   0.414506   0.793554  0.695926  1  0.301667
        0.440536  0.347593  0.367667  0          0.0714286  0.695948  0.651706  1  0.440536
        0.3975    0.297407  0.414506  0.0714286  0          0.709111  0.700497  1  0.3975
        0.808358  0.788166  0.793554  0.695948   0.709111   0         0.576613  1  0.795818
        0.748036  0.679282  0.695926  0.651706   0.700497   0.576613  0         1  0.748036
        1         1         1         1          1          1         1         0  1
        0.125     0.214375  0.301667  0.440536   0.3975     0.795818  0.748036  1  0
        --------  --------  --------  ---------  ---------  --------  --------  -  --------

        :return: The normalized pairwise distances of all segments in this object represented as an symmetric array.
        """
        return self._distances

    def similarityMatrix(self) -> numpy.ndarray:
        # noinspection PyUnresolvedReferences
        """
        Converts the distances into similarities using the knowledge about distance method and analysis type.

        The order of the matrix elements in each row and column is the same as in self.segments.

        >>> from tabulate import tabulate
        >>> from netzob.Model.Vocabulary.Messages.RawMessage import RawMessage
        >>> from inference.analyzers import Value
        >>>
        >>> bytedata = [
        ...     bytes([1, 2, 3, 4]),
        ...     bytes([   2, 3, 4]),
        ...     bytes([   1, 3, 4]),
        ...     bytes([   2, 4   ]),
        ...     bytes([   2, 3   ]),
        ...     bytes([20, 30, 37, 50, 69, 2, 30]),
        ...     bytes([        37,  5, 69       ]),
        ...     bytes([0, 0, 0, 0]),
        ...     bytes([3, 2, 3, 4])
        ...     ]
        >>> messages  = [RawMessage(bd) for bd in bytedata]
        >>> analyzers = [Value(message) for message in messages]
        >>> segments  = [MessageSegment(analyzer, 0, len(analyzer.message.data)) for analyzer in analyzers]
        >>> DistanceCalculator.debug = False
        >>> dc = DistanceCalculator(segments)
         . . .
         . .
         .
        <BLANKLINE>
        Calculated distances for 37 segment pairs in ... seconds.
        >>> print((numpy.diag(dc.similarityMatrix()) == 1).all())
        True
        >>> print(tabulate(dc.similarityMatrix()))
        --------  --------  --------  --------  --------  --------  --------  -  --------
        1         0.785625  0.698333  0.559464  0.6025    0.191642  0.251964  0  0.875
        0.785625  1         0.888889  0.652407  0.702593  0.211834  0.320718  0  0.785625
        0.698333  0.888889  1         0.632333  0.585494  0.206446  0.304074  0  0.698333
        0.559464  0.652407  0.632333  1         0.928571  0.304052  0.348294  0  0.559464
        0.6025    0.702593  0.585494  0.928571  1         0.290889  0.299503  0  0.6025
        0.191642  0.211834  0.206446  0.304052  0.290889  1         0.423387  0  0.204182
        0.251964  0.320718  0.304074  0.348294  0.299503  0.423387  1         0  0.251964
        0         0         0         0         0         0         0         1  0
        0.875     0.785625  0.698333  0.559464  0.6025    0.204182  0.251964  0  1
        --------  --------  --------  --------  --------  --------  --------  -  --------


        :return: The pairwise similarities of all segments in this object represented as an symmetric array.
        """
        similarityMatrix = 1 - self._distances
        return similarityMatrix

    @property
    def segments(self) -> List[MessageSegment]:
        """
        :return: All segments in this object.
        """
        return self._segments

    @property
    def rawSegments(self):
        """
        :return: All segments in this object.
        """
        return self._segments

    @property
    def offsets(self) -> Dict[Tuple[int, int], int]:
        # noinspection PyUnresolvedReferences
        """
        >>> from tabulate import tabulate
        >>> import math
        >>> from netzob.Model.Vocabulary.Messages.RawMessage import RawMessage
        >>> from inference.analyzers import Value
        >>>
        >>> bytedata = [
        ...     bytes([1, 2, 3, 4]),
        ...     bytes([   2, 3, 4]),
        ...     bytes([   1, 3, 4]),
        ...     bytes([   2, 4   ]),
        ...     bytes([   2, 3   ]),
        ...     bytes([20, 30, 37, 50, 69, 2, 30]),
        ...     bytes([        37,  5, 69       ]),
        ...     bytes([0, 0, 0, 0]),
        ...     bytes([3, 2, 3, 4])
        ...     ]
        >>> messages  = [RawMessage(bd) for bd in bytedata]
        >>> analyzers = [Value(message) for message in messages]
        >>> segments  = [MessageSegment(analyzer, 0, len(analyzer.message.data)) for analyzer in analyzers]
        >>> DistanceCalculator.debug = False
        >>> dc = DistanceCalculator(segments)
         . . .
         . .
         .
        <BLANKLINE>
        Calculated distances for 37 segment pairs in ... seconds.
        >>> print(tabulate(dc.offsets.items()))
        ------  -
        (4, 7)  0
        (4, 6)  0
        (3, 2)  1
        (4, 8)  1
        (4, 5)  5
        (2, 8)  1
        (6, 0)  1
        (7, 5)  0
        (0, 5)  3
        (4, 2)  0
        (3, 7)  0
        (2, 5)  4
        (8, 5)  3
        (1, 0)  1
        (6, 5)  2
        (4, 0)  1
        (2, 7)  0
        (6, 7)  0
        (6, 8)  1
        (1, 5)  4
        (3, 1)  0
        (3, 0)  1
        (3, 8)  1
        (2, 0)  1
        (1, 8)  1
        (3, 6)  0
        (1, 7)  0
        (3, 5)  5
        (4, 1)  0
        ------  -
        >>> offpairs = [["-"*(off*2) + dc.segments[pair[0]].bytes.hex(), dc.segments[pair[1]].bytes.hex()]
        ...                     for pair, off in dc.offsets.items()]
        >>> for opsub in range(1, int(math.ceil(len(offpairs)/5))):
        ...     print(tabulate(map(list,zip(*offpairs[(opsub-1)*5:opsub*5])), numalign="left"))
        --------  ------  ------  --------  --------------
        0203      0203    --0204  --0203    ----------0203
        00000000  250545  010304  03020304  141e253245021e
        --------  ------  ------  --------  --------------
        --------  --------  --------------  --------------  ------
        --010304  --250545  00000000        ------01020304  0203
        03020304  01020304  141e253245021e  141e253245021e  010304
        --------  --------  --------------  --------------  ------
        --------  --------------  --------------  --------  --------------
        0204      --------010304  ------03020304  --020304  ----250545
        00000000  141e253245021e  141e253245021e  01020304  141e253245021e
        --------  --------------  --------------  --------  --------------
        --------  --------  --------  --------  --------------
        --0203    010304    250545    --250545  --------020304
        01020304  00000000  00000000  03020304  141e253245021e
        --------  --------  --------  --------  --------------
        ------  --------  --------  --------  --------
        0204    --0204    --0204    --010304  --020304
        020304  01020304  03020304  01020304  03020304
        ------  --------  --------  --------  --------

        :return: In case of mixed-length distances, this returns a mapping of segment-index pairs to the
        positive or negative offset of the smaller segment from the larger segment start position.
        """
        # is set in the constructor and should therefore be always valid.
        return self._offsets

    def segments2index(self, segmentList: Iterable[AbstractSegment]) -> List[int]:
        # noinspection PyUnresolvedReferences
        """
        Look up the indices of the given segments.

        >>> from utils.baseAlgorithms import generateTestSegments
        >>> segments = generateTestSegments()
        >>> DistanceCalculator.debug = False
        >>> dc = DistanceCalculator(segments)
         . . .
         . .
         .
        <BLANKLINE>
        Calculated distances for 37 segment pairs in ... seconds.
        >>> msgI = numpy.random.randint(len(dc.segments), size=10)
        >>> msgS = dc.segments2index([dc.segments[i] for i in msgI])
        >>> numpy.all([i==s for i, s in zip(msgI, msgS)])
        True

        :param segmentList: List of segments
        :return: List of indices for the given segments
        """
        return [self._seg2idx[seg] for seg in segmentList]


    def pairDistance(self, A: MessageSegment, B: MessageSegment) -> numpy.float64:
        # noinspection PyUnresolvedReferences
        """
                Retrieve the distance between two segments.

        >>> from itertools import combinations
        >>> from netzob.Model.Vocabulary.Messages.RawMessage import RawMessage
        >>> from inference.analyzers import Value
        >>>
        >>> bytedata = [
        ...     bytes([1, 2, 3, 4]),
        ...     bytes([   2, 3, 4]),
        ...     bytes([   1, 3, 4]),
        ...     bytes([   2, 4   ]),
        ...     bytes([   2, 3   ]),
        ...     bytes([20, 30, 37, 50, 69, 2, 30]),
        ...     bytes([        37,  5, 69       ]),
        ...     bytes([0, 0, 0, 0]),
        ...     bytes([3, 2, 3, 4])
        ...     ]
        >>> messages  = [RawMessage(bd) for bd in bytedata]
        >>> analyzers = [Value(message) for message in messages]
        >>> segments  = [MessageSegment(analyzer, 0, len(analyzer.message.data)) for analyzer in analyzers]
        >>> DistanceCalculator.debug = False
        >>> dc = DistanceCalculator(segments)
         . . .
         . .
         .
        <BLANKLINE>
        Calculated distances for 37 segment pairs in ... seconds.
        >>> for a,b in list(combinations(range(len(dc.segments)), 2)):
        ...     if not dc.pairDistance(segments[a], segments[b]) == dc.distanceMatrix[a,b]:
        ...         print("Failed")

        :param A: Segment A
        :param B: Segment B
        :return: Distance between A and B
        """
        a = self._seg2idx[A]
        b = self._seg2idx[B]
        return self._distances[a,b]

    def distancesSubset(self, As: Sequence[AbstractSegment], Bs: Sequence[AbstractSegment] = None) \
            -> numpy.ndarray:
        """
        Retrieve a matrix of pairwise distances for two lists of segments.

        >>> from utils.baseAlgorithms import generateTestSegments
        >>> segments = generateTestSegments()
        >>> DistanceCalculator.debug = False
        >>> dc = DistanceCalculator(segments)
         . . .
         . .
         .
        <BLANKLINE>
        Calculated distances for 37 segment pairs in ... seconds.
        >>> dc.distancesSubset(segments[:3], segments[-3:])
        array([[ 0.74803615,  1.        ,  0.125     ],
               [ 0.67928229,  1.        ,  0.214375  ],
               [ 0.69592646,  1.        ,  0.30166667]])
        >>> (dc.distancesSubset(segments[:3], segments[-3:]) == dc.distanceMatrix[:3,-3:]).all()
        True

        :param As: List of segments
        :param Bs: List of segments
        :return: Matrix of distances: As are rows, Bs are columns.
        """
        clusterI = As
        clusterJ = As if Bs is None else Bs
        simtrx = numpy.ones((len(clusterI), len(clusterJ)))

        transformatorK = dict()  # maps indices i from clusterI to matrix rows k
        for i, seg in enumerate(clusterI):
            transformatorK[i] = self._seg2idx[seg]
        if Bs is not None:
            transformatorL = dict()  # maps indices j from clusterJ to matrix cols l
            for j, seg in enumerate(clusterJ):
                transformatorL[j] = self._seg2idx[seg]
        else:
            transformatorL = transformatorK

        for i,k in transformatorK.items():
            for j,l in transformatorL.items():
                simtrx[i,j] = self._distances[k,l]
        return simtrx

    def groupByLength(self) -> Dict[int, List[Tuple[int, int, Tuple[float]]]]:
        """
        Groups segments by value length.

        Used in constructor.

        >>> from pprint import pprint
        >>> from utils.baseAlgorithms import generateTestSegments
        >>> segments = generateTestSegments()
        >>> DistanceCalculator.debug = False
        >>> dc = DistanceCalculator(segments)
         . . .
         . .
         .
        <BLANKLINE>
        Calculated distances for 37 segment pairs in ... seconds.
        >>> pprint(dc.groupByLength())
        {2: [(3, 2, (2, 4)), (4, 2, (2, 3))],
         3: [(1, 3, (2, 3, 4)), (2, 3, (1, 3, 4)), (6, 3, (37, 5, 69))],
         4: [(0, 4, (1, 2, 3, 4)), (7, 4, (0, 0, 0, 0)), (8, 4, (3, 2, 3, 4))],
         7: [(5, 7, (20, 30, 37, 50, 69, 2, 30))]}


        :return: a dict mapping the length to the list of MessageSegments of that length.
        """
        segsByLen = dict()
        for seg in self._quicksegments:
            seglen = seg[1]
            if seglen not in segsByLen:
                segsByLen[seglen] = list()
            segsByLen[seglen].append(seg)
        return segsByLen

    @staticmethod
    def __prepareValuesMatrix(segments: List[Tuple[int, int, Tuple[float]]], method) -> Tuple[str, numpy.ndarray]:
        # noinspection PyUnresolvedReferences,PyProtectedMember
        """
        Prepare a values matrix as input for distance calculation. This means extracting the values from all segments
        and placing them into an array. The preparation also includes handling of cosine zero vectors.

        >>> from tabulate import tabulate
        >>> testdata =  [(0, 4, (1, 2, 3, 4)),
        ...              (1, 3, (2, 3, 4)),
        ...              (2, 3, (1, 3, 4)),
        ...              (3, 2, (2, 4)),
        ...              (4, 2, (2, 3)),
        ...              (5, 7, (20, 30, 37, 50, 69, 2, 30)),
        ...              (6, 3, (37, 5, 69)),
        ...              (7, 4, (0, 0, 0, 0)),
        ...              (8, 4, (3, 2, 3, 4))]
        >>> prepMtx = DistanceCalculator._DistanceCalculator__prepareValuesMatrix(testdata, 'canberra')
        >>> print(prepMtx[0])
        canberra
        >>> print(tabulate(prepMtx[1]))
        --  --  --  --  --  -  --
         1   2   3   4
         2   3   4
         1   3   4
         2   4
         2   3
        20  30  37  50  69  2  30
        37   5  69
         0   0   0   0
         3   2   3   4
        --  --  --  --  --  -  --


        # TODO if this should become is a performance killer, drop support for cosine and correlation
          of unsuitable segments altogether

        :param segments: List of the indices and length and values in format of self._quicksegments
        :param method: The distance method to use
        :return: A tuple of the method-str (may have been adjusted) and the values matrix as numpy array.
        """
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
        # noinspection PyProtectedMember,PyTypeChecker
        """
        Calculates pairwise distances for all input segments.
        The values of all segments have to be of the same length!

        >>> from pprint import pprint
        >>> testdata =  [(0, 4, (1, 2, 3, 4)),
        ...              (1, 4, (70, 42, 12, 230)),
        ...              (2, 4, (0, 0, 0, 0)),
        ...              (3, 4, (1, 2, 3, 4)),
        ...              (4, 4, (3, 2, 3, 4))]
        >>> pprint(DistanceCalculator._calcDistances(testdata, 'canberra'))
        [(0, 1, 3.4467338608183682),
         (0, 2, 4.0),
         (0, 3, 0.0),
         (0, 4, 0.5),
         (1, 2, 4.0),
         (1, 3, 3.4467338608183682),
         (1, 4, 3.3927110940809571),
         (2, 3, 4.0),
         (2, 4, 4.0),
         (3, 4, 0.5)]

        :param segments: list of segments to calculate their similarity/distance for.
        :param method: The method to use for distance calculation. See scipy.spatial.distance.pdist.
            defaults to 'canberra'.
        :return: List of all pairwise distances between segements.
        """
        if DistanceCalculator.debug:
            import time
            # tPrep = time.time()
            # print('Prepare values.', end='')
        method, segmentValuesMatrix = DistanceCalculator.__prepareValuesMatrix(segments, method)

        # if DistanceCalculator.debug:
            # tPdist = time.time()  # Does not take a noticable amount of time: mostly some milliseconds
            # print(' {:.3f}s\ncall pdist from scipy.'.format(tPdist-tPrep), end='')
        if len(segments) == 1:
            return [(segments[0][0], segments[0][0], 0)]
        # This is the poodle's core
        segPairSimi = scipy.spatial.distance.pdist(segmentValuesMatrix, method)

        # if DistanceCalculator.debug:
        #     tExpand = time.time()
        #     print(' {:.3f}s\nExpand compressed pairs.'.format(tExpand-tPdist), end='')
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
        # if DistanceCalculator.debug:
        #     tFinal = time.time()  # Does not take a noticable amount of time: mostly some milliseconds, seldom about half a second
        #     print(" {:.3f}s".format(tFinal-tExpand))
        return segPairs

    @staticmethod
    def _getDistanceMatrix(distances: List[Tuple[int, int, float]], segmentCount: int) -> numpy.ndarray:
        # noinspection PyProtectedMember
        """
        Arrange the representation of the pairwise similarities of the input parameter in an symmetric array.
        The order of the matrix elements in each row and column is the same as in self._segments.

        Distances for pair not included in the list parameter are considered incomparable and set to -1 in the
        resulting matrix.

        Used in constructor.

        >>> from tabulate import tabulate
        >>> from inference.templates import DistanceCalculator
        >>> testdata =  [(3, 3, 0.0),
        ...              (0, 3, 0.80835755871765202),
        ...              (5, 3, 1.0),
        ...              (1, 3, 0.78816607689353413),
        ...              (4, 3, 0.57661277498012198),
        ...              (2, 3, 0.7091107871720117),
        ...              (0, 5, 1.0),
        ...              (1, 0, 0.21437499999999998),
        ...              (1, 5, 1.0),
        ...              (4, 0, 0.74803614550403941),
        ...              (4, 5, 1.0),
        ...              (2, 0, 0.39749999999999996),
        ...              (2, 5, 1.0),
        ...              (1, 4, 0.67928228544666891),
        ...              (2, 1, 0.2974074074074074),
        ...              (2, 4, 0.70049738841405507),
        ...              (2, 2, 0.0)]
        >>> dm = DistanceCalculator._getDistanceMatrix(testdata, 7)
        >>> print(tabulate(dm))
        ---------  ---------  ---------  ---------  ---------  --  --
         0          0.214375   0.3975     0.808358   0.748036   1  -1
         0.214375   0          0.297407   0.788166   0.679282   1  -1
         0.3975     0.297407   0          0.709111   0.700497   1  -1
         0.808358   0.788166   0.709111   0          0.576613   1  -1
         0.748036   0.679282   0.700497   0.576613   0          1  -1
         1          1          1          1          1          0  -1
        -1         -1         -1         -1         -1         -1   0
        ---------  ---------  ---------  ---------  ---------  --  --

        :param distances: The pairwise similarities to arrange.
        :return: The distance matrix for the given similarities.
            -1 for each undefined element, 0 in the diagonal, even if not given in the input.
        """
        from inference.segmentHandler import matrixFromTpairs
        simtrx = matrixFromTpairs([(ise[0], ise[1], ise[2]) for ise in distances], range(segmentCount),
                                  incomparable=-1)  # TODO hanlde incomparable values (resolve and replace the negative value)
        return simtrx


    @staticmethod
    def embedSegment(shortSegment: Tuple[int, int, Tuple[float]], longSegment: Tuple[int, int, Tuple[float]],
                     method='canberra'):
        # noinspection PyTypeChecker
        """
        Embed shorter segment in longer one to determine a "partial" similarity-based distance between the segments.
        Enumerates all possible shifts of overlaying the short above of the long segment and returns the minimum
        distance of any of the distance calculations performed for each of these overlays.

        >>> testdata = [(0, 4, [1, 2, 3, 4]),
        ...             (1, 3,    [2, 3, 4]),
        ...             (2, 3,    [1, 3, 4]),
        ...             (3, 2,       [2, 4]),
        ...             (4, 2,    [2, 3]),
        ...             (5, 7, [20, 30, 37, 50, 69, 2, 30]),
        ...             (6, 3,         [37,  5, 69]),
        ...             (7, 4, [0, 0, 0, 0]),
        ...             (8, 4, [3, 2, 3, 4])]
        >>> DistanceCalculator.embedSegment(testdata[1], testdata[0], method='canberra')
        ('canberra', 1, (1, 0, 0.0))
        >>> DistanceCalculator.embedSegment(testdata[2], testdata[0])
        ('canberra', 1, (2, 0, 0.33333333333333331))
        >>> DistanceCalculator.embedSegment(testdata[3], testdata[2])
        ('canberra', 1, (3, 2, 0.20000000000000001))
        >>> DistanceCalculator.embedSegment(testdata[6], testdata[0])
        ('canberra', 1, (6, 0, 2.037846856340007))
        >>> DistanceCalculator.embedSegment(testdata[7], testdata[5])
        ('canberra', 0, (7, 5, 4.0))
        >>> DistanceCalculator.embedSegment(testdata[3], testdata[0])
        ('canberra', 1, (3, 0, 0.14285714285714285))
        >>> DistanceCalculator.embedSegment(testdata[4], testdata[0])
        ('canberra', 1, (4, 0, 0.0))
        >>> DistanceCalculator.embedSegment(testdata[6], testdata[5])
        ('canberra', 2, (6, 5, 0.81818181818181823))

        # # TODO: these are test calls for validating embedSegment -> doctest?!
        # m, s, inters = DistanceCalculator.embedSegment(segsByLen[4][50], segsByLen[8][50])
        #
        # overlay = ([None] * s + inters.segA.values, inters.segB.values)
        # from visualization.singlePlotter import SingleMessagePlotter
        # smp = SingleMessagePlotter(specimens, "test feature embedding", True)
        # smp.plotAnalysis(overlay)
        # smp.writeOrShowFigure()

        :param shortSegment: The shorter of the two input segments.
        :param longSegment: The longer of the two input segments.
        :param method: The distance method to use.
        :return: The minimal partial distance between the segments, wrapped in an "InterSegment-Tuple":
            (method, offset, (
                index of shortSegment, index of longSegment, distance betweent the two
            ))
        """
        assert longSegment[1] > shortSegment[1]

        if DistanceCalculator.offsetCutoff is not None:
            maxOffset = min(DistanceCalculator.offsetCutoff, longSegment[1] - shortSegment[1])
        else:
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
    def sigmoidThreshold(x: float, shift=0.5):
        """
        Standard sigmoid threshold function to transform x.

        >>> DistanceCalculator.sigmoidThreshold(.42)
        0.31226136010788613

        >>> DistanceCalculator.sigmoidThreshold(.23)
        0.065083072323092628

        :param x: The (distance) value to transform.
        :param shift: The shift of the threshold between 0 and 1.
        :return: The transformed (distance) value.
        """
        return 1 / (1 + numpy.exp(-numpy.pi ** 2 * (x - shift)))

    @staticmethod
    def neutralThreshold(x: float):
        """
        Neutral dummy threshold function to NOT transform x.

        >>> DistanceCalculator.neutralThreshold(42.23)
        42.23

        :param x: The (distance) value NOT to transform.
        :return: The original (distance) value.
        """
        return x


    def _embdedAndCalcDistances(self) -> \
            List[Tuple[int, int, float]]:
        """
        Embed all shorter Segments into all larger ones and use the resulting pairwise distances to generate a
        complete distance list of all combinations of the into segment list regardless of their length.

        >>> from tabulate import tabulate
        >>> from utils.baseAlgorithms import generateTestSegments
        >>> segments = generateTestSegments()
        >>> DistanceCalculator.debug = False
        >>> dc = DistanceCalculator(segments)
         . . .
         . .
         .
        <BLANKLINE>
        Calculated distances for 37 segment pairs in ... seconds.
        >>> print(tabulate([
        ...     [ min(segments[0].length, segments[1].length), segments[0].values, segments[1].values, dc.pairDistance(segments[0], segments[1]) ],
        ...     [ min(segments[0].length, segments[2].length), segments[0].values, segments[2].values, dc.pairDistance(segments[0], segments[2]) ],
        ...     [ min(segments[2].length, segments[3].length), segments[2].values, segments[3].values, dc.pairDistance(segments[2], segments[3]) ],
        ...     [ min(segments[6].length, segments[0].length), segments[6].values, segments[0].values, dc.pairDistance(segments[6], segments[0]) ],
        ...     [ min(segments[5].length, segments[7].length), segments[5].values, segments[7].values, dc.pairDistance(segments[5], segments[7]) ],
        ...     [ min(segments[0].length, segments[3].length), segments[0].values, segments[3].values, dc.pairDistance(segments[0], segments[3]) ],
        ...     [ min(segments[0].length, segments[4].length), segments[0].values, segments[4].values, dc.pairDistance(segments[0], segments[4]) ],
        ...     [ min(segments[5].length, segments[6].length), segments[5].values, segments[6].values, dc.pairDistance(segments[5], segments[6]) ],
        ...     [ min(segments[0].length, segments[7].length), segments[0].values, segments[7].values, dc.pairDistance(segments[0], segments[7]) ],
        ...     [ min(segments[3].length, segments[4].length), segments[3].values, segments[4].values, dc.pairDistance(segments[3], segments[4]) ],
        ...     [ min(segments[0].length, segments[8].length), segments[0].values, segments[8].values, dc.pairDistance(segments[0], segments[8]) ],
        ... ]))
        ...
        -  ---------------------------  ------------  ---------
        3  [1, 2, 3, 4]                 [2, 3, 4]     0.214375
        3  [1, 2, 3, 4]                 [1, 3, 4]     0.301667
        2  [1, 3, 4]                    [2, 4]        0.367667
        3  [37, 5, 69]                  [1, 2, 3, 4]  0.748036
        4  [20, 30, 37, 50, 69, 2, 30]  [0, 0, 0, 0]  1
        2  [1, 2, 3, 4]                 [2, 4]        0.440536
        2  [1, 2, 3, 4]                 [2, 3]        0.3975
        3  [20, 30, 37, 50, 69, 2, 30]  [37, 5, 69]   0.576613
        4  [1, 2, 3, 4]                 [0, 0, 0, 0]  1
        2  [2, 4]                       [2, 3]        0.0714286
        4  [1, 2, 3, 4]                 [3, 2, 3, 4]  0.125
        -  ---------------------------  ------------  ---------


        :return: List of Tuples
            (index of segment in self._segments), (segment length), (Tuple of segment analyzer values)
        """
        lenGrps = self.groupByLength()  # segment list is in format of self._quicksegments

        import time
        pit_start = time.time()

        distance = list()  # type: List[Tuple[int, int, float]]
        rslens = list(reversed(sorted(lenGrps.keys())))  # lengths, sorted by decreasing length
        for outerlen in rslens:
            self._outerloop(lenGrps, outerlen, distance, rslens)

            # profiler = cProfile.Profile()
            # int_start = time.time()
            # stats = profiler.runctx('self._outerloop(lenGrps, outerlen, distance, rslens)', globals(), locals())
            # int_runtime = time.time() - int_start
            # profiler.dump_stats("embdedAndCalcDistances-{:02.1f}-{}-i{:03d}.profile".format(
            #     int_runtime, self.segments[0].message.data[:5].hex(), outerlen))


        runtime = time.time() - pit_start
        print("Calculated distances for {} segment pairs in {:.2f} seconds.".format(len(distance), runtime))
        return distance


    def _outerloop(self, lenGrps, outerlen, distance, rslens):
        """
        explicit function for the outer loop of _embdedAndCalcDistances() for profiling it

        :return:
        """
        outersegs = lenGrps[outerlen]
        if DistanceCalculator.debug:
            print("    outersegs, length {}, segments {}".format(outerlen, len(outersegs)))
        # a) for segments of identical length: call _calcDistancesPerLen()
        # TODO something outside of _calcDistances takes a lot longer to return during the embedding loop. Investigate.
        ilDist = DistanceCalculator._calcDistances(outersegs, method=self._method)
        # # # # # # # # # # # # # # # # # # # # # # # #
        distance.extend([(i, l,
                          self.thresholdFunction(
                              d * self._normFactor(outerlen),
                              **self.thresholdArgs))
                         for i, l, d in ilDist])
        # # # # # # # # # # # # # # # # # # # # # # # #
        # b) on segments with mismatching length: embedSegment:
        #       for all length groups with length < current length
        for innerlen in rslens[rslens.index(outerlen) + 1:]:
            innersegs = lenGrps[innerlen]
            if DistanceCalculator.debug:
                print("        innersegs, length {}, segments {}".format(innerlen, len(innersegs)))
            else:
                print(" .", end="", flush=True)
            # for all segments in "shorter length" group
            #     for all segments in current length group
            for iseg in innersegs:
                # TODO performance improvement: embedSegment directly generates six (offset cutoff) ndarrays per run
                # and gets called a lot of times: e.g. for dhcp-10000 511008 times alone for outerlen = 9 taking 50 sec.
                # :
                # instead prepare one values matrix for all outersegs of one innerseg iteration at once (the "embedded"
                # lengths are all of innerlen, so they are compatible for one single run of cdist). Remeber offset for
                # each "embedding subsequence", i.e. each first slice from the outerseg values.
                # this then is a list of embedding options (to take the minimum distance from) for all outersegs
                # embedded into the one innerseg.

                for oseg in outersegs:
                    # embedSegment(shorter in longer)
                    embedded = DistanceCalculator.embedSegment(iseg, oseg, self._method)
                    # add embedding similarity to list of InterSegments
                    interseg = embedded[2]

                    distEmb = interseg[2] * self._normFactor(innerlen)  # d_e
                    ratio = (outerlen - innerlen) / outerlen  # ratio = 1 - (l_i/l_o)
                    penalty = innerlen / outerlen ** 2
                    # self._reliefFactor  # f

                    mlDistance = \
                        (1 - ratio) * distEmb \
                        + ratio \
                        + ratio * penalty * (1 - distEmb) \
                        - self._reliefFactor * ratio * (1 - distEmb)

                    # # # # # # # # # # # # # # # # # # # # # # # #
                    dlDist = (interseg[0], interseg[1], (
                        self.thresholdFunction(
                            mlDistance,
                            **self.thresholdArgs)
                    )
                              )  # minimum of dimensions
                    # # # # # # # # # # # # # # # # # # # # # # # #
                    distance.append(dlDist)
                    self._offsets[(interseg[0], interseg[1])] = embedded[1]
        if not DistanceCalculator.debug:
            print()


    def _normFactor(self, dimensions: int):
        """
        For a distance value with given number of dimensions, calculate the factor to normalize this distance.
        For further parameters for this calculation, it uses this objects configuration.

        :param dimensions: The number of dimensions of the shorter segment
        :return: Factor to multiply to the corresponding distance to normalize it to 1.
        """
        return DistanceCalculator.normFactor(self._method, dimensions, self._segments[0].analyzer.domain)


    @staticmethod
    def normFactor(method: str, dimensions: int, analyzerDomain: Tuple[float, float]):
        """
        For a distance value with given parameters, calculate the factor to normalize this distance.
        This is static for cosine and correlation, but dynamic for canberra and euclidean measures.

        >>> DistanceCalculator.normFactor("euclidean", 4, (0.0, 2.0))
        0.125
        >>> DistanceCalculator.normFactor("canberra", 4, (0.0, 2.0))
        0.25
        >>> DistanceCalculator.normFactor("cosine", 4, (0.0, 2.0))
        1.0

        :param method: distance calculation method. One of cosine, correlation, canberra, euclidean, sqeuclidean
        :param dimensions: The number of dimensions of the shorter segment
        :param analyzerDomain: the value domain of the used analyzer
        :return: Factor to multiply to the corresponding distance to normalize it to 1.
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

    def neigbors(self, segment: AbstractSegment, subset: List[MessageSegment]=None) -> List[Tuple[int, float]]:
        # noinspection PyUnresolvedReferences
        """

        >>> from netzob.Model.Vocabulary.Messages.RawMessage import RawMessage
        >>> from inference.analyzers import Value
        >>>
        >>> bytedata = [
        ...     bytes([1, 2, 3, 4]),
        ...     bytes([   2, 3, 4]),
        ...     bytes([   1, 3, 4]),
        ...     bytes([   2, 4   ]),
        ...     bytes([   2, 3   ]),
        ...     bytes([20, 30, 37, 50, 69, 2, 30]),
        ...     bytes([        37,  5, 69       ]),
        ...     bytes([0, 0, 0, 0]),
        ...     bytes([3, 2, 3, 4])
        ...     ]
        >>> messages  = [RawMessage(bd) for bd in bytedata]
        >>> analyzers = [Value(message) for message in messages]
        >>> segments  = [MessageSegment(analyzer, 0, len(analyzer.message.data)) for analyzer in analyzers]
        >>> DistanceCalculator.debug = False
        >>> dc = DistanceCalculator(segments)
         . . .
         . .
         .
        <BLANKLINE>
        Calculated distances for 37 segment pairs in ... seconds.
        >>> nbrs = dc.neigbors(segments[2], segments[3:7])
        >>> dsts = [dc.pairDistance(segments[2], segments[3]),
        ...         dc.pairDistance(segments[2], segments[4]),
        ...         dc.pairDistance(segments[2], segments[5]),
        ...         dc.pairDistance(segments[2], segments[6])]
        >>> nbrs
        [(0, 0.3676666666666667), (1, 0.41450617283950619), (3, 0.69592645998558034), (2, 0.7935542543548032)]
        >>> [dsts[a] for a,b in nbrs] == [a[1] for a in nbrs]
        True

        :param segment: Segment to get the neigbors for.
        :param subset: The subset of MessageSegments to use from this DistanceCalculator object, if any.
        :return: An ascendingly sorted list of neighbors of parameter segment
            from all the segments in this object (if subset is None)
            or from the segments in subset.
            The result is a list of tuples with
                * the index of the neigbor (from self.segments or the subset list, respectively) and
                * the distance to this neighbor
        """
        home = self._seg2idx[segment]
        if subset:
            mask = self.segments2index(subset)
            assert len(mask) == len(subset)
        else:
            # mask self identity by "None"-value if applicable
            mask = list(range(home)) + [None] + list(range(home + 1, self._distances.shape[0]))

        candNeigbors = self._distances[:, home]
        neighbors = sorted([(nidx, candNeigbors[n]) for nidx, n in enumerate(mask) if n is not None],
                           key=lambda x: x[1])
        return neighbors

    @staticmethod
    def _checkCacheFile(analysisTitle: str, tokenizer: str, pcapfilename: str):
        from os.path import splitext, basename, exists
        pcapName = splitext(basename(pcapfilename))[0]
        dccachefn = 'cache-dc-{}-{}-{}.{}'.format(analysisTitle, tokenizer, pcapName, 'dc')
        if not exists(dccachefn):
            return False, dccachefn
        else:
            return True, dccachefn

    @staticmethod
    def loadCached(analysisTitle: str, tokenizer: str, pcapfilename: str) -> Tuple[
            List[Tuple[MessageSegment]], 'MessageComparator', 'DistanceCalculator'
    ]:
        """
        Loads a cached DistanceCalculator instance from the filesystem. If none is found or the requested parameters
        differ, return None.

        :return: A cached DistanceCalculator or None
        """
        import pickle
        from validation.dissectorMatcher import MessageComparator

        dccacheexists, dccachefn = DistanceCalculator._checkCacheFile(analysisTitle, tokenizer, pcapfilename)

        if dccacheexists:
            print("Load distances from cache file {}".format(dccachefn))
            segmentedMessages, comparator, dc = pickle.load(open(dccachefn, 'rb'))
            if not (isinstance(comparator, MessageComparator)
                    and isinstance(dc, DistanceCalculator)
                    and isinstance(segmentedMessages, List)
            ):
                raise TypeError("Cached objects in file " + dccachefn + " are of unexpected type.")
            return segmentedMessages, comparator, dc
        else:
            raise FileNotFoundError("Cache file " + dccachefn + " not found.")

    def saveCached(self, analysisTitle: str, tokenizer: str,
                   comparator: 'MessageComparator', segmentedMessages: List[Tuple[MessageSegment]]):
        """
        cache the DistanceCalculator and necessary auxiliary objects comparator and segmentedMessages to the filesystem

        :param analysisTitle:
        :param tokenizer:
        :param comparator:
        :param segmentedMessages:
        :return:
        """
        import pickle
        dccacheexists, dccachefn = DistanceCalculator._checkCacheFile(
            analysisTitle, tokenizer, comparator.specimens.pcapFileName)
        if not dccacheexists:
            with open(dccachefn, 'wb') as f:
                pickle.dump((segmentedMessages, comparator, self), f, pickle.HIGHEST_PROTOCOL)
        else:
            raise FileExistsError("Cache file" + dccachefn + " already exists. Abort saving.")


    def findMedoid(self, segments: List[AbstractSegment]) -> AbstractSegment:
        """
        Find the medoid closest describing the given list of segments.

        :param segments: a list of segments that must be known to this template generator.
        :return: The MessageSegment from segments that is the list's medoid.
        """
        distSubMatrix = self.distancesSubset(segments)
        mid = distSubMatrix.sum(axis=1).argmin()
        return segments[mid]

    def _manipulateChars(self, charMatchGain = .5):
        """
        Manipulate (decrease) calculated distances for all char/char pairs.

        :param charMatchGain: Factor to multiply to each distance of a chars-chars pair in self.distanceMatrix.
            try 0.33 or 0.5 or x
        :return:
        """
        from itertools import combinations
        from inference.segmentHandler import filterChars

        assert all((isinstance(seg, AbstractSegment) for seg in self.segments))
        charsequences = filterChars(self.segments)
        charindices = self.segments2index(charsequences)

        # for all combinations of pairs from charindices
        for a, b in combinations(charindices, 2):
            # decrease distance by factor
            self._distances[a, b] = self._distances[a, b] * charMatchGain
            self._distances[b, a] = self._distances[b, a] * charMatchGain



class Template(AbstractSegment):
    """
    Represents a template for some group of similar MessageSegments
    A Templates values are either the values of a medoid or the mean of values per vector component.
    """

    def __init__(self, values: Union[List[Union[float, int]], numpy.ndarray, MessageSegment],
                 baseSegments: Iterable[AbstractSegment],
                 method='canberra'):
        """
        :param values: The values of the template (e. g. medoid or mean)
        :param baseSegments: The Segments this template represents and is based on.
        :param method: The distance method to use for calculating further distances if necessary.
        """
        super().__init__()
        if isinstance(values, MessageSegment):
            self._values = values.values
            self.medoid = values
        else:
            self._values = values
            self.medoid = None
        self.baseSegments = list(baseSegments)  # list/cluster of MessageSegments this template was generated from.
        self.checkSegmentsAnalysis()
        self._method = method
        self.length = len(self._values)


    @property
    def bytes(self):
        return bytes(self._values) if isinstance(self._values, Iterable) else None


    @property
    def analyzer(self):
        return self.baseSegments[0].analyzer


    def checkSegmentsAnalysis(self):
        """
        Validate that all base segments of this tempalte are configured with the same type of analysis.

        :raises: ValueError if not all analysis types and parameters of the base segments are identical.
        :return: Doesn't return anything if all is well. Raises a ValueError otherwise.
        """
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
        """
        TODO This is an old experimental method. Remove it?

        :param haystack:
        :param method:
        :return:
        """
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
        Calculate the normalized distances to the template (cluster center, not medoid).

        Only supports single length segments and does not support a threshold function.

        :return: Array of distances
        """
        baseValues = numpy.array([seg.values for seg in self.baseSegments])
        assert baseValues.shape[1] == self.values.shape[0], "cannot calc distances to mixed lengths segments"
        cenDist = scipy.spatial.distance.cdist(self.values.reshape((1, self.values.shape[0])),
                                               baseValues, self._method)
        normf = DistanceCalculator.normFactor(self._method, len(self.values), self.baseSegments[0].analyzer.domain)
        return normf * cenDist


    def distancesToMixedLength(self, dc: DistanceCalculator=None):
        # noinspection PyTypeChecker
        """
        Get distances to the medoid of this template.
        If no DistanceCalculator is given. does not support a threshold function.

        >>> from tabulate import tabulate
        >>> from scipy.spatial.distance import cdist
        >>> from utils.baseAlgorithms import generateTestSegments
        >>> from inference.templates import DistanceCalculator, Template
        >>> listOfSegments = generateTestSegments()
        >>> DistanceCalculator.debug = False
        >>> dc = DistanceCalculator(listOfSegments)
         . . .
         . .
         .
        <BLANKLINE>
        Calculated distances for 37 segment pairs in ... seconds.
        >>> center = [1,3,7,9]  # "mean"
        >>> tempearly = Template(center, listOfSegments)
        >>> dtml = tempearly.distancesToMixedLength(dc)
         . . .
         . .
         .
        <BLANKLINE>
        Calculated distances for 46 segment pairs in ... seconds.
        >>> cdist([center], [listOfSegments[0].values], "canberra")[0,0]/len(center) == dtml[0][0]
        True
        >>> print(tabulate(dtml))
        --------  --
        0.246154   0
        0.373087   0
        0.285795   0
        0.539909   1
        0.497917   0
        0.825697  -3
        0.682057   1
        1          0
        0.371154   0
        --------  --
        >>> center = listOfSegments[0]  # "medoid"
        >>> template = Template(center, listOfSegments)
        >>> print(tabulate(template.distancesToMixedLength(dc)))
        --------  --
        0          0
        0.214375   1
        0.301667   1
        0.440536   1
        0.3975     1
        0.808358  -3
        0.748036   1
        1          0
        0.125      0
        --------  --

        :param dc: DistanceCalculator to look up precomputed distances and offsets
        :return: List of Tuples of distances and offsets: -n for center, +n for segment
        """
        # create a map from the length groups to the baseSegments index
        lengthGroups = dict()
        for idx, bs in enumerate(self.baseSegments):
            if bs.length not in lengthGroups:
                lengthGroups[bs.length] = list()
            lengthGroups[bs.length].append(idx)
        if not (self.medoid and dc):
            # calculate distance to given template center values
            centerAnalysis = Value(AbstractMessage(bytes(self.values)))
            center = HelperSegment(centerAnalysis, 0, len(self.values))
            center.values = self.values
            dc = DistanceCalculator(self.baseSegments + [center], self._method)
        else:
            center = self.medoid

        # look up the distances between the base segments
        distances = dc.distancesSubset([center], self.baseSegments)
        offsets = [0] * len(self.baseSegments)
        medoidIndex = dc.segments2index([center])[0]
        globalIndices = [dc.segments2index([bs])[0] for bs in self.baseSegments]
        # collect offsets
        for bs, gs in enumerate(globalIndices):
            if (medoidIndex, gs) in dc.offsets:
                offsets[bs] = -dc.offsets[(medoidIndex, gs)]  # TODO check sign
            if (gs, medoidIndex) in dc.offsets:
                offsets[bs] = dc.offsets[(gs, medoidIndex)]
        distOffs = [(distances[0, idx], offsets[idx]) for idx in range(len(self.baseSegments))]
        return distOffs


    def maxDistToMedoid(self, dc: DistanceCalculator = None):
        return max(self.distancesToMixedLength(dc))[0]


    def distToNearest(self, segment: Union[MessageSegment, Sequence[MessageSegment]], dc: DistanceCalculator = None):
        if isinstance(segment, Sequence):
            segments = segment
        else:
            segments = [segment]
        return dc.distancesSubset(segments, self.baseSegments).min()

    def __hash__(self):
        """
        :return: Hash-representation of the template based on the values of it.
            (Templates are considered equal if their values are equal!)
        """
        return hash(tuple(self.values))


    def toColor(self):
        """
        The colors are not collision resistant.

        :return: A fixed-length color-coded visual representation of this template,
            NOT representing the template value itself!
        """
        oid = hash(self)
        # return '{:02x}'.format(oid % 0xffff)
        import visualization.bcolors as bcolors
        # Template
        return bcolors.colorizeStr('{:02x}'.format(oid % 0xffff), oid % 0xff)

    def __repr__(self):
        if self.values is not None and isinstance(self.values, (list, tuple)) and len(self.values) > 3:
            printValues = str(self.values[:3])[:-1] + '...'
        else:
            printValues = str(self.values)

        return "Template {} bytes: {} | #base {}".format(self.length, printValues, len(self.baseSegments))



class TypedTemplate(Template):

    def __init__(self, values: Union[List[Union[float, int]], numpy.ndarray, MessageSegment],
                 baseSegments: Iterable[AbstractSegment],
                 method='canberra'):
        from inference.segments import TypedSegment

        super().__init__(values, baseSegments, method)
        ftypes = {bs.fieldtype for bs in baseSegments if isinstance(bs, TypedSegment)}
        fcount = len(ftypes)
        if fcount == 1:
            self._fieldtype = ftypes.pop()
        elif fcount > 1:
            self._fieldtype = "[mixed]"
        else:
            self._fieldtype = "[unknown]"

    @property
    def fieldtype(self) -> str:
        """
        :return: One of the types defined in ParsedMessage.ParsingConstants.TYPELOOKUP
        """
        return self._fieldtype

    @fieldtype.setter
    def fieldtype(self, value: str):
        """
        mark segment with its true type

        :param value: One of the types defined in ParsedMessage.ParsingConstants.TYPELOOKUP
        """
        self._fieldtype = value



class FieldTypeTemplate(TypedTemplate, FieldTypeMemento):

    def __init__(self, baseSegments: Iterable[AbstractSegment], method='canberra'):
        """
        A new FieldTypeTemplate for the collection of base segments given.

        Per vector component, the mean, stdev, and the covariance matrix is calculated. Therefore the collection needs
        to be represented by a vector of one common vector space and thus a fixed number of dimensions.
        Thus for the calculation:
            * zero-only segments are ignored
            *


        :param baseSegments:
        :param method:
        """
        self.baseSegments = list(baseSegments)
        """:type List[AbstractSegment]"""
        self._baseOffsets = dict()
        relevantSegs = [seg for seg in self.baseSegments if set(seg.values) != {0}]
        segLens = {seg.length for seg in relevantSegs}

        if len(segLens) == 1:
            # all segments have equal length, so we simply can create an array from all of them
            segV = numpy.array([seg.values for seg in relevantSegs])
        elif len(segLens) > 1:
            # find the optimal shift/offset of each shorter segment to match the longest ones
            #   with shortest distance according to the method
            self._maxLen = max(segLens)
            # Better than the longest would be the most common length, but that would increase complexity a lot and
            #   we assume that for most use cases the longest segments will be the most frequent length.
            maxLenSegs = [(idx, seg.length, seg.values) for idx, seg in enumerate(relevantSegs, 1)
                          if seg.length == self._maxLen]
            segE = list()
            for seg in baseSegments:
                if set(seg.values) == {0}:
                    continue
                if seg.length < self._maxLen:
                    shortSeg = (0, seg.length, tuple(seg.values))
                    offsets = [DistanceCalculator.embedSegment(shortSeg, longSeg, method)[1] for longSeg in maxLenSegs]
                    offCount = Counter(offsets)
                    self._baseOffsets[seg] = offCount.most_common(1)[0][0]
                    paddedVals = self.paddedValues(seg)
                    if debug:
                        from tabulate import tabulate
                        print(tabulate((maxLenSegs[0][2], paddedVals)))
                    segE.append(paddedVals)
                else:
                    segE.append(list(seg.values))
            segV = numpy.array(segE)
        else:
            # handle situation when no base segment is relevant (all are zeros)
            self._maxLen = max({seg.length for seg in self.baseSegments})
            for bs in self.baseSegments:
                if bs.length == self._maxLen:
                    self._mean = numpy.array(bs.values)
                    self._stdev = numpy.zeros(self._mean.shape)
                    self._cov = numpy.zeros((self._mean.shape[0], self._mean.shape[0]))
                    break
            if not (isinstance(self._mean, numpy.ndarray) and isinstance(self._stdev, numpy.ndarray)
                    and isinstance(self._cov, numpy.ndarray)):
                raise RuntimeError("This collection of base segments is not suited to generate a FieldTypeTemplate.")
            super().__init__(self._mean, self.baseSegments, method)
            return

        self._mean = numpy.nanmean(segV, 0)
        self._stdev = numpy.nanstd(segV, 0)

        # for all components that have a stdev of 0 we need to wobble the values (randomly) to derive a
        #   covariance matrix that shows no linear dependent entries ("don't care positions")
        assert segV.shape == (len(relevantSegs), len(self._stdev))
        if any(self._stdev == 0):
            segV = segV.astype(float)
        for compidx, compstdev in enumerate(self._stdev):
            if compstdev == 0:
                segV[:,compidx] = numpy.random.random_sample((len(relevantSegs),)) * .5

        # print(segV)

        # self._cov = numpy.cov(segV, rowvar=False)
        # pandas cov allows for nans, numpy not
        pd = DataFrame(segV)
        self._cov = pd.cov().values
        self._picov = None  # fill on demand

        assert len(self._mean) == len(self._stdev) == len(self._cov.diagonal())
        super().__init__(self._mean, self.baseSegments, method)


    def paddedValues(self, segment: AbstractSegment):
        """
        :param segment: The base segment to get the padded values for
        :return: The values of the given base segment padded with nans to the length of the
            longest segment represented by this template.
        """
        shift = self._baseOffsets[segment] if segment in self._baseOffsets else 0
        vals = [numpy.nan] * shift + list(segment.values) + [numpy.nan] * (self._maxLen - shift - segment.length)
        return vals



class TemplateGenerator(object):
    """
    Generate templates for a list of segments according to their distance.
    """

    def __init__(self, dc: DistanceCalculator, clusterer: 'AbstractClusterer'):
        """
        Find similar segments, group them, and return a template for each group.

        :param dc: Segment distances to base the templates on.
        """
        self._dc = dc
        self._clusterer = clusterer

    @property
    def distanceCalculator(self):
        return self._dc

    @property
    def clusterer(self):
        return self._clusterer

    @staticmethod
    def generateTemplatesForClusters(dc: DistanceCalculator, segmentClusters: Iterable[List[MessageSegment]], medoid=True) \
            -> List[Template]:
        """
        Find templates representing the message segments in the input clusters.

        :param dc: Distance calculator to generate templates with
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
                center = dc.findMedoid(cluster)
            templates.append(Template(center, cluster))
        return templates


    def generateTemplates(self) -> List[Template]:
        # noinspection PyUnresolvedReferences
        """
        Generate templates for all clusters. Triggers a new clustering run.

        >>> from pprint import pprint
        >>> from netzob.Model.Vocabulary.Messages.RawMessage import RawMessage
        >>> from utils.loader import BaseLoader
        >>> from inference.analyzers import Value
        >>>
        >>> bytedata = [
        ...     bytes([1, 2, 3, 4]),
        ...     bytes([   2, 3, 4]),
        ...     bytes([   1, 3, 4]),
        ...     bytes([   2, 4   ]),
        ...     bytes([   2, 3   ]),
        ...     bytes([20, 30, 37, 50, 69, 2, 30]),
        ...     bytes([        37,  5, 69       ]),
        ...     bytes([70, 2, 3, 4]),
        ...     bytes([3, 2, 3, 4])
        ...     ]
        >>> messages  = [RawMessage(bd) for bd in bytedata]
        >>> analyzers = [Value(message) for message in messages]
        >>> segments  = [MessageSegment(analyzer, 0, len(analyzer.message.data)) for analyzer in analyzers]
        >>> specimens = BaseLoader(messages)
        >>> DistanceCalculator.debug = False
        >>> dc = DistanceCalculator(segments, thresholdFunction=DistanceCalculator.neutralThreshold, thresholdArgs=None)
         . . .
         . .
         .
        <BLANKLINE>
        Calculated distances for 37 segment pairs in ... seconds.
        >>> clusterer = DBSCANsegmentClusterer(dc, eps=1.0, min_samples=3)
        >>> tg = TemplateGenerator(dc, clusterer)
        >>> templates = tg.generateTemplates()
        DBSCAN epsilon: 1.0 minpts: 3
        >>> pprint([t.baseSegments for t in templates])
        [[MessageSegment 4 bytes: 01020304... | values: [1, 2, 3...,
          MessageSegment 3 bytes: 020304 | values: [2, 3, 4],
          MessageSegment 3 bytes: 010304 | values: [1, 3, 4],
          MessageSegment 2 bytes: 0204 | values: [2, 4],
          MessageSegment 2 bytes: 0203 | values: [2, 3],
          MessageSegment 4 bytes: 46020304... | values: [70, 2, 3...,
          MessageSegment 4 bytes: 03020304... | values: [3, 2, 3...]]
        >>> pprint(clusterer.getClusters())
        DBSCAN epsilon: 1.0 minpts: 3
        {-1: [MessageSegment 7 bytes: 141e253245021e... | values: [20, 30, 37...,
              MessageSegment 3 bytes: 250545 | values: [37, 5, 69]],
         0: [MessageSegment 4 bytes: 01020304... | values: [1, 2, 3...,
             MessageSegment 3 bytes: 020304 | values: [2, 3, 4],
             MessageSegment 3 bytes: 010304 | values: [1, 3, 4],
             MessageSegment 2 bytes: 0204 | values: [2, 4],
             MessageSegment 2 bytes: 0203 | values: [2, 3],
             MessageSegment 4 bytes: 46020304... | values: [70, 2, 3...,
             MessageSegment 4 bytes: 03020304... | values: [3, 2, 3...]}

        # Example, not to run by doctest:
        #
        labels = [-1]*len(segments)
        for i, t in enumerate(templates):
            for s in t.baseSegments:
                labels[segments.index(s)] = i
            labels[segments.index(t.medoid)] = "({})".format(i)
        from visualization.distancesPlotter import DistancesPlotter
        sdp = DistancesPlotter(specimens, 'distances-testcase', True)
        sdp.plotSegmentDistances(tg, numpy.array(labels))
        sdp.writeOrShowFigure()


        :return: A list of Templates for all clusters.
        """
        # retrieve all clusters and omit noise for template generation.
        allClusters = [cluster for label, cluster in self._clusterer.getClusters().items() if label > -1]
        return TemplateGenerator.generateTemplatesForClusters(self._dc, allClusters)








# # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # Clusterer classes # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

class AbstractClusterer(ABC):
    """
    Wrapper for any clustering implementation to select and adapt the autoconfiguration of the parameters.
    """
    def __init__(self, dc: DistanceCalculator, segments: Sequence[MessageSegment] = None):
        """

        :param dc:
        :param segments: subset of segments from dc to cluster, use all segments in dc if None
        """
        self._dc = dc  # type: DistanceCalculator
        if segments is None:
            self._distances = dc.distanceMatrix
            self._segments = dc.segments
        else:
            self._segments = segments
            self._distances = self._dc.distancesSubset(segments)

    @property
    def segments(self):
        return self._segments

    @property
    def distanceCalculator(self):
        return self._dc

    def clusterSimilarSegments(self, filterNoise=True) -> List[List[MessageSegment]]:
        """
        Find suitable discrimination between dissimilar segments.

        Works on representatives for segments of identical features.

        :param: if filterNoise is False, the first element in the returned list of clusters
            always is the (possibly empty) noise.
        :return: clusters of similar segments
        """
        clusters = self.getClusters()
        if filterNoise and -1 in clusters: # omit noise
            del clusters[-1]
        clusterlist = [clusters[l] for l in sorted(clusters.keys())]
        return clusterlist

    def getClusters(self) -> Dict[int, List[MessageSegment]]:
        """
        Do the initialization of the clusterer and perform the clustering of the list of segments contained in the
        distance calculator.

        Works on representatives for segments of identical features.

        :return: A dict of labels to lists of segments with that label.
        """
        try:
            labels = self.getClusterLabels()
        except ValueError as e:
            print(self._segments)
            # import tabulate
            # print(tabulate.tabulate(similarities))
            raise e
        assert isinstance(labels, numpy.ndarray)
        ulab = set(labels)

        segmentClusters = dict()
        for l in ulab:
            class_member_mask = (labels == l)
            segmentClusters[l] = [seg for seg in itertools.compress(self._segments, class_member_mask)]
        return segmentClusters

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
        # noinspection PyUnresolvedReferences,PyProtectedMember
        """
        see also DistanceCalculator.neighbors()
        In comparison to the general implementation in DistanceCalculator, this one does not return a sorted list,
        but just the closest neighbor and its index for all segments.

        numpy.array([[0.0, 0.5, 0.8],[0.1, 0.0, 0.9],[0.7,0.3,0.0]]

        This test uses a non-symmetric matrix to detect bugs if any. This is NOT a use case example!
        >>> from pprint import pprint
        >>> from netzob.Model.Vocabulary.Messages.RawMessage import RawMessage
        >>> from utils.loader import BaseLoader
        >>> from inference.analyzers import Value
        >>>
        >>> bytedata = [
        ...     bytes([1, 2, 3, 4]),
        ...     bytes([1, 2      ]),
        ...     bytes([3, 2, 3, 4])
        ...     ]
        >>> messages  = [RawMessage(bd) for bd in bytedata]
        >>> analyzers = [Value(message) for message in messages]
        >>> segments  = [MessageSegment(analyzer, 0, len(analyzer.message.data)) for analyzer in analyzers]
        >>> specimens = BaseLoader(messages)
        >>> DistanceCalculator.debug = False
        >>> dc = DistanceCalculator(segments)
         .
        <BLANKLINE>
        Calculated distances for 4 segment pairs in ... seconds.
        >>> clusterer = DBSCANsegmentClusterer(dc, eps=1, min_samples=2)
        >>> print(clusterer.distances)
        [[ 0.        0.3975    0.125   ]
         [ 0.3975    0.        0.548125]
         [ 0.125     0.548125  0.      ]]
        >>> clusterer._nearestPerNeigbor()
        [(2, 0.125), (0, 0.397499...), (0, 0.125)]

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
        npn = [self._dc.neigbors(seg) for seg in self._dc.segments]
        # iterate all the k-th neigbors up to 2 * log(#neigbors)
        dpnmln = list()
        for k in range(0, len(npn) - 1):
            kthNeigbors4is = [idn[k][1] for idn in npn if idn[k][1] > 0][:2 * lnN]
            if len(kthNeigbors4is) > 0:
                dpnmln.append(numpy.mean(kthNeigbors4is))
            else:
                dpnmln.append(numpy.nan)

        # enumerate the means of deltas starting from an offset of log(#neigbors)
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



class HDBSCANsegmentClusterer(AbstractClusterer):
    """
    Hierarchical Density-Based Spatial Clustering of Applications with Noise

    https://github.com/scikit-learn-contrib/hdbscan
    """

    def __init__(self, dc: DistanceCalculator, **kwargs):
        """

        :param dc:
        :param kwargs: e. g. epsilon: The DBSCAN epsilon value, if it should be fixed.
            If not given (None), it is autoconfigured.
        """
        super().__init__(dc)

        if len(kwargs) == 0:
            # from math import log
            # lnN = round(log(self.distances.shape[0]))
            self.min_cluster_size = self.steepestSlope()[0] # round(lnN * 1.5)
        elif 'min_cluster_size' in kwargs:
            self.min_cluster_size = kwargs['min_cluster_size']
        else:
            raise ValueError("Parameters for HDBSCAN without autoconfiguration missing. "
                             "Requires min_cluster_size.")
        self.min_samples = 3


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
        return dbscan.labels_


    def __repr__(self):
        return 'HDBSCAN mcs {} ms {}'.format(self.min_cluster_size, self.min_samples)



class DBSCANsegmentClusterer(AbstractClusterer):
    """
    Wrapper for DBSCAN from the sklearn.cluster module including autoconfiguration of the parameters.
    """

    def __init__(self, dc: DistanceCalculator, **kwargs):
        """
        :param dc:
        :param kwargs: e. g. epsilon: The DBSCAN epsilon value, if it should be fixed.
            If not given (None), it is autoconfigured.
        """
        super().__init__(dc)
        if len(kwargs) == 0:
            self.min_samples, self.eps = self._autoconfigure()
        else:
            if not 'eps' in kwargs or not 'min_samples' in kwargs:
                raise ValueError("Parameters for DBSCAN without autoconfiguration missing. "
                                 "Requires epsilon and min_cluster_size.")
            self.min_samples, self.eps = kwargs['min_samples'], kwargs['eps']

    def getClusterLabels(self) -> numpy.ndarray:
        """
        Cluster the entries in the similarities parameter by DBSCAN
        and return the resulting labels.

        :return: (numbered) cluster labels for each segment in the order given in the (symmetric) distance matrix
        """
        import sklearn.cluster

        if numpy.count_nonzero(self._distances) == 0:  # the distance matrix contains only identical segments
            return numpy.zeros_like(self._distances[0], int)

        dbscan = sklearn.cluster.DBSCAN(eps=self.eps, min_samples=self.min_samples)
        print("DBSCAN epsilon: {:0.3f}, minpts: {}".format(self.eps, self.min_samples))
        dbscan.fit(self._distances)
        return dbscan.labels_


    def __repr__(self):
        return 'DBSCAN eps {:0.3f} mpt'.format(self.eps, self.min_samples) \
            if self.eps and self.min_samples \
            else 'DBSCAN unconfigured (need to set epsilon and min_samples)'


    def _autoconfigure(self):
        """
        Auto configure the clustering parameters epsilon and minPts regarding the input data

        :return: min_samples, epsilon
        """
        return self._autoconfigureKneedle()


    def _autoconfigureMPC(self):
        """
        Auto configure the clustering parameters epsilon and minPts regarding the input data
        Maximum Positive Curvature

        :return: min_samples, epsilon
        """
        from utils.baseAlgorithms import autoconfigureDBSCAN
        neighbors = [self.distanceCalculator.neigbors(seg) for seg in self.distanceCalculator.segments]
        epsilon, min_samples, k = autoconfigureDBSCAN(neighbors)
        print("eps {:0.3f} autoconfigured from k {}".format(epsilon, k))
        return min_samples, epsilon


    def _maximumPositiveCurvature(self):
        """
        Use implementation of utils.baseAlgorithms to determine the maximum positive curvature
        :return: k, min_samples
        """
        from utils.baseAlgorithms import autoconfigureDBSCAN
        e, min_samples, k = autoconfigureDBSCAN(
            [self.distanceCalculator.neigbors(seg) for seg in self.distanceCalculator.segments])
        return k, min_samples


    def _autoconfigureKneedle(self):
        """
        Auto configure the clustering parameters epsilon and minPts regarding the input data

        knee is too far right/value too small to be useful:
            the clusters are small/zero size and few, perhaps density function too uneven in this use case?
        So we added a margin. Here selecting
            low factors resulted in only few small clusters. The density function seems too uneven for DBSCAN/Kneedle.

        :return: minpts, epsilon
        """
        # min_samples, k = self.steepestSlope()
        k, min_samples = self._maximumPositiveCurvature()
        print("dists of", self._distances.shape[0], "neighbors, k", k, "min_samples", min_samples)

        # get minpts-nearest-neighbor distance:
        neighdists = self._knearestdistance(  # add a margin relative to the remaining interval to the number of neigbors
            round(k + (self._distances.shape[0] - 1 - k) * .2))
            # round(minpts + 0.5 * (self.distances.shape[0] - 1 - min_samples))

        print("KneeLocator")
        # # knee by Kneedle alogithm: https://ieeexplore.ieee.org/document/5961514
        from kneed import KneeLocator
        kneel = KneeLocator(range(len(neighdists)), neighdists, curve='convex', direction='increasing')
        kneeX = kneel.knee

        # import matplotlib.pyplot as plt
        # kneel.plot_knee_normalized()
        # plt.show()

        if isinstance(kneeX, int):
            epsilon = neighdists[kneeX]
        else:
            print("Warning: Kneedle could not find a knee in {}-nearest distribution.".format(min_samples))
            epsilon = 0.0

        if not epsilon > 0.0:  # fallback if epsilon becomes zero
            lt = self.lowertriangle()
            epsilon = numpy.nanmean(lt) + numpy.nanstd(lt)

        return min_samples, epsilon


    def _autoconfigureEvaluation(self):
        """
        Auto configure the clustering parameters epsilon and minPts regarding the input data

        :return: minpts, epsilon
        """
        pass
        # # another alternative to get min_cluster_size estimation
        # import math
        # min_cluster_size = round(math.log(similarities.shape[0]))

        # # get mean of minpts-nearest-neighbors:
        # neighdists = self._knearestmean(minpts)
        # # it gives no significantly better results than the direct k-nearest distance,
        # # but requires more computation.

        # # knee calculation by rule of thumb
        # kneeX = self._kneebyruleofthumb(neighdists)
        # # result is far (too far) left of the actual knee

        # steepest-drop position:
        # kneeX = numpy.ediff1d(neighdists).argmax()
        # # better results than "any of the" knee values

        # # DEBUG and TESTING
        # #
        # # plots of k-nearest-neighbor distance histogram and "knee"
        # import matplotlib.pyplot as plt
        # fig, ax = plt.subplots(1, 2)
        #
        # axl, axr = ax.flat
        #
        # # for k in range(0, 100, 10):
        # #     alpha = .4
        # #     if k == minpts:
        # #         alpha = 1
        # #     plt.plot(sorted([dpn[k] for nid, dpn in npn]), alpha=alpha, label=k)
        #
        # # farthest
        # # plt.plot([max([dpn[k] for nid, dpn in npn]) for k in range(0, len(npn)-1)], alpha=.4)
        # # axl.plot(dpnmln, alpha=.4)
        # # plt.plot([self._knearestdistance(k) for k in range( round(0.5 * (self.distances.shape[0]-1)) )])
        # disttril = numpy.tril(self._distances)
        # alldist = [e for e in disttril.flat if e > 0]
        # axr.hist(alldist, 50)
        #
        # # plt.plot(smoothdists, alpha=.8)
        # # axl.axvline(minpts, linestyle='dashed', color='red', alpha=.4)
        # axl.axvline(steepslopeK, linestyle='dotted', color='blue', alpha=.4)
        # left = axl.get_xlim()[0]
        # bottom = axl.get_ylim()[1]
        # # axl.text(left, bottom,"mpt={}, eps={:0.3f}".format(minpts, epsilon))
        # # plt.axhline(neighdists[int(round(kneeX))], alpha=.4)
        # # plt.plot(range(len(numpy.ediff1d(smoothdists))), numpy.ediff1d(smoothdists), linestyle='dotted')
        # # plt.plot(range(len(numpy.ediff1d(neighdists))), numpy.ediff1d(neighdists), linestyle='dotted')
        # axl.legend()
        # # plt.text(0,0,'max {:.3f}, mean {:.3f}'.format(self.distances.max(), self.distances.mean()))
        # import time
        # # plt.show()
        # plt.savefig("reports/k-nearest_distance_{:0.0f}.pdf".format(time.time()))
        # plt.close('all')
        # plt.clf()
        #
        # # print(kneeX, smoothdists[kneeX], neighdists[kneeX])
        # # print(tabulate([neighdists[:10]], headers=[i for i in range(10)]))
        # # print(tabulate([dpn[:10] for nid, dpn in npn], headers=[i for i in range(10)]))
        # # import IPython; IPython.embed()
        # #
        # # DEBUG and TESTING

        # return minpts, epsilon


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
        neigbordistances = [self._dc.neigbors(seg)[k][1] for seg in self.segments]
        return sorted(neigbordistances)


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

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # END # # Clusterer classes # # # # # # # # # END # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # #




"""
Methods/properties (including super's) 

* working on/returning raw segment indices:
    * rawSegments@ / _rawSegments

* working on/returning representative's indices:
    * segments@ / _segments
    * _quicksegments
    * _offsets / offsets@
    * _seg2idx
    * distanceMatrix@ / distances
    * similarityMatrix()
    * groupByLength()

* returning translation from segment to representative's indices:
  * segments2index()
  * internally: pairDistance() / distancesSubset()
  * internally: neigbors
  * internally: findMedoid



"""
class DelegatingDC(DistanceCalculator):
    """
    Subclass to encapsulate (and redirect if necessary) segemnt lookups to representatives.
    This reduces the size of the matrix (considerable) by the amount of duplicate feature values
    and other segment types for which hypothesis-driven distance values are more appropriate (like textual fields).
    """

    def __init__(self, segments: Sequence[MessageSegment], reliefFactor=.33, manipulateChars = True):
        """
        Determine the distance between the given segments using representatives
        to delegate groups of similar segments to.

        >>> from tabulate import tabulate
        >>> segments = __testing_generateTestSegmentsWithDuplicates()
        >>> DistanceCalculator.debug = False
        >>> dc = DistanceCalculator(segments)
         . . .
         . .
         .
        <BLANKLINE>
        Calculated distances for 45 segment pairs in ... seconds.
        >>> ddc = DelegatingDC(segments)
         . . .
         . .
         .
        <BLANKLINE>
        Calculated distances for 23 segment pairs in ... seconds.
        >>> print(tabulate(enumerate(ddc.segments)))
        -  ---------------------------------------------------------
        0  MessageSegment 4 bytes: 00000000... | values: [0, 0, 0...
        1  MessageSegment 4 bytes: 01020304... | values: [1, 2, 3...
        2  MessageSegment 4 bytes: 03020304... | values: [3, 2, 3...
        3  MessageSegment 3 bytes: 020304 | values: [2, 3, 4]
        4  MessageSegment 3 bytes: 250545 | values: [37, 5, 69]
        5  Template 2 bytes: (2, 4) | #base 2
        6  Template 7 bytes: (20, 30, 37... | #base 3
        -  ---------------------------------------------------------
        >>> print(tabulate(enumerate(dc.segments)))
        -  ------------------------------------------------------------------
        0  MessageSegment 4 bytes: 01020304... | values: [1, 2, 3...
        1  MessageSegment 3 bytes: 020304 | values: [2, 3, 4]
        2  MessageSegment 2 bytes: 0204 | values: [2, 4]
        3  MessageSegment 2 bytes: 0204 | values: [2, 4]
        4  MessageSegment 7 bytes: 141e253245021e... | values: [20, 30, 37...
        5  MessageSegment 7 bytes: 141e253245021e... | values: [20, 30, 37...
        6  MessageSegment 3 bytes: 250545 | values: [37, 5, 69]
        7  MessageSegment 4 bytes: 00000000... | values: [0, 0, 0...
        8  MessageSegment 7 bytes: 141e253245021e... | values: [20, 30, 37...
        9  MessageSegment 4 bytes: 03020304... | values: [3, 2, 3...
        -  ------------------------------------------------------------------

        :param segments: List of segments to calculate pairwise distances from.
        """
        self._rawSegments = segments

        # prior to insert entries into the matrix, filter segments
        # for special treatment and generate templates for them
        uniqueSegments, deduplicatingTemplates, self.reprMap = DelegatingDC._templates4duplicates(segments)
        """
        :var self.reprMap: Mapping from each segment to its representative's index in (the later) self._segments
        """

        filteredSegments = uniqueSegments + deduplicatingTemplates
        super().__init__(filteredSegments, reliefFactor=reliefFactor, manipulateChars=manipulateChars)

        # assert symmetric matrix
        for i in range(self.distanceMatrix.shape[0]):
            for j in range(self.distanceMatrix.shape[1]):
                if self.distanceMatrix[i, j] != self.distanceMatrix[j, i]:
                    print("NOK", i, j)
                assert self.distanceMatrix[i, j] == self.distanceMatrix[j, i]



    @staticmethod
    def _templates4duplicates(segments: Sequence[MessageSegment]) -> Tuple[
           List[AbstractSegment], List[Template], Dict[MessageSegment, int]]:
        # noinspection PyProtectedMember,PyUnresolvedReferences
        """
        Filter segments that are identical regarding their feature vector and replace them by templates.

        >>> from pprint import pprint
        >>> segments = __testing_generateTestSegmentsWithDuplicates()
        >>> t4d = DelegatingDC._templates4duplicates(segments)
        >>> pprint(t4d[0])
        [MessageSegment 4 bytes: 00000000... | values: [0, 0, 0...,
         MessageSegment 4 bytes: 01020304... | values: [1, 2, 3...,
         MessageSegment 4 bytes: 03020304... | values: [3, 2, 3...,
         MessageSegment 3 bytes: 020304 | values: [2, 3, 4],
         MessageSegment 3 bytes: 250545 | values: [37, 5, 69]]
        >>> pprint(t4d[1])
        [Template 2 bytes: (2, 4) | #base 2, Template 7 bytes: (20, 30, 37... | #base 3]
        >>> pprint([sorted(((k, t4d[2][k]) for k in t4d[2].keys()), key=lambda x: x[1])])
        [[(MessageSegment 2 bytes: 0204 | values: [2, 4], 5),
          (MessageSegment 2 bytes: 0204 | values: [2, 4], 5),
          (MessageSegment 7 bytes: 141e253245021e... | values: [20, 30, 37..., 6),
          (MessageSegment 7 bytes: 141e253245021e... | values: [20, 30, 37..., 6),
          (MessageSegment 7 bytes: 141e253245021e... | values: [20, 30, 37..., 6)]]


        :param segments: Segments to filter
        :return: A tuple of two lists and a dict containing:
            1. The segments found to be unique according to their feature vectors
            2. The representatives of the segments that are duplicates according to their feature vectors
            3. The mapping of each duplicate segment to its representative's index
                in self.segments and self.distanceMatrix
        """
        # filter out identical segments
        uniqueFeatures = dict()
        for s in segments:
            svt = tuple(s.values)
            if svt not in uniqueFeatures:
                uniqueFeatures[svt] = list()
            uniqueFeatures[svt].append(s)
        filteredSegments = [s[0] for f, s in uniqueFeatures.items() if len(s) == 1]
        duplicates = {f: s for f, s in uniqueFeatures.items() if len(s) > 1}

        # generate template for each duplicate list
        typedSegs = False
        for seg in segments:
            if isinstance(seg, TypedSegment):
                typedSegs = True
        if typedSegs:
            templates = [TypedTemplate(f, s) for f, s in duplicates.items()]
        else:
            templates = [Template(f, s) for f, s in duplicates.items()]

        # # we need this here already to improve performance,
        # # although it is generated in the super-constructor also afterwards
        # seg2idx = {seg: idx for idx, seg in enumerate(segments)}

        uniqCount = len(filteredSegments)
        mapping = {s: tidx+uniqCount for tidx, t in enumerate(templates) for s in t.baseSegments}  # type: Dict[MessageSegment, int]
        """
        Mapping from each segment to (index in templates) + len(filteredSegments)
        """

        return filteredSegments, templates, mapping

    @staticmethod
    def _templates4allZeros(segments: Iterable[MessageSegment]):
        # filter out segments that contain no relevant byte data, i. e., all-zero byte sequences
        filteredSegments = [t for t in segments if t.bytes.count(b'\x00') != len(t.bytes)]

        # filter out segments that resulted in no relevant feature data, i. e.,
        # (0, .., 0) | (nan, .., nan) | or a mixture of both
        # noinspection PyUnusedLocal
        filteredSegments = [s for s in filteredSegments if
                            numpy.count_nonzero(s.values) - numpy.count_nonzero(numpy.isnan(s.values)) > 0]
        raise NotImplementedError()

    @staticmethod
    def _templates4Paddings(segments: Iterable[MessageSegment]):
        raise NotImplementedError()


    def segments2index(self, segmentList: Iterable[MessageSegment]):
        # noinspection PyUnresolvedReferences
        """
        Look up the indices of the given segments.
        Resolves MessageSegments directly and Templates generated of similar segments to their representative's index
        in the similarity matrix.

        >>> from tabulate import tabulate
        >>> segments = __testing_generateTestSegmentsWithDuplicates()
        >>> dc, ddc = __testing_generateTestDCandDDC(segments)
        >>> # segments and their representatives resolved by segments2index()
        >>> reprSegs = [(seg, ddc.segments[idx]) for idx, seg in zip(ddc.segments2index(segments), segments)]
        >>> # validates that each segment is referenced in its representative's Template's baseSegments
        >>> all([se in re.baseSegments for se, re in reprSegs if isinstance(re, Template)])
        True
        >>> # validates that each segment that is directly contained in the DelegatingDC is correctly resolved
        >>> all([se == re for se, re in reprSegs if not isinstance(re, Template)])
        True

        :param segmentList: List of segments
        :return: List of indices for the given segments
        """
        return [self._seg2idx[seg] if seg in self._seg2idx else self.reprMap[seg] for seg in segmentList]

    @property
    def segments(self):
        """
        >>> from tabulate import tabulate
        >>> segments = __testing_generateTestSegmentsWithDuplicates()
        >>> dc, ddc = __testing_generateTestDCandDDC(segments)
        >>> print("ddc:", len(ddc.segments), "dc:", len(dc.segments))
        ddc: 7 dc: 10
        >>> print(tabulate(enumerate(ddc.segments[-1].baseSegments)))
        -  ------------------------------------------------------------------
        0  MessageSegment 7 bytes: 141e253245021e... | values: [20, 30, 37...
        1  MessageSegment 7 bytes: 141e253245021e... | values: [20, 30, 37...
        2  MessageSegment 7 bytes: 141e253245021e... | values: [20, 30, 37...
        -  ------------------------------------------------------------------

        :return: All unique segments and representatives for "feature-identical" segments in this object.
        """
        return self._segments

    @property
    def rawSegments(self):
        """
        The original list of segments this DelegatingDC instance was created from.

        >>> from tabulate import tabulate
        >>> segments = __testing_generateTestSegmentsWithDuplicates()
        >>> dc, ddc = __testing_generateTestDCandDDC(segments)
        >>> ddc.rawSegments == segments
        True

        :return: All raw segments in this object.
        """
        return self._rawSegments

    def pairDistance(self, A: MessageSegment, B: MessageSegment) -> numpy.float64:
        """
        Retrieve the distance between two segments, resolving representatives internally if necessary.

        >>> from tabulate import tabulate
        >>> segments = __testing_generateTestSegmentsWithDuplicates()
        >>> dc, ddc = __testing_generateTestDCandDDC(segments)
        >>> ddc.pairDistance(segments[0], segments[9])
        0.125
        >>> dc.pairDistance(segments[0], segments[9])
        0.125

        :param A: Segment A
        :param B: Segment B
        :return: Distance between A and B
        """
        a = self._seg2idx[A] if A in self._seg2idx else self.reprMap[A]
        b = self._seg2idx[B] if B in self._seg2idx else self.reprMap[B]
        return self._distances[a, b]

    def distancesSubset(self, As: Sequence[MessageSegment], Bs: Sequence[MessageSegment] = None) \
            -> numpy.ndarray:
        """
        Retrieve a matrix of pairwise distances for two lists of segments, resolving representatives internally if necessary.

        >>> from tabulate import tabulate
        >>> segments = __testing_generateTestSegmentsWithDuplicates()
        >>> dc, ddc = __testing_generateTestDCandDDC(segments)
        >>> (ddc.distancesSubset(segments) == dc.distanceMatrix).all()
        True

        :param As: List of segments
        :param Bs: List of segments
        :return: Matrix of distances: As are rows, Bs are columns.
        """
        clusterI = As
        clusterJ = As if Bs is None else Bs
        simtrx = numpy.ones((len(clusterI), len(clusterJ)))

        transformatorK = {i: a for i, a in enumerate(self.segments2index(clusterI))}  # maps indices i from clusterI to matrix rows k
        if Bs is not None:
            transformatorL = {j: b for j, b in enumerate(self.segments2index(clusterJ))} # maps indices j from clusterJ to matrix cols l
        else:
            transformatorL = transformatorK

        for i, k in transformatorK.items():
            for j, l in transformatorL.items():
                simtrx[i, j] = self._distances[k, l]
        return simtrx

    def representativesSubset(self, As: Sequence[MessageSegment], Bs: Sequence[MessageSegment] = None) \
            -> Tuple[numpy.ndarray, List[AbstractSegment], List[AbstractSegment]]:
        # noinspection PyUnresolvedReferences
        """
        Retrieve a matrix of pairwise distances for two lists of segments, making use also of representatives if any.

        >>> from tabulate import tabulate
        >>> from itertools import combinations
        >>> # generate toy values
        >>> segments = __testing_generateTestSegmentsWithDuplicates()
        >>> dc, ddc = __testing_generateTestDCandDDC(segments)
        >>> segHalf = ddc.rawSegments[5:]
        >>> # retrieve a subset with unresolved representatives
        >>> reprMatrix, reprSegsA, reprSegsB = ddc.representativesSubset(segHalf)
        >>> reprSegsB is None
        True
        >>> # generate all possible combinations of toy subset "segHalf" and retrieve baseline distances for validation
        >>> combis = list(combinations(segHalf, 2))
        >>> baselineDists = [dc.pairDistance(a,b) for a,b in combis]
        >>> # map of toy segment list to indices of segments or representatives in reprMatrix/reprSegsA
        >>> subsetIndicesMap = {i: reprSegsA.index(i) if i in reprSegsA else
        ...    [a for a, j in enumerate(reprSegsA) if isinstance(j, Template) and i in j.baseSegments][0]
        ...     for i in segHalf}
        >>> reprDistances = [reprMatrix[subsetIndicesMap[a], subsetIndicesMap[b]] for a,b in combis]
        >>> reprDistances == baselineDists
        True

        :param As: List of segments
        :param Bs: List of segments
        :return: Matrix of distances: As are rows, Bs are columns.
        """
        transformatorK, clusterI = self.__genTransformator(As)
        if Bs is None:
            clusterJ = clusterI
            transformatorL = transformatorK
        else:
            transformatorL, clusterJ = self.__genTransformator(Bs)
        simtrx = numpy.ones((len(clusterI), len(clusterJ)))

        for k, i in transformatorK.items():
            for l, j in transformatorL.items():
                simtrx[i, j] = \
                    self._distances[k, l]
        return simtrx, clusterI, (clusterJ if Bs is not None else None)

    def __genTransformator(self, As: Sequence[MessageSegment]):
        """
        Generate a dictionary that points from the indices in self.segments to new indices in a list compressed to only
        hold the segments from As. This is aware of representatives.

        :param As: Input list of segments
        :return: For self.segments and As as inputs, the method returns:
            0. transformator, i. e. (index from self.segments) -> (index from cluster)
            1. cluster, with segments resolved to representatives/templates if applicable.
        """
        direct4I = {segi: self._seg2idx[segi] for segi in As if segi in self._seg2idx}  # As segments -> direct ID
        repres4I = {segi: self.reprMap[segi] for segi in As if segi in self.reprMap}  # As segments -> representative's ID
        unqreprI = list(set(repres4I.values()))  # unique representatives
        drctSegI = list(direct4I.values())  # remaining direct segments
        clusterI = [self.segments[idx] for idx in
                    drctSegI + unqreprI]  # resulting list of direct and represented segments from As
        transformatorKrepr = {orI: trI for trI, orI in enumerate(unqreprI, start=len(
            drctSegI))} # original index -> new index for represented
        transformatorK = {self._seg2idx[self.segments[a]]: i for i, a in
                          enumerate(drctSegI)}  # original index -> new index for direct segments
        transformatorK.update(transformatorKrepr)  # maps indices i from clusterI to matrix rows k
        return transformatorK, clusterI



def __testing_generateTestSegmentsWithDuplicates():
    """
    Generates dummy segments with duplicates for testing.

    :return: List of message segments.
    """
    from netzob.Model.Vocabulary.Messages.RawMessage import RawMessage
    from inference.analyzers import Value
    bytedata = [
        bytes([1, 2, 3, 4]),
        bytes([2, 3, 4]),
        bytes([2, 4]),
        bytes([2, 4]),
        bytes([20, 30, 37, 50, 69, 2, 30]),
        bytes([20, 30, 37, 50, 69, 2, 30]),
        bytes([37, 5, 69]),
        bytes([0, 0, 0, 0]),
        bytes([20, 30, 37, 50, 69, 2, 30]),
        bytes([3, 2, 3, 4])
    ]
    messages = [RawMessage(bd) for bd in bytedata]
    analyzers = [Value(message) for message in messages]
    segments = [MessageSegment(analyzer, 0, len(analyzer.message.data)) for analyzer in analyzers]
    return segments

def __testing_generateTestDCandDDC(segments: List[MessageSegment]):
    """
    Generates a DistanceCalculator and a DelegatingDC from the segments parameter for testing.

    :param segments: List of message segments.
    :return: Instances of DistanceCalculator and DelegatingDC from the same segments for comparison.
    """
    import os
    import sys
    stdout = sys.stdout
    f = open(os.devnull, 'w')
    sys.stdout = f

    DistanceCalculator.debug = False
    dc = DistanceCalculator(segments)
    ddc = DelegatingDC(segments)

    sys.stdout = stdout
    return dc, ddc

