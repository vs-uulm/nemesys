from typing import List, Dict, Union, Iterable, Sequence, Tuple, Iterator
from abc import ABC, abstractmethod
from os import cpu_count
from collections import Counter

from pandas import DataFrame
from kneed import KneeLocator
import numpy, scipy.spatial, itertools, kneed, math
from scipy import interpolate

from netzob.Model.Vocabulary.Messages.AbstractMessage import AbstractMessage
from sklearn.cluster import OPTICS

from nemere.inference.fieldTypes import FieldTypeMemento
from nemere.inference.analyzers import MessageAnalyzer, Value
from nemere.inference.segments import MessageSegment, AbstractSegment, CorrelatedSegment, HelperSegment, TypedSegment
from nemere.utils.baseAlgorithms import ecdf


debug = False

parallelDistanceCalc = True
"""
activate parallel/multi-processor calculation of dissimilarities 
in inference.templates.DistanceCalculator#_embdedAndCalcDistances
"""



class ClusterAutoconfException(Exception):
    """
    Exception to raise in case of an failed clusterer autoconfiguration.
    """
    def __init__(self, description: str):
        super().__init__(description)


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

        >>> from netzob.Model.Vocabulary.Messages.RawMessage import RawMessage
        >>> from nemere.inference.analyzers import Value
        >>>
        >>> bytedata = bytes([1,2,3,4])
        >>> message = RawMessage(bytedata)
        >>> analyzer = Value(message)
        >>> segments = [MessageSegment(analyzer, 0, 4)]
        >>> DistanceCalculator.debug = False
        >>> dc = DistanceCalculator(segments)
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
        :param reliefFactor: Increase the non-linearity of the dimensionality penalty for mismatching segment lengths.
        :return: A normalized distance matrix for all input segments.
            If necessary, performs an embedding of mixed-length segments to determine cross-length distances.
        """
        self._reliefFactor = reliefFactor
        self._offsetCutoff = DistanceCalculator.offsetCutoff
        self._method = method
        self.thresholdFunction = thresholdFunction if thresholdFunction else DistanceCalculator.neutralThreshold
        self.thresholdArgs = thresholdArgs if thresholdArgs else {}
        self._segments = list()  # type: List[AbstractSegment]
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
        self._distances = type(self)._getDistanceMatrix(self._embdedAndCalcDistances(), len(self._quicksegments))

        # prepare lookup for matrix indices
        self._seg2idx = {seg: idx for idx, seg in enumerate(self._segments)}  # type: Dict[AbstractSegment, int]

        if manipulateChars:
            # Manipulate calculated distances for all char/char pairs.
            self._manipulateChars()


    @property
    def distanceMatrix(self) -> numpy.ndarray:
        """
        The order of the matrix elements in each row and column is the same as in self.segments.

        >>> from tabulate import tabulate
        >>> from nemere.utils.baseAlgorithms import generateTestSegments
        >>> segments = generateTestSegments()
        >>> DistanceCalculator.debug = False
        >>> dc = DistanceCalculator(segments)
        Calculated distances for 37 segment pairs in ... seconds.
        >>> print((numpy.diag(dc.distanceMatrix) == 0).all())
        True
        >>> print(tabulate(dc.distanceMatrix, floatfmt=".2f"))
        ----  ----  ----  ----  ----  ----  ----  ----  ----
        0.00  0.21  0.30  0.44  0.40  0.81  0.75  1.00  0.12
        0.21  0.00  0.11  0.35  0.30  0.79  0.68  1.00  0.21
        0.30  0.11  0.00  0.37  0.41  0.79  0.70  1.00  0.30
        0.44  0.35  0.37  0.00  0.07  0.70  0.65  1.00  0.44
        0.40  0.30  0.41  0.07  0.00  0.71  0.70  1.00  0.40
        0.81  0.79  0.79  0.70  0.71  0.00  0.58  1.00  0.80
        0.75  0.68  0.70  0.65  0.70  0.58  0.00  1.00  0.75
        1.00  1.00  1.00  1.00  1.00  1.00  1.00  0.00  1.00
        0.12  0.21  0.30  0.44  0.40  0.80  0.75  1.00  0.00
        ----  ----  ----  ----  ----  ----  ----  ----  ----

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
        >>> from nemere.inference.analyzers import Value
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
        Calculated distances for 37 segment pairs in ... seconds.
        >>> print((numpy.diag(dc.similarityMatrix()) == 1).all())
        True
        >>> print(tabulate(dc.similarityMatrix(), floatfmt=".2f"))
        ----  ----  ----  ----  ----  ----  ----  ----  ----
        1.00  0.79  0.70  0.56  0.60  0.19  0.25  0.00  0.88
        0.79  1.00  0.89  0.65  0.70  0.21  0.32  0.00  0.79
        0.70  0.89  1.00  0.63  0.59  0.21  0.30  0.00  0.70
        0.56  0.65  0.63  1.00  0.93  0.30  0.35  0.00  0.56
        0.60  0.70  0.59  0.93  1.00  0.29  0.30  0.00  0.60
        0.19  0.21  0.21  0.30  0.29  1.00  0.42  0.00  0.20
        0.25  0.32  0.30  0.35  0.30  0.42  1.00  0.00  0.25
        0.00  0.00  0.00  0.00  0.00  0.00  0.00  1.00  0.00
        0.88  0.79  0.70  0.56  0.60  0.20  0.25  0.00  1.00
        ----  ----  ----  ----  ----  ----  ----  ----  ----


        :return: The pairwise similarities of all segments in this object represented as an symmetric array.
        """
        similarityMatrix = 1 - self._distances
        return similarityMatrix

    @property
    def segments(self) -> List[AbstractSegment]:
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
        >>> from nemere.utils.baseAlgorithms import generateTestSegments
        >>> from nemere.inference.templates import DistanceCalculator
        >>>
        >>> segments = generateTestSegments()
        >>> DistanceCalculator.debug = False
        >>> dc = DistanceCalculator(segments)
        Calculated distances for 37 segment pairs in ... seconds.
        >>> print(tabulate(dc.offsets.items()))
        ------  -
        (0, 5)  3
        (7, 5)  0
        (8, 5)  3
        (1, 5)  4
        (2, 5)  4
        (6, 5)  2
        (3, 5)  5
        (4, 5)  5
        (1, 0)  1
        (1, 7)  0
        (1, 8)  1
        (2, 0)  1
        (2, 7)  0
        (2, 8)  1
        (6, 0)  1
        (6, 7)  0
        (6, 8)  1
        (3, 0)  1
        (3, 7)  0
        (3, 8)  1
        (4, 0)  1
        (4, 7)  0
        (4, 8)  1
        (3, 1)  0
        (3, 2)  1
        (3, 6)  0
        (4, 1)  0
        (4, 2)  0
        (4, 6)  0
        ------  -
        >>> offpairs = [["-"*(off*2) + dc.segments[pair[0]].bytes.hex(), dc.segments[pair[1]].bytes.hex()]
        ...                     for pair, off in dc.offsets.items()]
        >>> for opsub in range(1, int(math.ceil(len(offpairs)/5))):
        ...     print(tabulate(map(list,zip(*offpairs[(opsub-1)*5:opsub*5])), numalign="left"))
        --------------  --------------  --------------  --------------  --------------
        ------01020304  00000000        ------03020304  --------020304  --------010304
        141e253245021e  141e253245021e  141e253245021e  141e253245021e  141e253245021e
        --------------  --------------  --------------  --------------  --------------
        --------------  --------------  --------------  --------  --------
        ----250545      ----------0204  ----------0203  --020304  020304
        141e253245021e  141e253245021e  141e253245021e  01020304  00000000
        --------------  --------------  --------------  --------  --------
        --------  --------  --------  --------  --------
        --020304  --010304  010304    --010304  --250545
        03020304  01020304  00000000  03020304  01020304
        --------  --------  --------  --------  --------
        --------  --------  --------  --------  --------
        250545    --250545  --0204    0204      --0204
        00000000  03020304  01020304  00000000  03020304
        --------  --------  --------  --------  --------
        --------  --------  --------  ------  ------
        --0203    0203      --0203    0204    --0204
        01020304  00000000  03020304  020304  010304
        --------  --------  --------  ------  ------

        :return: In case of mixed-length distances, this returns a mapping of segment-index pairs to the
        positive or negative offset of the smaller segment from the larger segment start position.
        """
        # is set in the constructor and should therefore be always valid.
        return self._offsets


    def segments2index(self, segmentList: Iterable[AbstractSegment]) -> List[int]:
        # noinspection PyUnresolvedReferences
        """
        Look up the indices of the given segments.

        >>> from nemere.utils.baseAlgorithms import generateTestSegments
        >>> segments = generateTestSegments()
        >>> DistanceCalculator.debug = False
        >>> dc = DistanceCalculator(segments)
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
        >>> from nemere.inference.analyzers import Value
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

        >>> from nemere.utils.baseAlgorithms import generateTestSegments
        >>> segments = generateTestSegments()
        >>> DistanceCalculator.debug = False
        >>> dc = DistanceCalculator(segments)
        Calculated distances for 37 segment pairs in ... seconds.
        >>> dc.distancesSubset(segments[:3], segments[-3:])
        array([[0.748 , 1.    , 0.125 ],
               [0.679 , 1.    , 0.2144],
               [0.696 , 1.    , 0.3018]], dtype=float16)
        >>> (dc.distancesSubset(segments[:3], segments[-3:]) == dc.distanceMatrix[:3,-3:]).all()
        True

        :param As: List of segments
        :param Bs: List of segments
        :return: Matrix of distances: As are rows, Bs are columns.
        """
        clusterI = As
        clusterJ = As if Bs is None else Bs
        # simtrx = numpy.ones((len(clusterI), len(clusterJ)))
        simtrx = MemmapDC.largeFilled((len(clusterI), len(clusterJ)))

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
        >>> from nemere.utils.baseAlgorithms import generateTestSegments
        >>> segments = generateTestSegments()
        >>> DistanceCalculator.debug = False
        >>> dc = DistanceCalculator(segments)
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
        --  --  ---  ---  ---  ---  ---
         1   2    3    4  nan  nan  nan
         2   3    4  nan  nan  nan  nan
         1   3    4  nan  nan  nan  nan
         2   4  nan  nan  nan  nan  nan
         2   3  nan  nan  nan  nan  nan
        20  30   37   50   69    2   30
        37   5   69  nan  nan  nan  nan
         0   0    0    0  nan  nan  nan
         3   2    3    4  nan  nan  nan
        --  --  ---  ---  ---  ---  ---



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

        max_len = max(len(seg[2]) for seg in segments)
        # Create a NumPy array with NaN values
        segmentValuesMatrix = numpy.full((len(segments), max_len), numpy.nan)
        if method == 'cosine':
            # comparing to zero vectors is undefined in cosine.
            # Its semantically equivalent to a (small) horizontal vector

            for i, seg in enumerate(segments):
                coseg = seg[2] if numpy.any(numpy.array(seg[2]) != 0) else numpy.full(len(seg[2]), 1e-16)
                segmentValuesMatrix[i, :len(coseg)] = coseg
        else:
            for i, seg in enumerate(segments):
                segmentValuesMatrix[i, :len(seg[2])] = seg[2]

        return method, segmentValuesMatrix


    @staticmethod
    def calcDistances(segments: List[Tuple[int, int, Tuple[float]]], method='canberra') -> List[
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
        >>> pprint(DistanceCalculator.calcDistances(testdata, 'canberra'))
        [(0, 1, 3.446...),
         (0, 2, 4.0),
         (0, 3, 0.0),
         (0, 4, 0.5),
         (1, 2, 4.0),
         (1, 3, 3.446...),
         (1, 4, 3.392...),
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
    def _getDistanceMatrix(distances: Iterable[Tuple[int, int, float]], segmentCount: int) -> numpy.ndarray:
        # noinspection PyProtectedMember
        """
        Arrange the representation of the pairwise similarities of the input parameter in an symmetric array.
        The order of the matrix elements in each row and column is the same as in self._segments.

        Distances for pair not included in the list parameter are considered incomparable and set to -1 in the
        resulting matrix.

        Used in constructor.

        >>> from tabulate import tabulate
        >>> from nemere.inference.templates import DistanceCalculator
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
        >>> print(tabulate(dm, floatfmt=".2f"))
        -----  -----  -----  -----  -----  -----  -----
         0.00   0.21   0.40   0.81   0.75   1.00  -1.00
         0.21   0.00   0.30   0.79   0.68   1.00  -1.00
         0.40   0.30   0.00   0.71   0.70   1.00  -1.00
         0.81   0.79   0.71   0.00   0.58   1.00  -1.00
         0.75   0.68   0.70   0.58   0.00   1.00  -1.00
         1.00   1.00   1.00   1.00   1.00   0.00  -1.00
        -1.00  -1.00  -1.00  -1.00  -1.00  -1.00   0.00
        -----  -----  -----  -----  -----  -----  -----

        :param distances: The pairwise similarities to arrange.
        :return: The distance matrix for the given similarities.
            -1 for each undefined element, 0 in the diagonal, even if not given in the input.
        """
        from nemere.inference.segmentHandler import matrixFromTpairs
        simtrx = matrixFromTpairs(distances, range(segmentCount),
                                  incomparable=-1)  # TODO handle incomparable values (resolve and replace the negative value)
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
        ('canberra', 1, (2, 0, 0.3333...))
        >>> DistanceCalculator.embedSegment(testdata[3], testdata[2])
        ('canberra', 1, (3, 2, 0.2...))
        >>> DistanceCalculator.embedSegment(testdata[6], testdata[0])
        ('canberra', 1, (6, 0, 2.037846856340007))
        >>> DistanceCalculator.embedSegment(testdata[7], testdata[5])
        ('canberra', 0, (7, 5, 4.0))
        >>> DistanceCalculator.embedSegment(testdata[3], testdata[0])
        ('canberra', 1, (3, 0, 0.1428...))
        >>> DistanceCalculator.embedSegment(testdata[4], testdata[0])
        ('canberra', 1, (4, 0, 0.0))
        >>> DistanceCalculator.embedSegment(testdata[6], testdata[5])
        ('canberra', 2, (6, 5, 0.8181...))

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
        # TODO overlapping offset from -1
        for offset in range(0, maxOffset + 1):
            offSegment = longSegment[2][offset:offset + shortSegment[1]]
            subsets.append((-1, shortSegment[1], offSegment))
        method, segmentValuesMatrix = DistanceCalculator.__prepareValuesMatrix(subsets, method)

        subsetsSimi = scipy.spatial.distance.cdist(segmentValuesMatrix, numpy.array([shortSegment[2]]), method)
        shift = subsetsSimi.argmin() # for debugging and evaluation
        # noinspection PyArgumentList
        distance = subsetsSimi.min()

        return method, shift, (shortSegment[0], longSegment[0], distance)

    @staticmethod
    def sigmoidThreshold(x: float, shift=0.5):
        """
        Standard sigmoid threshold function to transform x.

        >>> DistanceCalculator.sigmoidThreshold(.42)
        0.312261360...

        >>> DistanceCalculator.sigmoidThreshold(.23)
        0.065083072...

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
            Iterator[Tuple[int, int, float]]:
        """
        Embed all shorter Segments into all larger ones and use the resulting pairwise distances to generate a
        complete distance list of all combinations of the into segment list regardless of their length.

        >>> from tabulate import tabulate
        >>> from nemere.utils.baseAlgorithms import generateTestSegments
        >>> segments = generateTestSegments()
        >>> DistanceCalculator.debug = False
        >>> dc = DistanceCalculator(segments)
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
        ... ], floatfmt=".2f"))
        ...
        -  ---------------------------  ------------  ----
        3  (1, 2, 3, 4)                 (2, 3, 4)     0.21
        3  (1, 2, 3, 4)                 (1, 3, 4)     0.30
        2  (1, 3, 4)                    (2, 4)        0.37
        3  (37, 5, 69)                  (1, 2, 3, 4)  0.75
        4  (20, 30, 37, 50, 69, 2, 30)  (0, 0, 0, 0)  1.00
        2  (1, 2, 3, 4)                 (2, 4)        0.44
        2  (1, 2, 3, 4)                 (2, 3)        0.40
        3  (20, 30, 37, 50, 69, 2, 30)  (37, 5, 69)   0.58
        4  (1, 2, 3, 4)                 (0, 0, 0, 0)  1.00
        2  (2, 4)                       (2, 3)        0.07
        4  (1, 2, 3, 4)                 (3, 2, 3, 4)  0.12
        -  ---------------------------  ------------  ----

        :return: List of Tuples
            (index of segment in self._segments), (segment length), (Tuple of segment analyzer values)
        """
        from concurrent.futures.process import BrokenProcessPool
        import time

        dissCount = 0
        lenGrps = self.groupByLength()  # segment list is in format of self._quicksegments

        pit_start = time.time()

        rslens = list(reversed(sorted(lenGrps.keys())))  # lengths, sorted by decreasing length

        if not parallelDistanceCalc:
            for outerlen in rslens:
                for diss in self._outerloop(lenGrps, outerlen, rslens):
                    dissCount += 1
                    # split off the offset if present (None in case of equal length segments)
                    if diss[3] is not None:
                        self._offsets[(diss[0], diss[1])] = diss[3]
                    yield diss[0], diss[1], diss[2]

        #     profiler = cProfile.Profile()
        #     int_start = time.time()
        #     stats = profiler.runctx('self._outerloop(lenGrps, outerlen, distance, rslens)', globals(), locals())
        #     int_runtime = time.time() - int_start
        #     profiler.dump_stats("embdedAndCalcDistances-{:02.1f}-{}-i{:03d}.profile".format(
        #         int_runtime, self.segments[0].message.data[:5].hex(), outerlen))
        else:
            import concurrent.futures
            with concurrent.futures.ProcessPoolExecutor(max_workers=cpu_count()-1) as executor:   # Process # Thread
                futureDis = dict()
                for outerlen in rslens:
                    futureDis[executor.submit(self._outerloop, lenGrps, outerlen, rslens)] = outerlen
                futureRes = dict()
                for future in concurrent.futures.as_completed(futureDis.keys()):
                    try:
                        futureRes[futureDis[future]] = future.result()
                    except BrokenProcessPool as e:
                        import IPython
                        print("Process failed. outerlen ", outerlen)
                        print()
                        IPython.embed()
                        raise e
                for ol in rslens:
                    for diss in futureRes[ol]:
                        dissCount += 1
                        # split off the offset if present (None in case of equal length segments)
                        if diss[3] is not None:
                            self._offsets[(diss[0], diss[1])] = diss[3]
                        yield diss[0], diss[1], diss[2]

        runtime = time.time() - pit_start
        print("Calculated distances for {} segment pairs in {:.2f} seconds.".format(dissCount, runtime))


    def _outerloop(self, lenGrps, outerlen, rslens):
        """
        explicit function for the outer loop of _embdedAndCalcDistances() for profiling it

        :return:
        """
        dissimilarities = list()
        outersegs = lenGrps[outerlen]
        if DistanceCalculator.debug:
            print("    outersegs, length {}, segments {}".format(outerlen, len(outersegs)))
        # a) for segments of identical length: call _calcDistancesPerLen()
        # TODO something outside of calcDistances takes a lot longer to return during the embedding loop. Investigate.
        ilDist = DistanceCalculator.calcDistances(outersegs, method=self._method)
        # # # # # # # # # # # # # # # # # # # # # # # #
        dissimilarities.extend([
            (i, l, self.thresholdFunction( d * self._normFactor(outerlen), **self.thresholdArgs), None)
            for i, l, d in ilDist])
        # # # # # # # # # # # # # # # # # # # # # # # #
        # b) on segments with mismatching length: embedSegment:
        #       for all length groups with length < current length
        for innerlen in rslens[rslens.index(outerlen) + 1:]:
            innersegs = lenGrps[innerlen]
            if DistanceCalculator.debug:
                print("        innersegs, length {}, segments {}".format(innerlen, len(innersegs)))
            # else:
            #     print(" .", end="", flush=True)
            # for all segments in "shorter length" group
            #     for all segments in current length group
            for iseg in innersegs:
                # TODO performance improvement: embedSegment directly generates six (offset cutoff) ndarrays per run
                #  and gets called a lot of times: e.g. for dhcp-10000 511008 times alone for outerlen = 9 taking 50 sec.
                # :
                # instead, prepare one values matrix for all outersegs of one innerseg iteration at once (the "embedded"
                # lengths are all of innerlen, so they are compatible for one single run of cdist). Remember offset for
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
                    dlDist = (interseg[0], interseg[1],
                              ( self.thresholdFunction( mlDistance, **self.thresholdArgs ) ),
                              embedded[1]  # byte offset between interseg[0] and interseg[1] from embedding
                              )  # minimum of dimensions
                    # # # # # # # # # # # # # # # # # # # # # # # #
                    dissimilarities.append(dlDist)
        # if not DistanceCalculator.debug:
        #     print()
        return dissimilarities


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

    def neighbors(self, segment: AbstractSegment, subset: List[MessageSegment]=None) -> List[Tuple[int, float]]:
        # noinspection PyUnresolvedReferences
        """

        >>> from netzob.Model.Vocabulary.Messages.RawMessage import RawMessage
        >>> from nemere.inference.analyzers import Value
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
        Calculated distances for 37 segment pairs in ... seconds.
        >>> nbrs = dc.neighbors(segments[2], segments[3:7])
        >>> dsts = [dc.pairDistance(segments[2], segments[3]),
        ...         dc.pairDistance(segments[2], segments[4]),
        ...         dc.pairDistance(segments[2], segments[5]),
        ...         dc.pairDistance(segments[2], segments[6])]
        >>> nbrs
        [(0, 0.3677), (1, 0.4146), (3, 0.696), (2, 0.7935)]
        >>> [dsts[a] for a,b in nbrs] == [a[1] for a in nbrs]
        True

        :param segment: Segment to get the neighbors for.
        :param subset: The subset of MessageSegments to use from this DistanceCalculator object, if any.
        :return: An ascendingly sorted list of neighbors of parameter segment
            from all the segments in this object (if subset is None)
            or from the segments in subset.
            The result is a list of tuples with
                * the index of the neighbor (from self.segments or the subset list, respectively) and
                * the distance to this neighbor
        """
        home = self.segments2index([segment])[0]
        if subset:
            mask = self.segments2index(subset)
            assert len(mask) == len(subset)
        else:
            # mask self identity by "None"-value if applicable
            mask = list(range(home)) + [None] + list(range(home + 1, self._distances.shape[0]))

        candNeighbors = self._distances[:, home]
        neighbors = sorted([(nidx, candNeighbors[n]) for nidx, n in enumerate(mask) if n is not None],
                           key=lambda x: x[1])
        return neighbors

    @staticmethod
    def _checkCacheFile(analysisTitle: str, tokenizer: str, pcapfilename: str):
        from os.path import splitext, basename, exists, join
        from nemere.utils.evaluationHelpers import cacheFolder
        pcapName = splitext(basename(pcapfilename))[0]
        dccachefn = 'cache-dc-{}-{}-{}.{}'.format(analysisTitle, tokenizer, pcapName, 'dc')
        dccachefn = join(cacheFolder, dccachefn)
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
        from nemere.validation.dissectorMatcher import MessageComparator

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
        from nemere.inference.segmentHandler import filterChars

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
    Represents a template for some group of similar MessageSegments.
    A Templates values are either the values of a medoid or the mean of values per vector component.
    """

    def __init__(self, values: Union[Tuple[Union[float, int]], MessageSegment],
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
        if isinstance(self._values, numpy.ndarray):
            if any(numpy.isnan(self._values)):
                return None  # TODO relevant for #10
            bi = self._values.astype(int).tolist()
            # noinspection PyTypeChecker
            return bytes(bi)
        if isinstance(self._values, Iterable):
            if any(numpy.isnan(self._values)):
                return None  # TODO relevant for #10
            return bytes(self._values)
        return None


    @property
    def analyzer(self):
        return self.baseSegments[0].analyzer


    def checkSegmentsAnalysis(self):
        """
        Validate that all base segments of this template are configured with the same type of analysis.

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
        If no DistanceCalculator is given, does not support a threshold function.

        >>> from tabulate import tabulate
        >>> from scipy.spatial.distance import cdist
        >>> from nemere.utils.baseAlgorithms import generateTestSegments
        >>> from nemere.inference.templates import DistanceCalculator, Template, parallelDistanceCalc
        >>> DistanceCalculator.debug = False
        >>> listOfSegments = generateTestSegments()
        >>> # test both implementations: using parallel dissimilarity calculation and one single process.
        >>> for pdc in [False, True]:
        ...     parallelDistanceCalc = pdc
        ...     dc = DistanceCalculator(listOfSegments)
        ...     center = [1,3,7,9]  # "mean"
        ...     tempearly = Template(center, listOfSegments)
        ...     dtml = tempearly.distancesToMixedLength(dc)
        ...     print("Dissimilarity to Segment 0:", cdist([center], [listOfSegments[0].values], "canberra")[0,0]/len(center))
        ...     print("     ... from the function:", dtml[0][0])
        ...     print("Matches:",
        ...         round(cdist([center], [listOfSegments[0].values], "canberra")[0,0]/len(center), 2) == round(dtml[0][0], 2)
        ...         )
        ...     print("Dissimilarities and offsets for a mean center:")
        ...     print(tabulate(dtml))
        ...     center = listOfSegments[0]  # "medoid"
        ...     template = Template(center, listOfSegments)
        ...     print("Dissimilarities and offsets for a medoid center:")
        ...     print(tabulate(template.distancesToMixedLength(dc)))
        Calculated distances for 37 segment pairs in ... seconds.
        Calculated distances for 46 segment pairs in ... seconds.
        Dissimilarity to Segment 0: 0.24615384615384617
             ... from the function: 0.2461
        Matches: True
        Dissimilarities and offsets for a mean center:
        --------  --
        0.246...   0
        0.373...   0
        0.285...   0
        0.5...   1
        0.497...   0
        0.825...  -3
        0.682...   1
        1          0
        0.371...   0
        --------  --
        Dissimilarities and offsets for a medoid center:
        --------  --
        0          0
        0.214...   1
        0.301...   1
        0.440...   1
        0.397...   1
        0.808...  -3
        0.748...   1
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
        # noinspection PyArgumentList
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
        import nemere.visualization.bcolors as bcolors
        # Template
        return bcolors.colorizeStr('{:02x}'.format(oid % 0xffff), oid % 0xff)

    def __repr__(self):
        if self.values is not None and isinstance(self.values, (list, tuple)) and len(self.values) > 3:
            printValues = str(self.values[:3])[:-1] + '...'
        else:
            printValues = str(self.values)

        return "Template {} bytes: {} | #base {}".format(self.length, printValues, len(self.baseSegments))


class TypedTemplate(Template):
    """
    Template for the representation of a segment type.
    """

    def __init__(self, values: Union[Tuple[Union[float, int]], MessageSegment],
                 baseSegments: Iterable[AbstractSegment],
                 method='canberra'):
        from nemere.inference.segments import TypedSegment
        from nemere.utils.evaluationHelpers import unknown

        super().__init__(values, baseSegments, method)
        ftypes = {bs.fieldtype for bs in baseSegments if isinstance(bs, TypedSegment)}
        fcount = len(ftypes)
        if fcount == 1:
            self._fieldtype = ftypes.pop()
        elif fcount > 1:
            self._fieldtype = "[mixed]"
        else:
            self._fieldtype = unknown

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
    """
    Template and Memento for collecting base segments representing a common field type.
    Also serves to commonly analyze the collevtive base segments, and thus, to determine characteristics of the
    field type. This way the class is the basis to verify a segment cluster's suitability as a type template in the
    first place.
    """

    def __init__(self, baseSegments: Iterable[AbstractSegment], method='canberra'):
        """
        A new FieldTypeTemplate for the collection of base segments given.

        Per vector component, the mean, stdev, and the covariance matrix is calculated. Therefore the collection needs
        to be represented by a vector of one common vector space and thus a fixed number of dimensions.
        Thus for the calculation:
            * zero-only segments are ignored
            * nans are used for shorter segments at don't-care positions

        CAVEAT: components with a standard deviation of 0 are "scintillated" to slightly deviate from 0. Thus we do not
                fail to calculate a covariance matrix for linearly dependent entries at the price of a minor loss of
                numeric precision.

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
            # TODO overlapping offset from -1
            self._maxLen = next(iter(segLens))
        elif len(segLens) > 1:
            # find the optimal shift/offset of each shorter segment to match the longest ones
            #   with shortest distance according to the method
            # TODO overlapping offset from -1
            self._maxLen = max(segLens)
            # Better than the longest would be the most common length, but that would increase complexity a lot and
            #   we assume that for most use cases the longest segments will be the most frequent length.
            # TODO -1 shift could also be necessary for comparing two max-long segments

            # tuples of indices, lengths, and values of the segments that all are the longest in the input
            maxLenSegs = [(idx, seg.length, seg.values) for idx, seg in enumerate(relevantSegs, 1)
                          if seg.length == self._maxLen]
            segE = list()
            for seg in relevantSegs:
                if seg.length == self._maxLen:
                    # only-zero segments are irrelevant and maxLenSegs are processed afterwards.
                    continue
                shortSeg = (0, seg.length, tuple(seg.values))
                # offsets = [DistanceCalculator.embedSegment(shortSeg, longSeg, method)[1] for longSeg in maxLenSegs]

                embeddingsStraight = [DistanceCalculator.embedSegment(shortSeg, longSeg, method) for longSeg in
                                      maxLenSegs]
                if seg.length > 2:
                    evenShorter = (0, seg.length - 1, tuple(seg.values[1:]))
                    embeddingsTrunc = [DistanceCalculator.embedSegment(evenShorter, longSeg, method) for longSeg in
                                          maxLenSegs]
                    # method, shift, (shortSegment[0], longSegment[0], distance)
                    longStraightDistLookup = { es[2][1]: es[2][2] for es in embeddingsStraight }
                    longTruncDistLookup = { et[2][1]: et[2][2] for et in embeddingsTrunc }
                    truncMatchDists = [(longTruncDistLookup[longSeg[2][1]],
                                        longStraightDistLookup[longSeg[2][1]])
                                       for longSeg in embeddingsStraight]
                    if all(list(map(lambda x: x[0]<x[1]-1, truncMatchDists))) \
                            and all([es[1] == 0 for es in embeddingsTrunc]):
                        offsets = [-1] * len(embeddingsStraight)
                    else:
                        offsets = [es[1] for es in embeddingsStraight]
                else:
                    offsets = [es[1] for es in embeddingsStraight]

                # count and select which of the offsets for seg is most common amongst maxLenSegs when using the
                # dissimilarity as criterion for the best-match shift of the shorter segment within the set of longest.
                offCount = Counter(offsets)
                self._baseOffsets[seg] = offCount.most_common(1)[0][0]
                paddedVals = self.paddedValues(seg)
                if debug:
                    from tabulate import tabulate
                    print(tabulate((maxLenSegs[0][2], paddedVals)))
                segE.append(paddedVals)

            # for all short segments that can be truncated by one byte and still have at least a length of 2:
            if len(maxLenSegs) > 1 and self._maxLen > 2:
                embeddingsStraight = DistanceCalculator.calcDistances(maxLenSegs)
                longStraightDistLookup = {(es[0], es[1]): es[2] for es in embeddingsStraight}

                # iterate the longest segments and determine if truncating the shorter segments further reduces the
                # dissimilarity for any shift of the shorter within the longest segments.
                for segIdx, segLen, segVals in maxLenSegs:

                    seg = relevantSegs[segIdx - 1]
                    assert seg.values == segVals, "Wrong segment selected during maxLenSegs truncation."

                    evenShorter = (0, segLen - 1, tuple(segVals[1:]))
                    embeddingsTrunc = [DistanceCalculator.embedSegment(evenShorter, longSeg, method) for longSeg in
                                       maxLenSegs if longSeg[0] != segIdx]
                    longTruncDistLookup = { et[2][1]: et[2][2] for et in embeddingsTrunc }
                    truncMatchDists = [(longTruncDistLookup[longSeg[2][1]],
                                        longStraightDistLookup[
                                            (longSeg[2][1], segIdx) if (longSeg[2][1], segIdx) in longStraightDistLookup
                                            else (segIdx, longSeg[2][1])
                                        ],
                                        longSeg[2][1])
                                       for longSeg in embeddingsTrunc]

                    if method != 'canberra':
                        # TODO this "-1" is canberra specific. Other methods need different values.
                        raise NotImplementedError("Threshold for non-caberra dissimilarity improvement is not yet"
                                                  "defined.")

                    # DEBUG
                    if debug:
                        for truncD, straightD, longIdx in truncMatchDists:
                            if truncD < straightD - 1:
                                longSeg = next(mls for mls in maxLenSegs if mls[0] == longIdx)
                                offset = next(et[1] for et in embeddingsTrunc if et[2][1] == longIdx)
                                comp = [["({})".format(segVals[0])] + list(evenShorter[2]), [""] + list(longSeg[2])]
                                from tabulate import tabulate
                                print("Offset", offset, "- truncD", truncD, "- straightD", straightD)
                                print(tabulate(comp))
                                # import IPython
                                # IPython.embed()

                    if all(map(lambda x: x[0]<x[1]-1, truncMatchDists)) \
                            and all([es[1] == 0 for es in embeddingsTrunc]):
                        # offsets = [-1] * len(embeddingsStraight)
                        # offCount = Counter(offsets)
                        # self._baseOffsets[seg] = offCount.most_common(1)[0][0]
                        self._baseOffsets[seg] = -1
                        paddedVals = self.paddedValues(seg)
                        segE.append(paddedVals)
                    else:
                        segE.append(list(segVals))

            else:
                segE.extend([list(segT[2]) for segT in maxLenSegs])


            segV = numpy.array(segE)
        else:
            # handle situation when no base segment is relevant (all are zeros)
            # TODO overlapping offset from -1
            self._maxLen = max({seg.length for seg in self.baseSegments}) if len(self.baseSegments) > 0 else 0
            for bs in self.baseSegments:
                if bs.length == self._maxLen:
                    self._mean = numpy.array(bs.values)
                    self._stdev = numpy.zeros(self._mean.shape)
                    self._cov = numpy.zeros((self._mean.shape[0], self._mean.shape[0]))
                    break
            if not (isinstance(self._mean, numpy.ndarray) and isinstance(self._stdev, numpy.ndarray)
                    and isinstance(self._cov, numpy.ndarray)):
                raise RuntimeError("This collection of base segments is not suited to generate a FieldTypeTemplate.")
            # noinspection PyTypeChecker
            super().__init__(self._mean, self.baseSegments, method)
            return

        self._mean = numpy.nanmean(segV, 0)
        self._stdev = numpy.nanstd(segV, 0)

        # for all components that have a stdev of 0 we need to scintillate the values (randomly) to derive a
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
        if segV.shape[0] > 1:
            pd = DataFrame(segV)
            self._cov = pd.cov().values
        else:
            # handle cases that result in a runtime warning of numpy, 
            # since a single sequence of values cannot yield a cov.
            self._cov = numpy.empty((segV.shape[1],segV.shape[1]))
            self._cov[:] = numpy.nan
        self._picov = None  # fill on demand

        assert len(self._mean) == len(self._stdev) == len(self._cov.diagonal())
        super().__init__(self._mean, self.baseSegments, method)


    def paddedValues(self, segment: AbstractSegment=None):
        """
        :param segment: The base segment to get the padded values for,
            or None to return an array of all padded values of this Template
        :return: The values of the given base segment padded with nans to the length of the
            longest segment represented by this template.
        """
        if segment is None:
            return numpy.array([self.paddedValues(seg) for seg in self.baseSegments])

        shift = self._baseOffsets[segment] if segment in self._baseOffsets else 0
        # overlapping offset from -1
        segvals = list(segment.values) if shift >= 0 else list(segment.values)[-shift:]
        vals = [numpy.nan] * shift + segvals + [numpy.nan] * (self._maxLen - shift - segment.length)
        return vals

    @property
    def baseOffsetCounts(self):
        """
        :return: The amounts of relative offset values.
        """
        return Counter([o for o in self._baseOffsets.values()])

    def paddedPosition(self, segment: MessageSegment):
        """
        Only works with MessageSegments, i.e. Templates need to be resolved into their base segments
        when creating the FieldTypeTemplate.

        :return: The absolute positions (analog to offset and nextOffset) of the padded values for the given segment.
            The values may be before the message start or after the message end!
        """
        offset = segment.offset - self._baseOffsets.get(segment, 0)
        # TODO overlapping offset from -1
        nextOffset = offset + self._maxLen
        return offset, nextOffset

    @property
    def maxLen(self):
        return self._maxLen


class FieldTypeContext(FieldTypeTemplate):

    def __init__(self, baseSegments: Iterable[MessageSegment], method='canberra'):
        """
        FieldTypeTemplate-subclass which, instead of a nan-padded offset alignment,
        fills shorter segments with the values of the message at the respective position.

        :param baseSegments: Requires a List of MessageSegment not AbstractSegment!
            Templates must therefore be resolved beforehand!
        :param method: see :py:class:`FieldTypeTemplate`
        """
        super().__init__(baseSegments, method)

    def paddedValues(self, segment: MessageSegment=None):
        """
        :param segment: The base segment to get the padded values for,
            or None to return an array of all padded values of this Template
        :return: The values of the given base segment padded with values of the original message to the length of the
            longest segment represented by this template. If a padding would exceed the message data, padd with nans
        """
        if segment is None:
            # noinspection PyTypeChecker
            return numpy.array([self.paddedValues(seg) for seg in self.baseSegments])

        shift = self._baseOffsets[segment] if segment in self._baseOffsets else 0
        paddedOffset = segment.offset - shift
        # if padding reaches before the start of the message
        if paddedOffset < 0:
            toPrepend = [numpy.nan] * -paddedOffset
        else:
            toPrepend = []
        paddedNext = paddedOffset + self._maxLen
        # if padding reaches after the end of the message
        if paddedNext > len(segment.analyzer.values):
            toAppend = [numpy.nan] * (paddedNext - len(segment.analyzer.values))
        else:
            toAppend = []
        values = toPrepend + \
                 segment.analyzer.values[max(0, paddedOffset):min(len(segment.analyzer.values), paddedNext)] + \
                 toAppend

        assert len(values) == self._maxLen, "value padding failed"
        # overlapping offset from -1
        if shift >= 0:
            assert tuple(values[shift:shift + segment.length]) == segment.values, "value padding failed (positive shift)"
        else:
            assert tuple(values[0:shift + segment.length]) == segment.values[-shift:], "value padding failed (negative shift)"
        return values

    def baseOffset(self, segment: MessageSegment):
        """The offset of the given segment from the relative base offset of all the baseSegments in this object."""
        return self._baseOffsets[segment] if segment in self._baseOffsets else 0


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
        >>> from nemere.utils.loader import BaseLoader
        >>> from nemere.inference.analyzers import Value
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
        Calculated distances for 37 segment pairs in ... seconds.
        >>> clusterer = DBSCANsegmentClusterer(dc, eps=1.0, min_samples=3)
        >>> tg = TemplateGenerator(dc, clusterer)
        >>> templates = tg.generateTemplates()
        DBSCAN epsilon: 1.000, minpts: 3
        >>> pprint([t.baseSegments for t in templates])
        [[MessageSegment 4 bytes at (0, 4): 01020304 | values: (1, 2, 3...,
          MessageSegment 3 bytes at (0, 3): 020304 | values: (2, 3, 4),
          MessageSegment 3 bytes at (0, 3): 010304 | values: (1, 3, 4),
          MessageSegment 2 bytes at (0, 2): 0204 | values: (2, 4),
          MessageSegment 2 bytes at (0, 2): 0203 | values: (2, 3),
          MessageSegment 7 bytes at (0, 7): 141e253245021e | values: (20, 30, 37...,
          MessageSegment 3 bytes at (0, 3): 250545 | values: (37, 5, 69),
          MessageSegment 4 bytes at (0, 4): 46020304 | values: (70, 2, 3...,
          MessageSegment 4 bytes at (0, 4): 03020304 | values: (3, 2, 3...]]
        >>> pprint(clusterer.getClusters())
        {0: [MessageSegment 4 bytes at (0, 4): 01020304 | values: (1, 2, 3...,
             MessageSegment 3 bytes at (0, 3): 020304 | values: (2, 3, 4),
             MessageSegment 3 bytes at (0, 3): 010304 | values: (1, 3, 4),
             MessageSegment 2 bytes at (0, 2): 0204 | values: (2, 4),
             MessageSegment 2 bytes at (0, 2): 0203 | values: (2, 3),
             MessageSegment 7 bytes at (0, 7): 141e253245021e | values: (20, 30, 37...,
             MessageSegment 3 bytes at (0, 3): 250545 | values: (37, 5, 69),
             MessageSegment 4 bytes at (0, 4): 46020304 | values: (70, 2, 3...,
             MessageSegment 4 bytes at (0, 4): 03020304 | values: (3, 2, 3...]}

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

        :param filterNoise: if False, the first element in the returned list of clusters
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
        >>> from nemere.utils.loader import BaseLoader
        >>> from nemere.inference.analyzers import Value
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
        Calculated distances for 4 segment pairs in ... seconds.
        >>> clusterer = DBSCANsegmentClusterer(dc, eps=1, min_samples=2)
        >>> print(clusterer._distances)
        [[0.     0.3975 0.125 ]
         [0.3975 0.     0.5483]
         [0.125  0.5483 0.    ]]
        >>> clusterer._nearestPerNeigbor()
        [(2, 0.125), (0, 0.3975), (0, 0.125)]

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
        npn = [self._dc.neighbors(seg) for seg in self._dc.segments]
        # iterate all the k-th neighbors up to 2 * log(#neighbors)
        dpnmln = list()
        for k in range(0, len(npn) - 1):
            kthNeigbors4is = [idn[k][1] for idn in npn if idn[k][1] > 0][:2 * lnN]
            if len(kthNeigbors4is) > 0:
                dpnmln.append(numpy.mean(kthNeigbors4is))
            else:
                dpnmln.append(numpy.nan)

        # enumerate the means of deltas starting from an offset of log(#neighbors)
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

    def __init__(self, dc: DistanceCalculator, segments: Sequence[MessageSegment] = None, **kwargs):
        """

        :param dc:
        :param kwargs: e. g. epsilon: The DBSCAN epsilon value, if it should be fixed.
            If not given (None), it is autoconfigured.
        """
        super().__init__(dc, segments)

        if len(kwargs) == 0:
            # from math import log
            # lnN = round(log(self.distances.shape[0]))
            self.min_cluster_size = self.steepestSlope()[0] # round(lnN * 1.5)
        elif 'min_cluster_size' in kwargs:
            self.min_cluster_size = kwargs['min_cluster_size']
        else:
            raise ValueError("Parameters for HDBSCAN without autoconfiguration missing. "
                             "Requires min_cluster_size.")
        self.min_samples = round(math.sqrt(len(dc.segments)))

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
        dbscan.fit(self._distances.astype(float))
        return dbscan.labels_

    def __repr__(self):
        return 'HDBSCAN mcs {} ms {}'.format(self.min_cluster_size, self.min_samples)


class OPTICSsegmentClusterer(AbstractClusterer):
    """
    Ordering Points To Identify the Clustering Structure

    https://scikit-learn.org/stable/modules/generated/sklearn.cluster.OPTICS.html#sklearn.cluster.OPTICS
    """

    def __init__(self, dc: DistanceCalculator, segments: Sequence[MessageSegment] = None, **kwargs):
        """

        :param dc:
        :param kwargs: e. g. epsilon: The DBSCAN epsilon value, if it should be fixed.
            If not given (None), it is autoconfigured.
        """
        super().__init__(dc, segments)

        self.min_samples = round(math.sqrt(len(dc.segments)))
        self.max_eps = .4
        if 'min_samples' in kwargs:
            self.min_samples = kwargs['min_samples']
        if 'max_eps' in kwargs:
            self.max_eps = kwargs['max_eps']

    def getClusterLabels(self) -> numpy.ndarray:
        """
        Cluster the entries in the similarities parameter by OPTICS
        and return the resulting labels.

        :return: (numbered) cluster labels for each segment in the order given in the (symmetric) distance matrix
        """
        if numpy.count_nonzero(self._distances) == 0:  # the distance matrix contains only identical segments
            return numpy.zeros_like(self._distances[0], int)

        optics = OPTICS(metric='precomputed', min_samples=self.min_samples, max_eps=self.max_eps)
        print("OPTICS min samples:", self.min_samples, "max eps:", self.max_eps)
        optics.fit(self._distances)  #.astype(float)
        return optics.labels_

    def __repr__(self):
        return 'OPTICS ms {} maxeps {}'.format(self.min_samples, self.max_eps)


class DBSCANsegmentClusterer(AbstractClusterer):
    """
    Wrapper for DBSCAN from the sklearn.cluster module including autoconfiguration of the parameters.
    """

    def __init__(self, dc: DistanceCalculator, segments: Sequence[MessageSegment] = None,
                 interp_method="spline", **kwargs):
        """
        :param dc:
        :param segments: subset of segments from dc to cluster, use all segments in dc if None
        :param kwargs: e. g. epsilon: The DBSCAN epsilon value, if it should be fixed.
            If not given (None), it is autoconfigured.
            For autoconfiguration with Kneedle applied to the ECDF of dissimilarities,
            S is Kneedle's sensitivity parameter with a default of 0.8.
        """
        super().__init__(dc, segments)

        self._clusterlabelcache = None
        self.kneelocator = None  # type: Union[None, KneeLocator]

        self.S = kwargs["S"] if "S" in kwargs else 0.8
        self.k = kwargs["k"] if "k" in kwargs else 0
        if len(kwargs) == 0 or "S" in kwargs or "k" in kwargs:
            self.min_samples, self.eps = self._autoconfigure(interp_method=interp_method)
        else:  # eps and min_samples given, preventing autoconfiguration
            if not 'eps' in kwargs or not 'min_samples' in kwargs:
                raise ValueError("Parameters for DBSCAN without autoconfiguration missing. "
                                 "Requires eps and min_samples.")
            self.min_samples, self.eps = kwargs['min_samples'], kwargs['eps']

    def getClusterLabels(self, noCache=False) -> numpy.ndarray:
        """
        Cluster the entries in the similarities parameter by DBSCAN
        and return the resulting labels.

        :return: (numbered) cluster labels for each segment in the order given in the (symmetric) distance matrix
        """
        if self._clusterlabelcache is not None and noCache == False:
            return self._clusterlabelcache

        import sklearn.cluster

        if numpy.count_nonzero(self._distances) == 0:  # the distance matrix contains only identical segments
            return numpy.zeros_like(self._distances[0], int)

        dbscan = sklearn.cluster.DBSCAN(eps=self.eps, min_samples=self.min_samples, metric='precomputed')
        print("DBSCAN epsilon: {:0.3f}, minpts: {}".format(self.eps, int(self.min_samples)))
        dbscan.fit(self._distances)
        self._clusterlabelcache = dbscan.labels_
        return dbscan.labels_

    def __repr__(self):
        return 'DBSCAN eps {:0.3f} mpt {:0.0f}'.format(self.eps, self.min_samples) \
            if self.eps and self.min_samples \
            else 'DBSCAN unconfigured (need to set epsilon and min_samples)'

    def _autoconfigure(self, **kwargs):
        """
        Auto configure the clustering parameters epsilon and minPts regarding the input data

        :return: min_samples, epsilon
        """
        # return self._autoconfigureKneedle(**kwargs)
        return self._autoconfigureECDFKneedle(**kwargs)

    def _autoconfigureMPC(self):
        """
        Auto configure the clustering parameters epsilon and minPts regarding the input data
        Maximum Positive Curvature

        :return: min_samples, epsilon
        """
        from nemere.utils.baseAlgorithms import autoconfigureDBSCAN
        neighbors = [self.distanceCalculator.neighbors(seg) for seg in self.distanceCalculator.segments]
        epsilon, min_samples, k = autoconfigureDBSCAN(neighbors)
        print("eps {:0.3f} autoconfigured (MPC) from k {}".format(epsilon, k))
        return min_samples, epsilon

    def _maximumPositiveCurvature(self):
        """
        Use implementation of utils.baseAlgorithms to determine the maximum positive curvature
        :return: k, min_samples
        """
        from nemere.utils.baseAlgorithms import autoconfigureDBSCAN
        e, min_samples, k = autoconfigureDBSCAN(
            [self.distanceCalculator.neighbors(seg) for seg in self.distanceCalculator.segments])
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
        print("KneeLocator: dists of", self._distances.shape[0], "neighbors, k", k, "min_samples", min_samples)

        # get k-nearest-neighbor distances:
        neighdists = self._knearestdistance(k)
            # # add a margin relative to the remaining interval to the number of neighbors
            # round(k + (self._distances.shape[0] - 1 - k) * .2))
            # # round(minpts + 0.5 * (self.distances.shape[0] - 1 - min_samples))

        # # knee by Kneedle alogithm: https://ieeexplore.ieee.org/document/5961514
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

    kneeyThreshold = 0.1
    splineSmooth = 0.03

    def _autoconfigureECDFKneedle(self, interp_method="spline", recurse=True, trim=None):
        """

        >>> from kneed import KneeLocator
        >>> from scipy import interpolate
        >>> from itertools import chain
        >>> from tabulate import tabulate
        >>> import matplotlib.pyplot as plt
        >>>
        >>> from nemere.utils.loader import SpecimenLoader
        >>> from nemere.inference.segmentHandler import bcDeltaGaussMessageSegmentation
        >>> from nemere.utils.baseAlgorithms import ecdf
        >>> from nemere.inference.templates import DistanceCalculator, DBSCANsegmentClusterer
        >>> from nemere.inference.analyzers import MessageAnalyzer, Value
        >>>
        >>> specimens = SpecimenLoader("../input/deduped-orig/ntp_SMIA-20111010_deduped-100.pcap", 2, True)
        >>> segmentsPerMsg = MessageAnalyzer.convertAnalyzers(bcDeltaGaussMessageSegmentation(specimens, 1.2), Value)
        Segmentation by inflections of sigma-1.2-gauss-filtered bit-variance.
        >>> segments = list(chain.from_iterable(segmentsPerMsg))
        >>> dc = DistanceCalculator(segments)
        Calculated distances for 448879 segment pairs in ... seconds.
        >>> clusterer = DBSCANsegmentClusterer(dc, segments, S=24)
        DBSCANsegmentClusterer: eps 0.200 autoconfigured (Kneedle on ECDF with S 24) from k 2
        >>> print(clusterer.k)
        2
        >>> kneels = list()
        >>> for k in range(1,10):
        ...     neighdists = clusterer._knearestdistance(k, True)
        ...     knncdf = ecdf(neighdists, True)
        ...     tck = interpolate.splrep(knncdf[0], knncdf[1], s=DBSCANsegmentClusterer.splineSmooth)
        ...     Ds_y = interpolate.splev(knncdf[0], tck, der=0)
        ...     kneel = KneeLocator(knncdf[0], Ds_y, S=clusterer.S, curve='concave', direction='increasing')
        ...     kneels.append(kneel)
        >>>     plt.plot(knncdf[0], knncdf[1], label=f"k = {k}")  # doctest: +SKIP
        >>>     plt.plot(knncdf[0], Ds_y, label=f"k = {k} (smoothed)")  # doctest: +SKIP
        >>> kneelist = [(k, locator.all_knees) for k, locator in enumerate(kneels,1)]
        >>> print(tabulate(kneelist))
        -  ---------------------
        1  {0.18154389003537735}
        2  {0.19987050867290748}
        3  {0.22025666655513917}
        4  {0.237042781964657}
        5  {0.24426345351319875}
        6  {0.25393409495926683}
        7  {0.2658486707566462}
        8  {0.2716385690789474}
        9  {0.27354003906249996}
        -  ---------------------
        >>> plt.legend()  # doctest: +SKIP
        >>> plt.show()  # doctest: +SKIP


        kneel = KneeLocator(knncdf[0], knncdf[1], S=clusterer.S, curve='concave', direction='increasing',
                            interp_method='polynomial', polynomial_degree=5)

        knee_index = sum(knncdf[0]<clusterer.kneelocator.knee)
        trimX = knncdf[0][:knee_index]
        trimY = knncdf[1][:knee_index]
        tck = interpolate.splrep(trimX, trimY, s=0.03*len(trimX)/len(knncdf[0]))
        Ds_y = interpolate.splev(trimX, tck, der=0)
        kneel = KneeLocator(trimX, Ds_y, S=clusterer.S, curve='concave', direction='increasing')
        kneel.plot_knee()
        kneel.plot_knee_normalized()
        plt.show()


        :return: min samples and epsilon
        """
        from math import log
        import kneed
        from kneed import KneeLocator

        if len(self.segments) < 4:
            raise ClusterAutoconfException(f"No knee could be found: too few elements in input ({len(self.segments)}).")

        # only unique!
        min_samples = round(log(len(self.segments)))

        if self.k <= 0:
            self.k = self._selectK(min_samples)

        # only unique
        neighdists = self._knearestdistance(self.k, True)
        knncdf = ecdf(neighdists, True)
        if isinstance(trim, float):
            fullcdflen = len(knncdf)
            trim_index = sum(knncdf[0] < trim)
            knncdf = (knncdf[0][:trim_index], knncdf[1][:trim_index])
        try:
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                if kneed.__version__ < '0.7.0' or interp_method=='polynomial':
                    kneel = KneeLocator(knncdf[0], knncdf[1], S=self.S, curve='concave', direction='increasing',
                                        interp_method='polynomial')  # polynomial prevents errors at the fringes
                                                                     # interp1d does not smooth fringes sufficiently
                else:
                    # somewhere between version 0.4.2 and 0.7 of the kneedle implementation, the behaviour at fringes
                    # changed significantly. polynomial_degree=5 does help but still the resulting knee is a little higher.
                    if isinstance(trim, float):
                        smooth = type(self).splineSmooth * trim_index / fullcdflen
                    else:
                        smooth = type(self).splineSmooth
                    try:
                        tck = interpolate.splrep(knncdf[0], knncdf[1], s=smooth)
                        Ds_y = interpolate.splev(knncdf[0], tck, der=0)
                        kneel = KneeLocator(knncdf[0], Ds_y, S=self.S, curve='concave', direction='increasing')
                    except TypeError:
                        # should be dealt with by the exception on less than 4 segments,
                        #   but just to be sure this fails controlled
                        raise ClusterAutoconfException(
                            f"No knee could be found: too few elements in input ({len(self.segments)}).")
                        # import IPython; IPython.embed()
                    # handle false knees at the left fringe due to interpolation
                    if max(kneel.all_knees_y) < type(self).kneeyThreshold:
                        if len(kneel.minima_indices) < 2 or len(kneel.maxima_indices) < 1:
                            raise ClusterAutoconfException("No knee could be found: left fringe error")
                        # start new kneedle from the middle between second to last minimum and last maximum on
                        mi0 = kneel.maxima_indices[-1] - kneel.minima_indices[-2]
                        trimmedX = knncdf[0][mi0:]
                        trimmedY = Ds_y[mi0:]
                        kneel = KneeLocator(trimmedX, trimmedY, S=self.S, curve='concave', direction='increasing')
                        print(f"Fringe corrected to {trimmedX[0]:.3f} in ECDF for Kneedle."
                              f" - minima_indices {kneel.minima_indices} - maxima_indices {kneel.maxima_indices}")
                # finally in case, no knee or no probable knee (y < .3 / norm_knee_y)
                if kneel.knee is None or max(kneel.all_knees_y) < 0.3 or kneel.norm_knee_y == 0:
                    if recurse:
                        self.k = self._sharpestKnee(min_samples)
                        return self._autoconfigureECDFKneedle(recurse=False, trim=trim)
                    else:
                        if kneel.knee is None:
                            excStr = "No knee could be found."
                        else:
                            excStr = "No knee could be found: knee {:.3f} and norm_knee_y {:.3f}".format(
                                kneel.knee, kneel.norm_knee_y)
                        # # # #
                        try:
                            kneel.plot_knee()
                            import matplotlib.pyplot as plt
                            plt.plot(knncdf[0], knncdf[1], label="raw ECDF")
                            plt.savefig("knee-exception.pdf")
                        except:
                            pass
                        # # # #
                        raise ClusterAutoconfException(excStr)
        except ValueError as e:
            raise ClusterAutoconfException("No knee could be found. Original exception was:\n  " + str(e))
        # use the rightmost of multiple knees
        if len(kneel.all_knees) > 1:
            epsilon = max(kneel.all_knees)
        else:
            epsilon = kneel.knee
        self.kneelocator = kneel

        print("DBSCANsegmentClusterer: eps {:0.3f} autoconfigured (Kneedle on ECDF with S {}) from k {}".format(epsilon, self.S, self.k))
        return min_samples, epsilon


    def autoconfigureEvaluation(self, filename: str, markeps: float = False):
        """
        Auto configure the clustering parameters epsilon and minPts regarding the input data

        :return: minpts, epsilon
        """
        import numpy
        import matplotlib.pyplot as plt
        from math import ceil, log
        from scipy.ndimage.filters import gaussian_filter1d
        from kneed import KneeLocator

        from nemere.utils.baseAlgorithms import ecdf

        sigma = log(len(self.segments))/2
        # k, min_samples = self._maximumPositiveCurvature()
        # smoothknearest = dict()
        # seconddiffMax = dict()
        # maxD = 0
        # maxK = 0
        minD = 1
        minK = None
        minX = None
        # just the first... int(len(self.segments) * 0.7)
        for curvK in range(0, ceil(log(len(self.segments) ** 2))):
            neighdists = self._knearestdistance(curvK)
            knncdf = ecdf(neighdists, True)
            smoothknn = gaussian_filter1d(knncdf[1], sigma)
            diff2smooth = numpy.diff(smoothknn, 2) / numpy.diff(knncdf[0])[1:]
            mX = diff2smooth.argmin()
            if minD > diff2smooth[mX]:
                print(curvK, minD)
                minD = diff2smooth[mX]
                minK = curvK
                minX = knncdf[0][mX+1]

            # seconddiffMax[curvK] = seconddiff.max()
            # if seconddiffMax[curvK] > maxD:
            #     maxD = seconddiffMax[curvK]
            #     maxK = curvK

        k = minK
        # noinspection PyStatementEffect
        minX
        # epsilon = minX
        min_samples = sigma*2

        neighdists = self._knearestdistance(k)
        knncdf = ecdf(neighdists, True)
        # smoothknn = gaussian_filter1d(knncdf[1], sigma)
        kneel = KneeLocator(knncdf[0], knncdf[1], curve='concave', direction='increasing')
        epsilon = kneel.knee * 0.8

        print("selected k = {}; epsilon = {:.3f}; min_samples = {:.0f}".format(k, epsilon, min_samples))


        # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        # plots of k-nearest-neighbor distance histogram and "knee"
        plt.rc('legend', frameon=False)
        fig = plt.gcf()
        fig.set_size_inches(16, 9)

        numK = 100 # ceil(sigma**2))
        minK = max(0, k - numK//2)
        maxK = min( len(self.segments) - 1, k + numK//2)
        for curvK in range(minK, maxK):
            alpha = 1 if curvK == k else .4
            neighdists = self._knearestdistance(curvK)
            knncdf = ecdf(neighdists, True)

            if curvK == k:
                smoothknn = gaussian_filter1d(knncdf[1], sigma)

                # diff1smooth = numpy.gradient(smoothknn)
                diff2smooth = numpy.diff(smoothknn, 2)
                # diff3smooth = numpy.gradient(numpy.gradient(numpy.gradient(smoothknn)))

                diffknn = numpy.diff(smoothknn)
                mX = diffknn.argmax()
                mQ = 0.1 * numpy.diff(knncdf[0]).mean() * sum(diffknn) # diffknn[mX] * 0.15
                a = next(xA for xA in range(mX, -1, -1) if diffknn[xA] < mQ or xA == 0)
                b = next(xB for xB in range(mX, len(knncdf[0])-1) if diffknn[xB] < mQ or xB == len(knncdf[0]) - 2)
                # tozerone = cdist(numpy.array(knncdf).transpose(), numpy.array([[0, 1]])).argmin()

                diffshrink = 0.05

                plt.plot(knncdf[0], knncdf[1], alpha=alpha, label=curvK, color='lime')
                plt.plot(knncdf[0], smoothknn, label="$g(\cdot)$", color='red')
                plt.plot(knncdf[0][1:], diffshrink * diffknn / numpy.diff(knncdf[0]), linestyle="dashed", color='blue',
                         label="$\Delta(g(\cdot))$")
                plt.plot(knncdf[0][2:], 20 * diff2smooth, linestyle="-.", color='violet', label="$\Delta^2(g(\cdot))$")
                plt.scatter(knncdf[0][a+1], diffshrink * diffknn[a] / (knncdf[0][a+1] - knncdf[0][a]),
                            label="a = {:.3f}".format(knncdf[0][a+1]))
                plt.scatter(knncdf[0][b+1], diffshrink * diffknn[b] / (knncdf[0][b+1] - knncdf[0][b]),
                            label="b = {:.3f}".format(knncdf[0][b+1]) )
                # plt.plot(knncdf[0], diff3smooth * 100, linestyle="dotted",
                #             label="$\Delta^3(g(\cdot))$")
                plt.axvline(knncdf[0][diff2smooth.argmin()], color='indigo', label="$min(\Delta^2(\cdot))$")

            else:
                plt.plot(knncdf[0], knncdf[1], alpha=alpha)
        plt.axvline(epsilon, label="alt eps {:.3f}".format(epsilon), linestyle="dashed", color='lawngreen')
        # plt.axvline(sqrt(epsilon), label="sqrt(neps) {:.3f}".format(sqrt(epsilon)),
        #             linestyle="dashed", color='green')
        if markeps:
            plt.axvline(markeps, label="applied eps {:.3f}".format(markeps), linestyle="-.", color='orchid')

        plt.tight_layout(rect=[0,0,1,.95])
        plt.legend()
        plt.savefig(filename)

        # fig, ax = plt.subplots(1, 2)
        # axl, axr = ax.flat
        #
        # # farthest
        # # plt.plot([max([dpn[k] for nid, dpn in npn]) for k in range(0, len(npn)-1)], alpha=.4)
        # # axl.plot(dpnmln, alpha=.4)
        # # plt.plot([self._knearestdistance(k) for k in range( round(0.5 * (self.distances.shape[0]-1)) )])
        # disttril = numpy.tril(self._distances)
        # alldist = [e for e in disttril.flat if e > 0]
        # axr.hist(alldist, 50)
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

        plt.close('all')
        plt.clf()

        return min_samples, epsilon


    def _knearestmean(self, k: int):
        """
        it gives no significantly better results than the direct k-nearest distance,
        but requires more computation.

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


    def _knearestdistance(self, k: int, onlyUnique=False):
        """
        :param k: neighbor to be selected
        :param onlyUnique: if set to true, and dc.segments contains Templates that pool some of the self.segments,
            use the pooled distances only between unique segments,
        :return: The distances of the k-nearest neighbors for all distances of this clusterer instance.
        """
        if onlyUnique and isinstance(self.distanceCalculator, DelegatingDC):
            segments = self.distanceCalculator.segments
        else:
            segments = self.segments

        if not k < len(segments) - 1:
            raise IndexError("k={} exeeds the number of neighbors.".format(k))
        neighbordistances = [self._dc.neighbors(seg)[k][1] for seg in segments]
        return sorted(neighbordistances)


    @staticmethod
    def _kneebyruleofthumb(neighdists):
        """
        according to the paper
        (!? this is the wrong reference ?!) https://www.sciencedirect.com/science/article/pii/S0169023X06000218

        result is far (too far) left of the actual knee

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

    # def _selectK(self, maxK) -> int:
    #     """
    #     The k (as in in k-NN ECDF) to use.
    #
    #     :return: Passes through the maxK parameter unchanged.
    #     """
    #     return maxK

    def _selectK(self, maxK) -> int:
        """Searching for the sharpest knee in the ks up to maxK."""
        return self._sharpestKnee(maxK)

    def _sharpestKnee(self, maxK) -> int:
        """
        Determine the sharpest of the knees for the k-NN ECDFs for ks between 2 and self.k
        We define sharpness by the value of the y_difference at the maximum.

        :return: The k (as in in k-NN ECDF) with the sharpest knee.
        """
        sharpest = (maxK,0)
        for k in range(2,maxK+1):
            neighdists = self._knearestdistance(k, True)
            knncdf = ecdf(neighdists, True)
            tck = interpolate.splrep(knncdf[0], knncdf[1], s=type(self).splineSmooth)
            Ds_y = interpolate.splev(knncdf[0], tck, der=0)
            kneel = KneeLocator(knncdf[0], Ds_y, S=self.S, curve='concave', direction='increasing')
            ydmax = kneel.y_difference[kneel.maxima_indices[-1]]
            # print(f"Knee for k={k} has ydmax={ydmax:.5f}")
            if ydmax > sharpest[1]:
                sharpest = (k, ydmax)
        return sharpest[0]

    def preventLargeCluster(self):
        """Prevent a large cluster > 60 % of non-noise by searching for a smaller epsilon."""
        if self.kneelocator is None:
            # TODO this probably is worth raising an exception
            return
        clusterLabels = self.getClusterLabels()
        # if one cluster is larger than 60% of all non-noise segments...
        if type(self).largestClusterExceeds(clusterLabels, 0.6):
            print("Cluster larger 60% found. Trim the knncdf to the knee.")
            self.min_samples, self.eps = self._autoconfigureECDFKneedle(trim=self.kneelocator.knee)
            # force re-clustering without cache
            clusterLabels = self.getClusterLabels(True)

    @staticmethod
    def largestClusterExceeds(clusterLabels, threshold):
        clusterSizes = Counter(clusterLabels)
        return max(clusterSizes.values()) > sum(cs for cl, cs in clusterSizes.items() if cl != -1) * threshold


class DBSCANadjepsClusterer(DBSCANsegmentClusterer):
    """
    DBSCANsegmentClusterer with adjustment of the auto-configured epsilon to fix systematic deviations of the optimal
    epsilon value for heuristically determined segments with NEMESYS and zero-segmenter.

    The adjustment is aware of changes in the implementation of the kneed module between versions
    lower than and from version 0.7.
    """
    epsfrac = 3  # done: 5, 4,
    epspivot = 0.15

    def __init__(self, dc: DistanceCalculator, segments: Sequence[MessageSegment] = None, **kwargs):
        super().__init__(dc, segments, **kwargs)

    def _autoconfigureECDFKneedle(self, **kwargs):
        min_samples, autoeps = super()._autoconfigureECDFKneedle(**kwargs)
        if kneed.__version__ < '0.7.0':
            # reduce k if no realistic eps is detected
            if autoeps < 0.05:
                self.k //= 2
                self.min_samples, autoeps = self._autoconfigure(**kwargs)
            # adjust epsilon
            adjeps = autoeps + autoeps / type(self).epsfrac * (1 if autoeps < type(self).epspivot else -1)
        else:
            # adjust epsilon, depending on the sharpness of the knee:
            # the sharper (higher ydmax), the farther to the "right"
            ydmax = self.kneelocator.y_difference[self.kneelocator.maxima_indices[-1]]
            assert ydmax in self.kneelocator.y_difference_maxima
            # epsfact = 7*ydmax**2 + -3*ydmax + 0.8
            # epsfact = 17 * ydmax**2 + -12 * ydmax + 2.5
            # epsfact = 21 * ydmax ** 2 + -14.7 * ydmax + 2.7  # (instead of 3, just to improve ntp-1000)
            #
            # polyfit after iterated best epsilon
            epsfact = 16 * ydmax**2 - 10 * ydmax + 1.8
            adjeps = epsfact * autoeps
            print(f"DBSCANadjepsClusterer: eps adjusted to {adjeps:.3f} by {epsfact:.2f} based on y_dmax {ydmax:.2f}")
        return min_samples, adjeps

    def _selectK(self, maxK) -> int:
        """Searching sharpest knee in the ks up to maxK."""
        return self._sharpestKnee(maxK)



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
  * internally: neighbors
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
        Calculated distances for 45 segment pairs in ... seconds.
        >>> ddc = DelegatingDC(segments)
        Calculated distances for 23 segment pairs in ... seconds.
        >>> print(tabulate(enumerate(ddc.segments)))
        -  ----------------------------------------------------------------
        0  MessageSegment 4 bytes at (0, 4): 01020304 | values: (1, 2, 3...
        1  MessageSegment 3 bytes at (0, 3): 020304 | values: (2, 3, 4)
        2  MessageSegment 3 bytes at (0, 3): 250545 | values: (37, 5, 69)
        3  MessageSegment 4 bytes at (0, 4): 00000000 | values: (0, 0, 0...
        4  MessageSegment 4 bytes at (0, 4): 03020304 | values: (3, 2, 3...
        5  Template 2 bytes: (2, 4) | #base 2
        6  Template 7 bytes: (20, 30, 37... | #base 3
        -  ----------------------------------------------------------------
        >>> print(tabulate(enumerate(dc.segments)))
        -  -------------------------------------------------------------------------
        0  MessageSegment 4 bytes at (0, 4): 01020304 | values: (1, 2, 3...
        1  MessageSegment 3 bytes at (0, 3): 020304 | values: (2, 3, 4)
        2  MessageSegment 2 bytes at (0, 2): 0204 | values: (2, 4)
        3  MessageSegment 2 bytes at (0, 2): 0204 | values: (2, 4)
        4  MessageSegment 7 bytes at (0, 7): 141e253245021e | values: (20, 30, 37...
        5  MessageSegment 7 bytes at (0, 7): 141e253245021e | values: (20, 30, 37...
        6  MessageSegment 3 bytes at (0, 3): 250545 | values: (37, 5, 69)
        7  MessageSegment 4 bytes at (0, 4): 00000000 | values: (0, 0, 0...
        8  MessageSegment 7 bytes at (0, 7): 141e253245021e | values: (20, 30, 37...
        9  MessageSegment 4 bytes at (0, 4): 03020304 | values: (3, 2, 3...
        -  -------------------------------------------------------------------------

        :param segments: List of segments to calculate pairwise distances from.
        :param reliefFactor: Increase the non-linearity of the dimensionality penalty for mismatching segment lengths.
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
        [MessageSegment 4 bytes at (0, 4): 01020304 | values: (1, 2, 3...,
         MessageSegment 3 bytes at (0, 3): 020304 | values: (2, 3, 4),
         MessageSegment 3 bytes at (0, 3): 250545 | values: (37, 5, 69),
         MessageSegment 4 bytes at (0, 4): 00000000 | values: (0, 0, 0...,
         MessageSegment 4 bytes at (0, 4): 03020304 | values: (3, 2, 3...]
        >>> pprint(t4d[1])
        [Template 2 bytes: (2, 4) | #base 2, Template 7 bytes: (20, 30, 37... | #base 3]
        >>> pprint([sorted(((k, t4d[2][k]) for k in t4d[2].keys()), key=lambda x: x[1])])
        [[(MessageSegment 2 bytes at (0, 2): 0204 | values: (2, 4), 5),
          (MessageSegment 2 bytes at (0, 2): 0204 | values: (2, 4), 5),
          (MessageSegment 7 bytes at (0, 7): 141e253245021e | values: (20, 30, 37...,
           6),
          (MessageSegment 7 bytes at (0, 7): 141e253245021e | values: (20, 30, 37...,
           6),
          (MessageSegment 7 bytes at (0, 7): 141e253245021e | values: (20, 30, 37...,
           6)]]


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

    def segments2index(self, segmentList: Iterable[AbstractSegment]):
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
        -  -------------------------------------------------------------------------
        0  MessageSegment 7 bytes at (0, 7): 141e253245021e | values: (20, 30, 37...
        1  MessageSegment 7 bytes at (0, 7): 141e253245021e | values: (20, 30, 37...
        2  MessageSegment 7 bytes at (0, 7): 141e253245021e | values: (20, 30, 37...
        -  -------------------------------------------------------------------------



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

    def distancesSubset(self, As: Sequence[AbstractSegment], Bs: Sequence[AbstractSegment] = None) \
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
        # simtrx = numpy.ones((len(clusterI), len(clusterJ)))
        simtrx = MemmapDC.largeFilled((len(clusterI), len(clusterJ)))

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
        # simtrx = numpy.ones((len(clusterI), len(clusterJ)))
        simtrx = MemmapDC.largeFilled((len(clusterI), len(clusterJ)))

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


# noinspection PyAbstractClass
class MemmapDC(DelegatingDC):
    maxMemMatrix = 750000
    if parallelDistanceCalc:
        maxMemMatrix /= cpu_count()

    @staticmethod
    def _getDistanceMatrix(distances: Iterable[Tuple[int, int, float]], segmentCount: int) -> numpy.ndarray:
        # noinspection PyProtectedMember
        """
        Arrange the representation of the pairwise similarities of the input parameter in an symmetric array.
        The order of the matrix elements in each row and column is the same as in self._segments.

        Distances for pair not included in the list parameter are considered incomparable and set to -1 in the
        resulting matrix.

        Used in constructor.

        >>> from tabulate import tabulate
        >>> from nemere.inference.templates import DistanceCalculator
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
        >>> print(tabulate(dm, floatfmt=".2f"))
        -----  -----  -----  -----  -----  -----  -----
         0.00   0.21   0.40   0.81   0.75   1.00  -1.00
         0.21   0.00   0.30   0.79   0.68   1.00  -1.00
         0.40   0.30   0.00   0.71   0.70   1.00  -1.00
         0.81   0.79   0.71   0.00   0.58   1.00  -1.00
         0.75   0.68   0.70   0.58   0.00   1.00  -1.00
         1.00   1.00   1.00   1.00   1.00   0.00  -1.00
        -1.00  -1.00  -1.00  -1.00  -1.00  -1.00   0.00
        -----  -----  -----  -----  -----  -----  -----

        :param distances: The pairwise similarities to arrange.
        :return: The distance matrix for the given similarities.
            -1 for each undefined element, 0 in the diagonal, even if not given in the input.
        """
        from tempfile import NamedTemporaryFile
        from nemere.inference.segmentHandler import matrixFromTpairs

        tempfile = NamedTemporaryFile()
        distancesSwap = numpy.memmap(tempfile.name, dtype=numpy.float16, mode="w+", shape=(segmentCount,segmentCount))

        simtrx = matrixFromTpairs(distances, range(segmentCount),
                                  incomparable=-1, simtrx=distancesSwap)  # TODO handle incomparable values (resolve and replace the negative value)
        return simtrx


    @staticmethod
    def largeFilled(shape: Tuple[int,int], filler=1):
        if shape[0] * shape[1] > MemmapDC.maxMemMatrix:
            from tempfile import NamedTemporaryFile
            tempfile = NamedTemporaryFile()
            simtrx = numpy.memmap(tempfile.name, dtype=numpy.float16, mode="w+", shape=shape)
            simtrx.fill(filler)
        elif filler == 1:
            simtrx = numpy.ones(shape, dtype=numpy.float16)
        elif filler == 0:
            simtrx = numpy.zeros(shape, dtype=numpy.float16)
        else:
            simtrx = numpy.empty(shape, dtype=numpy.float16)
            simtrx.fill(filler)
        return simtrx


def __testing_generateTestSegmentsWithDuplicates():
    """
    Generates dummy segments with duplicates for testing.

    :return: List of message segments.
    """
    from netzob.Model.Vocabulary.Messages.RawMessage import RawMessage
    from nemere.inference.analyzers import Value
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

