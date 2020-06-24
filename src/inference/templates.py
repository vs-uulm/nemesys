from typing import List, Dict, Union, Iterable, Sequence, Tuple, Iterator
from os import cpu_count
import numpy, scipy.spatial, itertools

from netzob.Model.Vocabulary.Messages.AbstractMessage import AbstractMessage

from inference.analyzers import MessageAnalyzer, Value
from inference.segments import MessageSegment, AbstractSegment, CorrelatedSegment, HelperSegment, TypedSegment


debug = False

parallelDistanceCalc = False
"""
activate parallel/multi-processor calculation of dissimilarities 
in inference.templates.DistanceCalculator#_embdedAndCalcDistances
"""



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
        self._distances = type(self)._getDistanceMatrix(self._embdedAndCalcDistances(), len(self._quicksegments))

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

        >>> from utils.baseAlgorithms import generateTestSegments
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
        >>> from utils.baseAlgorithms import generateTestSegments
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
        from inference.segmentHandler import matrixFromTpairs
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
        >>> from utils.baseAlgorithms import generateTestSegments
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
        3  [1, 2, 3, 4]                 [2, 3, 4]     0.21
        3  [1, 2, 3, 4]                 [1, 3, 4]     0.30
        2  [1, 3, 4]                    [2, 4]        0.37
        3  [37, 5, 69]                  [1, 2, 3, 4]  0.75
        4  [20, 30, 37, 50, 69, 2, 30]  [0, 0, 0, 0]  1.00
        2  [1, 2, 3, 4]                 [2, 4]        0.44
        2  [1, 2, 3, 4]                 [2, 3]        0.40
        3  [20, 30, 37, 50, 69, 2, 30]  [37, 5, 69]   0.58
        4  [1, 2, 3, 4]                 [0, 0, 0, 0]  1.00
        2  [2, 4]                       [2, 3]        0.07
        4  [1, 2, 3, 4]                 [3, 2, 3, 4]  0.12
        -  ---------------------------  ------------  ----

        :return: List of Tuples
            (index of segment in self._segments), (segment length), (Tuple of segment analyzer values)
        """
        dissCount = 0
        lenGrps = self.groupByLength()  # segment list is in format of self._quicksegments

        import time
        pit_start = time.time()

        rslens = list(reversed(sorted(lenGrps.keys())))  # lengths, sorted by decreasing length

        if not parallelDistanceCalc:
            for outerlen in rslens:
                for diss in self._outerloop(lenGrps, outerlen, rslens):
                    dissCount += 1
                    yield diss

        #     profiler = cProfile.Profile()
        #     int_start = time.time()
        #     stats = profiler.runctx('self._outerloop(lenGrps, outerlen, distance, rslens)', globals(), locals())
        #     int_runtime = time.time() - int_start
        #     profiler.dump_stats("embdedAndCalcDistances-{:02.1f}-{}-i{:03d}.profile".format(
        #         int_runtime, self.segments[0].message.data[:5].hex(), outerlen))
        else:
            import concurrent.futures
            with concurrent.futures.ProcessPoolExecutor(max_workers=cpu_count()) as executor:   # Process # Thread
                futureDis = dict()
                for outerlen in rslens:
                    futureDis[executor.submit(self._outerloop, lenGrps, outerlen, rslens)] = outerlen
                futureRes = dict()
                for future in concurrent.futures.as_completed(futureDis.keys()):
                    futureRes[futureDis[future]] = future.result()
                for ol in rslens:
                    for diss in futureRes[ol]:
                        dissCount += 1
                        yield diss

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
        dissimilarities.extend([(i, l,
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
                    dlDist = (interseg[0], interseg[1], (
                        self.thresholdFunction(
                            mlDistance,
                            **self.thresholdArgs)
                    )
                              )  # minimum of dimensions
                    # # # # # # # # # # # # # # # # # # # # # # # #
                    dissimilarities.append(dlDist)
                    self._offsets[(interseg[0], interseg[1])] = embedded[1]
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
        Calculated distances for 37 segment pairs in ... seconds.
        >>> nbrs = dc.neigbors(segments[2], segments[3:7])
        >>> dsts = [dc.pairDistance(segments[2], segments[3]),
        ...         dc.pairDistance(segments[2], segments[4]),
        ...         dc.pairDistance(segments[2], segments[5]),
        ...         dc.pairDistance(segments[2], segments[6])]
        >>> nbrs
        [(0, 0.3677), (1, 0.4146), (3, 0.696), (2, 0.7935)]
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
        home = self.segments2index([segment])[0]
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
            return bytes(self._values.astype(int).tolist())
        if isinstance(self._values, Iterable):
            return bytes(self._values)
        return None


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
        If no DistanceCalculator is given, does not support a threshold function.

        >>> from tabulate import tabulate
        >>> from scipy.spatial.distance import cdist
        >>> from utils.baseAlgorithms import generateTestSegments
        >>> from inference.templates import DistanceCalculator, Template
        >>> listOfSegments = generateTestSegments()
        >>> DistanceCalculator.debug = False
        >>> dc = DistanceCalculator(listOfSegments)
        Calculated distances for 37 segment pairs in ... seconds.
        >>> center = [1,3,7,9]  # "mean"
        >>> tempearly = Template(center, listOfSegments)
        >>> dtml = tempearly.distancesToMixedLength(dc)
        Calculated distances for 46 segment pairs in ... seconds.
        >>> print(cdist([center], [listOfSegments[0].values], "canberra")[0,0]/len(center))
        0.24615384615384617
        >>> print(dtml[0][0])
        0.2461
        >>> round(cdist([center], [listOfSegments[0].values], "canberra")[0,0]/len(center), 2) == round(dtml[0][0], 2)
        True
        >>> print(tabulate(dtml))
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
        >>> center = listOfSegments[0]  # "medoid"
        >>> template = Template(center, listOfSegments)
        >>> print(tabulate(template.distancesToMixedLength(dc)))
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
    """
    Template for the representation of a segment type.
    """

    def __init__(self, values: Union[Tuple[Union[float, int]], MessageSegment],
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
        Calculated distances for 45 segment pairs in ... seconds.
        >>> ddc = DelegatingDC(segments)
        Calculated distances for 23 segment pairs in ... seconds.
        >>> print(tabulate(enumerate(ddc.segments)))
        -  ----------------------------------------------------------------
        0  MessageSegment 4 bytes at (0, 4): 01020304 | values: [1, 2, 3...
        1  MessageSegment 3 bytes at (0, 3): 020304 | values: [2, 3, 4]
        2  MessageSegment 3 bytes at (0, 3): 250545 | values: [37, 5, 69]
        3  MessageSegment 4 bytes at (0, 4): 00000000 | values: [0, 0, 0...
        4  MessageSegment 4 bytes at (0, 4): 03020304 | values: [3, 2, 3...
        5  Template 2 bytes: (2, 4) | #base 2
        6  Template 7 bytes: (20, 30, 37... | #base 3
        -  ----------------------------------------------------------------
        >>> print(tabulate(enumerate(dc.segments)))
        -  -------------------------------------------------------------------------
        0  MessageSegment 4 bytes at (0, 4): 01020304 | values: [1, 2, 3...
        1  MessageSegment 3 bytes at (0, 3): 020304 | values: [2, 3, 4]
        2  MessageSegment 2 bytes at (0, 2): 0204 | values: [2, 4]
        3  MessageSegment 2 bytes at (0, 2): 0204 | values: [2, 4]
        4  MessageSegment 7 bytes at (0, 7): 141e253245021e | values: [20, 30, 37...
        5  MessageSegment 7 bytes at (0, 7): 141e253245021e | values: [20, 30, 37...
        6  MessageSegment 3 bytes at (0, 3): 250545 | values: [37, 5, 69]
        7  MessageSegment 4 bytes at (0, 4): 00000000 | values: [0, 0, 0...
        8  MessageSegment 7 bytes at (0, 7): 141e253245021e | values: [20, 30, 37...
        9  MessageSegment 4 bytes at (0, 4): 03020304 | values: [3, 2, 3...
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
        [MessageSegment 4 bytes at (0, 4): 01020304 | values: [1, 2, 3...,
         MessageSegment 3 bytes at (0, 3): 020304 | values: [2, 3, 4],
         MessageSegment 3 bytes at (0, 3): 250545 | values: [37, 5, 69],
         MessageSegment 4 bytes at (0, 4): 00000000 | values: [0, 0, 0...,
         MessageSegment 4 bytes at (0, 4): 03020304 | values: [3, 2, 3...]
        >>> pprint(t4d[1])
        [Template 2 bytes: (2, 4) | #base 2, Template 7 bytes: (20, 30, 37... | #base 3]
        >>> pprint([sorted(((k, t4d[2][k]) for k in t4d[2].keys()), key=lambda x: x[1])])
        [[(MessageSegment 2 bytes at (0, 2): 0204 | values: [2, 4], 5),
          (MessageSegment 2 bytes at (0, 2): 0204 | values: [2, 4], 5),
          (MessageSegment 7 bytes at (0, 7): 141e253245021e | values: [20, 30, 37...,
           6),
          (MessageSegment 7 bytes at (0, 7): 141e253245021e | values: [20, 30, 37...,
           6),
          (MessageSegment 7 bytes at (0, 7): 141e253245021e | values: [20, 30, 37...,
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
        -  -------------------------------------------------------------------------
        0  MessageSegment 7 bytes at (0, 7): 141e253245021e | values: [20, 30, 37...
        1  MessageSegment 7 bytes at (0, 7): 141e253245021e | values: [20, 30, 37...
        2  MessageSegment 7 bytes at (0, 7): 141e253245021e | values: [20, 30, 37...
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
        from inference.segmentHandler import matrixFromTpairs

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


    @staticmethod
    def _templates4Paddings(segments: Iterable[MessageSegment]):
        raise NotImplementedError()


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

