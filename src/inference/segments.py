import math
from typing import Dict, List, Union, Type, Any, Tuple, Iterable
from abc import ABC, abstractmethod
import numpy

from netzob.Model.Vocabulary.Messages.AbstractMessage import AbstractMessage




### MessageAnalyzer base class #########################################

class MessageAnalyzer(ABC):
    """
    Subclasses fully describe a specific analysis upon a message and hold the results.
    """
    _analyzerCache = dict()

    U_BYTE = 0
    U_NIBBLE = 1

    _unit = U_BYTE
    _message = None
    _analysisArgs = None
    _values = None
    _startskip = 0

    # TODO per analysis method that does not generate one value per one byte: margins (head, tail) and skip

    def __init__(self, message: AbstractMessage, unit=U_BYTE):
        """
        Create object and set the message and the unit-size.

        :param message:
        :param unit:
        """
        self._message = message
        self._unit = unit

    @property
    def unit(self):
        return self._unit

    @property
    def values(self):
        """
        :return: The analysis values for this message, possibly prepended by NaN values
            in the amount of startskip (see there),
            after analyze() was called. None otherwise.
        """
        if self._values is None:
            return None
        return [numpy.nan] * self.startskip + self._values

    @property
    def valuesRaw(self):
        """
        :return: the values of this analysis without adding a startskip
        """
        return self._values

    @property
    def startskip(self):
        """
        :return: The number of bytes to skip to align the analysis result to actual bytes.
            Is relevant for calculations that need to accumulate some bytes before able to calculate something.
            E. g. mean, bit congruence
        """
        return self._startskip


    @property
    def message(self):
        """
        :return: The message this analyzer is related to.
        """
        return self._message


    @property
    def analysisParams(self):
        return self._analysisArgs


    def setAnalysisParams(self, *args):
        self._analysisArgs = args


    @abstractmethod
    def analyze(self):
        """
        Performs the analysis and writes the result (typically a list or ndarray of discrete values) to self.values
        :return:
        """
        MessageAnalyzer._analyzerCache[(type(self), self._unit, self._message, self._analysisArgs)] = self


    @staticmethod
    def findExistingAnalysis(analyzerclass: type, unit: int,
                             message: AbstractMessage, analysisArgs: Union[Any, Tuple]=None):
        """
        Efficiently obtain an analyzer by looking for an already existing identical object instance.

        :param analyzerclass: Required analyzer class
        :param unit: Whether to work on bytes or nibbles as basic unit
        :param message: The message to analyze
        :param analysisArgs: The arguments for the analyzer class, typically a tuple of parameter values
        :return: The requested analyzer for the message, if not in cache a new instance
        :rtype MessageAnalyzer
        """
        keytuple = (analyzerclass, unit, message, analysisArgs)
        if keytuple in MessageAnalyzer._analyzerCache:
            return MessageAnalyzer._analyzerCache[keytuple]
        else:
            ac = analyzerclass(message, unit)  # type: MessageAnalyzer
            if analysisArgs is None:
                analysisArgs = tuple()
            try:
                ac.setAnalysisParams(*analysisArgs)
            except TypeError as e:
                raise TypeError(str(e) + ' - Class: {}, Parameters: {}'.format(type(ac), analysisArgs))
            ac.analyze()
            return ac


    def ngrams(self, n: int):
        """
        :return: the ngrams of the message in order
        """
        from utils.baseAlgorithms import ngrams
        return ngrams(self._message.data, n)


    @staticmethod
    def nibblesFromBytes(bytesequence: bytes) -> List[int]:
        """
        Returns a byte sequence representing the nibbles of the input byte sequence.
        Only the 4 least significent bits of the output are set, the first 4 bits are zero.
        The most significant bits of each byte of the input is moved to a separate (half-)byte in big endian manner:
        (nibbleA | nibbleB) = (0000 | nibbleA), (0000 | nibbleB)

        :param bytesequence:
        :return:
        """
        from itertools import chain
        return list(chain.from_iterable([(by >> 4, by & 0x0f) for by in bytesequence]))


    @staticmethod
    def tokenDelta(tokenlist, unitsize=U_BYTE):
        """
        Relative differences between subsequent token values.

        :return:
        """
        if len(tokenlist) < 2:
            raise ValueError("Needs at least two tokens to determine a delta. Message is {}".format(tokenlist))

        if unitsize == MessageAnalyzer.U_NIBBLE:
            tokens = MessageAnalyzer.nibblesFromBytes(tokenlist)
        else:
            tokens = tokenlist

        return list(numpy.ediff1d(tokens))


    @staticmethod
    def localMaxima(sequence) -> Tuple[List, List]:
        """
        Determine local maxima of sequence.

        :param sequence: A sequence of subscriptable values.
        :return: a list of two lists:
            One of the sequence indices of the maxima,
            and one of the corresponding maxium values.
        """
        localmaxima = ([], [])
        if sequence[0] > sequence[1]:  # boundary value
            localmaxima[0].append(0)
            localmaxima[1].append(sequence[0])
        for index, center in enumerate(sequence[1:-1], 1):
            before = sequence[index - 1]
            after = sequence[index + 1]
            if before < center > after or numpy.isnan(before) and center > after:
                localmaxima[0].append(index)
                localmaxima[1].append(center)
        if sequence[-1] > sequence[-2]:  # boundary value
            localmaxima[0].append(len(sequence) - 1)
            localmaxima[1].append(sequence[-1])
        return localmaxima


    @staticmethod
    def localMinima(sequence) -> Tuple[List, List]:
        """
        Determine local minima of sequence.

        :param sequence: A sequence of subscriptable values.
        :return: a list of two lists:
            One of the sequence indices of the minima,
            and one of the corresponding minium values.
        """
        localminima = ([], [])
        if sequence[0] < sequence[1]:  # boundary value
            localminima[0].append(0)
            localminima[1].append(sequence[0])
        for index, center in enumerate(sequence[1:-1], 1):
            before = sequence[index - 1]
            after = sequence[index + 1]
            if before > center < after or numpy.isnan(before) and center < after:
                localminima[0].append(index)
                localminima[1].append(center)
        if sequence[-1] < sequence[-2]:  # boundary value
            localminima[0].append(len(sequence) - 1)
            localminima[1].append(sequence[-1])
        return localminima


    @staticmethod
    def zeroSequences(sequence) -> Tuple[List, List]:
        """
        Determine begins and ends of sequences of zeros in value.

        :param sequence: A sequence of subscriptable values.
        :return: a list of two lists:
            One of the sequence indices of the zero begin/ends,
            and one of the corresponding zero values.
        """
        zeros = ([], [])
        for index, center in enumerate(sequence[1:-1], 1):
            before = sequence[index - 1]
            after = sequence[index + 1]
            if center == 0:
                if before != 0 and after == 0 or before == 0 and after != 0:
                    zeros[0].append(index)
                    zeros[1].append(center)
        return zeros


    @staticmethod
    def plateouStart(sequence) -> Tuple[List, List]:
        """
        Determine begins of sequences of plateaus in value.

        :param sequence: A sequence of subscriptable values.
        :return: a list of two lists:
            (1) the sequence indices of the plateau begins,
            and (2) the corresponding values.
        """
        plateau = ([], [])
        for index, center in enumerate(sequence[1:-1], 1):
            before = sequence[index - 1]
            after = sequence[index + 1]
            if before != center == after:
                plateau[0].append(index)
                plateau[1].append(center)
        return plateau

    @staticmethod
    def separateNaNs(sequence) -> Tuple[List, List]:
        """
        Split sequences of NaNs off values to separate segments.

        :param sequence: input sequence of values
        :return: lists of cut positions and dummy 0.0-values
        """
        nansep = ([], [])
        if numpy.isnan(sequence[0]) and not numpy.isnan(sequence[1]):
            nansep[0].append(1)
            nansep[1].append(0.0)
        for idx, center in enumerate(sequence[1:-1], 1):
            before = sequence[idx - 1]
            after  = sequence[idx + 1]
            if numpy.isnan(center):
                if not numpy.isnan(before) and numpy.isnan(after):
                    nansep[0].append(idx)
                    nansep[1].append(0.0)
                elif numpy.isnan(before) and not numpy.isnan(after):
                    nansep[0].append(idx + 1)
                    nansep[1].append(0.0)
        if not numpy.isnan(sequence[-2]) and numpy.isnan(sequence[-1]):
            nansep[0].append(len(sequence) - 1)
            nansep[1].append(0.0)
        return nansep

    @staticmethod
    def reduceNoise(data: list, radius: int):
        """
        **depreciated**

        Reduce noise:

        With a sliding window around center item i,
        if the mean of values [i-r] and [i+r] is below "the" threshold, set i to the mean of [i-r, i+r]

        A better kind of noise reduction is most probably the gauss filter:
        :func:`scipy.ndimage.filters.gaussian_filter1d()`

        :param data: input
        :param radius: the distance before and ahead of a value to calculate the mean of
        :return: data without noise
        """
        # min(data)
        # max(data)
        stddev = numpy.std(data)
        # print(stddev)
        wonoise = data[:radius]
        if not isinstance(wonoise, list):
            wonoise = [wonoise]
        for index, item in enumerate(data[radius:-radius], radius):
            behind = data[index-radius:index]
            ahead = data[index:index+radius]
            if abs(numpy.mean(behind) - numpy.mean(ahead)) < 2*stddev:
                if not isinstance(behind, list):
                    behind = [behind]
                if not isinstance(ahead, list):
                    ahead = [ahead]
                wonoise.append(numpy.mean(behind+ahead))  # [item]+
            else:
                wonoise.append(item)
        finalradius = data[-radius:]
        if isinstance(finalradius, list):
            wonoise.extend(finalradius)
        else:
            wonoise.append(finalradius)
        return wonoise


    @staticmethod
    def calcEntropy(tokens):
        """
        Calculates the entropy within `tokens`.

        :return: entropy in token list
        """
        # unit = U_BYTE  nibble => ASCII ?!
        alphabet = dict()
        # get counts for each word of the alphabet
        for x in tokens:
            if x in alphabet:
                alphabet[x] += 1
            else:
                alphabet[x] = 1

        # len(alphabet) would give a dynamic alphabet length.
        # since we are working on bytes, we assume 256.
        alphabet_len = 2
        entropy = 0
        for x in alphabet:
            # probability of value in string
            p_x = float(alphabet[x]) / len(tokens)
            entropy += - p_x * math.log(p_x, alphabet_len)

        return entropy


    def valueDistance(self):
        """
        min/max/mean distance of bytes occurring multiple times

        unit = U_BYTE
        :return: tuple of statistics of value distances
        """
        # get all byte positions per value in the buckets
        bucket = dict()  # type: Dict[List]
        for pos, char in enumerate(self._message.data):
            if char in bucket:
                bucket[char].append(pos)
            else:
                bucket[char] = [pos]
        valuestats = dict()
        # the above enumeration produces a position sorted list, so do no explicit sorting
        for val, poslist in bucket.items():
            valuedists = numpy.diff(poslist)
            sortedvals = sorted(valuedists)
            minval = sortedvals[0]
            maxval = sortedvals[-1]
            mean = numpy.mean(sortedvals)
            valuestats[val] = (minval, maxval, mean)
        return valuestats




### MessageSegment classes #########################################

class AbstractSegment(ABC):

    CORR_SAD     = 2  # Sum of Absolute Differences
    CORR_COSINE  = 1  # Cosine Coefficient
    CORR_PEARSON = 0  # Pearson Product-Moment Correlation Coefficient

    # list of analysis result values for the segment scope
    values = None  # type: Union[List, numpy.ndarray]

    def correlate(self,
                  haystack: Iterable['MessageSegment'],
                  method: int='AbstractSegment.CORR_PEARSON'
                  ) -> List['CorrelatedSegment']:
        """
        The correlation of this object to each entry in haystack is calculated.

        :param haystack: a list of MessageSegments
        :param method: The method to correlate with (see class constants prefixed with CORR_ for available options)
        :return:
        """
        from utils.baseAlgorithms import ngrams
        import scipy.spatial.distance

        selfCorrMsgs = list()
        for hayhay in haystack:
            # selfConvMsgs[(self, hayhay)] = numpy.divide(  # normalize correlation to the length of self.values
            #     numpy.correlate(self.values, msgValues, 'valid'),
            #     len(self.values)*len(self.values)
            # )
            # normalizing: https://stackoverflow.com/questions/5639280/why-numpy-correlate-and-corrcoef-return-different-values-and-how-to-normalize
            # a = (self.values - numpy.mean(self.values)) / (numpy.std(self.values) * len(self.values))
            # v = (hayhay.values - numpy.mean(hayhay.values)) / numpy.std(hayhay.values)
            # corr = numpy.convolve(a,v,'valid')

            # selfConvMsgs[(self, hayhay)] =
            # a = pandas.Series(self.values)
            # v = pandas.Series(msgValues)
            # selfConvMsgs[(self, hayhay)] = \
            #     v.rolling(window=len(self.values)).corr(a)

            try:
                if method == AbstractSegment.CORR_PEARSON:
                    measure = CorrelatedSegment.MEASURE_SIMILARITY
                    corrvalues = [numpy.corrcoef(a, self.values)[0, 1]
                                  for a in ngrams(hayhay.values, len(self.values))]
                elif method == AbstractSegment.CORR_COSINE:
                    measure = CorrelatedSegment.MEASURE_DISTANCE
                    ng = numpy.array([a for a in ngrams(hayhay.values, len(self.values))])
                    corrarray = scipy.spatial.distance.cdist(numpy.array([self.values]),
                                                              ng,
                                                              'cosine')
                    corrvalues = list(corrarray[0])
                elif method == AbstractSegment.CORR_SAD:
                    measure = CorrelatedSegment.MEASURE_DISTANCE
                    corrvalues = [numpy.sum(numpy.abs(numpy.subtract(self.values, cand)))
                                  for cand in ngrams(hayhay.values, len(self.values))
                                  ]
                else:
                    raise ValueError('Invalid correlation method {} given.'.format(method))

                selfCorrMsgs.append(CorrelatedSegment(corrvalues, self, hayhay, measure))
            except RuntimeWarning as w:
                raise RuntimeWarning('Message {} with feature {} failed caused by {}'.format(
                    hayhay, self, w))
        return selfCorrMsgs


class CorrelatedSegment(AbstractSegment):
    MEASURE_DISTANCE = 0
    MEASURE_SIMILARITY = 1

    def __init__(self, values: Union[List, numpy.ndarray], feature: AbstractSegment, haystack: 'MessageSegment',
                 measure:int='CorrelatedSegment.MEASURE_SIMILARITY'):
        """
        :param values: List of analysis result values.
        :param feature: A segment that should be matched against.
        :param haystack: A segment where feature should be fitted into.
        :param measure: The similarity measure to use.
        """
        self.values = values  # type: Union[List, numpy.ndarray]
        self.feature = feature  # type: AbstractSegment
        self.haystack = haystack  # type: MessageSegment
        self.id = "{:02x}".format(hash(tuple(feature.values)) ^ hash(tuple(haystack.values)))
        self.measure=measure

    # def argmax(self):
    #     """
    #     :return: index of maximum correlation
    #     """
    #     return int(numpy.nanargmax(self.values))

    def bestMatch(self):
        """
        :return: index of best match
        """
        if self.measure == CorrelatedSegment.MEASURE_SIMILARITY:
            return int(numpy.nanargmax(self.values))
        else:
            return int(numpy.nanargmin(self.values))

    def fieldCandidate(self):
        """
        Cuts the best fit part from haystack into a separate segment.

        :return: A segment being the best field candidate according to this correlation.
        """
        from inference.analyzers import NoneAnalysis

        # TODO There could be more than one close match...
        # and multiple segments/field candidates of the same type, therefore matching the same feature
        return MessageSegment(NoneAnalysis(self.haystack.message), self.bestMatch(), len(self.feature.values))


class MessageSegment(AbstractSegment):
    """
    Represent a locatable part of a message and the analysis results related to this segment.
    """

    def __init__(self, analyzer: MessageAnalyzer,
                 offset: int, length: int):
        """
        :param analyzer: A subclass of MessageAnalyzer describing and containing the analysis performed on the message.
            The generated data is written to self.values
        :param offset: number of bytes from the beginning of the message denoted by the analyzer
            where this segment starts.
        :param length: number of bytes which this segment is long.
        """

        # calculate values by the given analysis method, if not provided
        if not analyzer.values:
            self.analyzer = MessageAnalyzer.findExistingAnalysis(type(analyzer), analyzer.unit, analyzer.message, analyzer.analysisParams)
        else:
            self.analyzer = analyzer  # type: MessageAnalyzer
            """kind of the generation method for the contained analysis values"""

        if not isinstance(offset, int):
            raise ValueError('Offset is not an int.')
        if not isinstance(length, int):
            raise ValueError('Length is not an int.')
        if offset >= len(self.message.data):
            raise ValueError('Offset {} too large for message of length {}.'.format(offset, len(self.message.data)))
        if offset+length-1 > len(self.message.data):
            raise ValueError('Length {} from offset {} too large for message of length {}.'.format(
                length, offset, len(self.message.data)))

        self.offset = offset  # type: int
        """byte offset of the first byte in message this segment is related to"""
        self.length = length
        """byte count of the segment this object represents in the originating message"""


    @property
    def values(self):
        if super().values:
            return super().values
        return self.analyzer.values[self.offset:self.offset+self.length]


    def valueat(self, absoluteBytePosition: int):
        """
        :param absoluteBytePosition: byte position from the start of the message
        :return: analysis value for the given absolute byte position in the message
        """
        return self.analyzer.values[absoluteBytePosition]


    @property
    def message(self):
        return self.analyzer.message


    @property
    def bytes(self) -> bytes:
        """
        Message bytes from offset to offset+length

        :return: The bytes of the message which this segment is representing.
        """
        return self.message.data[self.offset:self.offset+self.length]


    def mean(self):
        """
        To call this method, `values` must be set.
        :return: The mean of the values of this segment.
        """
        if self.values is None:
            raise ValueError('Value of MessageSegment instance must be set to calculate its mean.')
        return numpy.mean(self.values)


    def stdev(self):
        """
        To call this method, `values` must be set.
        :return: The mean of the values of this segment.
        """
        if self.values is None:
            raise ValueError('Value of MessageSegment instance must be set to calculate its standard deviation.')
        return numpy.std(self.values)


    def fillCandidate(self, candidate: Union['MessageSegment', AbstractMessage]):
        if isinstance(candidate, MessageSegment):
            # any correlation makes only sense for values originating from the same analysis method.
            if type(candidate.analyzer) != type(self.analyzer) \
                    or candidate.analyzer.analysisParams != self.analyzer.analysisParams:
                raise ValueError('The analysis methods of this MessageSegment ({}({})) and '
                                 'the haystack to correlate it to ({}({})) are not compatible.'.format(
                    self.analyzer.__name__, ', '.join([str(a) for a in self.analyzer.analysisParams]),
                    candidate.analyzer.__name__, ', '.join([str(a) for a in candidate.analyzer.analysisParams])
                ))
            if not candidate.values:
                candidate.analyzer.analyze()
        if isinstance(candidate, AbstractMessage):  # make a segment from the whole message
            analyzer = MessageAnalyzer.findExistingAnalysis(
                type(self.analyzer), self.analyzer.unit, candidate, self.analyzer.analysisParams)
            candidate = MessageSegment(analyzer, 0, len(candidate.data))
        return candidate


    def newAnalysis(self, analyzer: Type[MessageAnalyzer], *analysisParams):
        """
        :return: Another type of analysis for this message segment
            The analysis parameters need to be set afterwards, if necessary.
        """
        return MessageSegment(MessageAnalyzer.findExistingAnalysis(
            analyzer, self.analyzer.unit, self.analyzer.message, analysisParams), self.offset, self.length)


    def __repr__(self):
        if self.values and isinstance(self.values, list) and len(self.values) > 3:
            printValues = str(self.values[:3])[:-1] + '...'
        else:
            printValues = str(self.values)

        return 'MessageSegment {} bytes: {:.16}{}'.format(self.length, self.bytes.hex(),
            '...' if self.length > 3 else '') + \
            ' | values: {}'.format(printValues if printValues else 'not set')


class HelperSegment(MessageSegment):
    """
    Segment class to hold intermediate values that are not consistent with the analyzer.
    """

    def __init__(self, analyzer: MessageAnalyzer,
                 offset: int, length: int):
        super().__init__(analyzer, offset, length)
        self._values = None

    @property
    def values(self):
        return self._values

    @values.setter
    def values(self, values):
        self._values = values


class TypedSegment(MessageSegment):
    """
    Segment class that knows the type of field data contained
    """

    def __init__(self, analyzer: MessageAnalyzer,
                 offset: int, length: int, fieldtype: str = None):
        """

        :param analyzer:
        :param offset:
        :param length:
        :param fieldtype: mark segment with its true type
        """
        super().__init__(analyzer, offset, length)
        self._fieldtype = fieldtype

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


