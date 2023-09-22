"""
Class for the analysis of and to provide statistics about a single message.

**Intra-Message Analysis**

:author: Stephan Kleber
"""
import IPython
import numpy
import pandas
from bitstring import Bits
from typing import Dict, List, Tuple, Union, Type

from scipy.ndimage.filters import gaussian_filter1d
from netzob.Model.Vocabulary.Messages.AbstractMessage import AbstractMessage

# The analyzer implementations heavily depend on the MessageAnalyzer base class
# that itself is deeply intertwined with the MessageSegment class:
from nemere.inference.segments import MessageAnalyzer, MessageSegment, SegmentAnalyzer


class NothingToCompareError(ValueError):
    """
    Error to raise if one of a pair of data is missing for comparison.
    """
    pass


class ParametersNotSet(ValueError):
    """
    Error to raise if the necessary analysis parameters are not set.
    """
    pass


class NoneAnalysis(MessageAnalyzer):
    """
    Class denoting a non-transforming analysis.
    Values remain None and only message is set.
    """
    @property
    def domain(self):
        return None, None

    def analyze(self):
        pass


class BitCongruence(MessageAnalyzer):
    """
    Bitwise congruence: Simple Matching [Sokal & Michener].

    not unit-dependant, always byte-wise
    """
    def __init__(self, message: AbstractMessage, unit=MessageAnalyzer.U_BYTE):
        super().__init__(message, unit)
        self._startskip = 1

    @property
    def domain(self):
        return 0,1

    def analyze(self):
        """
        Bitwise congruence: Simple Matching [Sokal & Michener].
        other kinds of bit variances from http://btluke.com/binclus.html

        :return: list of congruences from index i = 1 to n between bits of i-1 and i
        """
        tokenlist = self._message.data
        self._values = BitCongruence.bitCongruenceBetweenTokens(tokenlist)
        super().analyze()


    @staticmethod
    def bitCongruenceBetweenTokens(tokenlist: Union[List, bytes]):
        """
        Bitwise congruence: Simple Matching [Sokal & Michener]

        not unit-dependent, token-dependent: always compares tokenwise

        :param tokenlist: list of tokens between which the bit congruence is calculated
        :return: list of congruences from index i = 1 to n between bits of i-1 and i
        """
        congruencelist = []  # tokenlist could also be list of ngrams.
        if len(tokenlist) < 2:
            raise NothingToCompareError(
                "Needs at least two tokens to determine a congruence. Token list is {}".format(tokenlist))
        try:  # We need a type that can be casted to byte. Do it as soon as possible to fail early and completely.
            for tokenA, tokenB in zip(tokenlist[:-1], tokenlist[1:]):
                # converting and failsafes.
                if not isinstance(tokenA, bytes):
                    tokenA = bytes( [ tokenA] )
                bitsA = Bits(bytes=tokenA)

                if not isinstance(tokenB, bytes):
                    tokenB = bytes( [tokenB] )
                bitsB = Bits(bytes=tokenB)

                bitlength = len(bitsA)
                if bitlength != len(bitsB):
                    raise IndexError(
                        "All tokens need to be of equal bit length. Offending tokens: {} and {}".format(tokenA, tokenB))

                # finally do the real work:
                # total number of times (bits) subsequent tokens agree.
                bAgree = ~ (bitsA ^ bitsB)   # type: Bits
                congruencelist.append(bAgree.count(1) / bitlength)
        except TypeError as e:
            raise TypeError("Tokens must be convertible to bytes, which failed because: {} ".format(e))
        return congruencelist


class BitCongruenceGauss(BitCongruence):
    """
    Noise reduced bitwise congruence: Simple Matching [Sokal & Michener].
    """
    def __init__(self, message: AbstractMessage, unit=MessageAnalyzer.U_BYTE):
        super().__init__(message, unit)
        self._bcvalues = None

    def setAnalysisParams(self, sigma=1.5):
        if isinstance(sigma, tuple):
            self._analysisArgs = sigma[0]
        elif isinstance(sigma, float):
            self._analysisArgs = sigma
        else:
            raise TypeError('Parameter sigma is not valid')

    def analyze(self):
        if not self._analysisArgs:
            raise ParametersNotSet('Analysis parameter missing: sigma.')
        sigma = self._analysisArgs
        super().analyze()
        self._bcvalues = self._values
        self._values = list(gaussian_filter1d(self._values, sigma))

    @property
    def bitcongruences(self):
        if self._bcvalues is None:
            return None
        return [0.0] * self.startskip + self._bcvalues

    def messageSegmentation(self) -> List[MessageSegment]:
        """
        Segment message by determining local minima of sigma-1.5-gauss-filtered bit-congruence.

        >>> from netzob.Model.Vocabulary.Messages.L4NetworkMessage import L4NetworkMessage
        >>> tstmsg = '19040aec0000027b000012850a6400c8d23d06a2535ed71ed23d09faa4673315d23d09faa1766325d23d09faa17b4b10'
        >>> l4m = L4NetworkMessage(bytes.fromhex(tstmsg))
        >>> hbg = BitCongruenceGauss(l4m)
        >>> hbg.setAnalysisParams()
        >>> hbg.analyze()
        >>> spm = hbg.messageSegmentation()
        >>> # noinspection PyUnresolvedReferences
        >>> print(b''.join([seg.bytes for seg in spm]).hex() == spm[0].message.data.hex())
        True

        :return: Segmentation of this message based on this analyzer's type.
        """
        if not self.values:
            if not self._analysisArgs:
                raise ValueError('No values or analysis parameters set.')
            self.analyze()

        bclmins = self.pinpointMinima()

        cutCandidates = [0] + [int(b) for b in bclmins] + [len(self._message.data)]  # add the message end
        cutPositions = [0] + [right for left, right in zip(
            cutCandidates[:-1], cutCandidates[1:]
        ) if right - left > 1]
        if cutPositions[-1] != cutCandidates[-1]:
            cutPositions[-1] = cutCandidates[-1]

        segments = list()
        for lmaxCurr, lmaxNext in zip(cutPositions[:-1], cutPositions[1:]):
            segments.append(MessageSegment(self, lmaxCurr, lmaxNext-lmaxCurr))
        return segments

    def pinpointMinima(self):
        """
        Pinpoint the exact positions of local minima within the scope of each smoothed local minimum.
        The exact position is looked for in self.bitcongruences.

        :return: One exact local minium m in the interval ( center(m_n-1, m_n), center(m_n, m_n+1) )
            for each n in (0, smoothed local minimum, -1)
        """
        localminima = MessageAnalyzer.localMinima(self.values)  # List[idx], List[min]
        # localmaxima = MessageAnalyzer.localMaxima(self.values)  # List[idx], List[max]
        # for lminix in range(len(localminima)):
        #     localminima[lminix]
        lminAO = [0] + localminima[0] + [len(self._message.data)]
        lminMed = (numpy.round(numpy.ediff1d(lminAO) / 2) + lminAO[:-1]).astype(int)
        bclmins = [medl + numpy.argmin(self.bitcongruences[medl:medr]) for medl, medr in zip(lminMed[:-1], lminMed[1:])]
        return bclmins


class BitCongruenceDelta(BitCongruence):

    def __init__(self, message: AbstractMessage, unit=MessageAnalyzer.U_BYTE):
        super().__init__(message, unit)
        self._startskip = self._startskip + 1
        self._bcvalues = None

    @property
    def domain(self):
        return -1,1

    def analyze(self):
        """
        Delta of bitwise congruence. see :func:`MessageAnalyzer.bitCongruence`

        not unit-dependant, always byte-wise

        :return: list of amplitudes of bit congruence from index i = 1 to n between bits of i-1 and i
        """
        super().analyze()
        self._bcvalues = self._values
        self._values = MessageAnalyzer.tokenDelta(self._values)
        assert self._startskip + len(self._values) == len(self._message.data), \
            "{} + {} != {}".format(self._startskip, len(self._values), len(self._message.data))

    @property
    def bitcongruences(self):
        """
        :return: basic bit congruences
        """
        if self._bcvalues is None:
            return None
        return [numpy.nan] * super().startskip + self._bcvalues


class BitCongruenceDeltaGauss(BitCongruenceDelta):
    # _sensitivity = 0.33
    # """Sensitivity threshold for the smoothed extrema."""

    def __init__(self, message: AbstractMessage, unit=MessageAnalyzer.U_BYTE):
        super().__init__(message, unit)
        self._bcdvalues = None

    def setAnalysisParams(self, sigma=1.5):
        self._analysisArgs = (sigma, )

    def analyze(self):
        from collections import Sequence
        if not self._analysisArgs or not isinstance(self._analysisArgs, Sequence):
            raise ParametersNotSet('Analysis parameter missing: sigma.')
        sigma, = self._analysisArgs
        super().analyze()
        self._bcdvalues = self._values
        bcv = numpy.array(self._values)
        assert not numpy.isnan(bcv).any()
        # bcv could be filtered by: [~numpy.isnan(bcv)]
        self._values = list(gaussian_filter1d(bcv, sigma)) # + [numpy.nan]
        assert self._startskip + len(self._values) == len(self._message.data), \
            "{} + {} != {}".format(self._startskip, len(self._values), len(self._message.data))

    @property
    def bcdeltas(self):
        """
        :return: bit congruence deltas without smoothing
        """
        if self._bcdvalues is None:
            return None
        return [numpy.nan] * self.startskip + self._bcdvalues

    def messageSegmentation(self) -> List[MessageSegment]:
        """
        Segment message by determining inflection points of sigma-s-gauss-filtered bit-congruence.
        The cut position is the delta max of the unsmoothed bcd in the scope of a min/max (rising) pair.

        additionally cut at high plateaus starts in the basic bc values.

        :return: Segmentation of this message based on this analyzer's type.
        """
        if not self.values:
            if not self._analysisArgs:
                raise ValueError('No values or analysis parameters set.')
            self.analyze()

        # cut one byte before the inflection
        inflectionPoints = self.inflectionPoints()
        inflectionCuts = [ int(i)-1 for i in inflectionPoints[0]]

        # # cut one byte before the plateau
        # # | has yielded mixed quality results (was better for dhcp, much worse for ntp and dns)
        # # | TODO probably having some kind of precedence whether inflection or plateau is to be kept
        # # | if both cut positions are near to each other might make this worthwhile.
        # highPlats = self.bcHighPlateaus()
        # highPlatCuts = [ int(i)-1 for i in highPlats[0]]
        # # below: sorted( + highPlatCuts)

        # get candidates to cut segments from message
        cutCandidates = [0] + inflectionCuts \
                        + [len(self._message.data)]  # add the message end
        # cut only where a segment is of a length larger than 1
        cutPositions = [0] + [right for left, right in zip(
                cutCandidates[:-1], cutCandidates[1:]
            ) if right - left > 1]
        # cutPositions = list(sorted(cutPositions + nansep[0]))
        # add the end of the message if its not already there
        if cutPositions[-1] != cutCandidates[-1]:
            cutPositions[-1] = cutCandidates[-1]

        segments = list()
        for cutCurr, cutNext in zip(cutPositions[:-1], cutPositions[1:]):
            segments.append(MessageSegment(self, cutCurr, cutNext-cutCurr))
        return segments

    def extrema(self) -> List[Tuple[int, bool]]:
        """
        :return: all extrema of the smoothed bcd, each described by a tuple of its index and bool (min is False)
        """
        bcdNR = self.values
        lmin = MessageAnalyzer.localMinima(bcdNR)
        lmax = MessageAnalyzer.localMaxima(bcdNR)
        nrExtrema = sorted(
            [(i, False) for i in lmin[0]] + [(i, True) for i in lmax[0]], key=lambda k: k[0])
        return nrExtrema

    def risingDeltas(self) -> List[Tuple[int, numpy.ndarray]]:
        """
        the deltas in the original bcd (so: 2nd delta) between minima and maxima in smoothed bcd

        :return: offset of and the bcd-delta values starting at this position in rising parts of the smoothed bcd.
            Thus, offset is a minimum + 1 and the array covers the indices up to the following maximum, itself included.
        """
        extrema = self.extrema()
        risingdeltas = [ ( i[0] + 1, numpy.ediff1d(self.bcdeltas[i[0]:j[0]+1]) )  # include index of max
                for i, j in zip(extrema[:-1], extrema[1:])
                   if i[1] == False and j[1] == True and j[0]+1 - i[0] > 1]
        # risingdeltas[-1][0] >= len(self.bcdeltas)
        return risingdeltas

    def inflectionPoints(self) -> Tuple[List[int], List[float]]:
        """
        adjusted approximation of the inflection points at rising edges of the smoothed bcd.
        The approximation is that we are using the maximum delta of the unsmoothed bcd
        in scope of the rising part of the graph.

        :return: The indices and values of the approximated inflections.
        """
        inflpt = [ offset + int(numpy.nanargmax(wd)) for offset, wd in self.risingDeltas() ]
        inflvl = [ self.values[pkt] for pkt in inflpt ]
        return inflpt, inflvl

    def bcHighPlateaus(self):
        """
        :return: Plateaus in the bit congruence at high level (> 0.8)
        """
        plateauElevation = 0.8
        plat = MessageAnalyzer.plateouStart(self.bitcongruences)

        # filter for plateaus of high bit congruence
        hiPlat = ([], [])
        for ix, vl in zip(plat[0], plat[1]):
            if vl > plateauElevation:
                hiPlat[0].append(ix)
                hiPlat[1].append(vl)
        return hiPlat

class BitCongruenceLE(BitCongruence):
    """
    Little Endian version of
    Bitwise congruence: Simple Matching [Sokal & Michener].

    not unit-dependant, always byte-wise
    """
    @property
    def values(self):
        """
        :return: The analysis values for this message, possibly prepended by NaN values
            in the amount of startskip (see there),
            after analyze() was called. None otherwise.
        """
        if self._values is None:
            return None
        return self._values[::-1] + [numpy.nan] * self.startskip

    def analyze(self):
        """
        Bitwise congruence: Simple Matching [Sokal & Michener].
        other kinds of bit variances from http://btluke.com/binclus.html

        :return: list of congruences from index i = 1 to n between bits of i-1 and i
        """
        tokenlist = self._message.data[::-1]
        self._values = BitCongruence.bitCongruenceBetweenTokens(tokenlist)
        super().analyze()

class BitCongruenceDeltaLE(BitCongruenceLE, BitCongruenceDelta):
    @property
    def bitcongruences(self):
        """
        :return: basic bit congruences
        """
        if self._bcvalues is None:
            return None
        return self._bcvalues[::-1] + [numpy.nan] * super().startskip

class BitCongruenceDeltaGaussLE(BitCongruenceDeltaLE, BitCongruenceDeltaGauss):
    @property
    def bcdeltas(self):
        """
        :return: bit congruence deltas without smoothing
        """
        if self._bcdvalues is None:
            return None
        return self._bcdvalues[::-1] + [numpy.nan] * self.startskip

    def messageSegmentation(self) -> List[MessageSegment]:
        """
        Segment message by determining inflection points of sigma-s-gauss-filtered bit-congruence.
        The cut position is the delta max of the unsmoothed bcd in the scope of a min/max (rising) pair.

        additionally cut at high plateaus starts in the basic bc values.

        :return: Segmentation of this message based on this analyzer's type.
        """
        if not self.values:
            if not self._analysisArgs:
                raise ValueError('No values or analysis parameters set.')
            self.analyze()

        # CAVE: all following is in reversed index order!

        # cut one byte before the inflection
        inflectionPoints = self.inflectionPoints()
        inflectionCuts = [ int(i)-1 for i in inflectionPoints[0]]

        # get candidates to cut segments from message
        cutCandidates = [0] + inflectionCuts \
                        + [len(self._message.data)]  # add the message end
        # cut only where a segment is of a length larger than 1
        cutPositions = [0] + [right for left, right in zip(
                cutCandidates[:-1], cutCandidates[1:]
            ) if right - left > 1]
        # cutPositions = list(sorted(cutPositions + nansep[0]))
        # add the end of the message if its not already there
        if cutPositions[-1] != cutCandidates[-1]:
            cutPositions[-1] = cutCandidates[-1]

        segments = list()
        # zip(cutPositions[::-1][:-1], cutPositions[::-1][1:]) is in simpler terms:
        for cutCurr, cutNext in zip(cutPositions[:0:-1], cutPositions[-2::-1]):
            # here we reverse the index order again, to reinstate the actual byte offsets
            offset = len(self.values) - cutCurr
            length = cutCurr - cutNext
            segments.append(MessageSegment(self, offset, length))
        return segments

    def extrema(self) -> List[Tuple[int, bool]]:
        """
        in reversed index order!
        :return: all extrema of the smoothed bcd, each described by a tuple of its index and bool (min is False)
        """
        bcdNR = self.values[::-1]  # values is in message byte order and with added nans for missing values
        lmin = MessageAnalyzer.localMinima(bcdNR)
        lmax = MessageAnalyzer.localMaxima(bcdNR)
        nrExtrema = sorted(
            [(i, False) for i in lmin[0]] + [(i, True) for i in lmax[0]], key=lambda k: k[0])
        return nrExtrema

    def risingDeltas(self) -> List[Tuple[int, numpy.ndarray]]:
        """
        in reversed index order!
        the deltas in the original bcd (so: 2nd delta) between minima and maxima in smoothed bcd

        :return: offset of and the bcd-delta values starting at this position in rising parts of the smoothed bcd.
            Thus, offset is a minimum + 1 and the array covers the indices up to the following maximum, itself included.
        """
        extrema = self.extrema()
        risingdeltas = [ ( i[0] + 1, numpy.ediff1d(self.bcdeltas[::-1][i[0]:j[0]+1]) )  # include index of max
                for i, j in zip(extrema[:-1], extrema[1:])
                   if i[1] == False and j[1] == True and j[0]+1 - i[0] > 1]
        # risingdeltas[-1][0] >= len(self.bcdeltas)
        return risingdeltas

    def inflectionPoints(self) -> Tuple[List[int], List[float]]:
        """
        in reversed index order!
        adjusted approximation of the inflection points at rising edges of the smoothed bcd.
        The approximation is that we are using the maximum delta of the unsmoothed bcd
        in scope of the rising part of the graph.

        :return: The indices and values of the approximated inflections.
        """
        inflpt = [ offset + int(numpy.nanargmax(wd)) for offset, wd in self.risingDeltas() ]
        inflvl = [ self.bcdeltas[::-1][pkt] for pkt in inflpt ]
        return inflpt, inflvl

    def bcHighPlateaus(self):
        """
        in reversed index order!
        :return: Plateaus in the bit congruence at high level (> 0.8)
        """
        plateauElevation = 0.8
        plat = MessageAnalyzer.plateouStart(self.bitcongruences[::-1])

        # filter for plateaus of high bit congruence
        hiPlat = ([], [])
        for ix, vl in zip(plat[0], plat[1]):
            if vl > plateauElevation:
                hiPlat[0].append(ix)
                hiPlat[1].append(vl)
        return hiPlat


class BitCongruence2ndDelta(BitCongruenceDelta):
    def __init__(self, message: AbstractMessage, unit=MessageAnalyzer.U_BYTE):
        super().__init__(message, unit)
        # self._startskip += 1

    def analyze(self):
        """
        2nd order delta of bitwise congruence. see :func:`MessageAnalyzer.bitCongruence()`

        not unit-dependant, always byte-wise

        :return: list of amplitudes of bit congruences from index i = 1 to n between bits of i-1 and i
        """
        super().analyze()
        self._values = MessageAnalyzer.tokenDelta(self._values)


class BitCongruenceBetweenNgrams(BitCongruence):
    """
    Bit congruence for all bits within consecutive ngrams.
    """
    def __init__(self, message: AbstractMessage, unit=MessageAnalyzer.U_BYTE):
        super().__init__(message, unit)
        self._n = None

    def setAnalysisParams(self, n: Union[int, Tuple[int]]):
        self._n = n if not isinstance(n, tuple) else n[0]
        self._startskip = self._n

    def analyze(self):
        """
        not unit-dependant

        bit congruence directly between ngrams
        """
        if not self._n:
            raise ParametersNotSet('Analysis parameter missing: N-gram size ("n").')
        tokenlist = list(self.ngrams(self._n))
        self._values = BitCongruence.bitCongruenceBetweenTokens(tokenlist)
        MessageAnalyzer.analyze(self)


class BitCongruenceNgramMean(BitCongruence):
    """
    Cumulated bit congruences within each ngram of the message,
    """
    def __init__(self, message: AbstractMessage, unit=MessageAnalyzer.U_BYTE):
        super().__init__(message, unit)
        self._n = None
        self._ngramMean = list()

    def setAnalysisParams(self, n: Union[int, Tuple[int]]):
        self._n = int(n) if not isinstance(n, tuple) else int(n[0])
        self._startskip = self._n

    def analyze(self):
        """
        not unit-dependant

        mean of byte-wise bit congruences of each ngram

        :return:
        """
        if not self._n:
            raise ParametersNotSet('Analysis parameter missing: N-gram size ("n").')
        from nemere.utils.baseAlgorithms import ngrams

        super().analyze()
        self._ngramMean = [float(numpy.mean(bcn)) for bcn in ngrams(self._values, self._n)]

    @property
    def values(self):
        if self._values is None:
            return None
        return [0.0] * self.startskip + self._ngramMean


class BitCongruenceNgramStd(BitCongruence):
    """
    Standard deviation of bit congruences for all bits within ngrams.
    """
    _n = None
    _ngramVar = list()

    def setAnalysisParams(self, n: Union[int, Tuple[int]]):
        self._n = int(n) if not isinstance(n, tuple) else int(n[0])
        self._startskip = self._n


    def analyze(self):
        """
        not unit-dependant

        deviation of bit congruence within ngrams

        :return:
        """
        if not self._n:
            raise ParametersNotSet('Analysis parameter missing: N-gram size ("n").')
        from ..utils.baseAlgorithms import ngrams

        super().analyze()
        self._ngramVar = [float(numpy.std(bcn)) for bcn in ngrams(self._values, self._n)]


    @property
    def values(self):
        if self._ngramVar is None:
            return None
        return [0.0] * self.startskip + self._ngramVar


    def messageSegmentation(self):
        """

        :return: Segmentation of this message based on this analyzer's type.
        """
        raise NotImplementedError('Unfinished implementation.')

        if not self._ngramVar:
            if not self._analysisArgs:
                raise ValueError('No values or analysis parameters set.')
            self.analyze()

        # TODO segmentation based on areas of similar congruence (not border detection):
        # factor * std(message) < abs(std(3gram_n) - std(3gram_(n-1))) -> cut segment
        # factor = 1 for now

        # find a threshold factor
        min(self._ngramVar)
        max(self._ngramVar)

        # prevent 1 byte segments, since they do not contain usable congruence!
        cutCandidates = [0] + [int(b) for b in bclmins] + [len(self._message.data)]  # add the message end
        cutPositions = [0] + [right for left, right in zip(
            cutCandidates[:-1], cutCandidates[1:]
        ) if right - left > 1]
        if cutPositions[-1] != cutCandidates[-1]:
            cutPositions[-1] = cutCandidates[-1]

        segments = list()
        for lmaxCurr, lmaxNext in zip(cutPositions[:-1], cutPositions[1:]):
            segments.append(MessageSegment(self, lmaxCurr, lmaxNext-lmaxCurr))
        return segments

        # TODO Areas of similarity: may also be feasible for bc, hbc, sliding2means, deltaProgression


class PivotBitCongruence(BitCongruence):
    """
    Repeatedly cut the message(segments) in half, calculate the mean/variance of bit congruence for each half,
    until the difference between the segments is below a™ threshold.

    Fixed Pivot Results:
    ====================
    Thresholds .1, .05, .02 show (pivotedbitvariancesParent.*_ntp_SMIA-20111010_deduped-100)
    that some messages get segmented arbitrarily deep, while others with clearly visible structure are
    not segmented at all. In this design the analysis is unsuitable.

    Fixed Pivot Results:
    ============
    Slide the pivot positions over the whole message (-segment)
    and use the one maximizing difference in the segments' congruence to recurse.

    Works comparatively well for DNS, is awful for ntp and dhcp.
    With the same parameters (different fixed and weighted threshold calculation strategies, fixed and weighted pivot
    selection condition), there is no correlation between fields and segment splits. Some areas of the message are too
    similar in mean and deviation to be splitted altough they should be, while others are to diverse within a field,
    so they get fragmented way to much.
    """
    def __init__(self, message: AbstractMessage, unit=MessageAnalyzer.U_BYTE):
        super().__init__(message, unit)
        self._meanThreshold = .02

    def setAnalysisParams(self, args):
        if isinstance(args, tuple):
            self._meanThreshold = args[0]
        else:
            self._meanThreshold = args

    @property
    def analysisParams(self):
        return self._meanThreshold,

    def _recursiveFixedPivotMean(self, segment: MessageSegment):
        """
        Recursively split the segment in half, calculate the mean for the values of each of the two resulting
        sub-segments, and compare each of them to the original segments mean. If a sub-segment is sufficiently
        different from its parent (meanThreshold = .02) further split the sub-segment.

        :param segment: One message segment that should be segmented.
        :return: List of segments after the splitting.
        """

        if not segment.values:
            segment.analyzer.analyze()
        mymean = segment.mean()

        if segment.length >= 4:  # we need two bytes for each segment to get a bit congruence of them
            pivot = segment.length//2
            leftSegment = MessageSegment(segment.analyzer, segment.offset, pivot)
            rightSegment = MessageSegment(segment.analyzer, segment.offset + pivot, segment.length - pivot)

            # test for recursion conditions
            returnSegments = list()
            if abs(leftSegment.mean() - mymean) > self._meanThreshold:  # still different
                returnSegments.extend(self._recursiveFixedPivotMean(leftSegment))
            else:
                returnSegments.append(leftSegment)
            if abs(rightSegment.mean() - mymean) > self._meanThreshold:  # still different
                returnSegments.extend(self._recursiveFixedPivotMean(rightSegment))
            else:
                returnSegments.append(rightSegment)
            # if abs(lsm - rsm) > .1:  # still different
            return returnSegments
        else:
            return [segment]

    def messageSegmentation(self) -> List[MessageSegment]:
        segments = self._recursiveDynamicSlidedMean(MessageSegment(BitCongruence(self.message), 0, len(self._message.data)))
        sortedSegments = sorted(segments, key=lambda x: x.offset)
        # varPerSeg = list()
        # for segment in sortedSegments:
        #     if segment.offset > len(varPerSeg):
        #         raise ValueError('Segment before offset {} missing for message with data ...{}...'.format(
        #             segment.offset, hex(self._message.data[len(varPerSeg):segment.offset])))
        #         # # instead of failing we could also add placeholders if something is missing.
        #         # # But is shouldn't happen: We do not have overlapping or omitted segments.
        #         # meanVarPerSeg.extend( [-1]*(segment.offset-len(meanVarPerSeg)) )
        #     # add mean value for all byte positions of one segment.
        #     varPerSeg.extend( [ segment.stdev() ]*segment.length )
        # self._values = varPerSeg
        if self.__debug:
            input('next message: ')
        return sortedSegments

    __debug = False

    def _recursiveDynamicPivotStd(self, segment: MessageSegment):
        """
        Recursively split the segment at positions shifting from 2 to n-2, calculate the standard deviation for the
        values of each of the two resulting sub-segments, and compare each of them to the original segments deviation.
        If a sub-segment is sufficiently different from its parent
        (varThreshold = 0.5 parentvar * min(len(vl), len(vr))/(len(vl) + len(vr))) further split the sub-segment.

        :param segment: One message segment that should be segmented.
        :return: List of segments after the splitting.
        """

        if not segment.values:
            segment.analyzer.analyze()
        myvar = segment.stdev()

        if segment.length >= 4:  # we need two bytes for each segment to get a bit congruence of them

            # select a suitable pivot: find the one yielding the highest deviation-difference from parent
            segmentSplit = dict()
            for pivot in range(2, segment.length-1):
                leftSegment = MessageSegment(segment.analyzer, segment.offset, pivot)
                rightSegment = MessageSegment(segment.analyzer, segment.offset + pivot, segment.length - pivot)
                # deviation needs to be higher towards the edges to be a probable splitting point
                lenweight = 2 * min(leftSegment.length, rightSegment.length) / segment.length
                # add splits: varDiff: (leftSegment, rightSegment)
                segmentSplit[abs(leftSegment.stdev() - rightSegment.stdev()) * lenweight] \
                    = (leftSegment, rightSegment)

            if self.__debug:
                from tabulate import tabulate
                print(tabulate(sorted([(wlrdiff, ls.offset, ls.stdev(), rs.offset, rs.stdev(), rs.offset + rs.length)
                                for wlrdiff, (ls, rs) in segmentSplit.items()], key=lambda x: x[0]), headers=[
                    'wlrdiff', 'l.o', 'lvar', 'r.o', 'rvar', 'r.b']))  #abs(x[3] - x[4])

            # use the segments splitted at selected pivot: search max varDiff in splits
            splitdiffmax = max(segmentSplit.keys())
            leftSegment, rightSegment = segmentSplit[splitdiffmax]
            # weightedThresh = 0.5 * myvar * min(leftSegment.length, rightSegment.length) / segment.length
            weightedThresh = 0.1 * myvar
            if self.__debug:
                print('parent segment stdev:', myvar)
                print('weighted threshold:', weightedThresh)

            # test for recursion conditions: recurse if above weightedThresh
            returnSegments = list()
            if abs(leftSegment.stdev() - myvar) > weightedThresh:  # still different
                if self.__debug:
                    print('split left', leftSegment.offset)
                returnSegments.extend(self._recursiveDynamicPivotStd(leftSegment))
            else:
                if self.__debug:
                    print('left finished', abs(rightSegment.stdev() - myvar))
                returnSegments.append(leftSegment)
            if abs(rightSegment.stdev() - myvar) > weightedThresh:  # still different
                if self.__debug:
                    print('split right', rightSegment.offset)
                returnSegments.extend(self._recursiveDynamicPivotStd(rightSegment))
            else:
                if self.__debug:
                    print('right finished', abs(rightSegment.stdev() - myvar))
                returnSegments.append(rightSegment)

            # if abs(lsm - rsm) > .1:  # still different
            return returnSegments
        else:
            return [segment]

    def _recursiveDynamicSlidedMean(self, segment: MessageSegment):
        """
        Recursively split the segment at positions shifting from 2 to n-2, calculate the mean for the
        values of each of the two resulting sub-segments, and compare each of them to the original segment's mean.
        If a sub-segment is sufficiently different from its parent
        (meanThreshold = 0.5 parentvar * min(len(vl), len(vr))/(len(vl) + len(vr))) further split the sub-segment.

        :param segment: One message segment that should be segmented.
        :return: List of segments after the splitting.
        """

        if not segment.values:
            segment.analyzer.analyze()
        parentMean = segment.mean()

        if segment.length >= 4:  # we need two bytes for each segment to get a bit congruence of them

            # select a suitable pivot: find the one yielding the highest deviation-difference from parent
            segmentSplit = dict()
            for pivot in range(2, segment.length-1):
                leftSegment = MessageSegment(segment.analyzer, segment.offset, pivot)
                rightSegment = MessageSegment(segment.analyzer, segment.offset + pivot, segment.length - pivot)
                # deviation needs to be higher towards the edges to be a probable splitting point
                lenweight = 2 * min(leftSegment.length, rightSegment.length) / segment.length
                # add splits: varDiff: (leftSegment, rightSegment)
                segmentSplit[abs(leftSegment.mean() - rightSegment.mean()) * lenweight] \
                    = (leftSegment, rightSegment)

            if self.__debug:
                from tabulate import tabulate
                print(tabulate(sorted([(wlrdiff, ls.offset, ls.mean(), rs.offset, rs.mean(), rs.offset + rs.length)
                                for wlrdiff, (ls, rs) in segmentSplit.items()], key=lambda x: x[0]), headers=[
                    'wlrdiff', 'l.o', 'lmean', 'r.o', 'rmean', 'r.b']))  #abs(x[3] - x[4])

            # use the segments splitted at selected pivot: search max varDiff in splits
            splitdiffmax = max(segmentSplit.keys())
            leftSegment, rightSegment = segmentSplit[splitdiffmax]
            # weightedThresh = 0.5 * parentMean * min(leftSegment.length, rightSegment.length) / segment.length
            weightedThresh = self._meanThreshold * parentMean
            if self.__debug:
                print('parent segment mean:', parentMean)
                print('weighted threshold:', weightedThresh)

            # test for recursion conditions: recurse if above weightedThresh
            returnSegments = list()
            if abs(leftSegment.mean() - parentMean) > weightedThresh:  # still different
                if self.__debug:
                    print('split left', leftSegment.offset)
                returnSegments.extend(self._recursiveDynamicSlidedMean(leftSegment))
            else:
                if self.__debug:
                    print('left finished', abs(rightSegment.mean() - parentMean))
                returnSegments.append(leftSegment)
            if abs(rightSegment.mean() - parentMean) > weightedThresh:  # still different
                if self.__debug:
                    print('split right', rightSegment.offset)
                returnSegments.extend(self._recursiveDynamicSlidedMean(rightSegment))
            else:
                if self.__debug:
                    print('right finished', abs(rightSegment.mean() - parentMean))
                returnSegments.append(rightSegment)

            # if abs(lsm - rsm) > .1:  # still different
            return returnSegments
        else:
            return [segment]


class SlidingNmeanBitCongruence(BitCongruence):
    """
    Slide a window of given size over the message and calculate means of the bit congruences of the windows.
    """

    def setAnalysisParams(self, halfWindow = (2,)):
        if isinstance(halfWindow, int):
            self._analysisArgs = (halfWindow,)
        else:
            self._analysisArgs = halfWindow
        self._startskip = halfWindow - 1

    def analyze(self):
        if not self._analysisArgs:
            raise ParametersNotSet('Analysis parameter missing: halfWindow.')
        halfWindow = self._analysisArgs[0]

        super().analyze()
        rollmean = pandas.Series(self._values).rolling(window=halfWindow).mean()  # type: pandas.Series

        self._values = rollmean.tolist()


class SlidingNbcGradient(SlidingNmeanBitCongruence):
    """
    Gradient (centered finite difference, h=1) with numpy method.
    """
    @property
    def domain(self):
        return -1,1

    def analyze(self):
        super().analyze()
        self._values = numpy.gradient(self._values).tolist()


class SlidingNbcDelta(SlidingNmeanBitCongruence):
    """
    Slide a window of given size over the message and calculate means of the bit congruences in (two equally long)
    halfs of the window. Compare each two means by calculating its difference.

    Delta (difference quotient with forward finite difference, h=1) for all values.

    Alternative Idea
    ====

        A difference quotient of n > 1 (8, 6, 4) may show regularly recurring 0s for consecutive fields
        of equal length and type.
    """
    def __init__(self, message: AbstractMessage, unit=MessageAnalyzer.U_BYTE):
        super().__init__(message, unit)
        self._startskip += 1

    @property
    def domain(self):
        return -1,1

    def analyze(self):
        super().analyze()
        halfWindow = self._analysisArgs[0]
        self._values = [r-l for l,r in zip(self._values[:-halfWindow], self._values[halfWindow:])] + [numpy.nan]
        # self._values = numpy.ediff1d(self._values).tolist() + [numpy.nan]
        # self._values = numpy.divide(numpy.diff(self._values, n=8), 8).tolist()

    @property
    def values(self):
        return super().values + [numpy.nan]

class SlidingNbcDeltaGauss(SlidingNbcDelta):
    """
    Gauss filtered sliding n horizon bit congruence deltas.
    """
    def __init__(self, message: AbstractMessage, unit=MessageAnalyzer.U_BYTE):
        super().__init__(message, unit)
        self._bcvalues = None
        self._sensitivity = 0.5
        """Sensitivity threshold for the smoothed extrema."""
        self._startskip += 1

    def setAnalysisParams(self, horizon=2, sigma=1.5):
        self._analysisArgs = (horizon, sigma)

    def analyze(self):
        from collections import Sequence
        if not self._analysisArgs or not isinstance(self._analysisArgs, Sequence):
            raise ParametersNotSet('Analysis parameter missing: horizon and sigma.')
        horizon, sigma = self._analysisArgs
        super().analyze()
        self._bcvalues = self._values
        bcv = numpy.array(self._values)
        self._values = list(gaussian_filter1d(bcv[~numpy.isnan(bcv)], sigma)) + [numpy.nan]

    @property
    def bitcongruences(self):
        """
        :return: sliding n horizon bit congruence deltas without smoothing
        """
        if self._bcvalues is None:
            return None
        return [numpy.nan] * (self.startskip - 1) + self._bcvalues

    def messageSegmentation(self) -> List[MessageSegment]:
        """
        Segment message by determining local extrema of sigma-s-gauss-filtered sliding n-byte-mean bit-congruence.

        >>> from netzob.Model.Vocabulary.Messages.L4NetworkMessage import L4NetworkMessage
        >>> tstmsg = '19040aec0000027b000012850a6400c8d23d06a2535ed71ed23d09faa4673315d23d09faa1766325d23d09faa17b4b10'
        >>> l4m = L4NetworkMessage(bytes.fromhex(tstmsg))
        >>> hbg = HorizonBitcongruenceGauss(l4m)
        >>> hbg.setAnalysisParams()
        >>> hbg.analyze()
        >>> spm = hbg.messageSegmentation()
        >>> print(b''.join([seg.bytes for seg in spm]).hex() == spm[0].message.data.hex())
        True

        :return: Segmentation of this message based on this analyzer's type.
        """
        if not self.values:
            if not self._analysisArgs:
                raise ValueError('No values or analysis parameters set.')
            self.analyze()

        bcd = MessageAnalyzer.findExistingAnalysis(BitCongruenceDelta, MessageAnalyzer.U_BYTE, self.message)

        # all local minima
        bclmins = self.pinpointMinima()
        # local maxima, if bc[e] < bc[e+1] or bc[e] > 2*s2mbc[e] for all e in cadidate indices
        bclmaxs = self.pinpointMaxima()
        bcdmaxs = [e for e in bclmaxs if bcd.values[e+1] > bcd.values[e] or bcd.values[e] > 2 * self.bitcongruences[e]]
        minmax = bclmins
        for bdm in bcdmaxs:  # only keep bcdmaxs if not in scope if min
            if bdm + 1 not in minmax and bdm - 1 not in minmax:
                minmax.append(bdm)
         # starts of plateaus of bit congruences
        bcplats = MessageAnalyzer.plateouStart(self.bitcongruences)[0]  # bcd.values
        for bps in bcplats:  # only keep platoustarts if not in scope if min or max
            if bps + 1 not in minmax and bps - 1 not in minmax:
                minmax.append(bps)

        # # separate nan-values
        # nansep = MessageAnalyzer.separateNaNs(self.values)
        relevantPositions = list(sorted(minmax))
        # get candidates to cut segments from message
        cutCandidates = [0] + [int(b) for b in relevantPositions if not numpy.isnan(b)] \
                        + [len(self._message.data)]  # add the message end
        # cut only where a segment is of a length larger than 1
        cutPositions = [0] + [right for left, right in zip(
                cutCandidates[:-1], cutCandidates[1:]
            ) if right - left > 1]
        # cutPositions = list(sorted(cutPositions + nansep[0]))
        # add the end of the message if its not already there
        if cutPositions[-1] != cutCandidates[-1]:
            cutPositions[-1] = cutCandidates[-1]

        segments = list()
        for lmaxCurr, lmaxNext in zip(cutPositions[:-1], cutPositions[1:]):
            segments.append(MessageSegment(self, lmaxCurr, lmaxNext-lmaxCurr))
        return segments

    def pinpointMinima(self):
        """
        Pinpoint the exact positions of local minima within the scope of each smoothed local minimum.
        The exact position is looked for in self.bitcongruences.
        Only those extrema of the smoothed graph are taken into account which are above the sensitivity threshold.

        :return: One exact local minium m in the interval ( center(m_n-1, m_n), center(m_n, m_n+1) )
            for each n in (0, smoothed local minimum, -1)
        """
        from itertools import compress

        localminima = MessageAnalyzer.localMinima(self.values)  # List[idx], List[min]
        allovermin = min(localminima[1])
        minSmsk = [True if e < self._sensitivity * allovermin else False for e in localminima[1]]
        lminAO = [0] + list(compress(localminima[0], minSmsk)) + [len(self._message.data)]
        lminMed = (numpy.round(numpy.ediff1d(lminAO) / 2) + lminAO[:-1]).astype(int)
        bclmins = [medl + numpy.argmin(self.bitcongruences[medl:medr]) for medl, medr in zip(lminMed[:-1], lminMed[1:])]
        return bclmins

    def pinpointMaxima(self):
        """
        Pinpoint the exact positions of local maxima within the scope of each smoothed local maximum.
        The exact position is looked for in self.bitcongruences.
        Only those extrema of the smoothed graph are taken into account which are above the sensitivity threshold.

        :return: One exact local maximum m in the interval ( center(m_n-1, m_n), center(m_n, m_n+1) )
            for each n in (0, smoothed local maximum, -1)
        """
        from itertools import compress

        localmaxima = MessageAnalyzer.localMaxima(self.values)  # List[idx], List[max]
        allovermax = max(localmaxima[1])
        maxSmsk = [True if e > self._sensitivity * allovermax else False for e in localmaxima[1]]
        lmaxAO = [0] + list(compress(localmaxima[0], maxSmsk)) + [len(self._message.data)]
        lmaxMed = (numpy.round(numpy.ediff1d(lmaxAO) / 2) + lmaxAO[:-1]).astype(int)
        bclmaxs = [medl + numpy.argmax(self.bitcongruences[medl:medr]) for medl, medr in zip(lmaxMed[:-1], lmaxMed[1:])]
        return bclmaxs



class SlidingNbc2ndDelta(SlidingNmeanBitCongruence):
    """
    2nd order difference quotient (forward finite difference, h=1) for all values.

    Field boundaries have no obvious property in this 2nd order difference quotient (NTP/DNS).
    """
    def __init__(self, message: AbstractMessage, unit=MessageAnalyzer.U_BYTE):
        super().__init__(message, unit)
        self._startskip += 1

    @property
    def domain(self):
        return -1,1

    def analyze(self):
        super().analyze()
        self._values = [x2-2*x1+x0 for x2,x1,x0 in zip(self._values[2:], self._values[1:-1], self._values[:-2])]
        # self._values = numpy.divide(numpy.diff(self._values, n=8), 8).tolist()


class HorizonBitcongruence(BitCongruence):
    """
    This is already the DELTA between the mean of the BC of 2 bytes to the left of n and the BC at n.

    >>> from nemere.validation.dissectorMatcher import MessageComparator
    >>> from nemere.utils.loader import SpecimenLoader
    >>> from nemere.inference.analyzers import *
    >>> specimens = SpecimenLoader("../input/maxdiff-fromOrig/smb_SMIA20111010-one-rigid1_maxdiff-100.pcap",
    ...     relativeToIP=True, layer=2)
    >>> # input/maxdiff-fromOrig/ntp_SMIA-20111010_maxdiff-100.pcap
    >>> comparator = MessageComparator(specimens, relativeToIP=True, layer=2)
    >>> l4, rm = next(iter(comparator.messages.items()))
    >>> analyzer = MessageAnalyzer.findExistingAnalysis(HorizonBitcongruence, MessageAnalyzer.U_BYTE, l4, (2,))
    >>> a = []
    >>> for l4,rm in comparator.messages.items():
    ...     a.append((len(l4.data), comparator.fieldEndsPerMessage(rm)[-1]))


    """
    def setAnalysisParams(self, horizon):
        if isinstance(horizon, tuple):
            self._analysisArgs = horizon
        else:
            self._analysisArgs = (int(horizon),)
        self._startskip += self._analysisArgs[0]

    @property
    def domain(self):
        return -1,1

    def analyze(self):
        """
        bit congruence compared to number of bytes of horizon backwards.

        :return:
        """
        if not self._analysisArgs:
            raise ParametersNotSet('Analysis parameter missing: horizon.')
        horizon = self._analysisArgs[0]

        tokenlist = self._message.data  # tokenlist could also be list of ngrams.
        bitcongruences = BitCongruence.bitCongruenceBetweenTokens(tokenlist)

        mbhBitVar = list()
        for idx, token in enumerate(bitcongruences[horizon:], horizon):
            congruenceUptoHorizon = numpy.mean(bitcongruences[idx-2:idx])
            mbVar = token - congruenceUptoHorizon
            mbhBitVar.append(mbVar)
        self._values = mbhBitVar

        # add this object to the cache
        MessageAnalyzer.analyze(self)


class HorizonBitcongruenceGauss(HorizonBitcongruence):
    """
    Gauss filtered multi-byte horizon bit congruence. This is already based on the DELTA between the mean of the BC of 2 bytes to the left of n and the BC at n.
    """
    def __init__(self, message: AbstractMessage, unit=MessageAnalyzer.U_BYTE):
        super().__init__(message, unit)
        self._bcvalues = None

    def setAnalysisParams(self, horizon=2, sigma=1.5):
        self._analysisArgs = (horizon, sigma)

    def analyze(self):
        from collections import Sequence
        if not self._analysisArgs or not isinstance(self._analysisArgs, Sequence):
            raise ParametersNotSet('Analysis parameter missing: horizon and sigma.')
        horizon, sigma = self._analysisArgs
        super().analyze()
        self._bcvalues = self._values
        self._values = list(gaussian_filter1d(self._values, sigma))

    @property
    def bitcongruences(self):
        if self._bcvalues is None:
            return None
        return [0.0] * self.startskip + self._bcvalues

    def messageSegmentation(self) -> List[MessageSegment]:
        """
        Segment message by determining local maxima of sigma-1.5-gauss-filtered 2-byte-horizon bit-congruence.

        >>> from netzob.Model.Vocabulary.Messages.L4NetworkMessage import L4NetworkMessage
        >>> tstmsg = '19040aec0000027b000012850a6400c8d23d06a2535ed71ed23d09faa4673315d23d09faa1766325d23d09faa17b4b10'
        >>> l4m = L4NetworkMessage(bytes.fromhex(tstmsg))
        >>> hbg = HorizonBitcongruenceGauss(l4m)
        >>> hbg.setAnalysisParams()
        >>> hbg.analyze()
        >>> spm = hbg.messageSegmentation()
        >>> # noinspection PyUnresolvedReferences
        >>> print(b''.join([seg.bytes for seg in spm]).hex() == spm[0].message.data.hex())
        True

        :return: Segmentation of this message based on this analyzer's type.
        """
        if not self.values:
            if not self._analysisArgs:
                raise ValueError('No values or analysis parameters set.')
            self.analyze()

        bclmins = self.pinpointMinima()

        # prevent 1 byte segments, since they do not contain usable congruence!
        cutCandidates = [0] + [int(b) for b in bclmins] + [len(self._message.data)]  # add the message end
        cutPositions = [0] + [right for left, right in zip(
            cutCandidates[:-1], cutCandidates[1:]
        ) if right - left > 1]
        if cutPositions[-1] != cutCandidates[-1]:
            cutPositions[-1] = cutCandidates[-1]


        segments = list()
        for lmaxCurr, lmaxNext in zip(cutPositions[:-1], cutPositions[1:]):
            segments.append(MessageSegment(self, lmaxCurr, lmaxNext-lmaxCurr))
        return segments

    def pinpointMinima(self):
        """
        Pinpoint the exact positions of local minima within the scope of each smoothed local minimum.
        The exact position is looked for in self.bitcongruences.

        :return: One exact local minium m in the interval ( center(m_n-1, m_n), center(m_n, m_n+1) )
            for each n in (0, smoothed local minimum, -1)
        """
        localminima = MessageAnalyzer.localMinima(self.values)  # List[idx], List[min]
        # localmaxima = MessageAnalyzer.localMaxima(self.values)  # List[idx], List[max]
        # for lminix in range(len(localminima)):
        #     localminima[lminix]
        lminAO = [0] + localminima[0] + [len(self._message.data)]
        lminMed = (numpy.round(numpy.ediff1d(lminAO) / 2) + lminAO[:-1]).astype(int)
        bclmins = [medl + numpy.argmin(self.bitcongruences[medl:medr]) for medl, medr in zip(lminMed[:-1], lminMed[1:])]
        return bclmins



class HorizonBitcongruenceGradient(HorizonBitcongruence):
    """
    Gradient (centered finite difference, h=1) with numpy method.
    """
    def analyze(self):
        super().analyze()
        self._values = numpy.gradient(self._values).tolist()


class HorizonBitcongruenceDelta(HorizonBitcongruence):
    """
    Difference quotient (forward finite difference, h=1) for all values.
    This is already the delta of the DELTA between the mean of the BC of 2 bytes to the left of n and the BC at n.

    NTP: field starts mostly at minimum (some FP) and at first of a sequence (at least 2) of values near 0.

    DNS: Zero sequences without structure. Non-zero is field - or peak after near zero sequence.
        High values between areas of low (almost 0) difference quotient (here)
        is ASCII-punctuation between ASCII-letters.

    Alternative Idea
    ====

    A difference quotient of n > 1 (8, 6, 4) may show regularly recurring 0s for consecutive fields
        of equal length ant type.
    """
    def __init__(self, message: AbstractMessage, unit=MessageAnalyzer.U_BYTE):
        super().__init__(message, unit)
        self._startskip += 1

    def analyze(self):
        super().analyze()
        self._values = numpy.ediff1d(self._values).tolist()
        # self._values = numpy.divide(numpy.diff(self._values, n=8), 8).tolist()



class HorizonBitcongruence2ndDelta(HorizonBitcongruence):
    """
    2nd order difference quotient (forward finite difference, h=1) for all values.
    This is already delta of the delta of the DELTA between the mean of the BC of 2 bytes to the left of n and the BC at n.

    Field boundaries have no obvious property in this 2nd order difference quotient (NTP/DNS).
    """
    def __init__(self, message: AbstractMessage, unit=MessageAnalyzer.U_BYTE):
        super().__init__(message, unit)
        self._startskip += 1

    def analyze(self):
        super().analyze()
        self._values = [x2-2*x1+x0 for x2,x1,x0 in zip(self._values[2:], self._values[1:-1], self._values[:-2])]
        # self._values = numpy.divide(numpy.diff(self._values, n=8), 8).tolist()


class Autocorrelation(MessageAnalyzer):
    """
    Correlate the analysis of this message to itself,
    shifting over all bytes in the message.
    """
    def __init__(self, message: AbstractMessage, unit=MessageAnalyzer.U_BYTE):
        super().__init__(message, unit)
        self._am = None  # type: Union[MessageAnalyzer, None]

    @property
    def domain(self):
        """
        The correct domain cannot be determined for segments based on autocorrelation.
        It is depenent on the length of the segment (dim):
            sum( [analysisdomain]_1..dim * [analysisdomain]_1..dim )

        :return: The domain for the whole message (>= domain of one segment {in} message)
        """
        return -len(self.message.data) * self._am.domain[0], \
                len(self.message.data) * self._am.domain[1]

    def setAnalysisParams(self, analysisMethod: Type[MessageAnalyzer], *analysisArgs):
        self._am = MessageAnalyzer.findExistingAnalysis(analysisMethod, MessageAnalyzer.U_BYTE,
                                                        self._message, analysisArgs)
        # self._am = analysisMethod(self._message)  # type: MessageAnalyzer
        # self._am.setAnalysisParams(*analysisArgs)

    @property
    def analysisParams(self):
        if isinstance(self._am.analysisParams, tuple):
            return (type(self._am),) + self._am.analysisParams
        return type(self._am), self._am.analysisParams

    def analyze(self):
        """
        Get the list of discrete values of the correlation result from reading self.values
        """
        if not self._am:
            raise ParametersNotSet('Analysis method missing.')
        results = self._am.valuesRaw
        correlation = numpy.correlate(results, results, 'full')
        self._values = [numpy.nan] * self._am.startskip + correlation.tolist()
        super().analyze()


class CumulatedValueProgression(MessageAnalyzer):
    """
    Cumulation of value progression. Shows the subsequently cumulated values.
    """
    @property
    def domain(self):
        return 0, (255 if self.unit == MessageAnalyzer.U_BYTE else 128) * len(self.message)

    def analyze(self):
        valuecumulation = [0]  # initial value
        if self._unit == MessageAnalyzer.U_NIBBLE:
            tokens = self.nibblesFromBytes(self._message.data)
        else:
            tokens = self._message.data

        prev = 0
        for token in tokens:
            prev += token
            valuecumulation.append(prev)
        self._values = valuecumulation


class CumulatedProgressionGradient(CumulatedValueProgression):
    """
    Gradient (centered finite difference, h=1) with numpy method.
    """
    @property
    def domain(self):
        return 0, 255 if self.unit == MessageAnalyzer.U_BYTE else 128

    def analyze(self):
        super().analyze()
        self._values = numpy.gradient(self._values).tolist()


class EntropyWithinNgrams(MessageAnalyzer):
    """
    Calculates the entropy of each message ngrams based on an alphabet of bytes or nibbles (4 bit).
    """

    @property
    def domain(self):
        from math import log
        return 0, log(len(self._message.data) - self._n + 1, 2)

    def __init__(self, message: AbstractMessage, unit=MessageAnalyzer.U_BYTE):
        super().__init__(message, unit)
        self._n = None

    def setAnalysisParams(self, n: Union[int, Tuple[int]]):
        self._n = int(n if not isinstance(n, tuple) else n[0])
        self._startskip = self._n

    def analyze(self):
        ngramEntropies = list()
        for gram in [gram for gram in self.ngrams(self._n)]:
            if self._unit == MessageAnalyzer.U_NIBBLE:
                tokens = MessageAnalyzer.nibblesFromBytes(gram)
            else:
                tokens = gram

            ngramEntropies.append(MessageAnalyzer.calcEntropy(tokens))  # should work for bytes
        self._values = ngramEntropies


class ValueVariance(MessageAnalyzer):
    """
    Shows the difference between subsequent values.

    The early analyzer ValueProgressionDelta, i. e., the differential value progression, is the inverse of
    ValueVariance: ValueVariance == inverted (minus) ValueProgressionDelta.
    We removed ValueProgressionDelta to prevent confusion.

    LOL. ValueVariance == CumulatedProgression2ndDelta. :-D

    Field boundaries have no obvious property in this 2nd order difference quotient (NTP/DNS).
    """
    def __init__(self, message: AbstractMessage, unit=MessageAnalyzer.U_BYTE):
        super().__init__(message, unit)
        self._startskip = 1
        self._analysisArgs = (200,)

    @property
    def domain(self):
        extrema = 255 if self.unit == MessageAnalyzer.U_BYTE else 128
        return -extrema, extrema

    def setAnalysisParams(self, steepness=(200,)):
        """
        200 is a decent tradeoff.
        A larger value (220) benefits only NTP.
        Other protocols rather benefit from lower values (150).

        see ScoreStatistics-VD-steepness.ods
        """
        if isinstance(steepness, tuple):
            self._analysisArgs = steepness
        else:
            self._analysisArgs = (int(steepness),)

    @property
    def steepness(self):
        return self._analysisArgs[0]

    def analyze(self):
        """
        Relative variance of single message bytes.
        """
        self._values = MessageAnalyzer.tokenDelta(list(self._message.data), self._unit)

    def messageSegmentation(self) -> List[MessageSegment]:
        if not self.values:
            self.analyze()

        # value drop or rise more than steepness threshold in one step, split at highest abs(value)
        sc = self.steepChanges()
        # and value drop to or rise from 0, split at the non-zero value
        zb = self.zeroBorders()

        cutat = numpy.add(sorted(set(sc + zb)), self._startskip).tolist()
        if cutat[0] != 0:
            cutat = [0] + cutat
        if cutat[-1] != len(self._message.data):
            cutat = cutat + [len(self._message.data)]


        segments = list()
        for cutCurr, cutNext in zip(cutat[:-1], cutat[1:]):  # add the message end
            segments.append(MessageSegment(self, cutCurr, cutNext-cutCurr))
        return segments

    def steepChanges(self):
        """
        value drop or rise more than steepness in one step, split at highest abs(value)

        :return:
        """
        return [ ix if abs(vl) > abs(vr) else ix+1
                 for ix, (vl, vr) in enumerate(zip(self._values[:-1], self._values[1:]))
                 if abs(vr-vl) > self.steepness]

    def zeroBorders(self):
        """
        value drop to or rise from 0, split at the non-zero value

        :return:
        """
        return [ ix if abs(vl) > abs(vr) else ix+1
                 for ix, (vl, vr) in enumerate(zip(self._values[:-1], self._values[1:]))
                 if (vr == 0) != (vl == 0)]

class VarianceAmplitude(MessageAnalyzer):
    @property
    def domain(self):
        return None

    def __init__(self, message: AbstractMessage, unit=MessageAnalyzer.U_BYTE):
        super().__init__(message, unit)
        self._startskip = 2

    def analyze(self):
        """
        "amplitude" of variance => threshold.

        change of the amplitude
            * intra message
            * TODO across messages
        """
        self._values = MessageAnalyzer.tokenDelta(
            MessageAnalyzer.tokenDelta(
                list(self._message.data), self._unit))


class ValueFrequency(MessageAnalyzer):
    @property
    def domain(self):
        return 0, len(self._message.data)

    @property
    def values(self) -> Dict[int, int]:
        return self._values

    def analyze(self):
        """
        frequency of byte values.

        only unit = U_BYTE

        :return: dict of value => frequency mappings.
        """
        bucket = dict()
        for char in self._message.data:
            if char in bucket:
                bucket[char] += 1
            else:
                bucket[char] = 1
        self._values = bucket

    def mostFrequent(self):
        """
        Most frequent byte values

        :return:
        """
        if not self._values:
            raise ValueError('Method analyze() was not called previously.')

        mostFreq = sorted([(frq, val) for val, frq in self._values.items()], key=lambda k: k[0])
        return mostFreq



class Value(MessageAnalyzer):
    """
    Simply returns the byte values of the message.

    LOL. this is CumulatedProgressionDelta == ValueProgression == Value. :-D

    Alternative Idea
    ====
    A difference quotient of n > 1 (8, 6, 4) may show regularly recurring 0s for consecutive fields
        of equal length ant type.
    """
    @property
    def domain(self):
        return 0, 255 if self.unit == MessageAnalyzer.U_BYTE else 128

    def analyze(self):
        """
        Does nothing.
        """
        pass

    @property
    def values(self):
        if self.unit == MessageAnalyzer.U_BYTE:
            return list(self.message.data)
        else:
            return MessageAnalyzer.nibblesFromBytes(self.message.data)

    def messageSegmentation(self) -> List[MessageSegment]:
        """
        produces very bad/unusable results.

        :return:
        """
        if not self.values:
            self.analyze()

        # sudden drop (inversion?) in progression delta steepness.
        sc = self.steepChanges(.3)  # TODO iterate best value

        cutat = numpy.add(sorted(set(sc)), self._startskip).tolist()
        if len(cutat) == 0 or cutat[0] != 0:
            cutat = [0] + cutat
        if len(cutat) == 0 or cutat[-1] != len(self._message.data):
            cutat = cutat + [len(self._message.data)]  # add the message end

        segments = list()
        for cutCurr, cutNext in zip(cutat[:-1], cutat[1:]):
            segments.append(MessageSegment(self, cutCurr, cutNext-cutCurr))
        return segments

    def steepChanges(self, epsfact: float=.1):
        """
        From the top of the value range to the bottom of the value range directly.

        :param epsfact: value deviation towards the middle considered to be at the limits of the value range.
        :return:
        """
        if epsfact > 1 or epsfact <= 0:
            raise ValueError('epsilon factor for allowed range deviation is below 0 or above 1.')

        vmin = min(self.values)
        vmax = max(self.values)
        epsilon = (vmax - vmin) * epsfact
        emin = vmin + epsilon
        emax = vmax - epsilon
        return [ ix
                 for ix, (vl, vr) in enumerate(zip(self.values[:-1], self.values[1:]))
                 if vl > emax and vr < emin]


class Entropy(SegmentAnalyzer):
    """
    Calculates the entropy of each message segment based on the alphabet of bytes or nibbles (4 bit) in this segment.
    This analyzer calculates the entropy of an already existing segment (subclass of SegmentAnalyzer) and is not a segmenter!
    """
    def value(self, start, end):
        if self._unit == MessageAnalyzer.U_NIBBLE:
            return [MessageAnalyzer.calcEntropy(MessageAnalyzer.nibblesFromBytes(self.message.data[start:end]))]
        else:
            return [MessageAnalyzer.calcEntropy(self.message.data[start:end])]

