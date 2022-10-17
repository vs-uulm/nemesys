from abc import ABC
from typing import Type, Union, Any, Tuple, Iterable, List, Dict

import numpy
import scipy.spatial

from netzob.Model.Vocabulary.Messages.AbstractMessage import AbstractMessage

from nemere.inference.analyzers import Value
from nemere.inference.segments import MessageAnalyzer, TypedSegment




class BaseTypeMemento(object):
    """
    Base class providing the means to identify a field type by a name and ID.
    """
    def __init__(self, fieldtype: str, length = None):
        # data type this field represents
        self._fieldtype = fieldtype
        self.__length = length

    @property
    def fieldtype(self):
        return self._fieldtype

    def __len__(self):
        if self.__length is None:
            raise ValueError("Call of len() on a BaseTypeMemento without this property.")
        return self.__length

    @property
    def typeID(self, short=True):
        """
        :param short: Use only the last half (4 bytes) of the hash
        :return: As an identifier use the hash of the fieldtype value
        """
        tid = "{:02x}".format(hash(self.fieldtype))
        return tid[-8:] if short else tid

    def __repr__(self):
        return "FieldTypeMemento " + self.typeID + " for " + self.fieldtype


class FieldTypeMemento(BaseTypeMemento):
    """
    Class to help persisting field type characteristics from a FieldTypeTemplate represented by mean and covariance.
    Contains methods to calculate the covariance matrix, mahalanobis distance to a given vector, and the "confidence"
    of a positive match.
    """

    def __init__(self, mean: numpy.ndarray, stdev: numpy.ndarray, cov: numpy.ndarray, fieldtype: str,
                 analyzerClass: Type[MessageAnalyzer] = Value, analysisParams: Union[Any, Tuple] = None,
                 unit=MessageAnalyzer.U_BYTE):
        super().__init__(fieldtype)
        self._mean = mean
        self._cov = cov
        self._picov = None
        self._stdev = stdev
        # for reference:
        self._analyzerClass = analyzerClass
        self._analysisParams = analysisParams
        self._unit = unit


    'from inference.templates import FieldTypeTemplate'
    # noinspection PyUnresolvedReferences
    @staticmethod
    def fromTemplate(ftt: 'FieldTypeTemplate'):
        ftm = FieldTypeMemento(ftt.mean, ftt.stdev, ftt.cov, ftt.fieldtype,
                               type(ftt.baseSegments[0].analyzer), ftt.baseSegments[0].analyzer.analysisParams,
                               ftt.baseSegments[0].analyzer.unit)
        return ftm

    @property
    def mean(self) -> numpy.ndarray:
        return self._mean

    @property
    def stdev(self) -> numpy.ndarray:
        return self._stdev

    @property
    def cov(self) -> numpy.ndarray:
        """
        There is some rounding error so the stdev is not entierely identical to the diagonal of the covariance matrix.

        >>> from nemere.inference.templates import FieldTypeTemplate
        >>> from nemere.utils.baseAlgorithms import generateTestSegments
        >>> bs = generateTestSegments()
        >>> ftt = FieldTypeTemplate(bs)
        >>> # numpy.round(ftt.stdev, 8) == numpy.round(ftt.cov.diagonal(), 8)
        >>> numpy.round(ftt.stdev, 8)
        array([ 0.        ,  0.        ,  0.        , 20.40067401, 31.70392545,
                0.49487166,  9.16292441])
        >>> numpy.round(ftt.cov.diagonal(), 8)
        array([...e-0..., ...e-0..., ...e-0..., 5.54916667e+02,
               1.20616667e+03, 2.85714290e-01, 9.79523810e+01])

        :return: The covariance matrix of the template.
        """
        return self._cov

    @property
    def picov(self) -> numpy.ndarray:
        """
        Often cov is a singular matrix in our use case, so we use the approximate Moore-Penrose pseudo-inverse
        from numpy.
        (G. Strang, Linear Algebra and Its Applications, 2nd Ed., Orlando, FL, Academic Press, Inc., 1980, pp. 139-142.)

        :return: pseudo-inverse of the covariance matrix.
        """
        if self._picov is None:
            self._picov = numpy.linalg.pinv(self.cov)
        return self._picov

    @property
    def upper(self) -> numpy.ndarray:
        return self._mean + self.stdev

    @property
    def lower(self) -> numpy.ndarray:
        return self._mean - self.stdev

    @property
    def analyzerClass(self) -> Type[MessageAnalyzer]:
        """
        :return: The type of the analyzer
        """
        return self._analyzerClass

    def recreateAnalyzer(self, message: AbstractMessage) -> MessageAnalyzer:
        """
        Recreate an analyzer of the type and configuration given in this memento instance.

        :param message: The message to create the analyzer for.
        :return: The newly created analyzer instance.
        """
        return MessageAnalyzer.findExistingAnalysis(self._analyzerClass, self._unit, message, self._analysisParams)

    def __len__(self):
        return len(self._mean)

    @property
    def typeID(self, short=True) -> str:
        """
        :param short: Use only the last half (4 bytes) of the hash
        :return: As an identifier use the hash of the mean values
        """
        tid = "{:02x}".format(hash(tuple(self.mean)))
        return tid[-8:] if short else tid

    @property
    def codePersist(self) -> str:
        """:return: Python code to persist this Memento"""
        return "{}(numpy.array({}), numpy.array({}), numpy.array({}), '{}', {}, {}, {})".format(
            type(self).__name__, self.mean.tolist(), self.stdev.tolist(), self.cov.tolist(), self._fieldtype,
            self._analyzerClass.__name__, self._analysisParams,
            "MessageAnalyzer.U_BYTE" if self._unit == MessageAnalyzer.U_BYTE else "MessageAnalyzer.U_NIBBLE")

    def mahalanobis(self, vector: Iterable[float]) -> numpy.ndarray:
        """
        Compute the Mahalanobis distance between this fieldtype's mean and the given vector using the
        covariance matrix contained in this object.

        Mahalanobis distance measures the distance of a vector from the mean in terms of the multivariate pendent to
        the standard deviation: zotero

        :param vector: The vector of which the distance to the mean shall be calculated.
        :return: The Mahalanobis distance between the field type mean and the given vector.
        """
        return scipy.spatial.distance.mahalanobis(self.mean, vector, self.picov)

    def confidence(self, vector: Iterable[float]) -> numpy.ndarray:
        """
        :param vector: A feature vector (e. g. byte values)
        :return: The confidence that the given vector is of the field type represented by this memento.
            Mostly this is equivalent to the mahalanobis distance between vector and FieldTypeMemento, but for
            the fieldtype "id" the confidence is reduced by factor 2 (smaller value => higher confidence).
        """
        conf = self.mahalanobis(vector)
        # TODO move to be a parameterizable property of the FieldTypeMemento class
        # make ids twice as unconfident
        if self.fieldtype == "id":
            conf *= 2
        return conf


class RecognizedVariableLengthField(object):
    """
    Lightweight representation a field of variable length recognized by a heuristic method.
    """

    def __init__(self, message: AbstractMessage, template: BaseTypeMemento,
                 position: int, end: int, confidence: float):
        self.message = message
        self.position = position
        self.end = end
        self.confidence = confidence
        self.template = template

    def __repr__(self) -> str:
        """
        :return: A textual representation of this heuristically recognized field.
        """
        return "RecognizedField of {} at ({}, {}) (c {:.2f})".format(
            self.template.fieldtype, self.position, self.end, self.confidence)


    def isOverlapping(self, otherField: 'RecognizedField') -> bool:
        """
        Determines whether the current recognized field overlaps with a given other field.

        :param otherField: The other field to check against.
        :return: Is overlapping or not.
        """
        if self.message == otherField.message \
                and (self.position <= otherField.position < self.end
                or otherField.position < self.end <= otherField.end):
            return True
        else:
            return False


    def toSegment(self, fallbackAnalyzer:Type[MessageAnalyzer]=Value,
                  fallbackUnit: int=MessageAnalyzer.U_BYTE, fallbackParams: Tuple=()) -> TypedSegment:
        """
        Convertes this object to a MessageSegment. Uses the analyzer stored in the object's template
        to (re-)create the segments.

        :param fallbackAnalyzer: Used if the object knows no template to extract an analyzer from.
        :param fallbackUnit: Used if the object knows no template to extract a unit from.
        :param fallbackParams: Used if the object knows no template to extract analyzer parameters from.
        :return: A segment anotated with this templates fieldtype.
        """
        if isinstance(self.template, FieldTypeMemento):
            analyzer = self.template.recreateAnalyzer(self.message)
        else:
            # With out a FieldTypeMemento we have no information about the original analyzer, and probably there never
            #   has existed any, e.g. for char and flag heuristics. Therefore define a fallback.
            analyzer = MessageAnalyzer.findExistingAnalysis(
                fallbackAnalyzer, fallbackUnit, self.message, fallbackParams)
        return TypedSegment(analyzer, self.position, len(self.template), self.template.fieldtype)


class RecognizedField(RecognizedVariableLengthField):
    """
    Represents a field of constant length recognized by a heuristic method.
    """
    def __init__(self, message: AbstractMessage, template: BaseTypeMemento,
                 position: int, confidence: float):
        """
        Create the representation of a heuristically recognized field.
        This object is much more lightweight than inference.segments.AbstractSegment.

        :param message: The message this field is contained in.
        :param template: The field type template that this field resembles.
        :param position: The byte position/offset in the message at which the field starts.
        :param confidence: The confidence (0 is best) of the recognition.
        """
        if isinstance(template, BaseTypeMemento):
            super().__init__(message, template, position, position + len(template), confidence)
        else:
            raise TypeError("Template needs to be a BaseTypeMemento")



class ValueFieldTypeRecognizer(ABC):
    """
    Provides the method #recognizedFields to find the most confident matches with the known field types to be recognized.
    The templates to recognize must be provided in a subclass into the class variable `fieldtypeTemplates` as
    list of `FieldTypeMemento`s.
    """
    fieldtypeTemplates = None  # type: List[FieldTypeMemento]

    def __init__(self, analyzer: MessageAnalyzer):
        self._analyzer = analyzer
        for ftt in type(self).fieldtypeTemplates:
            assert ftt.analyzerClass == type(analyzer), \
                "The given analyzer is not compatible to the existing templates."

    @property
    def message(self):
        return self._analyzer.message

    def findInMessage(self, fieldtypeTemplate: FieldTypeMemento):
        """

        :param fieldtypeTemplate:
        :return: list of (position, confidence) for all offsets.
        """
        assert fieldtypeTemplate.analyzerClass == type(self._analyzer)

        # position, confidence
        posCon = list()
        ftlen = len(fieldtypeTemplate)
        for offset in range(len(self._analyzer.values) - ftlen + 1):
            ngram = self._analyzer.values[offset:offset+ftlen]
            if set(ngram) == {0}:  # zero values do not give any information
                posCon.append(99)
            else:
                posCon.append(fieldtypeTemplate.confidence(ngram))  # / (ftlen**2 / 4)

        return posCon

    def charsInMessage(self) -> List[RecognizedField]:
        """
        Mark recognized char sequences.

        :return: list of recognized char sequences with the constant confidence of 0.2
        """
        from nemere.inference.segmentHandler import isExtendedCharSeq

        confidence = 0.2
        offset = 0
        minLen = 6
        recognizedChars = list()
        while offset < len(self.message.data) - minLen:
            # ignore chunks starting with 00 if its not likely an UTF16 string
            # if (self.message.data[offset:offset+1] != b"\x00"
            #         or self.message.data[offset+1:offset+2] != b"\x00"
            #         and self.message.data[offset+2:offset+3] == b"\x00"
            #     ) and ( ):
            if b"\x00" not in self.message.data[offset:offset + minLen] \
                    or self.message.data[offset:offset + minLen].count(b"\x00") > 1 \
                    and b"\x00\x00" not in self.message.data[offset:offset + minLen]:
                add2len = 0
                while offset + minLen + add2len <= len(self.message.data) \
                        and isExtendedCharSeq(self.message.data[offset:offset + minLen + add2len], minLen=6):
                    add2len += 1
                    # end chunk if zero byte or double zero for a likely UTF16 is hit
                    # // xx 00 00
                    # -3 -2 -1
                    # -4 -3 -2 -1
                    if self.message.data[offset + minLen + add2len - 1:offset + minLen + add2len] == b"\x00" \
                        and self.message.data[offset + minLen + add2len - 3:offset + minLen + add2len - 2] != b"\x00":
                            # or self.message.data[offset + minLen + add2len - 2:offset + minLen + add2len - 1] == b"\x00"):
                        add2len += 1
                        break
                chunkLen = minLen + add2len - 1
                chunk = self.message.data[offset:offset + chunkLen]
                if isExtendedCharSeq(chunk, minLen=6):
                    recognizedChars.append(
                        RecognizedField(self.message, BaseTypeMemento("chars", chunkLen), offset, confidence))
                    offset += minLen + add2len
                else:
                    offset += 1
            else:
                offset += 1
        return recognizedChars

    def flagsInMessage(self) -> List[RecognizedField]:
        """
        Mark probable flag byte pairs.
        Recognizes probable flag bytes by not exceeding the number of 2 bits set per byte.

        :return: list of recognized flag sequences with the constant confidence of 3.0
        """
        confidence = 3.0
        offset = 0
        minLen = 2
        bitset = 2

        recognizedFlags = list()
        while offset < len(self.message.data) - minLen:
            belowBitset = True
            bitSum = 0
            bitCountSeq = list()
            for bVal in self.message.data[offset:offset + minLen]:
                bitCount = bin(bVal).count("1")
                bitCountSeq.append(bitCount)
                bitSum += bitCount
                if bitCount > bitset:
                    belowBitset = False
                    break
            if belowBitset and bitSum > 0:
                # TODO find a valid dynamic generation by observing groundtruth
                # confidence = 4 * bitSum / bitset  # * minLen
                recognizedFlags.append(
                    RecognizedField(self.message, BaseTypeMemento("flags", minLen), offset, confidence))
            offset += 1
        return recognizedFlags

    def recognizedFields(self, confidenceThreshold = None) -> Dict[FieldTypeMemento, List[RecognizedField]]:
        """
        Most probable inferred field structure: The field template positions with the highest confidence for a match.
        Iterate starting from the most confident match and removing all further matches that overlap with this high
        confidence match.

        :param confidenceThreshold: Threshold to decide which confidence value is high enough to assume a match if no
            concurring comparison values ("no relevant matches") are in this message. If not given, a threshold is
            determined dynamically from 20/numpy.log(ftMemento.stdev.mean()).
            The rationale is that the stdev allows to estimate the original distance that the majority of base segments
            the template was created from resided in. TODO what about the log and factor 20?
        :return: Mapping of each field type to the list recognized fields for it.
        """
        applConfThre = confidenceThreshold
        mostConfident = dict()
        for ftMemento in type(self).fieldtypeTemplates:
            if confidenceThreshold is None:
                applConfThre = 20/numpy.log(ftMemento.stdev.mean())
            confidences = self.findInMessage(ftMemento)
            mostConfident[ftMemento] = [RecognizedField(self.message, ftMemento, pos, con)
                                        for pos, con in enumerate(confidences) if con < applConfThre]
        mostConfident[BaseTypeMemento("chars")] = self.charsInMessage()
        mostConfident[BaseTypeMemento("flags")] = self.flagsInMessage()
        return mostConfident


















