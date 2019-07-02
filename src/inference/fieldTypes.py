from typing import Type, Union, Any, Tuple, Iterable, List, Dict

import numpy
import scipy.spatial

from inference.analyzers import Value
from inference.segments import MessageAnalyzer, MessageSegment, TypedSegment
from netzob.Model.Vocabulary.Messages.AbstractMessage import AbstractMessage




class BaseTypeMemento(object):
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
        >>> from inference.templates import FieldTypeTemplate
        >>> from inference.segments import MessageSegment
        >>> bs = [MessageSegment(None, 0, 1)]
        >>> ftt = FieldTypeTemplate(bs)
        >>> numpy.round(ftt.stdev, 8) == numpy.round(ftt.cov.diagonal(), 8)

        :return: The covariance matrix of the template.
        """
        return self._cov

    @property
    def picov(self) -> numpy.ndarray:
        """
        Often cov is a singluar matrix in our use case, so we use the approximate Moore-Penrose pseudo-inverse
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
        the standard deviation: zoter

        :param vector: The vector of which the distance to the mean shall be calculated.
        :return: The Mahalanobis distance between the field type mean and the given vector.
        """
        return scipy.spatial.distance.mahalanobis(self.mean, vector, self.picov)

    def confidence(self, vector: Iterable[float]) -> numpy.ndarray:
        conf = self.mahalanobis(vector)
        # TODO move to be a parameterizable property of the FieldTypeMemento class
        # make ids twice as unconfident
        if self.fieldtype == "id":
            conf *= 2
        return conf


class RecognizedVariableLengthField(object):
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
    Represents a field recognized by a heuristic method.
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



class FieldTypeRecognizer(object):

    # TODO make extensible by moving to a ValueFieldTypeRecognizer subclass and making this class abstract.
    # from commit 475d179
    fieldtypeTemplates = [
        # timestamp
        FieldTypeMemento(numpy.array([205.75982532751092, 63.838427947598255, 24.375545851528383, 122.61135371179039, 133.65652173913043, 134.2608695652174, 147.52173913043478, 120.2695652173913]), numpy.array([25.70183334947299, 19.57641190835309, 53.50246711271789, 66.9864879316446, 71.84675313281585, 72.93261773499316, 72.72710729917598, 76.51925835899723]), numpy.array([[663.4815368114683, -472.49073010036244, -423.75589902704553, 139.7702826936337, 9.80247835746574, 3.6199532674480817, -242.52683291197417, 505.4362598636329], [-472.49073010036244, 384.91676243009425, 296.96445261625695, -111.35254347659536, 75.25739293648974, -4.666034628054849, 166.73218800275805, -345.0736612273047], [-423.75589902704553, 296.96445261625695, 2875.068873056001, 318.1860683367806, 261.37767179958604, 84.96889603922475, 157.02409407798976, -593.1297594422736], [139.7702826936337, -111.35254347659536, 318.1860683367806, 4506.870221405036, 442.0494330805179, -212.5338044893896, -498.76463265149766, 382.4062667585993], [9.80247835746574, 75.25739293648974, 261.37767179958604, 442.0494330805179, 5184.497228023539, -456.02790962597265, -181.6759065881906, 370.75238276058485], [3.6199532674480817, -4.666034628054849, 84.96889603922475, -212.5338044893896, -456.02790962597265, 5342.394531991645, -240.0144294664894, 118.43155496487557], [-242.52683291197417, 166.73218800275805, 157.02409407798976, -498.76463265149766, -181.6759065881906, -240.0144294664894, 5312.329219669637, -311.37269793051064], [505.4362598636329, -345.0736612273047, -593.1297594422736, 382.4062667585993, 370.75238276058485, 118.43155496487557, -311.37269793051064, 5880.76544522499]]), 'timestamp', Value, (), MessageAnalyzer.U_BYTE) ,
        # ipv4
        FieldTypeMemento(numpy.array([172.65573770491804, 23.098360655737704, 2.4098360655737703, 40.73770491803279]), numpy.array([3.561567374164002, 22.79682528805688, 1.5404603227322615, 68.36543600851412]), numpy.array([[12.896174863387957, 80.93442622950825, 3.726775956284154, 2.841530054644803], [80.93442622950825, 528.3568306010928, 19.475683060109265, 50.24289617486337], [3.726775956284154, 19.475683060109265, 2.4125683060109235, 19.059289617486336], [2.841530054644803, 50.24289617486337, 19.059289617486336, 4751.73005464481]]), 'ipv4', Value, (), MessageAnalyzer.U_BYTE) ,
        # macaddr
        FieldTypeMemento(numpy.array([0.0, 12.0, 41.0, 137.28571428571428, 123.17857142857143, 124.82142857142857]), numpy.array([0.0, 0.0, 0.0, 69.38446570834608, 77.84143845546744, 67.62293227439753]), numpy.array([[0.01691577109311472, 0.0017520133005179852, -0.0009323360045556071, 0.42935337597073336, 1.9917595051987003, -1.850657000447773], [0.0017520133005179852, 0.020892243107491996, -0.0016527739029342462, 0.6300638864312258, 1.8509234806353214, -3.728455646581064], [-0.0009323360045556071, -0.0016527739029342462, 0.0213386532068979, -0.7361112702560478, -0.9897013085501344, 1.1738129845129344], [0.42935337597073336, 0.6300638864312258, -0.7361112702560478, 4992.507936507936, 746.2804232804231, -19.428571428571523], [1.9917595051987003, 1.8509234806353214, -0.9897013085501344, 746.2804232804231, 6283.70767195767, 452.44047619047603], [-1.850657000447773, -3.728455646581064, 1.1738129845129344, -19.428571428571523, 452.44047619047603, 4742.22619047619]]), 'macaddr', Value, (), MessageAnalyzer.U_BYTE) ,
        # id
        FieldTypeMemento(numpy.array([123.72916666666667, 122.125, 136.45833333333334, 117.16666666666667]), numpy.array([71.22374942839565, 68.57861212992479, 72.74726751263965, 79.2507448265034]), numpy.array([[5180.754875886526, 1118.9707446808509, -236.4476950354611, 433.72695035460987], [1118.9707446808509, 4803.090425531915, -360.46276595744695, 696.5531914893616], [-236.4476950354611, -360.46276595744695, 5404.764184397164, 237.8794326241136], [433.72695035460987, 696.5531914893616, 237.8794326241136, 6414.312056737587]]), 'id', Value, (), MessageAnalyzer.U_BYTE) ,
        # float
        FieldTypeMemento(numpy.array([0.9509647848839462, 3.3029185443164613, 22.105263157894736, 116.69473684210526]), numpy.array([0.5588487293595614, 1.6796493488314423, 8.562431632256324, 72.70952644225558]), numpy.array([[0.31563436935261324, -0.04162196912699606, -0.5355585699839925, 4.230231223185803], [-0.04162196912699606, 2.851234934338717, 0.39996987465004535, -16.023069981731833], [-0.5355585699839925, 0.39996987465004535, 74.09518477043672, -41.43561030235157], [4.230231223185803, -16.023069981731833, -41.43561030235157, 5342.916461366181]]), 'float', Value, (), MessageAnalyzer.U_BYTE) ,

        # # int
        # FieldTypeMemento(numpy.array([193.8953488372093, 157.2325581395349]),
        #                  numpy.array([24.531436438079247, 32.666985830090965]),
        #                  numpy.array([[608.8712722298224, -276.293023255814], [-276.293023255814, 1079.6864569083446]]),
        #                  'int', Value, (), MessageAnalyzer.U_BYTE),
        # # checksum
        # FieldTypeMemento(numpy.array(
        #     [109.45454545454545, 93.0909090909091, 134.8181818181818, 100.18181818181819, 138.27272727272728,
        #      103.45454545454545, 138.63636363636363, 120.81818181818181]), numpy.array(
        #     [61.14269990821566, 59.946119057722534, 71.2317832301618, 37.48167596383399, 69.24933988407517,
        #      63.96925914611236, 73.5752833093396, 82.34276275734496]),
        #     numpy.array([[4112.272727272727, -202.74545454545452, 197.29090909090905, -876.5909090909092,
        #                   1638.7636363636366, -414.3272727272728, 115.38181818181832, 1327.9909090909096],
        #                  [-202.74545454545452, 3952.890909090909, 1656.6181818181817, 1572.9818181818182,
        #                   -1493.5272727272727, -1224.345454545455, -1531.1636363636364, 859.7181818181818],
        #                  [197.29090909090905, 1656.6181818181817, 5581.363636363636, 1590.636363636364,
        #                   -932.6454545454549, -1105.3090909090909, -2908.3727272727274, -871.1363636363635],
        #                  [-876.5909090909092,  1572.9818181818182,  1590.636363636364,  1545.3636363636363,
        #                   -1728.2545454545457,  -836.9909090909092,  -597.6272727272726,  -323.6636363636362],
        #                  [1638.7636363636366,  -1493.5272727272727,  -932.6454545454549,  -1728.2545454545457,
        #                   5275.018181818181,  -687.7363636363639,  -954.8909090909092,  1865.354545454546],
        #                  [-414.3272727272728,  -1224.345454545455,  -1105.3090909090909,  -836.9909090909092,
        #                   -687.7363636363639,  4501.272727272727,  1276.9818181818184,  2721.9909090909096],
        #                  [115.38181818181832,  -1531.1636363636364,  -2908.3727272727274,  -597.6272727272726,
        #                   -954.8909090909092,  1276.9818181818184,  5954.654545454545,  1005.1272727272728],
        #                  [1327.9909090909096,  859.7181818181818,  -871.1363636363635,  -323.6636363636362,
        #                   1865.354545454546,  2721.9909090909096,  1005.1272727272728,  7458.363636363637]]), 'checksum',
        #                  Value, (), MessageAnalyzer.U_BYTE),
    ]


    def __init__(self, analyzer: MessageAnalyzer):
        self._analyzer = analyzer
        for ftt in FieldTypeRecognizer.fieldtypeTemplates:
            assert ftt.analyzerClass == type(analyzer), "The given analyzer is not compatible to the existing templates."


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
        Mark recognized char sequences

        :return: list of recognized char sequences with the constant confidence of 0.2
        """
        from inference.segmentHandler import isExtendedCharSeq

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

        :return: list of recog nized flag sequences with the constant confidence of 0.2
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
        for ftMemento in FieldTypeRecognizer.fieldtypeTemplates:
            if confidenceThreshold is None:
                applConfThre = 20/numpy.log(ftMemento.stdev.mean())
            confidences = self.findInMessage(ftMemento)
            mostConfident[ftMemento] = [RecognizedField(self.message, ftMemento, pos, con)
                                        for pos, con in enumerate(confidences) if con < applConfThre]
        mostConfident[BaseTypeMemento("chars")] = self.charsInMessage()
        mostConfident[BaseTypeMemento("flags")] = self.flagsInMessage()
        return mostConfident


class FieldTypeQuery(object):

    def __init__(self, ftRecognizer: FieldTypeRecognizer):
        self.message = ftRecognizer.message
        self._recognizedTemplates = ftRecognizer.recognizedFields()

        # mapping from byte offsets to list of recognitions (RecognizedField)
        self._posTemplMap = dict()  # type: Dict[int, List[RecognizedField]]
        for ftm, poscon in self._recognizedTemplates.items():
            for recognized in poscon:
                for o in range(recognized.position, recognized.end):
                    if o not in self._posTemplMap:
                        self._posTemplMap[o] = list()
                    self._posTemplMap[o].append(recognized)


    def retrieve4position(self, offset: int) -> List:
        templateMatches = sorted(self._posTemplMap[offset], key=lambda x: x.confidence) \
            if offset in self._posTemplMap else []
        return templateMatches


    def mostConfident(self):
        allRecogFields = sorted(
            [recognized for ftm, poscon in self._recognizedTemplates.items() for recognized in poscon],
            key=lambda x: x.confidence
        )
        return allRecogFields


    def resolveConflicting(self):
        nonConflicting = list()
        mostConfident = self.mostConfident()
        while len(mostConfident) > 0:
            mc = mostConfident.pop(0)
            nonConflicting.append(mc)
            # in scope of the current field recognition...
            for position in range(mc.position, mc.end):
                # ... get all the templates at any position...
                for recog in self._posTemplMap[position]:
                    # .. and remove all (less confident) recognitions there
                    if recog in mostConfident:
                        # this is not very efficient using a list!
                        mostConfident.remove(recog)

        return sorted(nonConflicting, key=lambda n: n.position)


    def resolvedSegments(self):  # TODO unfinished
        """
        Fills up the recognized fields with MessageSegments to completely cover the message.

        :return: list of segments for this queries message
        """
        # we assume self.resolveConflicting() is sorted by position!
        segmentList = list()
        nonConflicting = [rf.toSegment() for rf in self.resolveConflicting()]
        analyzer = nonConflicting[0].analyzer if len(nonConflicting) > 0 else \
            MessageAnalyzer.findExistingAnalysis(Value, MessageAnalyzer.U_BYTE, self.message)  # fallback analyzer
        for recseg in nonConflicting:
            lastEnd = segmentList[-1].nextOffset if len(segmentList) > 0 else 0
            if lastEnd < recseg.offset:
                segmentList.append(MessageSegment(analyzer, lastEnd, recseg.offset - lastEnd))
            segmentList.append(recseg)
        # check to fill up rest
        lastEnd = segmentList[-1].nextOffset if len(segmentList) > 0 else 0
        if lastEnd < len(self.message.data):
            segmentList.append(MessageSegment(analyzer, lastEnd, len(self.message.data) - lastEnd))
        return segmentList


    @staticmethod
    def isOverlapping(aStart: int, aEnd: int, bStart: int, bEnd: int):
        return aStart <= bStart < aEnd or bStart < aEnd <= bEnd


    from inference.segments import TypedSegment
    def matchStatistics(self, segmentedMessage: List[TypedSegment]):
        """
        Generate lists for statistics of true and false positives and false negatives.
        In the false negative list, we omit those segments that are just zeros.

        :param segmentedMessage: The segments according to dissection as ground truth.
        :return: dict, mapping fieldtype-names to (truePos, falsePos, falseNeg)
        """
        assert segmentedMessage[0].message == self.message

        from inference.segmentHandler import segments2types
        typedSegments = segments2types(segmentedMessage)
        nonConflicting = self.resolveConflicting()

        dictOfFieldTypesMappedToTheirTrueAndFalsePositivesAndFalseNegatives = dict()
        for ftype, segments in typedSegments.items():
            truePos = list()
            falsePos = list()
            nc4type = [nc for nc in nonConflicting if nc.template.fieldtype == ftype]
            for recognizedField in nc4type:
                for seg in segments:
                    if FieldTypeQuery.isOverlapping(
                            recognizedField.position, recognizedField.end, seg.offset, seg.nextOffset):
                        truePos.append((recognizedField, seg))
                if recognizedField not in (inftseg[0] for inftseg in truePos):
                    # for reference add the true segment which the closest offset to the false positive recognizedField
                    offsetDiffSorted = sorted(segmentedMessage, key=lambda s: abs(s.offset - recognizedField.position))
                    falsePos.append((recognizedField, offsetDiffSorted[0]))
            # omit those segments that are just zeros
            #                                                  recognizedSegments
            falseNeg = [seg for seg in segments
                        if set(seg.bytes) != {0} and seg not in (inftseg[1] for inftseg in truePos)]

            dictOfFieldTypesMappedToTheirTrueAndFalsePositivesAndFalseNegatives[ftype] = (truePos, falsePos, falseNeg)

        return dictOfFieldTypesMappedToTheirTrueAndFalsePositivesAndFalseNegatives


















