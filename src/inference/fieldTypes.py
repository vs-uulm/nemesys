from typing import Type, Union, Any, Tuple, Iterable, List, Dict

import numpy
import scipy.spatial

from inference.analyzers import Value
from inference.segments import MessageAnalyzer
from netzob.Model.Vocabulary.Messages.AbstractMessage import AbstractMessage


class FieldTypeMemento(object):
    def __init__(self, mean: numpy.ndarray, stdev: numpy.ndarray, cov: numpy.ndarray, fieldtype: str,
                 analyzerClass: Type[MessageAnalyzer]=Value, analysisParams: Union[Any, Tuple]=None, unit=MessageAnalyzer.U_BYTE):
        self._mean = mean
        self._cov = cov
        self._stdev = stdev
        # data type this field represents
        self._fieldtype = fieldtype
        # for reference:
        self._analyzerClass = analyzerClass
        self._analysisParams = analysisParams
        self._unit = unit
        self._picov = None

    @staticmethod
    def fromTemplate(ftt: "FieldTypeTemplate"):
        ftm = FieldTypeMemento(ftt.mean, ftt.stdev, ftt.cov, ftt.fieldtype,
                               type(ftt.baseSegments[0].analyzer), ftt.baseSegments[0].analyzer.analysisParams,
                               ftt.baseSegments[0].analyzer.unit)
        return ftm

    @property
    def mean(self):
        return self._mean

    @property
    def stdev(self):
        return self._stdev

    @property
    def cov(self):
        return self._cov

    @property
    def picov(self):
        """
        Often cov is a singluar matrix in our use case, so we use the approximate pinv.
        :return:
        """
        if self._picov is None:
            self._picov = numpy.linalg.pinv(self.cov)
        return self._picov

    @property
    def upper(self):
        return self._mean + self.stdev

    @property
    def lower(self):
        return self._mean - self.stdev

    @property
    def analyzer(self):
        return self._analyzerClass

    @property
    def fieldtype(self):
        return self._fieldtype

    def __len__(self):
        return len(self._mean)

    @property
    def typeID(self, short=True):
        """
        :param short: Use only the last half (4 bytes) of the hash
        :return: As an identifier use the hash of the mean values
        """
        tid = "{:02x}".format(hash(tuple(self.mean)))
        return tid[-8:] if short else tid

    @property
    def codePersist(self):
        """:return: Python code to persist this Memento"""
        return "{}(numpy.array({}), numpy.array({}), numpy.array({}), '{}', {}, {}, {})".format(
            type(self).__name__, self.mean.tolist(), self.stdev.tolist(), self.cov.tolist(), self._fieldtype,
            self._analyzerClass.__name__, self._analysisParams,
            "MessageAnalyzer.U_BYTE" if self._unit == MessageAnalyzer.U_BYTE else "MessageAnalyzer.U_NIBBLE")

    def mahalanobis(self, vector: Iterable[float]):
        return scipy.spatial.distance.mahalanobis(self.mean, vector, self.picov)

    def confidence(self, vector: Iterable[float]):
        return self.mahalanobis(vector)

    def __repr__(self):
        return "FieldTypeMemento " + self.typeID + " for " + self.fieldtype


class RecognizedField(object):
    HEURISTIC_FIELDTYPES = ["chars"]

    def __init__(self, message: AbstractMessage, template: Union[FieldTypeMemento, Tuple[str, int]],
                 position: int, confidence: float):
        self.message = message
        self.position = position
        self.confidence = confidence
        if isinstance(template, FieldTypeMemento):
            self.template = template
            self.end = position + len(template)
        elif isinstance(template, tuple):
            assert template[0] in RecognizedField.HEURISTIC_FIELDTYPES
            assert isinstance(template[1], int), "length {} is no int".format(template[1])
            self.template = template[0]
            self.end = position + template[1]
        else:
            raise TypeError("Template needs to be a FieldTypeMemento or a tuple of a type-name and length")


class FieldTypeRecognizer(object):
    fieldtypeTemplates = [
        # macaddr
        FieldTypeMemento(numpy.array([0.0, 12.0, 41.0, 137.28571428571428, 123.17857142857143, 124.82142857142857]), numpy.array([0.0, 0.0, 0.0, 69.38446570834607, 77.84143845546744, 67.62293227439753]), numpy.array([[0.022103573766497436, 0.0046327614942338, -0.006504195407397688, 0.6591927243117947, -0.644376065770878, 0.9913785886639866], [0.0046327614942338, 0.021488881156709244, -0.001382945195792973, -1.1536744995748671, 5.150600064885553, 0.3555595274599619], [-0.006504195407397688, -0.001382945195792973, 0.026359085778108426, 1.0184065358245054, 0.6723490088910922, -0.3966508786750666], [0.6591927243117947, -1.1536744995748671, 1.0184065358245054, 4992.5079365079355, 746.2804232804233, -19.428571428571455], [-0.644376065770878, 5.150600064885553, 0.6723490088910922, 746.2804232804233, 6283.70767195767, 452.44047619047615], [0.9913785886639866, 0.3555595274599619, -0.3966508786750666, -19.428571428571455, 452.44047619047615, 4742.22619047619]]), 'macaddr', Value, (), MessageAnalyzer.U_BYTE) ,
        # timestamp
        FieldTypeMemento(numpy.array([205.75982532751092, 63.838427947598255, 24.375545851528383, 122.61135371179039, 133.65652173913043, 134.2608695652174, 147.52173913043478, 120.2695652173913]), numpy.array([25.70183334947299, 19.57641190835309, 53.50246711271789, 66.9864879316446, 71.84675313281585, 72.93261773499316, 72.72710729917598, 76.51925835899723]), numpy.array([[663.4815368114683, -472.49073010036244, -423.75589902704553, 139.7702826936337, 9.80247835746574, 3.6199532674480817, -242.52683291197417, 505.4362598636329], [-472.49073010036244, 384.91676243009425, 296.96445261625695, -111.35254347659536, 75.25739293648974, -4.666034628054849, 166.73218800275805, -345.0736612273047], [-423.75589902704553, 296.96445261625695, 2875.068873056001, 318.1860683367806, 261.37767179958604, 84.96889603922475, 157.02409407798976, -593.1297594422736], [139.7702826936337, -111.35254347659536, 318.1860683367806, 4506.870221405036, 442.0494330805179, -212.5338044893896, -498.76463265149766, 382.4062667585993], [9.80247835746574, 75.25739293648974, 261.37767179958604, 442.0494330805179, 5184.497228023539, -456.02790962597265, -181.6759065881906, 370.75238276058485], [3.6199532674480817, -4.666034628054849, 84.96889603922475, -212.5338044893896, -456.02790962597265, 5342.394531991645, -240.0144294664894, 118.43155496487557], [-242.52683291197417, 166.73218800275805, 157.02409407798976, -498.76463265149766, -181.6759065881906, -240.0144294664894, 5312.329219669637, -311.37269793051064], [505.4362598636329, -345.0736612273047, -593.1297594422736, 382.4062667585993, 370.75238276058485, 118.43155496487557, -311.37269793051064, 5880.76544522499]]), 'timestamp', Value, (), MessageAnalyzer.U_BYTE) ,
        # id
        FieldTypeMemento(numpy.array([122.91489361702128, 122.76595744680851, 136.2340425531915, 116.25531914893617]), numpy.array([69.99721264579163, 69.05662578282282, 74.7862806755008, 81.35382873860368]), numpy.array([[5006.123034227568, 1258.0448658649402, -284.06660499537475, 412.17437557816834], [1258.0448658649402, 4872.487511563368, -406.5527289546716, 674.6262719703982], [-284.06660499537475, -406.5527289546716, 5714.574468085107, 456.6345975948195], [412.17437557816834, 674.6262719703982, 456.6345975948195, 6762.324699352451]]), 'id', Value, (), MessageAnalyzer.U_BYTE) ,
        # float
        FieldTypeMemento(numpy.array([0.0, 2., 24., 113.]),
                         numpy.array([0.0, 0.7060598835273263, 16.235663701838636, 76.9843433664198]), numpy.array(
                [[0.021138394040146703, 0.018424928271229466, 0.2192364920149479, 0.32239294646470784],
                 [0.018424928271229466, 0.5036075036075054, 0.03442589156874873, 7.556380127808698],
                 [0.2192364920149479, 0.03442589156874873, 266.28653885796797, -84.0164914450628],
                 [0.32239294646470784, 7.556380127808698, -84.0164914450628, 5987.06452277881]]), 'float', Value, (),
                         MessageAnalyzer.U_BYTE),
        FieldTypeMemento(numpy.array([0.0, 0.08080808080808081, 23.828282828282827, 113.43434343434343]), numpy.array([0.0, 0.7060598835273263, 16.235663701838636, 76.9843433664198]), numpy.array([[0.021138394040146703, 0.018424928271229466, 0.2192364920149479, 0.32239294646470784], [0.018424928271229466, 0.5036075036075054, 0.03442589156874873, 7.556380127808698], [0.2192364920149479, 0.03442589156874873, 266.28653885796797, -84.0164914450628], [0.32239294646470784, 7.556380127808698, -84.0164914450628, 5987.06452277881]]), 'float', Value, (), MessageAnalyzer.U_BYTE) ,
        # int
        # FieldTypeMemento(numpy.array([0.0, 0.0, 107.29166666666667]), numpy.array([0.0, 0.0, 47.78029507257382]), numpy.array([[0.027493310152260777, -0.005639908063087656, -0.07922934857447499], [-0.005639908063087656, 0.018914313768380903, -2.358032476778992], [-0.07922934857447499, -2.358032476778992, 2382.215579710145]]), 'int', Value, (), MessageAnalyzer.U_BYTE) ,
        FieldTypeMemento(numpy.array([0.0, 0.0, 4.75, 127.84375]), numpy.array([0.0, 0.0, 4.743416490252569, 39.259640038307786]), numpy.array([[0.021960602414756426, 0.0007935582831639521, -0.22956507460690648, 0.6994209166124543], [0.0007935582831639521, 0.017292996674435825, -0.06886282798507015, 1.917984540011557], [-0.22956507460690648, -0.06886282798507015, 23.225806451612904, -30.427419354838708], [0.6994209166124543, 1.917984540011557, -30.427419354838708, 1591.039314516129]]), 'int', Value, (), MessageAnalyzer.U_BYTE) ,
        # ipv4
        FieldTypeMemento(numpy.array([172.0, 18.928571428571427, 2.3095238095238093, 26.071428571428573]), numpy.array([0.0, 1.4374722712498649, 0.6721711530234811, 24.562256039881753]), numpy.array([[0.022912984623871414, 0.03544250639873077, -0.0004582497404991295, -0.4596803115368301], [0.03544250639873077, 2.1167247386759582, 0.022648083623693426, 2.2491289198606266], [-0.0004582497404991295, 0.022648083623693426, 0.4628339140534262, 10.19686411149826], [-0.4596803115368301, 2.2491289198606266, 10.19686411149826, 618.0191637630664]]), 'ipv4', Value, (), MessageAnalyzer.U_BYTE) ,
    ]

    def __init__(self, analyzer: MessageAnalyzer):
        self._analyzer = analyzer


    @property
    def message(self):
        return self._analyzer.message


    def findInMessage(self, fieldtypeTemplate: FieldTypeMemento):
        """

        :param fieldtypeTemplate:
        :return: list of (position, confidence) for all offsets.
        """
        assert fieldtypeTemplate.analyzer == type(self._analyzer)

        # position, confidence
        posCon = list()
        ftlen = len(fieldtypeTemplate)
        for offset in range(len(self._analyzer.values) - ftlen):
            ngram = self._analyzer.values[offset:offset+ftlen]
            if set(ngram) == {0}:  # zero values do not give any information
                posCon.append(99)
            else:
                posCon.append(fieldtypeTemplate.confidence(ngram))

        return posCon


    def charsInMessage(self) -> List[RecognizedField]:
        from inference.segmentHandler import isExtendedCharSeq

        confidence = 0.0
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
                        RecognizedField(self.message, ("chars", chunkLen), offset, confidence))
                    offset += minLen + add2len
                else:
                    offset += 1
            else:
                offset += 1
        return recognizedChars


    def recognizedFields(self, confidenceThreshold = 2) -> Dict[FieldTypeMemento, List[RecognizedField]]:
        """
        Most probable inferred field structure: The field template positions with the highest confidence for a match.
        TODO How to decide which of any overlapping fields should be the recognized one?
        TODO How to decide which confidence value is high enough to assume a match if no concurring comparison values
            ("no relevant matches") are in this message?
        :return:
        """
        mostConfident = dict()
        for ftMemento in FieldTypeRecognizer.fieldtypeTemplates:
            confidences = self.findInMessage(ftMemento)
            mostConfident[ftMemento] = [RecognizedField(self.message, ftMemento, pos, con)
                                        for pos, con in enumerate(confidences) if con < confidenceThreshold]
        mostConfident["chars"] = self.charsInMessage()
        return mostConfident


class FieldTypeQuery(object):

    RECOGNIZABLE = FieldTypeRecognizer.fieldtypeTemplates + RecognizedField.HEURISTIC_FIELDTYPES  # type: List[Union[FieldTypeMemento, str]]

    def __init__(self, ftRecognizer: FieldTypeRecognizer):
        self._recognizedTemplates = ftRecognizer.recognizedFields()
        # self._recognizedTemplates["chars"] = ftRecognizer.charsInMessage()

        # mapping from byte offsets to list of recognitions that are (ftm, idx), to retrieve by self._recognizedTemplates[ftm][idx]
        self._posTemplMap = dict()  # type: Dict[int, List[RecognizedField]]
        for ftm, poscon in self._recognizedTemplates.items():
            for recognized in poscon:
                for o in range(recognized.position, recognized.end):
                    if o not in self._posTemplMap:
                        self._posTemplMap[o] = list()
                    self._posTemplMap[o].append(recognized)
        # self._charMap = {o: recogChars
        #                  for recogChars in self._recognizedChars
        #                  for o in range(recogChars.position, recogChars.end)}


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

