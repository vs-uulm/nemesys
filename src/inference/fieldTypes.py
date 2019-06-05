from typing import Type, Union, Any, Tuple, Iterable, List, Dict

import numpy
import scipy.spatial

from inference.analyzers import Value
from inference.segments import MessageAnalyzer
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
        self._stdev = stdev
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
        """

        There is some rounding error so the stdev is not entierely identical to the diagonal of the covariance matrix.
        >>> numpy.round(ftt.stdev, 8) == numpy.round(ftt.cov.diagonal(), 8)

        :return: The covariance matrix of the template.
        """
        return self._cov

    @property
    def picov(self):
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
    def upper(self):
        return self._mean + self.stdev

    @property
    def lower(self):
        return self._mean - self.stdev

    @property
    def analyzer(self):
        return self._analyzerClass

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
        """
        Compute the Mahalanobis distance between this fieldtype's mean and the given vector using the
        covariance matrix contained in this object.

        :param vector:
        :return:
        """
        return scipy.spatial.distance.mahalanobis(self.mean, vector, self.picov)

    def confidence(self, vector: Iterable[float]):
        conf = self.mahalanobis(vector)
        # TODO move to be a parameterizable property of the FieldTypeMemento class
        # make ids twice as unconfident
        if self.fieldtype == "id":
            conf *= 2
        return conf


class RecognizedField(object):
    def __init__(self, message: AbstractMessage, template: BaseTypeMemento,
                 position: int, confidence: float):
        self.message = message
        self.position = position
        self.confidence = confidence
        if isinstance(template, BaseTypeMemento):
            self.template = template
            self.end = position + len(template)
        else:
            raise TypeError("Template needs to be a BaseTypeMemento")


    def __repr__(self):
        return "RecognizedField of {} at ({}, {}) (c {:.2f})".format(
            self.template.fieldtype, self.position, self.end, self.confidence)


    def isOverlapping(self, otherField: 'RecognizedField'):
        if self.message == otherField.message \
                and (self.position <= otherField.position < self.end
                or otherField.position < self.end <= otherField.end):
            return True
        else:
            return False


class FieldTypeRecognizer(object):
    # from commit f442b9d
    fieldtypeTemplates = [
        # macaddr
        FieldTypeMemento(numpy.array([0.0, 12.0, 41.0, 137.28571428571428, 123.17857142857143, 124.82142857142857]),
                         numpy.array([0.0, 0.0, 0.0, 69.38446570834607, 77.84143845546744, 67.62293227439753]),
                         numpy.array(
                             [[0.021970046310679538, 0.0020136654623123394, -0.0026257581320540666, 4.213851338930285,
                               0.17211570097048762, -3.9634886782857977],
                              [0.0020136654623123394, 0.018063678574276265, -0.0006785532618061956, 2.689394908620589,
                               1.9445501397935412, -0.8648276565544288],
                              [-0.0026257581320540666, -0.0006785532618061956, 0.021345716783985814,
                               -0.5080691154469154,
                               0.2594949158918283, 3.1756906394784004],
                              [4.213851338930285, 2.689394908620589, -0.5080691154469154, 4992.5079365079355,
                               746.2804232804233,
                               -19.428571428571455],
                              [0.17211570097048762, 1.9445501397935412, 0.2594949158918283, 746.2804232804233,
                               6283.70767195767,
                               452.44047619047615],
                              [-3.9634886782857977, -0.8648276565544288, 3.1756906394784004, -19.428571428571455,
                               452.44047619047615,
                               4742.22619047619]]), 'macaddr', Value, (), MessageAnalyzer.U_BYTE),
        # # float
        # FieldTypeMemento(numpy.array([1.5325427466287367, 4.9510477851422445, 22.105263157894736, 116.69473684210526]),
        #                  numpy.array([0.25786246143446523, 0.5803090967897987, 8.562431632256324, 72.70952644225558]),
        #                  numpy.array([[0.06720042187892449, 0.01108737945671801, 0.1334931031456571, 4.069510452989349],
        #                               [0.01108737945671801, 0.34034118662355567, 0.7605770651397359, -0.08312808586398154],
        #                               [0.1334931031456571, 0.7605770651397359, 74.09518477043672, -41.43561030235157],
        #                               [4.069510452989349, -0.08312808586398154, -41.43561030235157, 5342.916461366181]]),
        #                  'float', Value, (), MessageAnalyzer.U_BYTE),
        # float
        FieldTypeMemento(numpy.array([1.0124348286955043, 2.9684767616240912, 22.105263157894736, 116.69473684210526]),
                         numpy.array([0.5859334443956004, 1.6856370764444046, 8.562431632256324, 72.70952644225558]),
                         numpy.array(
                             [[0.34697032042364634, -0.029258437144802, -0.19978884004957634, 0.22664826261354495],
                              [-0.029258437144802, 2.871599718946636, -1.4304488415634842, 19.928809693594715],
                              [-0.19978884004957634, -1.4304488415634842, 74.09518477043672, -41.43561030235157],
                              [0.22664826261354495, 19.928809693594715, -41.43561030235157, 5342.916461366181]]),
                         'float', Value, (), MessageAnalyzer.U_BYTE),
        # timestamp
        FieldTypeMemento(numpy.array(
            [205.75982532751092, 63.838427947598255, 24.375545851528383, 122.61135371179039, 133.65652173913043,
             134.2608695652174, 147.52173913043478, 120.2695652173913]), numpy.array(
            [25.70183334947299, 19.57641190835309, 53.50246711271789, 66.9864879316446, 71.84675313281585,
             72.93261773499316, 72.72710729917598, 76.51925835899723]),
            numpy.array([[663.4815368114683,
                          -472.49073010036244,
                          -423.75589902704553,
                          139.7702826936337, 9.80247835746574,
                          3.6199532674480817,
                          -242.52683291197417,
                          505.4362598636329],
                         [-472.49073010036244,
                          384.91676243009425,
                          296.96445261625695,
                          -111.35254347659536,
                          75.25739293648974, -4.666034628054849,
                          166.73218800275805,
                          -345.0736612273047],
                         [-423.75589902704553,
                          296.96445261625695, 2875.068873056001,
                          318.1860683367806, 261.37767179958604,
                          84.96889603922475, 157.02409407798976,
                          -593.1297594422736],
                         [139.7702826936337,
                          -111.35254347659536,
                          318.1860683367806, 4506.870221405036,
                          442.0494330805179, -212.5338044893896,
                          -498.76463265149766,
                          382.4062667585993],
                         [9.80247835746574, 75.25739293648974,
                          261.37767179958604, 442.0494330805179,
                          5184.497228023539,
                          -456.02790962597265,
                          -181.6759065881906,
                          370.75238276058485],
                         [3.6199532674480817,
                          -4.666034628054849, 84.96889603922475,
                          -212.5338044893896,
                          -456.02790962597265,
                          5342.394531991645, -240.0144294664894,
                          118.43155496487557],
                         [-242.52683291197417,
                          166.73218800275805,
                          157.02409407798976,
                          -498.76463265149766,
                          -181.6759065881906,
                          -240.0144294664894, 5312.329219669637,
                          -311.37269793051064],
                         [505.4362598636329, -345.0736612273047,
                          -593.1297594422736, 382.4062667585993,
                          370.75238276058485,
                          118.43155496487557,
                          -311.37269793051064,
                          5880.76544522499]]),
            'timestamp', Value, (), MessageAnalyzer.U_BYTE),
        # id
        FieldTypeMemento(numpy.array([122.0, 127.64444444444445, 140.22222222222223, 118.17777777777778]),
                         numpy.array([69.96538826845425, 66.7549933398429, 72.37952101063115, 78.82252607779043]),
                         numpy.array([[5006.409090909091, 1163.6590909090912, 0.36363636363636365, 830.5454545454546],
                                      [1163.6590909090912, 4557.507070707072, -678.6691919191921, 770.9055555555553],
                                      [0.36363636363636365, -678.6691919191921, 5357.858585858588, -94.7222222222223],
                                      [830.5454545454546, 770.9055555555553, -94.7222222222223, 6354.19494949495]]),
                         'id', Value, (), MessageAnalyzer.U_BYTE),
        # ipv4
        FieldTypeMemento(numpy.array([172.0, 18.979591836734695, 2.163265306122449, 22.755102040816325]),
                         numpy.array([0.0, 1.4355514608219568, 0.7380874008173525, 24.153024581160828]), numpy.array(
                [[0.019605839422040614, -0.027004344798568215, 0.02184009734406893, -0.27738019525778845],
                 [-0.027004344798568215, 2.103741496598639, 0.0034013605442177, 0.9740646258503398],
                 [0.02184009734406893, 0.0034013605442177, 0.5561224489795925, 11.769982993197274],
                 [-0.27738019525778845, 0.9740646258503398, 11.769982993197274, 595.5221088435375]]), 'ipv4', Value, (),
                         MessageAnalyzer.U_BYTE),
    ]

    # # from commit 4b69075
    # fieldtypeTemplates = [
    #     # macaddr
    #     FieldTypeMemento(numpy.array([0.0, 12.0, 41.0, 137.28571428571428, 123.17857142857143, 124.82142857142857]), numpy.array([0.0, 0.0, 0.0, 69.38446570834607, 77.84143845546744, 67.62293227439753]), numpy.array([[0.022103573766497436, 0.0046327614942338, -0.006504195407397688, 0.6591927243117947, -0.644376065770878, 0.9913785886639866], [0.0046327614942338, 0.021488881156709244, -0.001382945195792973, -1.1536744995748671, 5.150600064885553, 0.3555595274599619], [-0.006504195407397688, -0.001382945195792973, 0.026359085778108426, 1.0184065358245054, 0.6723490088910922, -0.3966508786750666], [0.6591927243117947, -1.1536744995748671, 1.0184065358245054, 4992.5079365079355, 746.2804232804233, -19.428571428571455], [-0.644376065770878, 5.150600064885553, 0.6723490088910922, 746.2804232804233, 6283.70767195767, 452.44047619047615], [0.9913785886639866, 0.3555595274599619, -0.3966508786750666, -19.428571428571455, 452.44047619047615, 4742.22619047619]]), 'macaddr', Value, (), MessageAnalyzer.U_BYTE) ,
    #     # timestamp
    #     FieldTypeMemento(numpy.array([205.75982532751092, 63.838427947598255, 24.375545851528383, 122.61135371179039, 133.65652173913043, 134.2608695652174, 147.52173913043478, 120.2695652173913]), numpy.array([25.70183334947299, 19.57641190835309, 53.50246711271789, 66.9864879316446, 71.84675313281585, 72.93261773499316, 72.72710729917598, 76.51925835899723]), numpy.array([[663.4815368114683, -472.49073010036244, -423.75589902704553, 139.7702826936337, 9.80247835746574, 3.6199532674480817, -242.52683291197417, 505.4362598636329], [-472.49073010036244, 384.91676243009425, 296.96445261625695, -111.35254347659536, 75.25739293648974, -4.666034628054849, 166.73218800275805, -345.0736612273047], [-423.75589902704553, 296.96445261625695, 2875.068873056001, 318.1860683367806, 261.37767179958604, 84.96889603922475, 157.02409407798976, -593.1297594422736], [139.7702826936337, -111.35254347659536, 318.1860683367806, 4506.870221405036, 442.0494330805179, -212.5338044893896, -498.76463265149766, 382.4062667585993], [9.80247835746574, 75.25739293648974, 261.37767179958604, 442.0494330805179, 5184.497228023539, -456.02790962597265, -181.6759065881906, 370.75238276058485], [3.6199532674480817, -4.666034628054849, 84.96889603922475, -212.5338044893896, -456.02790962597265, 5342.394531991645, -240.0144294664894, 118.43155496487557], [-242.52683291197417, 166.73218800275805, 157.02409407798976, -498.76463265149766, -181.6759065881906, -240.0144294664894, 5312.329219669637, -311.37269793051064], [505.4362598636329, -345.0736612273047, -593.1297594422736, 382.4062667585993, 370.75238276058485, 118.43155496487557, -311.37269793051064, 5880.76544522499]]), 'timestamp', Value, (), MessageAnalyzer.U_BYTE) ,
    #     # id
    #     FieldTypeMemento(numpy.array([122.91489361702128, 122.76595744680851, 136.2340425531915, 116.25531914893617]), numpy.array([69.99721264579163, 69.05662578282282, 74.7862806755008, 81.35382873860368]), numpy.array([[5006.123034227568, 1258.0448658649402, -284.06660499537475, 412.17437557816834], [1258.0448658649402, 4872.487511563368, -406.5527289546716, 674.6262719703982], [-284.06660499537475, -406.5527289546716, 5714.574468085107, 456.6345975948195], [412.17437557816834, 674.6262719703982, 456.6345975948195, 6762.324699352451]]), 'id', Value, (), MessageAnalyzer.U_BYTE) ,
    #     # # float
    #     # FieldTypeMemento(numpy.array([0.0, 2., 24., 113.]),
    #     #                  numpy.array([0.0, 0.7060598835273263, 16.235663701838636, 76.9843433664198]), numpy.array(
    #     #         [[0.021138394040146703, 0.018424928271229466, 0.2192364920149479, 0.32239294646470784],
    #     #          [0.018424928271229466, 0.5036075036075054, 0.03442589156874873, 7.556380127808698],
    #     #          [0.2192364920149479, 0.03442589156874873, 266.28653885796797, -84.0164914450628],
    #     #          [0.32239294646470784, 7.556380127808698, -84.0164914450628, 5987.06452277881]]), 'float', Value, (),
    #     #                  MessageAnalyzer.U_BYTE),
    #     # FieldTypeMemento(numpy.array([0.0, 0.08080808080808081, 23.828282828282827, 113.43434343434343]), numpy.array([0.0, 0.7060598835273263, 16.235663701838636, 76.9843433664198]), numpy.array([[0.021138394040146703, 0.018424928271229466, 0.2192364920149479, 0.32239294646470784], [0.018424928271229466, 0.5036075036075054, 0.03442589156874873, 7.556380127808698], [0.2192364920149479, 0.03442589156874873, 266.28653885796797, -84.0164914450628], [0.32239294646470784, 7.556380127808698, -84.0164914450628, 5987.06452277881]]), 'float', Value, (), MessageAnalyzer.U_BYTE) ,
    #     # # int
    #     # # FieldTypeMemento(numpy.array([0.0, 0.0, 107.29166666666667]), numpy.array([0.0, 0.0, 47.78029507257382]), numpy.array([[0.027493310152260777, -0.005639908063087656, -0.07922934857447499], [-0.005639908063087656, 0.018914313768380903, -2.358032476778992], [-0.07922934857447499, -2.358032476778992, 2382.215579710145]]), 'int', Value, (), MessageAnalyzer.U_BYTE) ,
    #     # FieldTypeMemento(numpy.array([0.0, 0.0, 4.75, 127.84375]), numpy.array([0.0, 0.0, 4.743416490252569, 39.259640038307786]), numpy.array([[0.021960602414756426, 0.0007935582831639521, -0.22956507460690648, 0.6994209166124543], [0.0007935582831639521, 0.017292996674435825, -0.06886282798507015, 1.917984540011557], [-0.22956507460690648, -0.06886282798507015, 23.225806451612904, -30.427419354838708], [0.6994209166124543, 1.917984540011557, -30.427419354838708, 1591.039314516129]]), 'int', Value, (), MessageAnalyzer.U_BYTE) ,
    #     # ipv4
    #     FieldTypeMemento(numpy.array([172.0, 18.928571428571427, 2.3095238095238093, 26.071428571428573]), numpy.array([0.0, 1.4374722712498649, 0.6721711530234811, 24.562256039881753]), numpy.array([[0.022912984623871414, 0.03544250639873077, -0.0004582497404991295, -0.4596803115368301], [0.03544250639873077, 2.1167247386759582, 0.022648083623693426, 2.2491289198606266], [-0.0004582497404991295, 0.022648083623693426, 0.4628339140534262, 10.19686411149826], [-0.4596803115368301, 2.2491289198606266, 10.19686411149826, 618.0191637630664]]), 'ipv4', Value, (), MessageAnalyzer.U_BYTE) ,
    # ]

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
        for offset in range(len(self._analyzer.values) - ftlen + 1):
            ngram = self._analyzer.values[offset:offset+ftlen]
            if set(ngram) == {0}:  # zero values do not give any information
                posCon.append(99)
            else:
                posCon.append(fieldtypeTemplate.confidence(ngram))  # / (ftlen**2 / 4)

        return posCon


    def charsInMessage(self) -> List[RecognizedField]:
        """
        # mark recognized char sequences  (TODO confidence?)
        :return:
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
        # confidence = 0.2
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
                confidence = 4.0  # 4 * bitSum / bitset  # * minLen
                recognizedFlags.append(
                    RecognizedField(self.message, BaseTypeMemento("flags", minLen), offset, confidence))
            offset += 1
        return recognizedFlags

    # = 2
    def recognizedFields(self, confidenceThreshold = None) -> Dict[FieldTypeMemento, List[RecognizedField]]:
        """
        Most probable inferred field structure: The field template positions with the highest confidence for a match.
        TODO How to decide which of any overlapping fields should be the recognized one?
        TODO How to decide which confidence value is high enough to assume a match if no concurring comparison values
            ("no relevant matches") are in this message?
        :return:
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


    # def mostConfident4Type(self, ftMemento: FieldTypeMemento):
    #     confidences = self.findInMessage(ftMemento)
    #     pass


    @staticmethod
    def isOverlapping(aStart: int, aEnd: int, bStart: int, bEnd: int):
        return aStart <= bStart < aEnd or bStart < aEnd <= bEnd


    from inference.segments import TypedSegment
    def matchStatistics(self, segmentedMessage: List[TypedSegment]):
        """
        in the falseNeg, we omit those segments that are just zeros.

        :param segmentedMessage:
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


















