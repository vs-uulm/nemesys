from typing import Type, Union, Any, Tuple, Iterable, List, Dict

import numpy
import scipy.spatial

from inference.analyzers import Value
from inference.segments import MessageAnalyzer, MessageSegment, TypedSegment
from netzob.Model.Vocabulary.Messages.AbstractMessage import AbstractMessage




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



class FieldTypeRecognizer(object):
    """
    Class containing persisted FieldTypeMementos to search for.
    Provides the method #recognizedFields to find the most confident matches with the known field types to be recognized.
    """

    # TODO make extensible by moving to a ValueFieldTypeRecognizer subclass and making this class abstract.

    fieldtypeTemplates = [
        # ipv4
        FieldTypeMemento(numpy.array([172.0, 19.133333333333333, 2.2, 43.51111111111111]),
                         numpy.array([0.0, 1.3432961119739928, 0.7774602526460401, 70.0297150333651]), numpy.array(
                [[0.02347338518143448, 0.006234149492898358, 0.010234076284326404, 0.9784377514408501],
                 [0.006234149492898358, 1.8454545454545468, 0.31363636363636377, 15.43030303030303],
                 [0.010234076284326404, 0.31363636363636377, 0.6181818181818184, 28.327272727272724],
                 [0.9784377514408501, 15.43030303030303, 28.327272727272724, 5015.619191919192]]), 'ipv4', Value, (),
                         MessageAnalyzer.U_BYTE),
        FieldTypeMemento(numpy.array([10.0, 110.0, 48.0, 174.66666666666666]),
                         numpy.array([0.0, 0.0, 0.0, 40.36362498862339]), numpy.array(
                [[0.011104288378642276, 0.003538891411820982, -0.004262199819749322, -0.4343913464330041],
                 [0.003538891411820982, 0.010787999355940582, 0.0026758247924099465, -3.6322251755584882],
                 [-0.004262199819749322, 0.0026758247924099465, 0.018235137273659847, 0.6231713459814735],
                 [-0.4343913464330041, -3.6322251755584882, 0.6231713459814735, 1955.0666666666673]]), 'ipv4', Value,
                         (), MessageAnalyzer.U_BYTE),
        FieldTypeMemento(numpy.array([192.0, 168.0, 1.0909090909090908, 133.27272727272728]),
                         numpy.array([0.0, 0.0, 1.443137078762504, 69.48000485305225]), numpy.array(
                [[0.028575235054388845, 0.0006935874260185519, 0.11175698520236568, -4.375053014787284],
                 [0.0006935874260185519, 0.017161633352611793, 0.07628635183326875, 2.6779650034086036],
                 [0.11175698520236568, 0.07628635183326875, 2.290909090909091, -60.92727272727274],
                 [-4.375053014787284, 2.6779650034086036, -60.92727272727274, 5310.218181818182]]), 'ipv4', Value, (),
                         MessageAnalyzer.U_BYTE),
        # macaddr
        FieldTypeMemento(numpy.array([0.0, 19.958333333333332, 55.125, 141.875, 115.33333333333333, 106.83333333333333]), numpy.array([0.0, 14.11405672449357, 32.16663968047227, 79.42885312235934, 72.9781169636183, 63.90400787292]), numpy.array([[0.028705911279892384, -0.23394767895272492, 0.14099914430429114, -3.675999598837579, 4.398666004054766, -1.0886476504107736], [-0.23394767895272492, 207.86775362318838, 304.39673913043475, -605.4836956521739, 279.18840579710144, -446.52898550724643], [0.14099914430429114, 304.39673913043475, 1079.679347826087, -293.76630434782606, 88.30434782608681, -782.7173913043476], [-3.675999598837579, -605.4836956521739, -293.76630434782606, 6583.244565217391, -1866.782608695652, 2614.282608695652], [4.398666004054766, 279.18840579710144, 88.30434782608681, -1866.782608695652, 5557.362318840577, 302.7101449275363], [-1.0886476504107736, -446.52898550724643, -782.7173913043476, 2614.282608695652, 302.7101449275363, 4261.27536231884]]), 'macaddr', Value, (), MessageAnalyzer.U_BYTE) ,
        # id
        FieldTypeMemento(numpy.array([114.01408450704226, 101.32394366197182, 110.94366197183099, 97.87323943661971]), numpy.array([68.31709269091691, 80.28554228465102, 78.61922628415738, 77.41691222179269]), numpy.array([[4733.899798792751, 646.181086519115, -305.8706237424549, 1283.3732394366195], [646.181086519115, 6537.8507042253505, 1910.575653923541, 2877.284507042252], [-305.8706237424549, 1910.575653923541, 6269.282494969821, 2121.2498993963786], [1283.3732394366195, 2877.284507042252, 2121.2498993963786, 6078.997987927564]]), 'id', Value, (), MessageAnalyzer.U_BYTE) ,
        # # int - unmodified
        # FieldTypeMemento(numpy.array([0.0, 0.0, 22.68208092485549, 139.83236994219652]), numpy.array([0.0, 0.0, 27.24492999988537, 56.82770526015298]), numpy.array([[0.019983059129803384, 0.0006681303969208007, 0.1602835987215403, 0.6138917528186394], [0.0006681303969208007, 0.018373348352696096, 0.5862990610502694, -0.3527773481697946], [0.1602835987215403, 0.5862990610502694, 746.6018282027158, -82.25127705336736], [0.6138917528186394, -0.3527773481697946, -82.25127705336736, 3248.1635972576955]]), 'int', Value, (), MessageAnalyzer.U_BYTE) ,
        # int - "scintillated" first two components
        FieldTypeMemento(numpy.array([0.976238924634935, 2.868796973312969, 22.68208092485549, 139.83236994219652]),
                         numpy.array([0.559790187655955, 1.6780123746408333, 27.24492999988537, 56.82770526015298]),
                         numpy.array([[0.3151869440458654, 0.07331153126264293, 0.19281378603891056, 2.40780772380801],
                                      [0.07331153126264293, 2.8320960267119997, 2.926437577484135, 0.07329099552698337],
                                      [0.19281378603891056, 2.926437577484135, 746.6018282027158, -82.25127705336736],
                                      [2.40780772380801, 0.07329099552698337, -82.25127705336736, 3248.1635972576955]]),
                         'int', Value, (), MessageAnalyzer.U_BYTE),

        # timestamp
        FieldTypeMemento(numpy.array([200.49175824175825, 68.46978021978022, 49.08516483516483, 145.29945054945054, 108.03021978021978, 122.93956043956044, 130.50824175824175, 112.33791208791209]), numpy.array([35.91767813212723, 31.652242754349523, 69.18243837794303, 73.74511270619739, 72.5432926365016, 70.38556823093352, 75.25489607077313, 82.59751873473633]), numpy.array([[1293.6335407017216, -601.2922607695357, -624.8436487754672, 306.0892516574335, 834.7922607695347, -114.98948021675288, -700.7024112251385, 1061.544114975933], [-601.2922607695357, 1004.6244286017051, 710.2601565101568, -135.66447945993403, -668.8324175824176, 90.6097387461025, 560.4768337722884, -846.6302561075287], [-624.8436487754672, 710.2601565101568, 4799.394930826749, -353.43328641055894, -870.3965201465198, 346.6442799624617, 535.7279462961285, -1278.8911164592969], [306.0892516574335, -135.66447945993403, -353.43328641055894, 5453.323305482399, -254.67574092574094, 50.125586534677375, -220.1526125389761, -82.00780280325735], [834.7922607695347, -668.8324175824176, -870.3965201465198, -254.67574092574094, 5277.026632458456, -493.63453213453204, -908.9079632488721, 1122.940173462901], [-114.98948021675288, 90.6097387461025, 346.6442799624617, 50.125586534677375, -493.63453213453204, 4967.775951321401, 129.1602791148241, 172.69816547089266], [-700.7024112251385, 560.4768337722884, 535.7279462961285, -220.1526125389761, -908.9079632488721, 129.1602791148241, 5678.900758332568, -766.1419111191834], [1061.544114975933, -846.6302561075287, -1278.8911164592969, -82.00780280325735, 1122.940173462901, 172.69816547089266, -766.1419111191834, 6841.144454030807]]), 'timestamp', Value, (), MessageAnalyzer.U_BYTE) ,
        # checksum
        FieldTypeMemento(numpy.array([123.7, 163.6, 116.2, 142.1, 160.6, 112.4, 154.1, 111.7]), numpy.array([76.14729148170669, 62.72830302184174, 77.60515446798621, 64.75407323095591, 48.18132418271628, 80.04023987970051, 54.70731212552852, 57.33245154360661]), numpy.array([[6442.677777777776, -3501.466666666667, 1916.8444444444447, -483.1888888888885, -399.3555555555555, 1279.6888888888889, -2335.411111111111, -1453.1000000000001], [-3501.466666666667, 4372.044444444444, -1350.4666666666667, -872.2888888888888, -123.73333333333332, 16.511111111111152, 1799.2666666666667, 2020.0888888888887], [1916.8444444444447, -1350.4666666666667, 6691.733333333332, -206.46666666666647, -2321.1333333333328, 396.0222222222222, -449.24444444444435, -3348.488888888889], [-483.1888888888885, -872.2888888888888, -206.46666666666647, 4658.988888888889, 1967.3777777777775, 2087.0666666666666, 1587.1000000000001, -18.07777777777769], [-399.3555555555555, -123.73333333333332, -2321.1333333333328, 1967.3777777777775, 2579.3777777777786, 652.5111111111111, 304.15555555555545, 1130.311111111111], [1279.6888888888889, 16.511111111111152, 396.0222222222222, 2087.0666666666666, 652.5111111111111, 7118.2666666666655, 1863.1777777777775, 728.4666666666666], [-2335.411111111111, 1799.2666666666667, -449.24444444444435, 1587.1000000000001, 304.15555555555545, 1863.1777777777775, 3325.433333333333, -312.0777777777778], [-1453.1000000000001, 2020.0888888888887, -3348.488888888889, -18.07777777777769, 1130.311111111111, 728.4666666666666, -312.0777777777778, 3652.233333333333]]), 'checksum', Value, (), MessageAnalyzer.U_BYTE) ,
    ]

    # from commit 475d179
    # [
    #     # timestamp
    #     FieldTypeMemento(numpy.array([205.75982532751092, 63.838427947598255, 24.375545851528383, 122.61135371179039, 133.65652173913043, 134.2608695652174, 147.52173913043478, 120.2695652173913]), numpy.array([25.70183334947299, 19.57641190835309, 53.50246711271789, 66.9864879316446, 71.84675313281585, 72.93261773499316, 72.72710729917598, 76.51925835899723]), numpy.array([[663.4815368114683, -472.49073010036244, -423.75589902704553, 139.7702826936337, 9.80247835746574, 3.6199532674480817, -242.52683291197417, 505.4362598636329], [-472.49073010036244, 384.91676243009425, 296.96445261625695, -111.35254347659536, 75.25739293648974, -4.666034628054849, 166.73218800275805, -345.0736612273047], [-423.75589902704553, 296.96445261625695, 2875.068873056001, 318.1860683367806, 261.37767179958604, 84.96889603922475, 157.02409407798976, -593.1297594422736], [139.7702826936337, -111.35254347659536, 318.1860683367806, 4506.870221405036, 442.0494330805179, -212.5338044893896, -498.76463265149766, 382.4062667585993], [9.80247835746574, 75.25739293648974, 261.37767179958604, 442.0494330805179, 5184.497228023539, -456.02790962597265, -181.6759065881906, 370.75238276058485], [3.6199532674480817, -4.666034628054849, 84.96889603922475, -212.5338044893896, -456.02790962597265, 5342.394531991645, -240.0144294664894, 118.43155496487557], [-242.52683291197417, 166.73218800275805, 157.02409407798976, -498.76463265149766, -181.6759065881906, -240.0144294664894, 5312.329219669637, -311.37269793051064], [505.4362598636329, -345.0736612273047, -593.1297594422736, 382.4062667585993, 370.75238276058485, 118.43155496487557, -311.37269793051064, 5880.76544522499]]), 'timestamp', Value, (), MessageAnalyzer.U_BYTE) ,
    #     # ipv4
    #     FieldTypeMemento(numpy.array([172.65573770491804, 23.098360655737704, 2.4098360655737703, 40.73770491803279]), numpy.array([3.561567374164002, 22.79682528805688, 1.5404603227322615, 68.36543600851412]), numpy.array([[12.896174863387957, 80.93442622950825, 3.726775956284154, 2.841530054644803], [80.93442622950825, 528.3568306010928, 19.475683060109265, 50.24289617486337], [3.726775956284154, 19.475683060109265, 2.4125683060109235, 19.059289617486336], [2.841530054644803, 50.24289617486337, 19.059289617486336, 4751.73005464481]]), 'ipv4', Value, (), MessageAnalyzer.U_BYTE) ,
    #     # macaddr
    #     FieldTypeMemento(numpy.array([0.0, 12.0, 41.0, 137.28571428571428, 123.17857142857143, 124.82142857142857]), numpy.array([0.0, 0.0, 0.0, 69.38446570834608, 77.84143845546744, 67.62293227439753]), numpy.array([[0.01691577109311472, 0.0017520133005179852, -0.0009323360045556071, 0.42935337597073336, 1.9917595051987003, -1.850657000447773], [0.0017520133005179852, 0.020892243107491996, -0.0016527739029342462, 0.6300638864312258, 1.8509234806353214, -3.728455646581064], [-0.0009323360045556071, -0.0016527739029342462, 0.0213386532068979, -0.7361112702560478, -0.9897013085501344, 1.1738129845129344], [0.42935337597073336, 0.6300638864312258, -0.7361112702560478, 4992.507936507936, 746.2804232804231, -19.428571428571523], [1.9917595051987003, 1.8509234806353214, -0.9897013085501344, 746.2804232804231, 6283.70767195767, 452.44047619047603], [-1.850657000447773, -3.728455646581064, 1.1738129845129344, -19.428571428571523, 452.44047619047603, 4742.22619047619]]), 'macaddr', Value, (), MessageAnalyzer.U_BYTE) ,
    #     # id
    #     FieldTypeMemento(numpy.array([123.72916666666667, 122.125, 136.45833333333334, 117.16666666666667]), numpy.array([71.22374942839565, 68.57861212992479, 72.74726751263965, 79.2507448265034]), numpy.array([[5180.754875886526, 1118.9707446808509, -236.4476950354611, 433.72695035460987], [1118.9707446808509, 4803.090425531915, -360.46276595744695, 696.5531914893616], [-236.4476950354611, -360.46276595744695, 5404.764184397164, 237.8794326241136], [433.72695035460987, 696.5531914893616, 237.8794326241136, 6414.312056737587]]), 'id', Value, (), MessageAnalyzer.U_BYTE) ,
    #     # float
    #     FieldTypeMemento(numpy.array([0.9509647848839462, 3.3029185443164613, 22.105263157894736, 116.69473684210526]), numpy.array([0.5588487293595614, 1.6796493488314423, 8.562431632256324, 72.70952644225558]), numpy.array([[0.31563436935261324, -0.04162196912699606, -0.5355585699839925, 4.230231223185803], [-0.04162196912699606, 2.851234934338717, 0.39996987465004535, -16.023069981731833], [-0.5355585699839925, 0.39996987465004535, 74.09518477043672, -41.43561030235157], [4.230231223185803, -16.023069981731833, -41.43561030235157, 5342.916461366181]]), 'float', Value, (), MessageAnalyzer.U_BYTE) ,
    # ]

    # [
    #     # int
    #     FieldTypeMemento(numpy.array([193.8953488372093, 157.2325581395349]),
    #                      numpy.array([24.531436438079247, 32.666985830090965]),
    #                      numpy.array([[608.8712722298224, -276.293023255814], [-276.293023255814, 1079.6864569083446]]),
    #                      'int', Value, (), MessageAnalyzer.U_BYTE),
    #     # checksum
    #     FieldTypeMemento(numpy.array(
    #         [109.45454545454545, 93.0909090909091, 134.8181818181818, 100.18181818181819, 138.27272727272728,
    #          103.45454545454545, 138.63636363636363, 120.81818181818181]), numpy.array(
    #         [61.14269990821566, 59.946119057722534, 71.2317832301618, 37.48167596383399, 69.24933988407517,
    #          63.96925914611236, 73.5752833093396, 82.34276275734496]),
    #         numpy.array([[4112.272727272727, -202.74545454545452, 197.29090909090905, -876.5909090909092,
    #                       1638.7636363636366, -414.3272727272728, 115.38181818181832, 1327.9909090909096],
    #                      [-202.74545454545452, 3952.890909090909, 1656.6181818181817, 1572.9818181818182,
    #                       -1493.5272727272727, -1224.345454545455, -1531.1636363636364, 859.7181818181818],
    #                      [197.29090909090905, 1656.6181818181817, 5581.363636363636, 1590.636363636364,
    #                       -932.6454545454549, -1105.3090909090909, -2908.3727272727274, -871.1363636363635],
    #                      [-876.5909090909092,  1572.9818181818182,  1590.636363636364,  1545.3636363636363,
    #                       -1728.2545454545457,  -836.9909090909092,  -597.6272727272726,  -323.6636363636362],
    #                      [1638.7636363636366,  -1493.5272727272727,  -932.6454545454549,  -1728.2545454545457,
    #                       5275.018181818181,  -687.7363636363639,  -954.8909090909092,  1865.354545454546],
    #                      [-414.3272727272728,  -1224.345454545455,  -1105.3090909090909,  -836.9909090909092,
    #                       -687.7363636363639,  4501.272727272727,  1276.9818181818184,  2721.9909090909096],
    #                      [115.38181818181832,  -1531.1636363636364,  -2908.3727272727274,  -597.6272727272726,
    #                       -954.8909090909092,  1276.9818181818184,  5954.654545454545,  1005.1272727272728],
    #                      [1327.9909090909096,  859.7181818181818,  -871.1363636363635,  -323.6636363636362,
    #                       1865.354545454546,  2721.9909090909096,  1005.1272727272728,  7458.363636363637]]), 'checksum',
    #                      Value, (), MessageAnalyzer.U_BYTE),
    # ]


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
        Mark recognized char sequences.

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
    """
    Class to encapsulate the retrieving of recognized field types for single positions in messages and whole messages.
    Moreover, statistics about the matching quality can be generated from here.
    """

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
        """
        Retrieve all template matches for a specific byte position in the message referenced by this object.
        :param offset: The byte position in the message.
        :return: List of template matches sorted by their confidence.
        """
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
        """
        Resolve conflicting recognitions by selecting the most confident match from the set of recognizable field types.

        :return: List of recognized fields sorted by their position in the message.
        """
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


















