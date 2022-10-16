from typing import Type, Union, Any, Tuple, Iterable

import numpy
import scipy.spatial

from netzob.Model.Vocabulary.Messages.AbstractMessage import AbstractMessage

from nemere.inference.analyzers import Value
from nemere.inference.segments import MessageAnalyzer




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


