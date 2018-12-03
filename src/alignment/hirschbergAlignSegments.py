from typing import List
from abc import ABC, abstractmethod
import numpy


class Alignment(ABC):

    SCORE_GAP = 0  # TODO: use as factor, to multiply with e.g. the maximum of the given similairty matrix.
    SCORE_MATCH = 5  # overruled by similarity matrix
    SCORE_MISMATCH = -5

    def __init__(self, similarityMatrix):
        """
        TODO: similarityMatrix needs to contain values for pairs of non-equal length segments, too!
        """

        self._similarities = similarityMatrix
        """
        matrix of similarities: higher values denote closer match

        TODO: usable value domain and/or 
        "mismatch penalty" that reaches into negative values for bad matches needs to be determined
        """

    @abstractmethod
    def align(self, message0: List[int], message1: List[int]):
        raise NotImplementedError()


class HirschbergOnSegmentSimilarity(Alignment):
    """
    Hirschberg on similarity matrix of segments
    """

    def align(self, message0: List[int], message1: List[int]):
        """

        >>> hirsch = HirschbergOnSegmentSimilarity(simtx)
        >>> haligned = hirsch.align(m0,m1)
        >>> tabulate(haligned)

        :param message0: list of indices of the similarity matrix denoting the columns representing a specific segment.
        :param message1: list of indices of the similarity matrix denoting the rows representing a specific segment.
        """
        from tabulate import tabulate

        # Peter H. Sellers: On the Theory and Computation of Evolutionary Distances.
        # In: SIAM Journal on Applied Mathematics. Band 26, Nr. 4, Juni 1974, S. 787–793, JSTOR:2099985.
        #
        # implemented from pseudo code in https://en.wikipedia.org/wiki/Hirschberg%27s_algorithm
        messageA = []
        messageB = []

        if len(message0) == 0:
            for y in message1:
                messageA.append(-1)  # gap
                messageB.append(y)
        elif len(message1) == 0:
            for x in message0:
                messageA.append(x)
                messageB.append(-1)  # gap
        elif len(message0) == 1 or len(message1) == 1:
            nwalign = NWonSegmentSimilarity(self._similarities)
            haligned = nwalign.align(message0, message1)
            # print("NW:")
            # print(tabulate(haligned))
            return haligned
        else:
            xmid = len(message0) // 2
            scoreL = self._nwScore(message0[:xmid], message1)
            scoreR = self._nwScore(message0[:xmid-1:-1], message1[::-1])
            scoreSum = scoreL[::-1] + scoreR  # vector sum of both alignments
            ymid = scoreSum.shape[0] - numpy.argmax(scoreSum) - 1  # last maximum to resemble NW more literally
            # # original:
            # scoreSum = scoreL + scoreR[::-1]  # vector sum of both alignments
            # ymid = numpy.argmax(scoreSum)

            leftA, leftB = self.align(message0[:xmid], message1[:ymid])
            rghtA, rghtB = self.align(message0[xmid:], message1[ymid:])

            # print("recurse:")
            # print(tabulate((leftA,leftB)))
            # print(tabulate((rghtA, rghtB)))

            messageA = leftA + rghtA
            messageB = leftB + rghtB

        print("return:")
        print(tabulate((messageA,messageB)))

        return messageA, messageB


    def _nwScore(self, tokensX: List[int], tokensY: List[int]):
        """
        calculate a nwscore for two lists of tokens.

        >>> from alignment.hirschbergAlignSegments import HirschbergOnSegmentSimilarity
        >>> from tabulate import tabulate
        >>> import numpy
        >>> simtx = numpy.array([
        ...     [5.0,1.0,0.4,1.8,2.5],
        ...     [0.7,5.0,1.0,2.8,1.5],
        ...     [2.4,0.9,5.0,0.8,2.5],
        ...     [0.0,1.0,1.8,5.0,2.5],
        ...     [0.0,1.0,1.8,2.5,5.0]
        ... ])
        >>> m0 = [2,4,3,0]
        >>> m1 = [1,4,0]
        >>> hirsch = HirschbergOnSegmentSimilarity(simtx)
        >>> score = hirsch._nwScore(m0,m1)
        >>> print(tabulate([score]))
        -  -  --  -
        2  4   3  0
        1  4  -1  0
        -  -  --  -


        :param tokensX:
        :param tokensY:
        :return:
        """
        score = numpy.empty([2,len(tokensY)+1])  # 2*length(Y) array
        score[0,] = 0

        # # penalize gaps at beginning and end
        # score[0,0] = 0
        # for j in range(1, len(tokensY)+1):
        #     score[0,j] = score[0,j-1] + Alignment.SCORE_GAP
         
        for x in range(1, len(tokensX)+1):  # init array
            score[1,0] = score[0,0] + Alignment.SCORE_GAP
            for y in range(1, len(tokensY)+1):
                scoreSub = score[0,y-1] + self._similarities[tokensX[x-1], tokensY[y-1]]
                scoreDel = score[0,y] + Alignment.SCORE_GAP
                scoreIns = score[1,y-1] + Alignment.SCORE_GAP
                score[1,y] = max(scoreSub, scoreDel, scoreIns)
            # copy Score[1] to Score[0]
            score[0,] = score[1,]

        # from tabulate import tabulate
        # print("score messages:")
        # print(tabulate((tokensX, tokensY, score[0,], score[1,])))

        return score[-1,]  # LastLine


class NWonSegmentSimilarity(Alignment):
    """
    Needleman-Wunsch on similarity matrix of segments
    """

    def align(self, message0: List[int], message1: List[int]):
        """

        >>> from alignment.hirschbergAlignSegments import NWonSegmentSimilarity, HirschbergOnSegmentSimilarity
        >>> import numpy
        >>> simtx = numpy.array([
        ...     [ 5.0,-5.0,-5.0,-5.0,-5.0],
        ...     [-5.0, 5.0,-5.0,-5.0,-5.0],
        ...     [-5.0,-5.0, 5.0,-5.0,-5.0],
        ...     [-5.0,-5.0,-5.0, 5.0,-5.0],
        ...     [-5.0,-5.0,-5.0,-5.0, 5.0]
        ... ])
        >>> m0 = [2,4,3,0]
        >>> m1 = [1,4,0]
        >>> nwalign = NWonSegmentSimilarity(simtx)
        --  --  -  --  -
        -1   2  4   3  0
         1  -1  4  -1  0
        --  --  -  --  -


        :param message0: list of indices of the similarity matrix denoting the columns representing a specific segment.
        :param message1: list of indices of the similarity matrix denoting the rows representing a specific segment.
        """
        # Peter H. Sellers: On the Theory and Computation of Evolutionary Distances.
        # In: SIAM Journal on Applied Mathematics. Band 26, Nr. 4, Juni 1974, S. 787–793, JSTOR:2099985.
        #
        # from pseudo code on https://en.wikipedia.org/wiki/Needleman%E2%80%93Wunsch_algorithm

        scores = self._scoreMatrix(message0, message1)

        alignmentA = []
        alignmentB = []
        i = len(message0)
        j = len(message1)
        while i > 0 or j > 0:
            m0v = message0[i - 1]
            m1v = message1[j - 1]
            if i > 0 and j > 0 and scores[i, j] == scores[i - 1, j - 1] + self._similarities[m0v, m1v]:
                alignmentA.append(m0v)
                alignmentB.append(m1v)
                i -= 1
                j -= 1
            elif i > 0 and scores[i, j] == scores[i - 1, j] + Alignment.SCORE_GAP:
                alignmentA.append(m0v)
                alignmentB.append(-1)
                i -= 1
            else:
                alignmentA.append(-1)
                alignmentB.append(m1v)
                j -= 1
        return alignmentA[::-1], alignmentB[::-1]


    def _scoreMatrix(self, message0, message1):
        scores = numpy.empty([len(message0)+1, len(message1)+1])
        scores[0,] = 0
        scores[:,0] = 0
        for i in range(1,scores.shape[0]):
            for j in range(1,scores.shape[1]):
                scoreSub = scores[i-1,j-1] + self._similarities[message0[i-1], message1[j-1]]
                scoreDel = scores[i-1,j] + Alignment.SCORE_GAP
                scoreIns = scores[i,j-1] + Alignment.SCORE_GAP
                scores[i,j] = max(scoreSub, scoreDel, scoreIns)
        return scores
