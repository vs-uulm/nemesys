from typing import List
import numpy


class HirschbergOnSegmentSimilarity(object):
    
    SCORE_GAP = 0
    SCORE_MATCH = 5;  # overruled by similarity matrix
    SCORE_MISMATCH = -5;

    

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
        # self._score = None
        

    def align(self, message0: List[int], message1: List[int]):
        """
        
        :param message0: list of indices of the similarity matrix denoting the columns representing a specific segment.
        :param message1: list of indices of the similarity matrix denoting the rows representing a specific segment.
        """
        # self._score = numpy.empty([len(message1), len(message0)])
        
        
        # 
        # Peter H. Sellers: On the Theory and Computation of Evolutionary Distances. 
        # In: SIAM Journal on Applied Mathematics. Band 26, Nr. 4, Juni 1974, S. 787–793, JSTOR:2099985.
        #
        # implemented from pseudo code in https://en.wikipedia.org/wiki/Hirschberg%27s_algorithm
        pass


    def _nwScore(self, tokensX: List[int], tokensY: List[int]):
        score = numpy.empty([2,len(tokensY)])  # 2*length(Y) array
        score[0,0] = 0
        
        for j in range(1, len(tokensY)):
            score[0,j] = score[0,j-1] + HirschbergOnSegmentSimilarity.SCORE_GAP
         
        for x in range(1, len(tokensX)):  # init array
            score[1,0] = score[0,0] + HirschbergOnSegmentSimilarity.SCORE_GAP
            for y in range(1, len(tokensY)):
                scoreSub = score[0,y-1] + self._similarities[x, y]
                scoreDel = score[0,y] + HirschbergOnSegmentSimilarity.SCORE_GAP
                scoreIns = score[1,y-1] + HirschbergOnSegmentSimilarity.SCORE_GAP
                score[1,y] = max(scoreSub, scoreDel, scoreIns)
            # copy Score[1] to Score[0]
            score[0,] = score[1,]
        return score[-1,]  # LastLine
    