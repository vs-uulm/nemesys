from typing import List



class NWonSegmentSimilarity(object):

    def __init__(self, similarityMatrix):
        self._similarities = similarityMatrix
    

    def align(self, message0: List[int], message1: List[int]):
        """
        
        :param message0: list of indices of the similarity matrix denoting the columns representing a specific segment.
        :param message1: list of indices of the similarity matrix denoting the rows representing a specific segment.
        """
        
        # 
        # Peter H. Sellers: On the Theory and Computation of Evolutionary Distances. 
        # In: SIAM Journal on Applied Mathematics. Band 26, Nr. 4, Juni 1974, S. 787–793, JSTOR:2099985.
        
