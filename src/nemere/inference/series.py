import numpy


class AnalysisSeries(dict):
    """
    Class to hold multiple analysis results related to the same message or segment.
    """
    # segments = dict()  # type: List[MessageSegment]

    ID = 'id'
    FEATURE = 'feature'
    CANDIDATE = 'candidate'
    CORRELATE = 'correlation'

    # def __init__(self, **kwargs):
    #     dict.__init__(self, **kwargs)

    @staticmethod
    def fromlist(correlation):
        """

        :param correlation: dict with keys defined in the constants: ID, FEATURE, CANDIDATE, CORRELATE
        :return:
        """
        import humanhash

        return AnalysisSeries({ humanhash.humanize('{:02x}'.format(ser[AnalysisSeries.ID])): ser for ser in correlation })


    def cand(self, humhash):
        """
        convenience method

        :param humhash:
        :return: values of candidate of analysis series for human hash humhash
            at the correlated position and the length of the feature
        """
        position = numpy.argmax(self.corr(humhash))
        length = len(self.feat(humhash))
        return self[humhash][AnalysisSeries.CANDIDATE].values[position:position+length]


    def candFull(self, humhash):
        """
        convenience method

        :param humhash:
        :return: values of feature of analysis series at human hash humhash
        """
        return self[humhash][AnalysisSeries.CANDIDATE].values


    def feat(self, humhash):
        """
        convenience method

        :param humhash:
        :return: values of feature of analysis series at human hash humhash
        """
        return self[humhash][AnalysisSeries.FEATURE].values


    def shiftFeature(self, humhash):
        """
        values of feature of analysis series at human hash humhash

        :param humhash:
        :return:
        """
        fshift = numpy.mean(self.cand(humhash)) - numpy.mean(self.feat(humhash))
        return numpy.add(self.feat(humhash), fshift)


    def corr(self, humhash):
        """
        convenience method

        :param humhash:
        :return: values of correlation of analysis series at human hash humhash
        """
        return self[humhash][AnalysisSeries.CORRELATE].values


