from typing import List, Union, Tuple, Dict
import numpy
import matplotlib.pyplot as plt

from netzob.Model.Vocabulary.Symbol import Symbol

from visualization.plotter import MessagePlotter
from validation.dissectorMatcher import MessageComparator
from utils.loader import SpecimenLoader


# noinspection PyMethodMayBeStatic
class SingleMessagePlotter(MessagePlotter):
    """
    Different methods to plot data of one or more messages in one figure.

    For using subplots for individual messages see MultiMessagePlotter
    """

    def __init__(self, specimens: SpecimenLoader, analysisTitle: str, isInteractive: bool=False):
        super().__init__(specimens, analysisTitle, isInteractive)
        plt.rc('xtick', labelsize=8)  # fontsize of the tick labels
        plt.rc('ytick', labelsize=8)  # fontsize of the tick labels


    def plotAnalysis(self, analysisResults, compareValue = None, fieldEnds = None, labels = None):
        """

        :param analysisResults: the main analysis results to be plotted.
        :param compareValue: another sequence of values to be plotted in comparison of the analysis.
        :param fieldEnds: real field ends of the message to be marked in the plot.
        :param labels: labels per subplot
        """
        pass


    def plotColormesh(self, analysisResults: Union[List[List[float]], numpy.ndarray], fieldEnds: List[List[int]]=None):
        if isinstance(analysisResults, numpy.ndarray):
            paddedar = analysisResults
        else:
            # fill up all messages to a common length
            mslen = max(len(line) for line in analysisResults)  # get maximum message length
            paddedar = numpy.array(
                [line + [numpy.nan]*(mslen - len(line)) for line in analysisResults]
            )

        plt.pcolormesh(paddedar)
        if fieldEnds:
            for msgnr, fe in enumerate(fieldEnds):
                plt.scatter(fe, [msgnr + 0.5] * len(fe), color='black', marker='.', s=2)
        plt.autoscale(tight=True)


    def plotWithAxOverlays(self, ax, mainData, labels=None, *overlayData):
        pass


    def cumulatedFieldEnds(self, fieldEnds: List[List[int]]):
        """
        Plot all distinct field ends of all messages combined to the figure.

        :param fieldEnds: True field ends to plot for reference to each message.
        :return: deduplicated set of field ends.
        """
        dedupFends = set()
        # remove duplicates
        for fe in fieldEnds:
            dedupFends.update(fe)
        for fe in dedupFends:
            plt.axvline(x=fe, linewidth=.5, linestyle='--', alpha=.4)
        plt.xticks(sorted(dedupFends))
        return dedupFends


    def heatMapFieldComparison(self, comparator: MessageComparator, symbols: List[Symbol]):
        from validation.dissectorMatcher import DissectorMatcher
        from collections import OrderedDict, Counter

        matchers = OrderedDict()  # type: OrderedDict[symbol, Tuple[List, List, DissectorMatcher]]
        for ctr, symbol in enumerate(symbols):
            if ctr % 100 == 0:
                print(". ", end="", flush=True)
            # TODO retrieve for all messages in symbol (dict key?)
            dm = DissectorMatcher(comparator, symbol)
            matchers[symbol] = (
                dm.inferredFields, dm.dissectionFields,
                dm)

        #########################################

        # a list of lists of distances of true to inferred FEs, one per abstracted (true) field end of all messages
        # an abstracted field end is one that is at the same sequential position of all messages regardless of the
        # particular byte position.
        distancesPerGeneralTrueFE = list()
        for symbol, (inferredFEs, trueFEs, dm) in matchers.items():
            # inferredForTrueFEs = dm.inferredInDissectorScopes()
            # dict(true field ends: List[signed inferred distances])
            distancesToTrueFEs = dm.distancesFromDissectorFieldEnds()  # type: Dict[int, List[int]]

            # iterate trueFEs, which are sorted
            for idx, tfe in enumerate(trueFEs[:-1]):  # without final message byte
                if idx >= len(distancesPerGeneralTrueFE): # we enumerate indices,
                    # so a missing index is always solved by adding one item
                    distancesPerGeneralTrueFE.append(list())
                if tfe not in distancesToTrueFEs: # there is no inferred field for this true field
                    distancesPerGeneralTrueFE[idx].append(numpy.nan)
                    # print('No inferred for true field: {},\n{}'.format(tfe,
                    #                tabulate((inferredFEs, trueFEs))))
                else:
                    # get distances for all inferred fields
                    distancesPerGeneralTrueFE[idx].extend(distancesToTrueFEs[tfe])

        #########################################

        # basis for the heatmap is list of lists:
        # [[0, 1, 0, 3, 0, 0] , [0, 7, 0, 0]]
        # each list is per abstracted true field end,
        # each item in them is a distance from the minimum to the maximum for this field end for all messages.
        # the value of the item is the count of inferred field ends having this distance to its true field end.
        distanceArrays = [None] * len(distancesPerGeneralTrueFE)
        for (idx, distancesToFE) in enumerate(distancesPerGeneralTrueFE):
            dCounter = Counter(distancesToFE)
            if numpy.isnan(list(dCounter.keys())).all():
                distanceArrays[idx] = numpy.zeros((2,1), int)
            else:
                dstmin, dstmax = numpy.nanmin(list(dCounter.keys())), numpy.nanmax(list(dCounter.keys()))
                dx = list(range(int(dstmin), int(dstmax) + 1))
                dc = [dCounter[distance] if distance in dCounter else 0
                    for distance in dx]
                distanceArrays[idx] = numpy.array([dx, dc])
            # TODO check what happens in the plot if outside
            # min(dCounter.keys()) < center < max(dCounter.keys())

        # print(distanceCount)
        # print(distanceCenter)

        #########################################

        # add intermediate "empty" bytes (to set the results into the right context)
        #
        # get the maximum distance per true fields of all messages
        trueFEs = [tfes for ifes, tfes, mch in matchers.values()]
        trueLengths = [[tfer-tfel for tfel, tfer in zip([0] + tfes[:-1], tfes)] for tfes in trueFEs[:-1]]
        maxTrueLens = list()
        for tfls in trueLengths:  # type: List[int]
            for feidx, tfl in enumerate(tfls):
                if len(maxTrueLens) <= feidx:
                    maxTrueLens.append(0)
                maxTrueLens[feidx] = max(tfl, maxTrueLens[feidx])

        # align inferred distances on field ends in one array
        assert len(maxTrueLens) - 1 == len(distanceArrays)
        combinedDistances = numpy.ndarray((2,0), int)
        offset = maxTrueLens[0]
        for distances, skip in zip(distanceArrays, maxTrueLens[1:]):  # type: numpy.ndarray
            offset += skip
            combinedDistances = numpy.append(
                combinedDistances,
                numpy.array([distances[0]+offset, distances[1]]),
                axis=1)

        #########################################

        # fig, axes = plt.subplots(1, len(distanceArrays), sharex='all', sharey='row')  # type: plt.Figure, numpy.ndarray
        # for ax, darray in \
        #         zip(axes.flat, distanceArrays):  # type: plt.Axes, List[int], int
        #     ax.bar(darray[0], darray[1])
        #     # ax.pcolormesh([dcount])
        #     # ax.plot([dcenter], [0.5], color='black', marker='.', markersize=6)
        #     ax.axvline(x=[0], **MessagePlotter.STYLE_FIELDENDLINE)
        # TODO change to one continuous bar plot
        plt.bar(combinedDistances[0], combinedDistances[1], width=1.0, color="green")
        maxtrueticks = list()
        mintickdist = combinedDistances[0,-1] * 0.04  # the x coordinate of the last of the plot columns
        offset = maxTrueLens[0]
        for skip in maxTrueLens[1:]:
            offset += skip
            if not maxtrueticks or maxtrueticks[-1] + mintickdist < offset:
                maxtrueticks.append(offset)
            plt.axvline(x=[offset], **MessagePlotter.STYLE_FIELDENDLINE)
        plt.xticks(sorted(maxtrueticks))
        plt.autoscale(tight=True)


    def histogramFieldEnds(self, symbols: List[Symbol]):
        """
        Histogram of the absolute byte positions in all messages of all input symbols.

        :param symbols:
        :return:
        """
        from validation.dissectorMatcher import MessageComparator
        from collections import Counter
        from netzob.Model.Vocabulary.Messages.AbstractMessage import AbstractMessage

        maxLen = 0
        cumulatedFieldEnds = Counter()
        for symbol in symbols:
            for message in symbol.messages:  # type: AbstractMessage
                maxLen = max(maxLen, len(message.data))
                cumulatedFieldEnds.update(
                    MessageComparator.fieldEndsPerSymbol(symbol, message)[:-1])  # omit message end

        countAt = list()
        for bytepos in range(maxLen):
            if bytepos in cumulatedFieldEnds.keys():
                countAt.append(cumulatedFieldEnds[bytepos])
            else:
                countAt.append(0)

        plt.bar(list(range(maxLen)), countAt, width=1.0, color="green")
        plt.autoscale(tight=True)


