"""
Intermediate experimentation implementation for the development of new analysis methods.

Individual functions are to be integrated into the inference.analyzers module if working state is reached.
"""

import argparse
from typing import Iterable
from os.path import isfile

import numpy as np
import matplotlib.pyplot as plt
from itertools import compress

import visualization.simplePrint as sprint
from validation.dissectorMatcher import MessageComparator
from utils.loader import SpecimenLoader
from inference.analyzers import *
from visualization.plotter import MessagePlotter
from visualization.singlePlotter import SingleMessagePlotter
from visualization.multiPlotter import MultiMessagePlotter

debug = True




def iterhbvRadius():
    global radius

    # from the looks a value of about 1.5 is suited for ntp, dns, dhcp
    # TODO compare # of fields to # of maxima to get a quantitative estimation.
    for radius in [np.round(rad, 1) for rad in np.arange(0.5, 2.5, 0.1)]:
        plotter = leftHorizonBcPlot()
        plotter.writeOrShowFigure()
    return None


def obtainData(comparator: MessageComparator, analyzerClass: Type[MessageAnalyzer], *analysisArgs, noiseRadius=None) \
        -> Union[Tuple[List[Iterable], List[Iterable], List[List[float]]], Tuple[List[Iterable], List[List[float]]]]:
    """
    Obtains data about each message in specimens by calling analysisFunction(*analysisArgs) on every message.

    :param comparator: Object containing the dissections of the message specimens to analyze and compare to.
    :param analyzerClass: The type of analysis to perform, described by a subclass of MessageAnalyzer.
    :param analysisArgs: The arguments for the analysis if any, else None.
    :param noiseRadius: If smoothing is desired as overlay for comparison,
        set the sigma for the gauss filter to be applied to the data.
    :return: Tuple of analysisResults and real field end indices for each message
    """
    analysisResults = list()
    noiseReduced = list()
    fieldEnds = list()
    for l4msg, rmsg in comparator.messages.items():
        try:
            analyzer = MessageAnalyzer.findExistingAnalysis(analyzerClass, MessageAnalyzer.U_BYTE, l4msg, analysisArgs)
        except NothingToCompareError as e:  # non-critical error that occurs if a message is too short to do any analyses
            print('Non-critical:', e)
            # simply ignore this message
            continue
        analysisResults.append(analyzer.values)
        if noiseRadius:
            noiseReduced.append([numpy.nan] * analyzer.startskip + list(gaussian_filter1d(analyzer.valuesRaw, noiseRadius)))

        # # If we ever want to support U_NIBBLE again, start from here:
        # x = [val/2 if analyzer._unit == MessageAnalyzer.U_NIBBLE else val
        #      for val in range(len(analresu))]

        # get field ends per message
        mfmt = comparator.dissections[rmsg]
        fieldEnds.append(MessageComparator.fieldEndsFromLength([flen for t, flen in mfmt]))
    if noiseRadius:
        return analysisResults, noiseReduced, fieldEnds
    else:
        return analysisResults, fieldEnds


def entropyWithinNgrams(n=3):
    """
    Just as verification. No surprises to expect

    :return: MultiMessagePlotter
    """
    # obtain data
    entropy, smoothed, fieldEnds = obtainData(comparator, EntropyWithinNgrams, n, noiseRadius=radius)

    # visualize data
    cols = 8 if specimens.maximumMessageLength <= 80 else 4  # set to 4 figures per row for traces with longer messages
    mmp = MultiMessagePlotter(specimens, 'entropyWithin{}grams'.format(n), 5, cols, args.interactive)
    mmp.plotSubfigs(entropy, compareValue=smoothed, fieldEnds=fieldEnds)
    return mmp


##################
# Value Progression

def vp():
    """
    Value progression differences

    :return: MultiMessagePlotter
    """
    # obtain data
    analysisResults, noiseReduced, fieldEnds = obtainData(
        comparator, ValueProgressionDelta,
        noiseRadius=radius
    )

    # visualize data
    cols = 8 if specimens.maximumMessageLength <= 80 else 4  # set to 4 figures per row for traces with longer messages
    mmp = MultiMessagePlotter(specimens, 'valueprogressionDelta', 5, cols, args.interactive)
    mmp.plotSubfigs(analysisResults, compareValue=noiseReduced, fieldEnds=fieldEnds)
    return mmp


def vpCumulated():
    """
    value progression cumulation.

    A beginning plateau often denotes the beginning of a numeric (counter) field,
    since the most significant bytes are zero or near zero values in the majority of messages.

    # TODO find a good value for the gauss sigma and the threshold between smoothed and original progression,
    # to match most boundary candidates
    To get all "sharp" changes in progression, compare the values to the gaussian filtered progression.
    Interpret as field border candidates every x at which the difference to the filtered progression is below
    some™ threshold.

    :return: MultiMessagePlotter
    """
    # obtain data
    analysisResults, noiseReduced, fieldEnds = obtainData(
        comparator, CumulatedValueProgression,
        noiseRadius=radius
    )

    # visualize data
    cols = 8 if specimens.maximumMessageLength <= 80 else 4  # set to 4 figures per row for traces with longer messages
    mmp = MultiMessagePlotter(specimens, 'valueprogressionCumulated', 5, cols, args.interactive)
    mmp.plotSubfigs(analysisResults, compareValue=noiseReduced, fieldEnds=fieldEnds)
    mmp.plotCompareFill(analysisResults, noiseReduced)
    return mmp


def vpCumulatedGradient():
    """
    Value progression cumulation.

    :return: MultiMessagePlotter
    """
    # obtain data
    analysisResults, noiseReduced, fieldEnds = obtainData(
        comparator, CumulatedProgressionGradient,
        noiseRadius=radius
    )

    # visualize data
    cols = 8 if specimens.maximumMessageLength <= 80 else 4  # set to 4 figures per row for traces with longer messages
    mmp = MultiMessagePlotter(specimens, 'valueprogressionCumulatedGradient', 5, cols, args.interactive)
    mmp.plotSubfigs(analysisResults, compareValue=noiseReduced, fieldEnds=fieldEnds)
    mmp.plotCompareFill(analysisResults, noiseReduced)
    return mmp


def vpCumulatedDelta():
    """
    Value progression cumulation.

    :return: MultiMessagePlotter
    """
    # obtain data
    analysisResults, noiseReduced, fieldEnds = obtainData(
        comparator, CumulatedProgressionDelta,
        noiseRadius=radius
    )

    # visualize data
    cols = 8 if specimens.maximumMessageLength <= 80 else 4  # set to 4 figures per row for traces with longer messages
    mmp = MultiMessagePlotter(specimens, 'valueprogressionCumulatedDelta', 5, cols, args.interactive)
    mmp.plotSubfigs(analysisResults, compareValue=noiseReduced, fieldEnds=fieldEnds)
    mmp.plotCompareFill(analysisResults, noiseReduced)
    return mmp


def vpCumulated2ndDelta():
    """
    Value progression cumulation.

    :return: MultiMessagePlotter
    """
    # obtain data
    analysisResults, noiseReduced, fieldEnds = obtainData(
        comparator, CumulatedProgression2ndDelta,
        noiseRadius=radius
    )

    # visualize data
    cols = 8 if specimens.maximumMessageLength <= 80 else 4  # set to 4 figures per row for traces with longer messages
    mmp = MultiMessagePlotter(specimens, 'valueprogressionCumulated2ndDelta', 5, cols, args.interactive)
    mmp.plotSubfigs(analysisResults, compareValue=noiseReduced, fieldEnds=fieldEnds)
    mmp.plotCompareFill(analysisResults, noiseReduced)
    return mmp



##################
# Bit Congruence

def bcColormesh():
    """
    Bit congruence

    very nice first results: reports/bitvariances_*.pdf

    :return: SingleMessagePlotter
    """
    # obtain data
    analysisResults, fieldEnds = obtainData(comparator, BitCongruence)

    # visualize data
    smp = SingleMessagePlotter(specimens, 'bitcongruencesColormesh', args.interactive)
    smp.plotColormesh(analysisResults, fieldEnds)
    return smp


def bcDeltaColormesh():
    """
    Bit congruence deltas

    very nice first results: reports/bitvarAmps_*.pdf

    :return: SingleMessagePlotter
    """
    # obtain data
    analysisResults, fieldEnds = obtainData(comparator, BitCongruenceDelta)

    # visualize data
    smp = SingleMessagePlotter(specimens, 'bitCongruenceDeltaColormesh', args.interactive)
    smp.plotColormesh(analysisResults, fieldEnds)
    return smp


def bcPlot():
    """
    Bit congruences

    very nice first results: reports/bitvariances_*.pdf

    Inclined to have local maxima around field boundaries, but not only there.

    :return: MultiMessagePlotter
    """
    # obtain data
    analysisResults, noiseReduced, fieldEnds = obtainData(
        comparator, BitCongruence,
        noiseRadius=radius
    )

    plat = [MessageAnalyzer.plateouStart(am) for am in analysisResults]

    platmaskd = list()
    for m in plat:
        mres = ([], [])
        for ix, vl in zip(m[0], m[1]):
            if vl > 0.8:
                mres[0].append(ix)
                mres[1].append(vl)
        platmaskd.append(mres)

    # visualize data
    cols = 8 if specimens.maximumMessageLength <= 80 else 4  # set to 4 figures per row for traces with longer messages
    mmp = MultiMessagePlotter(specimens, 'bitCongruence', 5, cols, args.interactive)
    mmp.plotSubfigs(analysisResults, compareValue=noiseReduced, fieldEnds=fieldEnds)
    mmp.scatterInEachAx(platmaskd, '^')
    mmp.printMessageBytes(list(specimens.messagePool.keys()))
    return mmp


def bcDeltaPlot():
    """
    very nice first results: reports/bitvarAmps_*.pdf

    Inclined to have local maxima around field boundaries, but not only there.
    Rather use as pattern to search for (via convolution), not so much for message segmentation.

    :return: MultiMessagePlotter
    """
    from tabulate import tabulate

    # obtain data
    axzeros = list()
    fieldEnds = list()
    inflections = list()
    inflvals = list()
    noiseReduced = list()
    analysisResults = list()
    extrema = list()
    for l4msg, rmsg in specimens.messagePool.items():
        mfmt = comparator.dissections[rmsg]
        fieldEnds.append(MessageComparator.fieldEndsFromLength([flen for t, flen in mfmt]))

        bcdgAnalyzer = MessageAnalyzer.findExistingAnalysis(BitCongruenceDeltaGauss,
                                                             MessageAnalyzer.U_BYTE, l4msg,
                                                             (radius,))  # type: BitCongruenceDeltaGauss
        analysisResults.append(bcdgAnalyzer.bcdeltas)

        # zeros = [i for i in range(1,len(bcdgAnalyzer.bcdeltas)-1)
        #          if bcdgAnalyzer.bcdeltas[i-1] < 0.0 < bcdgAnalyzer.bcdeltas[i+1]
        #          and (bcdgAnalyzer.bcdeltas[i+1] - bcdgAnalyzer.bcdeltas[i-1]) > .6]
        # axzeros.append((zeros, len(zeros)*[0]))

        # smoothed graph and extrema thereof
        bcdNR = bcdgAnalyzer.values
        noiseReduced.append(bcdNR)
        nrExtrema = bcdgAnalyzer.extrema()

        lmin = MessageAnalyzer.localMinima(bcdNR)
        lmax = MessageAnalyzer.localMaxima(bcdNR)
        # nrExtrema = sorted(
        #     [(i, 0) for i in lmin[0]] + [(i, 1) for i in lmax[0]], key=lambda k: k[0])
        extrema.append((lmin[0] + lmax[0], lmin[1] + lmax[1]))

        # # filter minima with small rise to the following maximum
        # avgextdelta = np.nanmean([ bcdNR[j[0]] - bcdNR[i[0]]
        #         for i, j in zip(nrExtrema[:-1], nrExtrema[1:]) if i[1] == 0 and j[1] == 1])
        # minrise = 0.33*avgextdelta
        # lextmask = [True if el[1] == 1 or
        #                     bcdNR[er[0]] - bcdNR[el[0]] > minrise else False
        #             for el, er in zip(nrExtrema[:-1], nrExtrema[1:])] + [True]
        # bcdNRextrema = list(compress(nrExtrema,lextmask))
        # # don't filter
        bcdNRextrema = nrExtrema

        # somewhat messy approximate of inflection points
        # variant 1 using the deltas between minima and maxima in smoothed graph
        # wpgf = [ 1+i[0] + np.nanargmax(np.ediff1d(bcdNR[i[0]:j[0]]))
        #        for i, j in zip(bcdNRextrema[:-1], bcdNRextrema[1:]) if i[1] == 0 and j[1] == 1
        #        and j[0] - i[0] > 1]

        # variant 2 using the deltas in original bit congruences deltas between minima and maxima in smoothed bcd
        # index and the bcd-deltas at the position in rising parts of the smoothed bcd
        # risingdeltas = [ (i[0] + 1, np.ediff1d(bcdgAnalyzer.bcdeltas[i[0]:j[0]+1]))  # include index of max
        #         for i, j in zip(bcdNRextrema[:-1], bcdNRextrema[1:])
        #            if i[1] == 0 and j[1] == 1 and j[0]+1 - i[0] > 1]
        # risingdeltas = bcdgAnalyzer.risingDeltas()

        # inflpt = [ offset + np.nanargmax(wd) for offset, wd in risingdeltas ]
        # wpvals = [np.nanmax(wd) for offset, wd in risingdeltas]
        # inflections.append((inflpt, [bcdAnalyzer.values[pkt] for pkt in inflpt]))
        inflpt = bcdgAnalyzer.inflectionPoints()
        inflections.append(inflpt)

        # values before the inflections for comparison in graph
        prewp = [i - 1 for i in inflpt[0]]
        inflvals.append((prewp, [bcdgAnalyzer.bcdeltas[i] for i in prewp]))

        # # correct way to find inflections by 2nd derivate
        # bc2dNR = np.ediff1d(bcdNR)
        # bc3dNR = np.ediff1d(bc2dNR)
        #
        # inflmask = 2*[False] + [True if not np.isnan(ptl) and not np.isnan(ptr) and
        #                         ptl > 0.0 > ptr else False for ptl, ptr in zip(bc3dNR[:-1], bc3dNR[1:])] \
        #            + [False]
        # # print(len(inflmask))
        # inflections.append((list(compress(range(0, len(bcdNR)), inflmask)), list(compress(bcdNR, inflmask))))

        # with open('reports/bc2ndDelta{}.csv'.format(l4msg.id), 'w') as csvfile:
        #     import csv
        #     cw = csv.writer(csvfile)
        #     cw.writerows([bcdAnalyzer.values, bcdNR, ['nan'] + bc2dNR.tolist(),
        #                 2*['nan'] + bc3dNR.tolist(), inflmask])


    # print(tabulate.tabulate(analysisResults))

    # visualize data
    cols = 8 if specimens.maximumMessageLength <= 80 else 4  # set to 4 figures per row for traces with longer messages
    mmp = MultiMessagePlotter(specimens, 'bitCongruenceDelta', 5, cols, args.interactive)
    mmp.plotSubfigs(analysisResults, compareValue=noiseReduced, fieldEnds=fieldEnds)
    # mmp.scatterInEachAx(axzeros, 'v')
    mmp.scatterInEachAx(inflvals, 'v')
    # mmp.scatterInEachAx(extrema, '_')
    mmp.printValues(inflections)
    mmp.printValues(inflvals)
    mmp.printMessageBytes(list(specimens.messagePool.keys()))
    return mmp


def bc2ndDeltaPlot():
    """
    very nice first results: reports/bitvarAmps_*.pdf

    Inclined to have local maxima around field boundaries, but not only there.
    Rather use as pattern to search for (via convolution), not so much for message segmentation.

    :return: MultiMessagePlotter
    """
    # obtain data
    analysisResults, noiseReduced, fieldEnds = obtainData(
        comparator, BitCongruence2ndDelta,
        noiseRadius=radius
    )

    import tabulate
    print(tabulate.tabulate(analysisResults))

    # visualize data
    cols = 8 if specimens.maximumMessageLength <= 80 else 4  # set to 4 figures per row for traces with longer messages
    mmp = MultiMessagePlotter(specimens, 'bitCongruence2ndDelta', 5, cols, args.interactive)
    mmp.plotSubfigs(analysisResults, compareValue=noiseReduced, fieldEnds=fieldEnds)
    return mmp


def bcDistributionHist():
    """
    very nice first results: reports/bitvariances_*.pdf

    :return: List of value bucket
    """
    # obtain data
    analysisResults, fieldEnds = obtainData(comparator, BitCongruence)

    # prepare data
    bitVarsFends = list()
    bitVarsRest = list()
    for msgar, msgfe in zip(analysisResults, fieldEnds):
        for pos, bv in enumerate(msgar):
            if pos in msgfe:
                bitVarsFends.append(bv)
            else:
                bitVarsRest.append(bv)

    # visualize data
    plt.hist(
        [ bitVarsRest, bitVarsFends ],
        density=True
    )
    plt.autoscale(tight=True)
    return analysisResults


def bcAmplitudeDistributionHist():
    """
    Bit congruence

    very nice first results: reports/bitvariances_*.pdf

    :return: List of value buckets
    """
    # obtain data
    analysisResults = list()
    fieldEnds = list()
    for l4msg, rmsg in specimens.messagePool.items():

        analyzer = MessageAnalyzer.findExistingAnalysis(BitCongruence, MessageAnalyzer.U_BYTE, l4msg)  # , MessageAnalyzer.U_NIBBLE
        analysisResults.append(analyzer.values)

        # get field ends per message
        mfmt = comparator.dissections[rmsg]
        fieldEnds.append(MessageComparator.fieldEndsFromLength([flen for t, flen in mfmt]))

    varmean = np.mean([[var for var in msgar] for msgar in analysisResults])

    # prepare data
    bitVarsFends = list()
    bitVarsRest = list()
    for msgar, msgfe in zip(analysisResults, fieldEnds):
        for pos, bv in enumerate(msgar):
            if pos in msgfe:
                bitVarsFends.append(bv - varmean)
            else:
                bitVarsRest.append(bv - varmean)

    # visualize data
    plt.hist(
        [ bitVarsRest, bitVarsFends ],
        density=True
    )
    plt.autoscale(tight=True)
    return analysisResults


def bcBetweenNgramsColormesh(n=3):
    """
    Bit congruence within ngrams

    very nice first results: reports/bitvariancesPer3gram_*.pdf

    :return: SingleMessagePlotter
    """
    # obtain data
    analysisResults, noiseReduced, fieldEnds = obtainData(
        comparator, BitCongruenceBetweenNgrams,
        n, noiseRadius=radius)

    # visualize data
    smp = SingleMessagePlotter(specimens, 'bitcongruencesBetween{}gramsColormesh'.format(n), args.interactive)
    smp.plotColormesh(analysisResults, fieldEnds)
    return smp


def bcBetweenNgramsPlot(n=3):
    """
    Bit congruence between each consecutive ngram. Not so much relevant compared to byte-wise Bit congruence, since
    there is only the last byte of each ngram that can change.

    very nice first results: reports/bitvariancesPer3gram_*.pdf

    :return: MultiMessagePlotter
    """
    # obtain data
    analysisResults, noiseReduced, fieldEnds = obtainData(
        comparator, BitCongruenceBetweenNgrams,
        n, noiseRadius=radius)

    # visualize data
    cols = 8 if specimens.maximumMessageLength <= 80 else 4  # set to 4 figures per row for traces with longer messages
    mmp = MultiMessagePlotter(specimens, 'bitcongruencesBetween{}grams'.format(n), 4, cols, args.interactive)
    mmp.plotSubfigs(analysisResults, compareValue=noiseReduced, fieldEnds=fieldEnds)
    return mmp


def bcNgramsStdPlot(n=3) -> MultiMessagePlotter:
    """
    Standard deviation of bit congruence for each ngram of the message.

    Some field borders are at local minima (or the end of a minimum area), but not very conclusive.
    TODO Needs more work.

    :return: Plot of the first messages.
    """
    # obtain data
    analysisResults, noiseReduced, fieldEnds = obtainData(
        comparator, BitCongruenceNgramStd,
        n, noiseRadius=radius)

    # visualize data
    cols = 8 if specimens.maximumMessageLength <= 80 else 4  # set to 4 figures per row for traces with longer messages
    smp = MultiMessagePlotter(specimens, 'bitcongruences{}gramVars'.format(n), 4, cols, args.interactive)
    smp.plotSubfigs(analysisResults, compareValue=noiseReduced, fieldEnds=fieldEnds,
                    subfigName=["{:.3f}".format(np.std(ar)) for ar in analysisResults])
    return smp


def bcNgramMeansPlot(n=3) -> MultiMessagePlotter:
    """
    Mean of bit congruence for each ngram of the message.

    Some structure visible per field, but not very conclusive. TODO Needs more work.

    :return: Plot of the first messages.
    """
    # obtain data
    analysisResults, noiseReduced, fieldEnds = obtainData(
        comparator, BitCongruenceNgramMean,
        n, noiseRadius=radius)

    # visualize data
    cols = 8 if specimens.maximumMessageLength <= 80 else 4  # set to 4 figures per row for traces with longer messages
    mmp = MultiMessagePlotter(specimens, 'bitcongruences{}gramMeans'.format(n), 4, cols, args.interactive)
    mmp.plotSubfigs(analysisResults, compareValue=noiseReduced, fieldEnds=fieldEnds,
                    subfigName=["{:.3f}".format(np.mean(ar)) for ar in analysisResults])
    return mmp


##################
# Value Variance, Frequency

def valueFrequency() -> SingleMessagePlotter:
    """
    value frequency

    mess[vals]
    """
    # obtain data
    analyzers = list()  # type: List[ValueFrequency]
    for l4msg, rmsg in specimens.messagePool.items():
        analyzer = ValueFrequency(l4msg)
        analyzer.analyze()
        analyzers.append(analyzer)

    # print 5 most frequent
    mostFrqPerMsg = [a.mostFrequent()[:-6:-1] for a in analyzers]
    sprint.printMatrix(sorted(mostFrqPerMsg, key=lambda k: k[0][0]),
                       headers=['Most', '2nd', '3rd', '4th', '5th'])  # frq, val
    # for ntp: only value (specific dat) is relevant for frequent byte values
    # for smtp: byte values only below 128 (- control chars, ...) => ASCII

    # prepare data
    analysisResults = [a.values for a in analyzers]  # type: List[Dict[int, int]]
    valFrq = np.zeros( (len(analysisResults), 256) )  # initialize empty array
    for mid, mess in enumerate(analysisResults):
        for val, frq in mess.items():
            valFrq.itemset((mid, val), frq)

    # visualize data
    smp = SingleMessagePlotter(specimens, 'valuefrequency', args.interactive)
    smp.plotColormesh(valFrq)  #, bins=valFrq.shape)  # x: byte-value, y: message
    plt.xlabel('byte value')
    plt.ylabel('message')
    return smp


def valueVariance():
    """
    value variance

    Having a variance below ±128 is denoting a textual protocol.

    :return: MultiMessagePlotter
    """
    # obtain data
    analysisResults, noiseReduced, fieldEnds = obtainData(
        comparator, ValueVariance,
        noiseRadius=radius
    )

    # visualize data
    cols = 8 if specimens.maximumMessageLength <= 80 else 4  # set to 4 figures per row for traces with longer messages
    mmp = MultiMessagePlotter(specimens, 'valuevariance', 5, cols, args.interactive)
    mmp.plotSubfigs(analysisResults, compareValue=noiseReduced, fieldEnds=fieldEnds)
    return mmp


def valueVarianceAmplitude():
    """
    value amplitude. no specific properties around field boundaries.

    May show patterns for convolution.

    :return: MultiMessagePlotter
    """
    # obtain data
    analysisResults, noiseReduced, fieldEnds = obtainData(
        comparator, VarianceAmplitude,
        noiseRadius=radius
    )

    # visualize data
    cols = 8 if specimens.maximumMessageLength <= 80 else 4  # set to 4 figures per row for traces with longer messages
    mmp = MultiMessagePlotter(specimens, 'valuevarianceAmplitude', 5, cols, args.interactive)
    mmp.plotSubfigs(analysisResults, compareValue=noiseReduced, fieldEnds=fieldEnds)
    return mmp


def leftHorizonBcPlot():
    """
    very nice first results: reports/

    Inclined to have local maxima around field boundaries, with fewer exceptions than non-horizon (bit)variances.

    :return: MultiMessagePlotter
    """
    noiseReduced = None
    if radius:
        # obtain data
        analysisResults, noiseReduced, fieldEnds = obtainData(
            comparator, HorizonBitcongruence,
            horizon, noiseRadius=radius)

        # # prepare data
        deviation = [np.std(nr)*2 for nr in noiseReduced]
        north = [nr+dev for nr, dev in zip(noiseReduced, deviation)]
        south = [nr-dev for nr, dev in zip(noiseReduced, deviation)]
    else:
        # obtain data
        analysisResults, fieldEnds = obtainData(
            comparator, HorizonBitcongruence,
            horizon, noiseRadius=radius)


    # visualize data
    cols = 8 if specimens.maximumMessageLength <= 80 else 4  # set to 4 figures per row for traces with longer messages
    mmp = MultiMessagePlotter(specimens, 'lefthorizon{}bitcongruences{}'.format(
        horizon, 'NR{}'.format(radius) if radius else ''), 5, cols, args.interactive)
    if radius:
        mmp.plotSubfigs(noiseReduced, compareValue=analysisResults, fieldEnds=fieldEnds)
        mmp.plotInEachAx(north, linestyle=MessagePlotter.STYLE_COMPARELINE)
        mmp.plotInEachAx(south, linestyle=MessagePlotter.STYLE_COMPARELINE)
        mmp.plotCompareFill(north, south)
    else:
        mmp.plotSubfigs(analysisResults, fieldEnds=fieldEnds)
    return mmp


def leftHorizonBcAutocorrelationPlot():
    correlations = list()
    for l4msg, rmsg in specimens.messagePool.items():
        analyzer = MessageAnalyzer.findExistingAnalysis(Autocorrelation, MessageAnalyzer.U_BYTE, l4msg,
                                                        (HorizonBitcongruenceGauss, horizon, radius))
        correlations.append(analyzer.values)

    cols = 8 if specimens.maximumMessageLength <= 80 else 4  # set to 4 figures per row for traces with longer messages
    mmp = MultiMessagePlotter(specimens, 'lefthorizon{}bitcongruencesNR{}AutoCorrelation'.format(horizon, radius),
                              5, cols, args.interactive)
    mmp.plotSubfigs(correlations)
    return mmp


def leftHorizonBcGradientPlot():
    analysisTitle = 'lefthorizonBitcongruencesGradient'

    # obtain data
    bitvariance, fieldEnds = obtainData(
        comparator, HorizonBitcongruence,
        horizon)
    gradient, fieldEnds = obtainData(
        comparator, HorizonBitcongruenceGradient,
        horizon)

    # visualize data
    cols = 8 if specimens.maximumMessageLength <= 80 else 4  # set to 4 figures per row for traces with longer messages
    mmp = MultiMessagePlotter(specimens, analysisTitle, 5, cols, args.interactive)
    mmp.plotSubfigs(gradient, compareValue=bitvariance, fieldEnds=fieldEnds)
    return mmp



def leftHorizonBcDeltaPlot():
    analysisTitle = 'lefthorizonBitcongruencesDelta'

    # obtain data
    bitvariance, fieldEnds = obtainData(
        comparator, HorizonBitcongruence,
        horizon)
    diffquot, fieldEnds = obtainData(
        comparator, HorizonBitcongruenceDelta,
        horizon)

    # visualize data
    cols = 8 if specimens.maximumMessageLength <= 80 else 4  # set to 4 figures per row for traces with longer messages
    mmp = MultiMessagePlotter(specimens, analysisTitle, 5, cols, args.interactive)
    mmp.plotSubfigs(diffquot, compareValue=bitvariance, fieldEnds=fieldEnds)
    return mmp


def leftHorizonBc2ndDeltaPlot():
    analysisTitle = 'lefthorizonBitcongruences2ndDelta'

    # obtain data
    bitvariance, fieldEnds = obtainData(
        comparator, HorizonBitcongruence,
        horizon)
    diffquot, fieldEnds = obtainData(
        comparator, HorizonBitcongruence2ndDelta,
        horizon)

    # visualize data
    cols = 8 if specimens.maximumMessageLength <= 80 else 4  # set to 4 figures per row for traces with longer messages
    mmp = MultiMessagePlotter(specimens, analysisTitle, 5, cols, args.interactive)
    mmp.plotSubfigs(diffquot, compareValue=bitvariance, fieldEnds=fieldEnds)
    return mmp


def pivotedBcPlot(meanThreshold = .02):
    """
    Repeatedly cut the message(segments) in half, calculate the mean/variance of bit congruence for each half,
    until the difference between the segments is below a™ threshold.

    :return: MultiMessagePlotter
    """
    # obtain data
    pivotbv, fieldEnds = obtainData(comparator, PivotBitCongruence, meanThreshold)
    bitvariance, fieldEnds2 = obtainData(comparator, BitCongruence)

    # visualize data
    # pivotbv and bitvariance.
    cols = 8 if specimens.maximumMessageLength <= 80 else 4  # set to 4 figures per row for traces with longer messages
    mmp = MultiMessagePlotter(specimens, 'pivoted{}bitcongruences'.format(meanThreshold), 5, cols, args.interactive)
    mmp.plotSubfigs(pivotbv, compareValue=bitvariance, fieldEnds=fieldEnds)
    return mmp


def slidingMeansBc():
    analysisTitle = 'sliding{}meansBitcongruences (+ BitCongruence)'.format(horizon)

    # obtain data
    bc, fieldEnds = obtainData(
        comparator, BitCongruence,
        horizon)
    sliding2mean, fieldEnds = obtainData(
        comparator, SlidingNmeanBitCongruence,
        horizon)

    slidingmins = list()
    bcDelta = list()
    slidingdeltamskd = list()
    deltamskd = list()
    for l4msg, rmsg in comparator.messages.items():
        s2mbcAnalyzer = MessageAnalyzer.findExistingAnalysis(SlidingNmeanBitCongruence, MessageAnalyzer.U_BYTE, l4msg,
                                                            (horizon, ))
        s2mbcdAnalyzer = MessageAnalyzer.findExistingAnalysis(SlidingNbcDelta, MessageAnalyzer.U_BYTE, l4msg,
                                                            (horizon, ))
        bcAnalyzer = MessageAnalyzer.findExistingAnalysis(BitCongruence, MessageAnalyzer.U_BYTE, l4msg,
                                                        (horizon,))
        lmins = MessageAnalyzer.localMinima(s2mbcAnalyzer.values)
        # S2MBC min only if BC[n-1] < BC[n+1]
        minmsk = [True if bcAnalyzer.values[i-1] < bcAnalyzer.values[i+1] else False for i in lmins[0]]
        slidingmins.append((list(compress(lmins[0], minmsk)), list(compress(lmins[1], minmsk))))

        # s2mbcdmax = MessageAnalyzer.localMaxima(s2mbcdAnalyzer.values)
        # deltamsk = [True if i not in s2mbcdmax[0] else False for i in lmins[0]]
        # slidingdeltamskd.append((list(compress(lmins[0], deltamsk)), list(compress(lmins[1], deltamsk))))

        bcDeltaanalyzer = MessageAnalyzer.findExistingAnalysis(BitCongruenceDelta, MessageAnalyzer.U_BYTE, l4msg)
        bcDelta.append(bcDeltaanalyzer.values)

        bcdmax = MessageAnalyzer.localMaxima(bcDeltaanalyzer.values)
        deltamsk = [True if i not in bcdmax[0] else False for i in lmins[0]]
        deltamskd.append((list(compress(lmins[0], deltamsk)), list(compress(lmins[1], deltamsk))))


    # visualize data
    cols = 8 if specimens.maximumMessageLength <= 80 else 4  # set to 4 figures per row for traces with longer messages
    mmp = MultiMessagePlotter(specimens, analysisTitle, 5, cols, args.interactive)
    mmp.plotSubfigs(sliding2mean, compareValue=bc, fieldEnds=fieldEnds)
    mmp.plotInEachAx(bcDelta, linestyle=MessagePlotter.STYLE_BLUMAINLINE)
    mmp.printMessageBytes(list(specimens.messagePool.keys()))
    mmp.scatterInEachAx(slidingmins, 'v')
    mmp.scatterInEachAx(deltamskd, '^')
    return mmp


def slidingMeansBcDelta():
    analysisTitle = 'sliding{}meansBitcongruencesDelta'.format(horizon)

    # obtain data
    lhorizonbc, fieldEnds = obtainData(
        comparator, HorizonBitcongruence,
        horizon)
    sliding2mean, fieldEnds = obtainData(
        comparator, SlidingNbcDelta,
        horizon)

    # visualize data
    cols = 8 if specimens.maximumMessageLength <= 80 else 4  # set to 4 figures per row for traces with longer messages
    mmp = MultiMessagePlotter(specimens, analysisTitle, 5, cols, args.interactive)
    mmp.plotSubfigs(sliding2mean, compareValue=lhorizonbc, fieldEnds=fieldEnds)
    mmp.printMessageBytes(list(specimens.messagePool.keys()))
    return mmp


def slidingMeansBcGradient():
    """
    Value progression cumulation.

    :return: MultiMessagePlotter
    """
    # obtain data
    bitvariance, fieldEnds = obtainData(
        comparator, SlidingNmeanBitCongruence,
        horizon)
    sliding2mean, fieldEnds = obtainData(
        comparator, SlidingNbcGradient,
        horizon)

    # visualize data
    cols = 8 if specimens.maximumMessageLength <= 80 else 4  # set to 4 figures per row for traces with longer messages
    mmp = MultiMessagePlotter(specimens, 'sliding{}meansBitcongruencesGradient'.format(horizon), 5, cols, args.interactive)
    mmp.plotSubfigs(sliding2mean, compareValue=bitvariance, fieldEnds=fieldEnds)
    mmp.plotCompareFill(sliding2mean, bitvariance)
    return mmp


def slidingMeansBcDeltaGauss():
    """
    Value progression cumulation.

    :return: MultiMessagePlotter
    """
    # obtain data
    slidingNbcDelta, fieldEnds = obtainData(
        comparator, SlidingNbcDelta,
        horizon)

    sliding2meanSmooth = list()
    barebv = list()
    slidingmins = list()
    slidingmaxs = list()
    slidingzero = list()
    slidingplat = list()
    slidingnans = list()
    msgAnalyzers = list()
    for l4msg, rmsg in specimens.messagePool.items():
        try:
            analyzer = MessageAnalyzer.findExistingAnalysis(SlidingNbcDeltaGauss, MessageAnalyzer.U_BYTE, l4msg,
                                                            (horizon, radius))
            bvanalyzer = MessageAnalyzer.findExistingAnalysis(BitCongruenceDelta, MessageAnalyzer.U_BYTE, l4msg)
        except NothingToCompareError as e:  # non-critical error that occurs if a message is too short to do any analyses
            print('Non-critical:', e)
            # simply ignore this message
            continue
        sliding2meanSmooth.append(analyzer.values)
        barebv.append(bvanalyzer.values)
        msgAnalyzers.append(analyzer)
        lmins = analyzer.localMinima(analyzer.values)
        lmaxs = analyzer.localMaxima(analyzer.values)
        # the frequent matches of the zero sequence ends is only due to the 2-byte window.
        lzero = analyzer.zeroSequences(analyzer.bitcongruences)
        plats = analyzer.plateouStart(analyzer.bitcongruences)
        nans = analyzer.separateNaNs(analyzer.bitcongruences)

        # create masks to compress lists to values above sensitivity (distance from 0)
        amin = min(lmins[1])
        amax = max(lmaxs[1])
        sens = 0.5
        minSmsk = [True if e < sens * amin else False for e in lmins[1]]
        maxSmsk = [True if e > sens * amax else False for e in lmaxs[1]]

        # # create masks to remove extrema in smoothed around zeros in hard bv
        # lminidx = set(list(np.subtract(lmins[1], 1)) + lmins[1] + list(np.add(lmins[1], 1)))
        # lmaxidx = set(list(np.subtract(lmaxs[1], 1)) + lmaxs[1] + list(np.add(lmaxs[1], 1)))
        # minZmsk = [True if i in lminidx else False for i in lmins[1]]
        # maxZmsk = [True if i in lmaxidx else False for i in lmaxs[1]]

        # minmask = [a and b for a, b in zip(minSmsk, minZmsk)]
        # maxmask = [a and b for a, b in zip(maxSmsk, maxZmsk)]
        minmask = minSmsk
        maxmask = maxSmsk

        slidingmins.append((list(compress(lmins[0], minmask)), list(compress(lmins[1], minmask))))
        slidingmaxs.append((list(compress(lmaxs[0], maxmask)), list(compress(lmaxs[1], maxmask))))
        slidingzero.append(lzero)
        slidingplat.append(plats)
        slidingnans.append(nans)

    # visualize data
    cols = 8 if specimens.maximumMessageLength <= 80 else 4  # set to 4 figures per row for traces with longer messages
    mmp = MultiMessagePlotter(specimens, 'sliding{}meansBitcongruencesDeltaGauss'.format(horizon), 5, cols, args.interactive)
    mmp.plotSubfigs(sliding2meanSmooth, compareValue=slidingNbcDelta, fieldEnds=fieldEnds, fieldEndMarks=False)
    # mmp.nameEachAx([man.message.data.hex()[:28] for man in msgAnalyzers])
    mmp.plotInEachAx(barebv, linestyle=MessagePlotter.STYLE_COMPARELINE)
    mmp.scatterInEachAx(slidingmins, 'v')
    mmp.scatterInEachAx(slidingmaxs, '^')
    # mmp.scatterInEachAx(slidingzero, '.')
    # mmp.scatterInEachAx(slidingplat, '*')
    # mmp.scatterInEachAx(slidingnans, 'x')
    mmp.plotCompareFill(sliding2meanSmooth, slidingNbcDelta)
    mmp.printMessageBytes(list(specimens.messagePool.keys()))
    return mmp


def slidingMeansBc2ndDelta():
    """
    Value progression cumulation.

    :return: MultiMessagePlotter
    """
    # obtain data
    bitvariance, fieldEnds = obtainData(
        comparator, SlidingNmeanBitCongruence,
        horizon)
    sliding2mean, fieldEnds = obtainData(
        comparator, SlidingNbc2ndDelta,
        horizon)

    # visualize data
    cols = 8 if specimens.maximumMessageLength <= 80 else 4  # set to 4 figures per row for traces with longer messages
    mmp = MultiMessagePlotter(specimens, 'sliding{}meansBitcongruences2ndDelta'.format(horizon), 5, cols, args.interactive)
    mmp.plotSubfigs(sliding2mean, compareValue=bitvariance, fieldEnds=fieldEnds)
    mmp.plotCompareFill(sliding2mean, bitvariance)
    return mmp








if __name__ == '__main__':

    # available analysis methods
    analyses = {
        'entropyNgrams':    entropyWithinNgrams,

        'bc':               bcPlot,
        'bcColormesh':      bcColormesh,
        'bcp3g':            bcBetweenNgramsPlot,
        'bcp3gColormesh':   bcBetweenNgramsColormesh,
        'bc3gstd':          bcNgramsStdPlot,
        'bc3gmean':         bcNgramMeansPlot,

        'bcDelta':          bcDeltaPlot,
        'bcDelta2':         bc2ndDeltaPlot,
        'bcDeltaColormesh': bcDeltaColormesh,

        'hbc':              leftHorizonBcPlot,
        'hbcnrIterRadius':  iterhbvRadius,
        'hbcnrAutoCorr':    leftHorizonBcAutocorrelationPlot,

        'hbcGrad':          leftHorizonBcGradientPlot,
        'hbcDelta':         leftHorizonBcDeltaPlot,
        'hbcDelta2':        leftHorizonBc2ndDeltaPlot,

        's2meanbc':         slidingMeansBc,
        's2meanbcGrad':     slidingMeansBcGradient,
        's2meanbcDelta':    slidingMeansBcDelta,
        's2meanbcDeltaGauss': slidingMeansBcDeltaGauss,
        's2meanbcDelta2':   slidingMeansBc2ndDelta,

        'pivotBc':          pivotedBcPlot,

        'variance':         valueVariance,
        'varAmp':           valueVarianceAmplitude,

        'vpcumu':           vpCumulated,
        'vpcumuGrad':       vpCumulatedGradient,
        'vpcumuDelta':      vpCumulatedDelta,
        'vpcumuDelta2':     vpCumulated2ndDelta,
        'vp': vp,

        'frequency':  valueFrequency

        # # # # # # # # # # # # # # #
        # allover-message stuff

        # analysisTitle = 'bitvariancesDistribution'
        # bcDistributionHist()

        # analysisTitle = 'bitvarAmpDistribution'
        # bcAmplitudeDistributionHist()
        # # # # # # # # # # # # # # #
    }

    parameters = {
        'pivotbv': .02
    }

    parser = argparse.ArgumentParser(
        description='Compare netzob inference and a scapy dissector for a set of messages (pcap).')
    parser.add_argument('pcapfilename', help='Name of PCAP file to load.')
    parser.add_argument('analysis', help='Type of the analysis to perform. Available options are:\n' +
                                         ", ".join(analyses.keys()), )
    parser.add_argument('-i', '--interactive',help='show interactive plot instead of writing output to file.',
                        action="store_true")
    parser.add_argument('-m', type=int)
    args = parser.parse_args()
    if not isfile(args.pcapfilename):
        print('File not found: ' + args.pcapfilename)
        exit(1)

    analysisType = args.analysis
    horizon = 2
    radius = 0.7
    # radius = 2
    # radius = None

    print()
    if analysisType not in analyses:
        print('Analysis type unknown: ' + args.analysis)
        exit(2)

    print("Load messages ...")
    specimens = SpecimenLoader(args.pcapfilename)
    print("Dissect messages...")
    comparator = MessageComparator(specimens, debug=debug)

    print("Analyze messages ({})...".format(analysisType))
    if analysisType in parameters:
        plotter = analyses[analysisType](parameters[analysisType])
    else:
        plotter = analyses[analysisType]()

    if plotter:
        plotter.writeOrShowFigure()
