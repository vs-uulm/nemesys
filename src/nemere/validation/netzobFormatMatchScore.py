"""
Methods for reformatting, pretty printing, and plotting of Format Match Score values.
"""

import csv
import os
import time
from os.path import abspath, isdir
from typing import Dict, Tuple, List
from concurrent.futures.process import ProcessPoolExecutor
from concurrent.futures import TimeoutError as FutureTOError


from netzob import all as netzob
from netzob.Common.Utils.MatrixList import MatrixList
from netzob.Model.Vocabulary.Messages.AbstractMessage import AbstractMessage

from nemere.utils.loader import SpecimenLoader
from nemere.validation.dissectorMatcher import FormatMatchScore, MessageComparator, \
    stop_process_pool, messageparsetimeout

def printFMS(
        formatmatchmetrics: Dict[Tuple[int, netzob.Symbol, List[tuple]], Tuple[int, int, int, int, int]],
        withsymbols = True):
    """
    Print the of the quality metric of each symbol/format as a table of symbols and their quality values.

    :param formatmatchmetrics: Expects a structure of::

            dict (
                (similarity, symbol, format) :
                     (quality, fieldcount, exactf, nearf, uospecific)
                )

    :param withsymbols: if true, also print the symbols themselves.
    """
    siSyFoQu = dict()
    for (similarity, symbol, mformat), symbolsquality in formatmatchmetrics.items():
        if similarity not in siSyFoQu:
            siSyFoQu[similarity] = dict()
        if symbol not in siSyFoQu[similarity]:
            siSyFoQu[similarity][symbol] = dict()
        siSyFoQu[similarity][symbol][mformat] = symbolsquality

    for similarity, syFoQu in siSyFoQu.items():
        print("### " + str(similarity) + " " + "#" * 60)
        if withsymbols:
            for symbol, foQu in syFoQu.items():
                print("~~~ " + symbol.name)
                print(symbol)
                print("\n")
        # list of (similaritythreshold, symbol, quality, fieldcount, exactf, nearf, uospecific)
        # for this similarity threshold
        matchprecisionlist = [ (symbol.name, "Format {:d}".format(fcnt), quality, fieldcount, exactf, nearf, uospecific)
                               for symbol, foQu in syFoQu.items()
                               for fcnt, (mformat, (quality, fieldcount, exactf, nearf, uospecific))
                               in enumerate(foQu.items()) ]
        # sort input symbols by quality
        matchprecisionlist.sort(key=lambda t: t[2])

        printResultMatrix(matchprecisionlist)
        print("\n")


class MessageScoreStatistics(object):

    def __init__(self, comparator: MessageComparator):
        self.__comparator = comparator
        self.__formats = comparator.dissections
        self.__l4lookup = {l4msg: self.__formats[rmsg] for l4msg, rmsg in comparator.messages.items()}


    def messageCountPerFormat(self, symbol: netzob.Symbol, fmt: List[tuple]):
        return len([True for m in symbol.messages if self.__l4lookup[m] == fmt])

    # (thresh, msg) : fms
    @staticmethod
    def minMaxMean(formatmatchmetrics: Dict[Tuple[int, AbstractMessage], FormatMatchScore]) \
            -> Dict[int, Tuple[float, float, float]]:
        """
        Basic statistics from the given formatmatchmetrics.

        :param formatmatchmetrics:
        :return: dict of min, max, mean per threshold
        """
        import numpy

        countEmpty = 0
        thrScores = dict()
        for (th, msg), fms in formatmatchmetrics.items():
            # ignore parsing errors
            if fms.score is None:
                countEmpty += 1
                continue
            if th not in thrScores:
                thrScores[th] = list()
            thrScores[th].append(fms.score)

        print("Empty inferences ignored:", countEmpty)
        return {th: (numpy.min(sc), numpy.max(sc), numpy.mean(sc)) for th, sc in thrScores.items()}

    @classmethod
    def printMinMax(cls, formatmatchmetrics: Dict[Tuple[int, AbstractMessage], FormatMatchScore]):
        """
        Print the Format Match Score min/max per threshold.

        :param formatmatchmetrics: Dict[Threshold, Message], FormatMatchScore]
        """
        mmm = cls.minMaxMean(formatmatchmetrics)

        qualmatrix = [["Thresh"], ["min"], ["max"], ["mean"]]
        for th, (minft, maxft, meanft) in mmm.items():
            qualmatrix[0].append(str(th))
            qualmatrix[1].append("{:03.3f}".format(minft))
            qualmatrix[2].append("{:03.3f}".format(maxft))
            qualmatrix[3].append("{:03.3f}".format(meanft))

        ml = MatrixList()
        ml.headers = qualmatrix[0]
        ml.extend(qualmatrix[1:])
        print('Overall format matching score statistics:')
        print(ml)


def writeReport(formatmatchmetrics: Dict[Tuple[int, AbstractMessage], FormatMatchScore], plot,
                runtimes: Dict[int, float],
                specimens: SpecimenLoader, comparator: MessageComparator, folder="reports"):
    """
    Write a report about this analysis run into a folder with several files containing aspects of the results.

    :param runtimes: runtimes per threshold
    :param formatmatchmetrics: The Format Match Scores grouped by thresholds, symbols, and formats
    :param comparator: To compare the inference results to real message formats
    :param plot: A figure (of the Format Match Scores) that should be saved with the report
    :param specimens: The representation of all original specimens
    :param folder: The folder to write to. Default is the relative folder "reports"
    """
    absFolder = abspath(folder)
    if not isdir(absFolder):
        raise NotADirectoryError("The reports folder {} is not a directory. Reports cannot be written there.".format(absFolder))
    pcapName = os.path.splitext(os.path.basename(specimens.pcapFileName))[0]
    reportFolder = os.path.join(absFolder, pcapName + "_clByAlign_" + time.strftime("%Y%m%d-%H%M%S", time.localtime()))
    os.makedirs(reportFolder)

    print('Write report to ' + reportFolder)

    mss = MessageScoreStatistics(comparator)

    # write Format Match Score and Metrics to csv
    with open(os.path.join(reportFolder, 'FormatMatchMetrics.csv'), 'w') as csvfile:
        fmmcsv = csv.writer(csvfile)
        fmmcsv.writerow(["Threshold", "Message", "Score", 'I', 'M', 'N', 'S', 'MG', 'SP'])
        fmmcsv.writerows( [
            (similaritythreshold, message.data.hex(),
             fms.score, fms.inferredCount, fms.exactCount, fms.nearCount, fms.specificy, fms.matchGain, fms.specificyPenalty)
            for (similaritythreshold, message), fms
            in formatmatchmetrics.items()] )

    with open(os.path.join(reportFolder, 'ScoreStatistics.csv'), 'w') as csvfile:
        fmmcsv = csv.writer(csvfile)
        fmmcsv.writerow(["Threshold", "min", "max", "mean", "runtime"])
        fmmcsv.writerows( [
            (th, minf, maxf, meanf, runtimes[th]) for th, (minf, maxf, meanf) in mss.minMaxMean(formatmatchmetrics).items()
        ])

    # write Symbols to csvs
    uniqueSymbols = {(similaritythreshold, q.symbol)    # by the set comprehension,
                        for (similaritythreshold, message), q # remove identical symbols due to multiple formats
                        in formatmatchmetrics.items()}
    for cnt, (thresh, symbol) in enumerate(uniqueSymbols):
        fileNameS = 'Symbol_{:d}_Thresh{:d}_{:s}'.format(cnt, thresh, symbol.name)
        with open(os.path.join(reportFolder, fileNameS + '.csv'), 'w') as csvfile:
            symbolcsv = csv.writer(csvfile)
            symbolcsv.writerow([field.name for field in symbol.fields])
            # wait only messageparsetimeout seconds for Netzob's MessageParser to return the result
            with ProcessPoolExecutor(max_workers=1) as executor:
                try:
                    future = executor.submit(symbol.getCells)
                    cells = future.result(messageparsetimeout)
                    symbolcsv.writerows([[val.hex() for val in msg] for msg in cells])
                except FutureTOError:
                    stop_process_pool(executor)
                    symbolcsv.writerow(["Parsing of symbol", symbol.name, "timed out after",
                                         messageparsetimeout, "seconds. Omitting", len(symbol.messages),
                                        "messages in this symbol."])

    # thrSym = dict()
    # for thr, sym in uniqueSymbols:
    #     if thr not in thrSym:
    #         thrSym[thr] = list()
    #     thrSym[thr].append(sym)
    # for cnt, (thresh, syms) in enumerate(thrSym.items()):
    #     fileNameS = '{:d}_thresh{:d}'.format(cnt, thresh)
    #     with open(os.path.join(reportFolder, 'interleaved-inference_{}.tikz'.format(fileNameS)), 'w') as tikzfile:
    #         tikzcode = comparator.tprintInterleaved(syms)
    #         tikzfile.write(tikzcode)

    # write tshark-dissection to csv
    # currently only unique formats. For a specific trace a baseline could be determined
    # by a one time run of per ParsedMessage
    with open(os.path.join(reportFolder, 'tshark-dissections.csv'), 'w') as csvfile:
        formatscsv = csv.writer(csvfile)
        revmsg = {l2m: l5m for l5m, l2m in specimens.messagePool.items()}  # get L5 messages for the L2 in tformats
        formatscsv.writerows([(revmsg[l2m].data.hex(), f) for l2m, f in comparator.dissections.items()])

    # write graph to svg
    plot.savefig(os.path.join(reportFolder, 'FormatMatchMetrics.svg'))

    return reportFolder

def printResultMatrix(matchprecision):
    """
    Prettyprint the content of a dict of symbols and their quality parameters.

    :param matchprecision: a list of tuples of symbols/formats and their
       (qualitymetric, fieldcount, exactf, nearf, uospecific)
    :type matchprecision: list of tuples: [ (netzob.Symbol.name, formatname, float, int, int, int, int) ]
    :rtype: None
    """

    # qualmatrix = [[] for i in range(0,6)]
    qualmatrix = [ [""], ["Quality"], ["Under-/Over-Specified"],
                   ["Near Field Matches"], ["Exact Field Matches"], ["Field Count"] ]
    for symbol, mformat, quality, fieldcount, exactf, nearf, uospecific in matchprecision:
        qualmatrix[0].append(symbol + "/" + mformat)
        qualmatrix[1].append(round(quality,3))
        qualmatrix[2].append(uospecific)
        qualmatrix[3].append(nearf)
        qualmatrix[4].append(exactf)
        qualmatrix[5].append(fieldcount)

        # print "{:s}: {:f}".format(symbol.name, quality)
        # print("Field Under- (pos.)/Over- (neg.) Specification: {:d}".format(uospecific))
        # print("Near Field Matches: {:d}".format(nearf))
        # print("Exact Field Matches: {:d}".format(exactf))
        # print("Field Count: {:d}".format(fieldcount))

    ml = MatrixList()
    ml.headers = qualmatrix[0]
    ml.extend(qualmatrix[1:])
    print(ml)


def getSimilarityquality(
        formatmatchmetrics: Dict[ Tuple[int, netzob.Symbol], FormatMatchScore ]
        ) -> List[Tuple[int, str, Tuple, float, int, int, int, int]]:
    """
    :param formatmatchmetrics: output of DissectorMatcher.thresymbolListsFMS()
    :return: A flat list of tuples from the input-dict:

        (similaritythreshold, symbol.name, mformat, quality, fieldcount, exactf, nearf, uospecific)
    """
    return [(similaritythreshold, symbol.name, fms.trueFormat, fms.score, fms.inferredCount,
             fms.exactCount, fms.nearCount, fms.specificy)
        for (similaritythreshold, symbol), fms
        in formatmatchmetrics.items()
            ]

def getSpecificity(similarityquality):
    """
    :param similarityquality: Output of getSimilarityquality()
    :return: list of specificity-values in input order.
    """
    return [uospecific
        for similaritythreshold, symbol, mformat, quality, fieldcount, exactf, nearf, uospecific
        in similarityquality
            ]

def getFormatMatchScore(similarityquality):
    """
    :param similarityquality: Output of getSimilarityquality()
    :return: list of quality-values (Format Match Scores) in input order.
    """
    return [quality
        for similaritythreshold, symbol, mformat, quality, fieldcount, exactf, nearf, uospecific
        in similarityquality
            ]    # # Format Match Score

def getSimilarity(similarityquality):
    """
    :param similarityquality: Output of getSimilarityquality()
    :return: list of similarity-parameters in input order.
    """
    return [similaritythreshold
        for similaritythreshold, symbol, mformat, quality, fieldcount, exactf, nearf, uospecific
        in similarityquality
            ]
