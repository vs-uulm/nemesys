"""
Write Format Match Score report for a list of analysed messages.
"""

import os
import time
import csv
import numpy
from typing import Dict, Tuple, Iterable
from os.path import abspath, isdir, splitext, basename, join
from itertools import chain

from netzob.Model.Vocabulary.Messages.AbstractMessage import AbstractMessage

from nemere.utils.loader import SpecimenLoader
from nemere.validation.dissectorMatcher import FormatMatchScore, MessageComparator


def calcScoreStats(scores: Iterable[float]) -> Tuple[float, float, float, float, float]:
    """
    :param scores: An Iterable of FMS values.
    :return: min, meankey, max, mediankey, standard deviation of the scores,
        where meankey is the value in scores closest to the mean of its values,
        and median is the value in scores closest to the mean of its values.
    """
    scores = sorted(scores)
    fmsmin, fmsmean, fmsmax, fmsmedian, fmsstd = \
        numpy.min(scores), numpy.mean(scores), numpy.max(scores), numpy.median(scores), numpy.std(scores)
    # get quality key closest to mean
    fmsmeankey = 1
    if len(scores) > 2:
        for b,a in zip(scores[:-1], scores[1:]):
            if a < fmsmean:
                continue
            fmsmeankey = b if fmsmean - b < a - fmsmean else a
            break
    return float(fmsmin), float(fmsmeankey), float(fmsmax), float(fmsmedian), float(fmsstd)


def getMinMeanMaxFMS(scores: Iterable[float]) -> Tuple[float, float, float]:
    """
    :param scores: An Iterable of FMS values.
    :return: min, meankey, and max of the scores,
        where meankey is the value in scores closest to the means of its values.
    """
    return calcScoreStats(scores)[:3]


def countMatches(quality: Iterable[FormatMatchScore]):
    """
    :param quality: List of FormatMatchScores
    :return: count of exact matches, off-by-one near matches, off-by-more-than-one matches
    """
    exactcount = 0
    offbyonecount = 0
    offbymorecount = 0
    for fms in quality:  # type: FormatMatchScore
        exactcount += fms.exactCount
        offbyonecount += sum(1 for truf, inff in fms.nearMatches.items() if abs(truf - inff) == 1)
        offbymorecount += sum(1 for truf, inff in fms.nearMatches.items() if abs(truf - inff) > 1)
    return exactcount, offbyonecount, offbymorecount


def writeReport(formatmatchmetrics: Dict[AbstractMessage, FormatMatchScore],
                runtime: float,
                specimens: SpecimenLoader, comparator: MessageComparator,
                inferenceTitle: str, folder="reports"):

    absFolder = abspath(folder)
    if not isdir(absFolder):
        raise NotADirectoryError("The reports folder {} is not a directory. Reports cannot be written there.".format(
            absFolder))
    pcapName = splitext(basename(specimens.pcapFileName))[0]
    reportFolder = join(absFolder, pcapName + "_{}_{}".format(
        inferenceTitle, time.strftime("%Y%m%d-%H%M%S", time.localtime())))
    os.makedirs(reportFolder)

    print('Write report to ' + reportFolder)

    # write Format Match Score and Metrics to csv
    with open(os.path.join(reportFolder, 'FormatMatchMetrics.csv'), 'w') as csvfile:
        fmmcsv = csv.writer(csvfile)
        fmmcsv.writerow(["Message", "Score", 'I', 'M', 'N', 'S', 'MG', 'SP'])
        fmmcsv.writerows( [
            (message.data.hex(), fms.score,
             fms.inferredCount, fms.exactCount, fms.nearCount, fms.specificy, fms.matchGain, fms.specificyPenalty)
            for message, fms in formatmatchmetrics.items()] )

    scoreStats = calcScoreStats([q.score for q in formatmatchmetrics.values()])
    matchCounts = countMatches(formatmatchmetrics.values())

    with open(os.path.join(reportFolder, 'ScoreStatistics.csv'), 'w') as csvfile:
        fmmcsv = csv.writer(csvfile)
        fmmcsv.writerow(["inference", "min", "mean", "max", "median", "std",
                         "exactcount", "offbyonecount", "offbymorecount", "runtime"])
        fmmcsv.writerow( [ inferenceTitle,
                           *scoreStats, *matchCounts,
                           runtime] )

    # write Symbols to csvs
    multipleSymbolCSVs = False
    if multipleSymbolCSVs:
        for cnt, symbol in enumerate(  # by the set comprehension,
                { quality.symbol  # remove identical symbols due to multiple formats
                for quality
                in formatmatchmetrics.values() } ):
            fileNameS = 'Symbol_{:s}_{:d}'.format(symbol.name, cnt)
            with open(os.path.join(reportFolder, fileNameS + '.csv'), 'w') as csvfile:
                symbolcsv = csv.writer(csvfile)
                symbolcsv.writerow([field.name for field in symbol.fields])
                symbolcsv.writerows([val.hex() for val in msg] for msg in symbol.getCells())
    else:
        fileNameS = 'Symbols'
        with open(os.path.join(reportFolder, fileNameS + '.csv'), 'w') as csvfile:
            symbolcsv = csv.writer(csvfile)
            msgcells = chain.from_iterable([sym.getCells() for sym in  # unique symbols by set
                {fms.symbol for fms in formatmatchmetrics.values()}])
            symbolcsv.writerows(
                [val.hex() for val in msg] for msg in msgcells
            )

    # # write tshark-dissection to csv
    # # currently only unique formats. For a specific trace a baseline could be determined
    # # by a one time run of per ParsedMessage
    # with open(os.path.join(reportFolder, 'tshark-dissections.csv'), 'w') as csvfile:
    #     formatscsv = csv.writer(csvfile)
    #     revmsg = {l2m: l5m for l5m, l2m in specimens.messagePool.items()}  # get L5 messages for the L2 in tformats
    #     formatscsv.writerows([(revmsg[l2m].data.hex(), f) for l2m, f in tformats.items()])


    # FMS : Symbol
    score2symbol = {fms.score: fms.symbol for fms in formatmatchmetrics.values()}

    tikzcode = comparator.tprintInterleaved(score2symbol[mmm] for mmm in scoreStats[:3])

    # write Format Match Score and Metrics to csv
    with open(join(reportFolder, 'example-inference-minmeanmax.tikz'), 'w') as tikzfile:
        tikzfile.write(tikzcode)