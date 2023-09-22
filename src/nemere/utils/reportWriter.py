"""
Write Format Match Score report for a list of analysed messages.
"""

import os
import csv
from abc import ABC, abstractmethod

import IPython
import numpy
from typing import Dict, Tuple, Iterable, TypeVar, Hashable, List, Union, Any, Sequence
from os.path import isdir, splitext, basename, join, exists
from itertools import chain
from collections import Counter, defaultdict, OrderedDict

from netzob.Model.Vocabulary.Messages.AbstractMessage import AbstractMessage
from netzob.Model.Vocabulary.Symbol import Symbol

from nemere.inference.segments import AbstractSegment, TypedSegment, MessageSegment
from nemere.inference.templates import Template, TypedTemplate, FieldTypeTemplate
from nemere.utils.evaluationHelpers import StartupFilecheck, reportFolder, segIsTyped, unknown
from nemere.utils.loader import SpecimenLoader
from nemere.validation.clusterInspector import SegmentClusterCauldron
from nemere.validation.dissectorMatcher import FormatMatchScore, MessageComparator, BaseDissectorMatcher
from nemere.visualization.simplePrint import FieldtypeComparingPrinter, ComparingPrinter


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
    :return: Count of exact matches, off-by-one near matches, off-by-more-than-one matches
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
                runtime: float, comparator: MessageComparator,
                inferenceTitle: str, folder="reports", withTitle=False):
    """
    Write report of the message segmentation quality measured in FMS to files.

    :param formatmatchmetrics: FMS for each message.
    :param runtime: Runtime of the analysis.
    :param comparator: Comparator to relate the inference to the ground truth.
    :param inferenceTitle: The title to use in the report to refer to the analysis run.
    :param folder: Folder to store the report to.
    :param withTitle: Flag to use the inferenceTitle in the file names of the report.
    """

    if not isdir(folder):
        raise NotADirectoryError("The reports folder {} is not a directory. Reports cannot be written there.".format(
            folder))
    print('Write report to ' + folder)

    # write Format Match Score and Metrics to csv
    fn = f'FormatMatchMetrics_{inferenceTitle}.csv' if withTitle else 'FormatMatchMetrics.csv'
    with open(os.path.join(folder, fn), 'w') as csvfile:
        fmmcsv = csv.writer(csvfile)
        fmmcsv.writerow(["Message", "Score", 'I', 'M', 'N', 'S', 'MG', 'SP'])
        fmmcsv.writerows( [
            (message.data.hex(), fms.score,
             fms.inferredCount, fms.exactCount, fms.nearCount, fms.specificy, fms.matchGain, fms.specificyPenalty)
            for message, fms in formatmatchmetrics.items()] )

    scoreStats = calcScoreStats([q.score for q in formatmatchmetrics.values()])
    matchCounts = countMatches(formatmatchmetrics.values())

    fn = os.path.join(folder, 'ScoreStatistics.csv')
    writeHeader = not exists(fn)
    with open(fn, 'a') as csvfile:
        fmmcsv = csv.writer(csvfile)
        if writeHeader:
            fmmcsv.writerow(["inference", "min", "mean", "max", "median", "std",
                             "exactcount", "offbyonecount", "offbymorecount", "runtime"])
        fmmcsv.writerow( [ inferenceTitle,
                           *scoreStats, *matchCounts,
                           runtime] )

    # write Symbols to csvs
    if all(isinstance(quality.symbol, Symbol) or quality.symbol is None for quality in formatmatchmetrics.values()):
        multipleSymbolCSVs = False  # Has no effect if not using Netzob Symbols
        if multipleSymbolCSVs:
            for cnt, symbol in enumerate(  # by the set comprehension,
                    { quality.symbol  # remove identical symbols due to multiple formats
                    for quality
                    in formatmatchmetrics.values() } ):
                fileNameS = 'Symbol_{:s}_{:d}'.format(symbol.name, cnt)
                with open(os.path.join(folder, fileNameS + '.csv'), 'w') as csvfile:
                    symbolcsv = csv.writer(csvfile)
                    symbolcsv.writerow([field.name for field in symbol.fields])
                    symbolcsv.writerows([val.hex() for val in msg] for msg in symbol.getCells())
        else:
            fileNameS = f'Symbols_{inferenceTitle}' if withTitle else 'Symbols'
            with open(os.path.join(folder, fileNameS + '.csv'), 'w') as csvfile:
                symbolcsv = csv.writer(csvfile)
                msgcells = chain.from_iterable([sym.getCells() for sym in  # unique symbols by set
                    {fms.symbol for fms in formatmatchmetrics.values()} if sym is not None])
                symbolcsv.writerows(
                    [val.hex() for val in msg] for msg in msgcells
                )
    elif all(isinstance(quality.symbol, Sequence) for quality in formatmatchmetrics.values()):
        if all(isinstance(segment, int) for quality in formatmatchmetrics.values() for segment in quality.symbol):
            # in case we have inferred field bounds (0, ..., n-1) in symbol
            fileNameS = f'Symbols_{inferenceTitle}' if withTitle else 'Symbols'
            with open(os.path.join(folder, fileNameS + '.csv'), 'w') as csvfile:
                symbolcsv = csv.writer(csvfile)
                msgcells = [ [fms.message.data[offstart:offend].hex() for offstart,offend
                              in zip(fms.symbol[:-1], fms.symbol[0])] for fms in formatmatchmetrics.values()]
                symbolcsv.writerows(msgcells)
        elif all(isinstance(segment, MessageSegment) for quality
                 in formatmatchmetrics.values() for segment in quality.symbol):
            # in case we have segment objects in symbol
            fileNameS = f'Symbols_{inferenceTitle}' if withTitle else 'Symbols'
            with open(os.path.join(folder, fileNameS + '.csv'), 'w') as csvfile:
                symbolcsv = csv.writer(csvfile)
                msgcells = [[seg.bytes.hex() for seg in fms.symbol] for fms in formatmatchmetrics.values()]
                symbolcsv.writerows(msgcells)
        else:
            print("Inconsistent inference information for reporting. Omitting symbol writing.")
    else:
        print("No Symbols of inference information found for reporting. Omitting symbol writing.")

    # # write tshark-dissection to csv
    # # currently only unique formats. For a specific trace a baseline could be determined
    # # by a one time run of per ParsedMessage
    # with open(os.path.join(reportFolder, 'tshark-dissections.csv'), 'w') as csvfile:
    #     formatscsv = csv.writer(csvfile)
    #     revmsg = {l2m: l5m for l5m, l2m in specimens.messagePool.items()}  # get L5 messages for the L2 in tformats
    #     formatscsv.writerows([(revmsg[l2m].data.hex(), f) for l2m, f in tformats.items()])

    # FMS : Symbol (we just need one example for each score, so we do not care overwriting multiple identical ones)
    score2symbol = {fms.score: fms.symbol for fms in formatmatchmetrics.values() if fms.symbol is not None}
    if scoreStats[1] in score2symbol.keys():
        # symsMinMeanMax = [score2symbol[mmm] for mmm in scoreStats[:3]]
        # add some context symbols
        scoreSorted = sorted(score2symbol.keys())
        meanI = scoreSorted.index(scoreStats[1])
        scoreSelect = scoreSorted[-3:][::-1] + scoreSorted[meanI:meanI+3][::-1] + scoreSorted[:3][::-1]
        symsMinMeanMax = [score2symbol[mmm] for mmm in scoreSelect]
        tikzcode = None
        if all(isinstance(mmm, Symbol) for mmm in symsMinMeanMax):
            tikzcode = "FMS values: " + ", ".join(f"{fms:0.2f}" for fms in scoreSelect) + "\n\n"
            tikzcode += comparator.tprintInterleaved(symsMinMeanMax)
        elif all(isinstance(mmm, Sequence) for mmm in symsMinMeanMax) \
                and all(isinstance(seg, MessageSegment) for mmm in symsMinMeanMax for seg in mmm):
            tikzcode = "FMS values: " + ", ".join(f"{fms:0.2f}" for fms in scoreSelect) + "\n\n"
            cprinter = ComparingPrinter(comparator, symsMinMeanMax)
            tikzcode += cprinter.toTikz()
        if tikzcode:
            # write Format Match Score and Metrics to csv
            fn = f'example-inference-minmeanmax_{inferenceTitle}.tikz' if withTitle else 'example-inference-minmeanmax.tikz'
            with open(join(folder, fn), 'w') as tikzfile:
                tikzfile.write(tikzcode)
        else:
            print("Inconsistent inference information. Cannot print examples.")
    else:
        print("No symbols in FMS given. Cannot print examples.")


def writeSegmentedMessages2CSV(segmentsPerMsg: Sequence[Sequence[MessageSegment]], folder="reports"):
    """
    Write the given segmentation into a CSV file.

    :param segmentsPerMsg: List of messages as a list of segments
    :param folder: Folder to store the report to.
    """
    import csv
    fileNameS = 'SegmentedMessages'
    with open(os.path.join(folder, fileNameS + '.csv'), 'w') as csvfile:
        symbolcsv = csv.writer(csvfile)
        symbolcsv.writerows(
            [seg.bytes.hex() for seg in msg] for msg in segmentsPerMsg
        )


def writeFieldTypesTikz(comparator: MessageComparator, segmentedMessages: List[Tuple[MessageSegment]],
                        fTypeTemplates: List[FieldTypeTemplate], filechecker: StartupFilecheck):
    """
    Visualization of segments from clusters in messages as tikz file.
    see also nemere/visualization/simplePrint.py

    :param comparator: Comparator to relate the inference to the ground truth.
    :param segmentedMessages: List of messages as a list of segments.
    :param fTypeTemplates: Field type templates of the inferred fields.
    :param filechecker: Filechecker to determine a suitable folder to write the file to.
    """
    # select the messages to print by quality: three of around fmsmin, fmsmean, fmsmax each
    fmslist = [BaseDissectorMatcher(comparator, msg).calcFMS() for msg in segmentedMessages]
    fmsdict = {fms.score: fms for fms in fmslist}  # type: Dict[float, FormatMatchScore]
    scoreSorted = sorted(fmsdict.keys())
    fmsmin, fmsmean, fmsmax, fmsmedian, fmsstd = calcScoreStats(scoreSorted)
    meanI = scoreSorted.index(fmsmean)
    scoreSelect = scoreSorted[-3:] + scoreSorted[meanI - 1:meanI + 2] + scoreSorted[:3]
    symsMinMeanMax = [fmsdict[mmm].message for mmm in scoreSelect]

    # visualization of segments from clusters in messages.
    cp = FieldtypeComparingPrinter(comparator, fTypeTemplates)
    tikzcode = cp.fieldtypes(symsMinMeanMax)
    # write tikz code to file
    with open(join(filechecker.reportFullPath, 'example-messages-fieldtypes.tikz'), 'w') as tikzfile:
        tikzfile.write(tikzcode)


def writeSemanticTypeHypotheses(cauldron: SegmentClusterCauldron, filechecker: StartupFilecheck):
    """
    Write report with the semantic type hypothesis as CSV.

    :param cauldron: Clusters of of the inferred fields.
    :param filechecker: Filechecker to determine a suitable folder to write the file to.
    """
    cauldron.regularClusters.shapeStats(filechecker)
    semanticHeaders = ["regCluI", "Cluster Label", "semantic type"]
    semanticHypotheses = cauldron.regularClusters.semanticTypeHypotheses()
    # print(tabulate([(i, cauldron.regularClusters.clusterLabel(i), h) for i, h in
    #                 semanticHypotheses.items()], headers=semanticHeaders))
    reportFile = join(filechecker.reportFullPath, "semanticTypeHypotheses-" + filechecker.pcapstrippedname + ".csv")
    print("Write semantic type hypotheses to", reportFile)
    with open(reportFile, 'a') as csvfile:
        statisticscsv = csv.writer(csvfile)
        statisticscsv.writerow(semanticHeaders)
        statisticscsv.writerows([( i, cauldron.regularClusters.clusterLabel(i), h )
                                 for i, h in semanticHypotheses.items()])


Element = TypeVar('Element', AbstractMessage, AbstractSegment)
class Report(ABC):
    """
    A base class for writing reports of various inference aspects into files.
    """
    statsFile = "statistics"

    def __init__(self, groundtruth, pcap: Union[str, StartupFilecheck], reportPath: str=None):
        """
        :param groundtruth: Lookup for Segment : true type string
        :param pcap: Reference to the PCAP file to report for.
        """
        self.groundtruth = groundtruth
        self.pcap = pcap
        self.reportPath = reportPath if reportPath is not None else reportFolder
        if not isdir(self.reportPath):
            raise FileNotFoundError(f"The report folder {self.reportPath} needs to exist. It does not. Aborting.")
        self.runtitle = None

    @abstractmethod
    def write(self, inference, runtitle: Union[str, Dict]):
        """To be implemented by a subclass."""
        raise NotImplementedError()

class ClusteringReport(Report, ABC):
    """
    Calculate conciseness, correctness = precision, and recall for the given clusters compared to some groundtruth.
    Applicable to clusters of AbstractMessage or AbstractSegment elements.
    """
    messagetypeStatsFile = None
    segmenttypeStatsFile = None

    @abstractmethod
    def write(self, clusters: Dict[Hashable, List[Element]], runtitle: Union[str, Dict]):
        """
        :param clusters: clusterlabel : List of Segments (not Templates!)
        :param runtitle: Label to identify the inference run with, e. g.
            "{}-{}-eps={:.2f}-min_samples={}-split".format(tokenizer,
            type(clusterer).__name__, clusterer.eps, clusterer.min_samples)
        """
        raise NotImplementedError()

    @abstractmethod
    def _writeCSV(self, runtitle: Union[str, Dict]):
        """
        :param runtitle: Label to identify the inference run with, e. g.
            "{}-{}-eps={:.2f}-min_samples={}-split".format(tokenizer,
            type(clusterer).__name__, clusterer.eps, clusterer.min_samples)
        """
        raise NotImplementedError()
        # TODO in implementing subclasses: add sigma, refinement type,
        # TODO split runtitle into columns for tokenizer, type(clusterer).__name__, clusterer.eps, clusterer.min_samples

    def _printMessage(self, outfile: str):
        """Print a user notification about whats happening."""
        print('Write {} cluster statistics to {}...'.format(
            "message" if self.statsFile == type(self).messagetypeStatsFile else "segment",
            outfile))

    @staticmethod
    def inferenceColumns(inferenceParams: Dict[str, str]):
        """
        :param inferenceParams: Parameters of the inference to be included in each line of the report.
        :return: List of additional columns in the report about the subject inference.
        """
        infCols = OrderedDict()
        infCols["tokenrefine"] = inferenceParams["tokenizer"]
        if inferenceParams["tokenParams"] is not None: infCols["tokenrefine"] += "-" + inferenceParams["tokenParams"]
        if inferenceParams["refinement"] is not None: infCols["tokenrefine"] += "-" + inferenceParams["refinement"]
        infCols["clustering"] = inferenceParams["clusterer"] + "-" +  inferenceParams["clusterParams"]
        infCols["postProcess"] = inferenceParams["postProcess"] if inferenceParams["postProcess"] is not None else ""
        return infCols

class IndividualClusterReport(ClusteringReport):
    """from writeIndividualClusterStatistics"""
    messagetypeStatsFile = "messagetype-cluster-statistics"
    segmenttypeStatsFile = "segment-cluster-statistics"

    def __init__(self, groundtruth: Dict[Element, str], pcap: Union[str, StartupFilecheck]):
        super().__init__(groundtruth, pcap)
        # set filename for CSV depending on element type (message or segment)
        ClusteringReport.statsFile = IndividualClusterReport.messagetypeStatsFile \
            if isinstance(next(iter(groundtruth.keys())), AbstractMessage) \
            else IndividualClusterReport.segmenttypeStatsFile
        self.hasNoise = False
        self.noiseTypes, self.ratioNoise, self.numNoise = [None] * 3
        self.conciseness, self.precisionRecallList = [None] * 2
        self._additionalColumns = OrderedDict()  # type: Dict[str, Dict[Hashable, Any]]

    def addColumn(self, colData: Dict[Hashable, Any], header: str):
        """add data to a new column. colData contains the cluster label (or "Noise") to determine the row.
        The order of the columns in the table is the same as they were added here."""
        self._additionalColumns[header] = colData

    def write(self, clusters: Dict[Hashable, List[Element]], runtitle: Union[str, Dict]):
        """
        Write the report with the individual cluster statistics for the given clusters to the file system.

        :param clusters: Clusters to generate the report for.
        :param runtitle: Title by which this analysis can be identified.
        """
        numSegs = 0
        prList = []

        # handle noise
        noise = None
        noisekey = 'Noise' if 'Noise' in clusters else -1 if -1 in clusters else None
        if noisekey:
            self.hasNoise = True
            prList.append(None)
            noise = clusters[noisekey]
            clusters = {k: v for k, v in clusters.items() if k != noisekey}  # remove the noise

        # cluster statistics
        numClusters = len(clusters)
        numTypesOverall = Counter(self.groundtruth.values())
        numTypes = len(numTypesOverall)
        self.conciseness = numClusters / numTypes
        for label, cluster in clusters.items():
            # we assume correct Tuples of MessageSegments with all objects in one Tuple originating from the same message
            typeFrequency = Counter([self.groundtruth[element] for element in cluster])
            mostFreqentType, numMFTinCluster = typeFrequency.most_common(1)[0]
            numSegsinCuster = len(cluster)
            numSegs += numSegsinCuster

            precision = numMFTinCluster / numSegsinCuster
            recall = numMFTinCluster / numTypesOverall[mostFreqentType]

            prList.append((label, mostFreqentType, precision, recall, numSegsinCuster))
        self.precisionRecallList = prList

        # noise statistics
        if noise:
            self.numNoise = len(noise)
            numSegs += self.numNoise
            self.ratioNoise = self.numNoise / numSegs
            self.noiseTypes = {self.groundtruth[element] for element in noise}

        self._writeCSV(runtitle)

    def _writeCSV(self, runtitle: Union[str, Dict]):
        """Add the report to the appropriate CSV. Appends rows, if the CSV already exists."""
        outfile = join(self.reportPath, self.statsFile + ".csv")
        self._printMessage(outfile)

        headers = [ 'trace', 'conciseness', 'cluster_label', 'most_freq_type', 'precision', 'recall', 'cluster_size' ]
        if not isinstance(runtitle, str):
            infCols = IndividualClusterReport.inferenceColumns(runtitle)
            headers = list(infCols.keys()) + headers
            infParams = list(infCols.values())
        else:
            headers = ['run_title'] + headers
            infParams = [runtitle]
        headers += list(self._additionalColumns.keys())

        csvWriteHead = False if os.path.exists(outfile) else True
        with open(outfile, 'a') as csvfile:
            clStatscsv = csv.writer(csvfile)  # type: csv.writer
            if csvWriteHead:
                # in "pagetitle": "seg_length", "analysis", "dist_measure", 'min_cluster_size'
                clStatscsv.writerow(headers)
            if self.hasNoise:
                additionalCells = [colData.get("Noise", "") for colData in self._additionalColumns.values()]

                # noinspection PyUnboundLocalVariable
                clStatscsv.writerow([
                    *infParams,
                    self.pcap.pcapstrippedname if isinstance(self.pcap, StartupFilecheck) else self.pcap,
                    self.conciseness,
                    'NOISE', str(self.noiseTypes),
                    'ratio:', self.ratioNoise,
                    self.numNoise] + additionalCells)
            clStatscsv.writerows([
                [*infParams,
                 self.pcap.pcapstrippedname if isinstance(self.pcap, StartupFilecheck) else self.pcap,
                 self.conciseness, *pr]
                 + [colData.get(pr[0], "") for colData in self._additionalColumns.values()]  # additional columns
                 for pr in self.precisionRecallList if pr is not None
            ])

class CombinatorialClustersReport(ClusteringReport):
    """from writeCollectiveClusteringStaticstics"""
    messagetypeStatsFile = "messagetype-combined-cluster-statistics"
    segmenttypeStatsFile = "segment-combined-cluster-statistics"

    def __init__(self, groundtruth: Dict[Element, str], pcap: Union[str, StartupFilecheck]):
        super().__init__(groundtruth, pcap)
        # set filename for CSV depending on element type (message or segment)
        ClusteringReport.statsFile = CombinatorialClustersReport.messagetypeStatsFile \
            if isinstance(next(iter(groundtruth.keys())), AbstractMessage) \
            else CombinatorialClustersReport.segmenttypeStatsFile
        self.tp, self.tpfp, self.fn, self.tnfn, self.fn = [None] * 5
        self.numNoise, self.numUnknown, self.segTotal, self.segUniqu = [None] * 4

    @property
    def precision(self):
        # return self.tp / (self.tp + self.fp)
        if self.tp == 0:
            return 0
        return self.tp / self.tpfp

    @property
    def recall(self):
        return self.tp / (self.tp + self.fn)

    def write(self, clusters: Dict[Hashable, List[Element]], runtitle: Union[str, Dict], ignoreUnknown=True):
        """
        Write the report with the individual cluster statistics for the given clusters to the file system.

        Precision and recall for the whole clustering interpreted as number of draws from pairs of messages.

        For details see: https://nlp.stanford.edu/IR-book/html/htmledition/evaluation-of-clustering-1.html
        How to calculate the draws is calculated for the Rand index in the document.

        Writes a CSV with tp, fp, fn, tn, pr, rc
            for (a) all clusters and for (b) clusters that have a size of at least 1/40 of the number of samples/messages.

        'total segs' and 'unique segs' are including 'unknown' and 'noise'

        :param clusters: Clusters to generate the report for.
        :param runtitle: Title by which this analysis can be identified.
        """
        from collections import Counter
        from itertools import combinations, chain
        from scipy.special import binom

        self.segTotal = sum(
            sum(len(el.baseSegments) if isinstance(el, Template) else 1 for el in cl)
            for cl in clusters.values())
        self.segUniqu = sum(len(cl) for cl in clusters.values())

        if ignoreUnknown:
            unknownKeys = [unknown, "[mixed]"]
            self.numUnknown = len([gt for gt in self.groundtruth.values() if gt in unknownKeys])
            clustersTemp = {lab: [el for el in clu if self.groundtruth[el] not in unknownKeys] for lab, clu in
                            clusters.items()}
            clusters = {lab: elist for lab, elist in clustersTemp.items() if len(elist) > 0}
            groundtruth = {sg: gt for sg, gt in self.groundtruth.items() if gt not in unknownKeys}
        else:
            groundtruth = self.groundtruth
            self.numUnknown = "n/a"

        noise = []
        noisekey = 'Noise' if 'Noise' in clusters else -1 if -1 in clusters else None
        # print("noisekey", noisekey)
        if noisekey is not None:
            noise = clusters[noisekey]
            clusters = {k: v for k, v in clusters.items() if k != noisekey}  # remove the noise
        self.numNoise = len(noise)

        """
        # # # # # # # #
        # test case
        >>> groundtruth = {
        >>>     "x0": "x", "x1": "x", "x2": "x", "x3": "x", "x4": "x", "x5": "x", "x6": "x", "x7": "x",
        >>>     "o0": "o", "o1": "o", "o2": "o", "o3": "o", "o4": "o",
        >>>     "#0": "#", "#1": "#", "#2": "#", "#3": "#"
        >>> }
        >>> clusters = { "A": ["x1", "x2", "x3", "x4", "x5", "o1"],
        >>>              "B": ["x6", "o2", "o3", "o4", "o0", "#1"],
        >>>              "C": ["x7", "x0", "#2", "#3", "#0"],
        >>>              }
        >>> typeFrequencies = [Counter([groundtruth[element] for element in c])
                                 for c in clusters.values()]
        # # # # # # # #
        """

        # numTypesOverall = Counter(groundtruth[comparator.messages[element[0].message]]
        #                           for c in clusters.values() for element in c)
        numTypesOverall = Counter(groundtruth.values())
        # number of types per cluster
        typeFrequencies = [Counter([groundtruth[element] for element in c])
                           for c in clusters.values()]
        noiseTypes = Counter([groundtruth[element] for element in noise])

        self.tpfp = sum(binom(len(c), 2) for c in clusters.values())
        self.tp = sum(binom(t, 2) for c in typeFrequencies for t in c.values())
        self.tnfn = sum(map(lambda n: n[0] * n[1], combinations(
            (len(c) for c in chain.from_iterable([clusters.values(), [noise]])), 2))) + \
               sum(binom(noiseTypes[typeName], 2) for typeName, typeTotal in noiseTypes.items())
        # import IPython; IPython.embed()
        # fn = sum(((typeTotal - typeCluster[typeName]) * typeCluster[typeName]
        #           for typeCluster in typeFrequencies + [noiseTypes]
        #           for typeName, typeTotal in numTypesOverall.items() if typeName in typeCluster))//2
        #
        # # noise handling: consider all elements in noise as false negatives
        self.fn = sum(((typeTotal - typeCluster[typeName]) * typeCluster[typeName]
                  for typeCluster in typeFrequencies
                  for typeName, typeTotal in numTypesOverall.items() if typeName in typeCluster)) // 2 + \
             sum((binom(noiseTypes[typeName], 2) +
                  (
                          (typeTotal - noiseTypes[typeName]) * noiseTypes[typeName]
                  ) // 2
                  for typeName, typeTotal in numTypesOverall.items() if typeName in noiseTypes))

        self._writeCSV(runtitle)

    def _writeCSV(self, runtitle: Union[str, Dict]):
        """Add the report to the appropriate CSV. Appends rows, if the CSV already exists."""
        outfile = join(self.reportPath, self.statsFile + ".csv")
        self._printMessage(outfile)

        headers = [ 'trace', 'true positives', 'false positives', 'false negatives', 'true negatives',
                'precision', 'recall', 'noise', 'unknown', 'total segs', 'unique segs' ]
        if not isinstance(runtitle, str):
            infCols = IndividualClusterReport.inferenceColumns(runtitle)
            headers = list(infCols.keys()) + headers
            infParams = list(infCols.values())
        else:
            headers = ['run_title'] + headers
            infParams = [runtitle]

        row = [*infParams,
               self.pcap.pcapstrippedname if isinstance(self.pcap, StartupFilecheck) else self.pcap,
               self.tp,
               self.tpfp - self.tp,
               self.fn,
               self.tnfn - self.fn,
               self.precision,
               self.recall,
               self.numNoise,
               self.numUnknown,
               self.segTotal,
               self.segUniqu]

        csvWriteHead = False if os.path.exists(outfile) else True
        with open(outfile, 'a') as csvfile:
            clStatscsv = csv.writer(csvfile)  # type: csv.writer
            if csvWriteHead:
                clStatscsv.writerow(headers)
            clStatscsv.writerow(row)


def plotMultiSegmentLines(segmentGroups: List[Tuple[str, List[Tuple[str, TypedSegment]]]],
                          specimens: SpecimenLoader, pagetitle=None, colorPerLabel=False,
                          typeDict: Dict[str, List[MessageSegment]] = None,
                          isInteractive=False):
    """
    This is a not awfully important helper function saving the writing of a few lines of code.

    :param segmentGroups: Groups of clusters of segments that should be plotted.
    :param specimens: Specimen object to link the plot to the source trace.
    :param pagetitle: Title to include in the plot name.
    :param colorPerLabel: Flag to select whether segments should be colored accorrding to their label.
    :param typeDict: dict of types (str-keys: list of segments) present in the segmentGroups
    :param isInteractive: Use a interactive windows or write to file.
    """
    from nemere.visualization.multiPlotter import MultiMessagePlotter

    mmp = MultiMessagePlotter(specimens, pagetitle, len(segmentGroups), isInteractive=isInteractive)
    mmp.plotMultiSegmentLines(segmentGroups, colorPerLabel)

    # TODO Think about replacing this implicit writing of the report CSV to an explicit one by the caller, then, accept
    #   the IndividualClusterReport instance as parameter to retrieve the precision and recall values for the plot.
    if typeDict:
        # mapping from each segment in typeDict to the corresponding cluster and true type,
        # considering representative templates
        groundtruth = {seg: ft for ft, segs in typeDict.items() for seg in segs}
        clusters = defaultdict(list)
        for label, segList in segmentGroups:
            for tl, seg in segList:
                if isinstance(seg, Template):
                    clusters[label].extend(seg.baseSegments)
                else:
                    clusters[label].append(seg)

        # calculate conciseness, correctness = precision, and recall
        report = IndividualClusterReport(groundtruth, splitext(basename(specimens.pcapFileName))[0])
        report.write(clusters, pagetitle)

        mmp.textInEachAx(["precision = {:.2f}\n"  # correctness
                          "recall = {:.2f}".format(pr[2], pr[3]) if pr else None for pr in report.precisionRecallList])

    mmp.writeOrShowFigure()
    del mmp


class SegmentClusterReport(ClusteringReport):
    """Clustered elements report for field type clustering with available ground truth to compare to."""
    statsFile = "segmentclusters"

    def __init__(self, pcap: Union[str, StartupFilecheck], reportPath: str=None):
        super().__init__(None, pcap, reportPath)

    def write(self, clusters: Dict[str, List[Union[MessageSegment, Template]]], runtitle: Union[str, Dict]=None):
        """
        Write the report with the cluster statistics for the given clusters to the file system.

        :param clusters: Clusters to generate the report for.
        :param runtitle: Title by which this analysis can be identified.
        """
        self._writeCSV(clusters, runtitle)

    def _writeCSV(self, clusters: Dict[str, List[Union[MessageSegment, Template]]], runtitle: Union[str, Dict]=None):
        outfile = self._buildOutFilename(runtitle)
        self._printMessage(outfile)

        with open(outfile, "a") as segfile:
            segcsv = csv.writer(segfile)
            segcsv.writerow(["Cluster", "Hex", "Bytes", "occurrence"])
            for cLabel, segments in clusters.items():  # type: Tuple[str, Union[MessageSegment, Template]]
                segcsv.writerows({
                    (cLabel, seg.bytes.hex(), seg.bytes, 1 if not isinstance(seg, Template) else len(seg.baseSegments))
                    for seg in segments
                })

    def _printMessage(self, outfile: str):
        """Print a user notification about whats happening."""
        wora = "Append" if os.path.exists(outfile) else "Write"
        print(f'{wora} field type cluster elements to {outfile}...')

    def _buildOutFilename(self, runtitle: Union[str, Dict]=None):
        return join(self.reportPath, self.statsFile + (
            "-" + runtitle if runtitle is not None else ""
        ) + "-" + (
            self.pcap.pcapstrippedname if isinstance(self.pcap, StartupFilecheck) else self.pcap
        ) + ".csv")

class SegmentClusterGroundtruthReport(SegmentClusterReport):
    """Clustered elements report for field type clustering with available ground truth to compare to."""
    statsFile = "segmentclusters"

    def __init__(self, comparator: MessageComparator, segments: List[AbstractSegment],
                 pcap: Union[str, StartupFilecheck], reportPath: str=None):
        """

        :param comparator: The comparator providing the ground truth
        :param segments: List of segments to write statistics for
        :param pcap: The filename or StartupFilecheck object pointing to the pcap
        :param reportPath: If None, automatically determine a path in the report folder using pcap.reportFullPath
            if available else the globally defined reportFolder
        """
        self._comparator = comparator
        self._segments = segments
        self._typedMatchSegs, self._typedMatchTemplates = self._matchSegments()
        reportPath = reportPath if reportPath is not None else pcap.reportFullPath \
            if isinstance(pcap, StartupFilecheck) else reportFolder
        super().__init__(pcap, reportPath)
        self.groundtruth = {rawSeg: typSeg[1].fieldtype if typSeg[0] > 0.5 else unknown
                            for rawSeg, typSeg in self.typedMatchTemplates.items()}

    def write(self, clusters: Dict[str, Union[MessageSegment, Template]], runtitle: Union[str, Dict]=None):
        """
        Write the report with the cluster statistics for the given clusters to the file system.

        :param clusters: Clusters to generate the report for.
        :param runtitle: Title by which this analysis can be identified.
        """
        self._writeCSV(clusters, runtitle)

    def _writeCSV(self, clusters: Dict[str, Union[MessageSegment, Template]], runtitle: Union[str, Dict]=None):
        """Add the report to the appropriate CSV. Appends rows, if the CSV already exists."""
        outfile = self._buildOutFilename(runtitle)
        self._printMessage(outfile)

        typedMatchTemplates = self.typedMatchTemplates  # type: Dict[Union[Template, MessageSegment], Tuple[float, Union[TypedSegment, TypedTemplate, Template, MessageSegment]]]

        with open(outfile, "a") as segfile:
            segcsv = csv.writer(segfile)
            segcsv.writerow(["Cluster", "Hex", "Bytes", "occurrence", "Data Type", "Overlap",
                             "| Field Name (Example)", "rStart", "rEnd"])  # r means "relative to the inferred segment"
            for cLabel, segments in clusters.items():  # type: Tuple[str, Union[MessageSegment, Template]]
                # if dc.segments != clusterer.segments:
                #     # Templates resolved to single Segments
                #     segcsv.writerows({(cLabel, seg.bytes.hex(), seg.bytes,
                #     typedMatchSegs[seg][1].fieldtype, typedMatchSegs[seg][0])
                #                       for seg in segments})
                # else:
                # Templates as is
                segcsv.writerows({
                    (
                        cLabel, seg.bytes.hex(), seg.bytes,
                        len(seg.baseSegments) if isinstance(seg, Template) else 1,
                        typedMatchTemplates[seg][1].fieldtype if segIsTyped(typedMatchTemplates[seg][1]) else unknown,
                        typedMatchTemplates[seg][0],
                        self._comparator.lookupField(
                            typedMatchTemplates[seg][1].baseSegments[0] if isinstance(typedMatchTemplates[seg][1],
                                                                                      Template)
                            else typedMatchTemplates[seg][1])[1],
                        *self.relativeOffsets(seg)
                    ) for seg in segments
                })

    @staticmethod
    def segIsTyped(someSegment):
        return segIsTyped(someSegment)

    def relativeOffsets(self, infSegment):
        """(Matched templates have offsets and lengths identical to seg (inferred) and not the true one.)"""
        infSegment = infSegment.baseSegments[0] if isinstance(infSegment, Template) else infSegment
        overlapRatio, overlapIndex, overlapStart, overlapEnd = self._comparator.fieldOverlap(infSegment)
        return infSegment.offset - overlapStart, infSegment.nextOffset - overlapEnd

    def _matchSegments(self):
        # mark segment matches with > 50% overlap with the prevalent true data type for the nearest boundaries.
        # list of tuples of overlap ratio ("intensity of match") and segment
        typedMatchSegs = dict()  # type: Dict[Union[Template, MessageSegment], Tuple[float, Union[TypedSegment, MessageSegment]]]
        typedMatchTemplates = dict()  # type: Dict[Union[Template, MessageSegment], Tuple[float, Union[TypedSegment, TypedTemplate, Template, MessageSegment]]]
        for seg in self._segments:
            # create typed segments/templates per cluster to get the inferred assignment
            if isinstance(seg, MessageSegment):
                typedMatchSegs[seg] = self._comparator.segment2typed(seg)
                typedMatchTemplates[seg] = self._comparator.segment2typed(seg)
            elif isinstance(seg, Template):
                typedBaseSegments = [self._comparator.segment2typed(bs) for bs in seg.baseSegments]
                typedMatchSegs.update({bs: ts for bs, ts in zip(seg.baseSegments, typedBaseSegments)})
                if any(not isinstance(baseS, TypedSegment) for ratio, baseS in typedBaseSegments):
                    typedMatchTemplates[seg] = (-1.0, seg)
                    # we have no info about this segment's gt
                    continue

                typeRatios = defaultdict(list)
                for ratio, baseS in typedBaseSegments:
                    typeRatios[baseS.fieldtype].append(ratio)
                # currently we need this only if there is only one matching type, but "for future use" calc all means.
                meanRatios = {ft: numpy.mean(ro) for ft, ro in typeRatios.items()}
                ftkeys = sorted(typeRatios.keys(), key=lambda x: -meanRatios[x])
                machingType = ftkeys[0]

                if len(typeRatios) > 1:
                    # print("Segment's matching field types are not the same in template, e. g., "
                    #   "{} and {} ({})".format( machingType, tempTyped.fieldtype, tempTyped.bytes.hex() ))
                    typedMatchTemplates[seg] = (0.0, seg)
                else:
                    typedMatchTemplates[seg] = (float(meanRatios[machingType]),
                                                TypedTemplate(seg.values, [ts for _, ts in typedBaseSegments],
                                                              seg._method))
        return typedMatchSegs, typedMatchTemplates

    @property
    def typedMatchSegs(self):
        return self._typedMatchSegs

    @property
    def typedMatchTemplates(self):
        return self._typedMatchTemplates

