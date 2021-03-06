"""
Module encapsulating evaluation parameters and helper functions to validate aspects of the
NEMESYS and NEMETYL approaches.
"""
from typing import TypeVar, Hashable, Sequence, Callable, Iterable
from netzob.all import RawMessage
from itertools import chain
import os, csv, pickle, time

from utils.loader import SpecimenLoader
from validation.dissectorMatcher import MessageComparator
from inference.analyzers import *
from inference.segmentHandler import segmentsFromLabels, bcDeltaGaussMessageSegmentation, \
    refinements, segmentsFixed
from inference.segments import MessageAnalyzer, TypedSegment, MessageSegment, AbstractSegment
from inference.templates import DistanceCalculator, DelegatingDC, MemmapDC, Template

Element = TypeVar('Element')


# available analysis methods
analyses = {
    'bcpnm': BitCongruenceNgramMean,
    # 'bcpnv': BitCongruenceNgramStd,  in branch inference-experiments
    'bc': BitCongruence,
    'bcd': BitCongruenceDelta,
    'bcdg': BitCongruenceDeltaGauss,
    'mbhbv': HorizonBitcongruence,

    'variance': ValueVariance,  # Note: VARIANCE is the inverse of PROGDIFF
    'progdiff': ValueProgressionDelta,
    'progcumudelta': CumulatedProgressionDelta,
    'value': Value,
    'ntropy': EntropyWithinNgrams,
    'stropy': Entropy,  # TODO check applicability of (cosine) distance calculation to this feature
}


# raw nemesys - cft-121 "withoutrefinement"
sigmapertrace = {
    "dhcp_SMIA2011101X_deduped-100.pcap"        : 0.6,
    "dns_ictf2010_deduped-100.pcap"             : 0.6,
    "dns_ictf2010-new-deduped-100.pcap"         : 0.6,
    "nbns_SMIA20111010-one_deduped-100.pcap"    : 1.0,
    "ntp_SMIA-20111010_deduped-100.pcap"        : 1.2,
    "smb_SMIA20111010-one_deduped-100.pcap"     : 0.6,
    "dhcp_SMIA2011101X_deduped-1000.pcap"       : 0.6,
    "dns_ictf2010_deduped-982-1000.pcap"        : 0.6,
    "dns_ictf2010-new-deduped-1000.pcap"        : 1.0,
    "nbns_SMIA20111010-one_deduped-1000.pcap"   : 1.0,
    "ntp_SMIA-20111010_deduped-1000.pcap"       : 1.2,
    "smb_SMIA20111010-one_deduped-1000.pcap"    : 0.6,

    # assumptions derived from first traces
    "dhcp_SMIA2011101X-filtered_maxdiff-100.pcap": 0.6,
    "dns_ictf2010_maxdiff-100.pcap": 0.6,
    "dns_ictf2010-new_maxdiff-100.pcap": 0.6,
    "nbns_SMIA20111010-one_maxdiff-100.pcap": 1.0,
    "ntp_SMIA-20111010_maxdiff-100.pcap": 1.2,
    "smb_SMIA20111010-one-rigid1_maxdiff-100.pcap": 0.6,
    "dhcp_SMIA2011101X-filtered_maxdiff-1000.pcap": 0.6,
    "dns_ictf2010_maxdiff-1000.pcap": 0.6,
    "dns_ictf2010-new_maxdiff-1000.pcap": 0.6,
    "nbns_SMIA20111010-one_maxdiff-1000.pcap": 1.0,
    "ntp_SMIA-20111010_maxdiff-1000.pcap": 1.2,
    "smb_SMIA20111010-one-rigid1_maxdiff-1000.pcap": 0.6,
}

epspertrace = {
    "dhcp_SMIA2011101X_deduped-100.pcap" : 1.8,
    "nbns_SMIA20111010-one_deduped-100.pcap" : 1.8, # or anything between 1.8 - 2.6
    "smb_SMIA20111010-one_deduped-100.pcap" : 1.6,
#    "dns_ictf2010_deduped-100.pcap" : 1.0,  # None/Anything, "nothing" to cluster
    "ntp_SMIA-20111010_deduped-100.pcap" : 1.5,
    "dhcp_SMIA2011101X_deduped-1000.pcap": 2.4,
    "nbns_SMIA20111010-one_deduped-1000.pcap": 2.4, # or anything between 1.0 - 2.8
    "smb_SMIA20111010-one_deduped-1000.pcap": 2.2,
    "dns_ictf2010_deduped-982-1000.pcap" : 2.4, # or anything between 1.6 - 2.8
    "ntp_SMIA-20111010_deduped-1000.pcap": 2.8
}

message_epspertrace = {
    "dhcp_SMIA2011101X_deduped-100.pcap" : 0.16, # tested only on 1000s
    "nbns_SMIA20111010-one_deduped-100.pcap" : 0.08, # tested only on 1000s
    "smb_SMIA20111010-one_deduped-100.pcap" : 0.2, # tested only on 1000s
    "dns_ictf2010_deduped-100.pcap" : 0.06,   # tested only on 1000s
    "ntp_SMIA-20111010_deduped-100.pcap" : 0.18,  # tested only on 1000s
    "dhcp_SMIA2011101X_deduped-1000.pcap": 0.16, # similar to 0.14
    "nbns_SMIA20111010-one_deduped-1000.pcap": 0.08, # identical to 0.07
    "smb_SMIA20111010-one_deduped-1000.pcap": 0.2,
    "dns_ictf2010_deduped-982-1000.pcap" : 0.06, # or 0.1
    "ntp_SMIA-20111010_deduped-1000.pcap": 0.18, # identical to 0.2
    "dhcp_SMIA2011101X_deduped-10000.pcap": 0.16,  # tested only on 1000s
    "nbns_SMIA20111010-one_deduped-10000.pcap": 0.08,  # tested only on 1000s
    "smb_SMIA20111010-one_deduped-10000.pcap": 0.2,  # tested only on 1000s
    "dns_ictf2010_deduped-9911-10000.pcap": 0.06,  # tested only on 1000s
    "ntp_SMIA-20111010_deduped-9995-10000.pcap": 0.18,  # tested only on 1000s
}

epsdefault = 2.4

reportFolder = "reports"
cacheFolder = "cache"
clStatsFile = os.path.join(reportFolder, 'messagetype-cluster-statistics.csv')
ccStatsFile = os.path.join(reportFolder, 'messagetype-combined-cluster-statistics.csv')
scStatsFile = os.path.join(reportFolder, 'segment-cluster-statistics.csv')
coStatsFile = os.path.join(reportFolder, 'segment-collective-cluster-statistics.csv')


unknown = "[unknown]"


def annotateFieldTypes(analyzerType: type, analysisArgs: Union[Tuple, None], comparator,
                       unit=MessageAnalyzer.U_BYTE) -> List[Tuple[TypedSegment]]:
    """
    :return: list of lists of segments that are annotated with their field type.
    """
    segmentedMessages = [segmentsFromLabels(
        MessageAnalyzer.findExistingAnalysis(analyzerType, unit,
                                             l4msg, analysisArgs), comparator.dissections[rmsg])
        for l4msg, rmsg in comparator.messages.items()]
    return segmentedMessages


def writeIndividualMessageClusteringStaticstics(
        clusters: Dict[Hashable, List[Tuple[MessageSegment]]], groundtruth: Dict[RawMessage, str],
        runtitle: str, comparator: MessageComparator):
    """
    calculate conciseness, correctness = precision, and recall

    """
    abstrMsgClusters = {lab : [comparator.messages[element[0].message] for element in segseq]
                        for lab, segseq in clusters.items()}
    return writeIndividualClusterStatistics(abstrMsgClusters, groundtruth, runtitle, comparator)


def writeIndividualClusterStatistics(
        clusters: Dict[Hashable, List[Element]], groundtruth: Dict[Element, str],
        runtitle: str, comparator: MessageComparator):
    # clusters: clusterlabel : List of Segments (not Templates!)
    # groundtruth: Lookup for Segment : true type string
    from collections import Counter

    outfile = clStatsFile if isinstance(next(iter(groundtruth.keys())), AbstractMessage) else scStatsFile
    print('Write {} cluster statistics to {}...'.format(
        "message" if isinstance(next(iter(groundtruth.keys())), AbstractMessage) else "segment",
        outfile))

    numSegs = 0
    prList = []
    noise = None
    if 'Noise' in clusters:
        noisekey = 'Noise'
    elif -1 in clusters:
        noisekey = -1
    else:
        noisekey = None

    if noisekey:
        prList.append(None)
        noise = clusters[noisekey]
        clusters = {k: v for k, v in clusters.items() if k != noisekey}  # remove the noise

    numClusters = len(clusters)
    numTypesOverall = Counter(groundtruth.values())
    numTypes = len(numTypesOverall)
    conciseness = numClusters / numTypes

    for label, cluster in clusters.items():
        # we assume correct Tuples of MessageSegments with all objects in one Tuple originating from the same message
        typeFrequency = Counter([groundtruth[element] for element in cluster])
        mostFreqentType, numMFTinCluster = typeFrequency.most_common(1)[0]
        numSegsinCuster = len(cluster)
        numSegs += numSegsinCuster

        precision = numMFTinCluster / numSegsinCuster
        recall = numMFTinCluster / numTypesOverall[mostFreqentType]

        prList.append((label, mostFreqentType, precision, recall, numSegsinCuster))

    # noise statistics
    if noise:
        numNoise = len(noise)
        numSegs += numNoise
        ratioNoise = numNoise / numSegs
        noiseTypes = {groundtruth[element] for element in noise}

    csvWriteHead = False if os.path.exists(outfile) else True
    with open(outfile, 'a') as csvfile:
        clStatscsv = csv.writer(csvfile)  # type: csv.writer
        if csvWriteHead:
            # in "pagetitle": "seg_length", "analysis", "dist_measure", 'min_cluster_size'
            clStatscsv.writerow([
                'run_title', 'trace', 'conciseness', 'cluster_label', 'most_freq_type', 'precision', 'recall', 'cluster_size'])
        if noise:
            # noinspection PyUnboundLocalVariable
            clStatscsv.writerow([
                runtitle, comparator.specimens.pcapFileName, conciseness, 'NOISE', str(noiseTypes), 'ratio:', ratioNoise, numNoise])
        clStatscsv.writerows([
            (runtitle, comparator.specimens.pcapFileName, conciseness, *pr) for pr in prList if pr is not None
        ])

    return prList, conciseness


def writeCollectiveClusteringStaticstics(
        clusters: Dict[Hashable, List[Element]], groundtruth: Dict[Element, str],
        runtitle: str, comparator: MessageComparator, ignoreUnknown=True):
    """
    Precision and recall for the whole clustering interpreted as number of draws from pairs of messages.

    For details see: https://nlp.stanford.edu/IR-book/html/htmledition/evaluation-of-clustering-1.html
    How to calculate the draws is calculated for the Rand index in the document.

    Writes a CSV with tp, fp, fn, tn, pr, rc
        for (a) all clusters and for (b) clusters that have a size of at least 1/40 of the number of samples/messages.

    'total segs' and 'unique segs' are including 'unknown' and 'noise'
    """
    from collections import Counter
    from itertools import combinations, chain
    from scipy.special import binom

    outfile = ccStatsFile if isinstance(next(iter(groundtruth.keys())), AbstractMessage) else coStatsFile
    print('Write message cluster statistics to {}...'.format(outfile))

    segTotal = sum(
        sum(len(el.baseSegments) if isinstance(el, Template) else 1 for el in cl)
        for cl in clusters.values() )
    segUniqu = sum(len(cl) for cl in clusters.values())

    if ignoreUnknown:
        unknownKeys = ["[unknown]", "[mixed]"]
        numUnknown = len([gt for gt in groundtruth.values() if gt in unknownKeys])
        clustersTemp = {lab: [el for el in clu if groundtruth[el] not in unknownKeys] for lab, clu in clusters.items()}
        clusters = {lab: elist for lab, elist in clustersTemp.items() if len(elist) > 0}
        groundtruth = {sg: gt for sg,gt in groundtruth.items() if gt not in unknownKeys}
    else:
        numUnknown = "n/a"

    noise = []
    noisekey = 'Noise' if 'Noise' in clusters else -1 if -1 in clusters else None
    if noisekey is not None:
        noise = clusters[noisekey]
        clusters = {k: v for k, v in clusters.items() if k != noisekey}  # remove the noise

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

    tpfp = sum(binom(len(c), 2) for c in clusters.values())
    tp = sum(binom(t,2) for c in typeFrequencies for t in c.values())
    tnfn = sum(map(lambda n: n[0] * n[1], combinations(
        (len(c) for c in chain.from_iterable([clusters.values(), [noise]])), 2))) + \
           sum(binom(noiseTypes[typeName],2) for typeName, typeTotal in noiseTypes.items())
    # import IPython; IPython.embed()
    # fn = sum(((typeTotal - typeCluster[typeName]) * typeCluster[typeName]
    #           for typeCluster in typeFrequencies + [noiseTypes]
    #           for typeName, typeTotal in numTypesOverall.items() if typeName in typeCluster))//2
    #
    # # noise handling: consider all elements in noise as false negatives
    fn = sum(((typeTotal - typeCluster[typeName]) * typeCluster[typeName]
              for typeCluster in typeFrequencies
              for typeName, typeTotal in numTypesOverall.items() if typeName in typeCluster))//2 + \
         sum((binom(noiseTypes[typeName],2) +
              (
                      (typeTotal - noiseTypes[typeName]) * noiseTypes[typeName]
              )//2
              for typeName, typeTotal in numTypesOverall.items() if typeName in noiseTypes))


    # precision = tp / (tp + fp)
    precision = tp / tpfp
    recall = tp / (tp + fn)

    head = [ 'run_title', 'trace', 'true positives', 'false positives', 'false negatives', 'true negatives',
             'precision', 'recall', 'noise', 'unknown', 'total segs', 'unique segs']
    row = [ runtitle, comparator.specimens.pcapFileName, tp, tpfp-tp, fn, tnfn-fn,
            precision, recall, len(noise), numUnknown, segTotal, segUniqu ]

    csvWriteHead = False if os.path.exists(outfile) else True
    with open(outfile, 'a') as csvfile:
        clStatscsv = csv.writer(csvfile)  # type: csv.writer
        if csvWriteHead:
            clStatscsv.writerow(head)
        clStatscsv.writerow(row)


def writeCollectiveMessageClusteringStaticstics(
        messageClusters: Dict[Hashable, List[Tuple[MessageSegment]]], groundtruth: Dict[RawMessage, str],
        runtitle: str, comparator: MessageComparator):
    abstrMsgClusters = {lab : [comparator.messages[element[0].message] for element in segseq]
                        for lab, segseq in messageClusters.items()}
    return writeCollectiveClusteringStaticstics(abstrMsgClusters, groundtruth, runtitle, comparator)


def plotMultiSegmentLines(segmentGroups: List[Tuple[str, List[Tuple[str, TypedSegment]]]],
                          specimens: SpecimenLoader, pagetitle=None, colorPerLabel=False,
                          typeDict: Dict[str, List[MessageSegment]] = None,
                          isInteractive=False):
    """
    This is a not awfully important helper function saving the writing of a few lines code.

    :param segmentGroups:
    :param specimens:
    :param pagetitle:
    :param colorPerLabel:
    :param typeDict: dict of types (str-keys: list of segments) present in the segmentGroups
    :param isInteractive:
    :return:
    """
    from visualization.multiPlotter import MultiMessagePlotter

    mmp = MultiMessagePlotter(specimens, pagetitle, len(segmentGroups), isInteractive=isInteractive)
    mmp.plotMultiSegmentLines(segmentGroups, colorPerLabel)

    # TODO redundant code of writeIndividualMessageClusteringStaticstics
    if typeDict:  # calculate conciseness, correctness = precision, and recall
        import os, csv
        from collections import Counter
        from inference.templates import Template

        # mapping from each segment in typeDict to the corresponding cluster and true type,
        # considering representative templates
        segment2type = {seg: ft for ft, segs in typeDict.items() for seg in segs}
        clusters = list()
        for label, segList in segmentGroups:
            cluster = list()
            for tl, seg in segList:
                if isinstance(seg, Template):
                    cluster.extend((tl, bs) for bs in seg.baseSegments)
                else:
                    cluster.append((tl, seg))
            clusters.append(cluster)

        numSegs = len(segment2type)
        prList = []
        noise = None
        if 'Noise' in segmentGroups[0][0]:
            noise, *clusters = clusters  # remove the noise
            prList.append(None)

        numClusters = len(clusters)
        numFtypes = len(typeDict)
        conciseness = numClusters / numFtypes

        for clusterSegs in clusters:
            # type from typeDict
            typeKey, numMFTinCluster = Counter(segment2type[seg] for tl, seg in clusterSegs).most_common(1)[0]
            # number of segments for the prevalent type in the trace
            numMFToverall = len(typeDict[typeKey])
            numSegsinCuster = len(clusterSegs)

            precision = numMFTinCluster / numSegsinCuster
            recall = numMFTinCluster / numMFToverall

            # # rather do not repeat the amount in the label
            # mostFrequentType = "{}: {} Seg.s".format(typeKey, numMFTinCluster)
            mostFrequentType = typeKey
            prList.append((mostFrequentType, precision, recall, numSegsinCuster))

        mmp.textInEachAx(["precision = {:.2f}\n"  # correctness
                          "recall = {:.2f}".format(pr[1], pr[2]) if pr else None for pr in prList])

        # noise statistics
        if noise:
            numNoise = len(noise)
            ratioNoise = numNoise / numSegs
            noiseTypes = {ft for ft, seg in noise}


        csvWriteHead = False if os.path.exists(scStatsFile) else True
        with open(scStatsFile, 'a') as csvfile:
            clStatscsv = csv.writer(csvfile)  # type: csv.writer
            if csvWriteHead:
                # in "pagetitle": "seg_length", "analysis", "dist_measure", 'min_cluster_size'
                clStatscsv.writerow(['run_title', 'trace', 'conciseness', 'most_freq_type',
                                     'precision', 'recall', 'cluster_size'])
            if noise:
                # noinspection PyUnboundLocalVariable
                clStatscsv.writerow([pagetitle, specimens.pcapFileName, conciseness, 'NOISE',
                                     str(noiseTypes), ratioNoise, numNoise])
            clStatscsv.writerows([
                (pagetitle, specimens.pcapFileName, conciseness, *pr) for pr in prList if pr is not None
            ])

    mmp.writeOrShowFigure()
    del mmp


def labelForSegment(segGrpHier: List[Tuple[str, List[Tuple[str, List[Tuple[str, TypedSegment]]]]]],
                    seg: AbstractSegment) -> Union[str, bool]:
    """
    Determine group label of an segment from deep hierarchy of segment clusters/groups.

    A more advanced variant of `numpy.array([seg.fieldtype for seg in tg.segments])`

    see #segments2clusteredTypes()

    :param segGrpHier: Hierarchy of segment groups
    :param seg: The segment to label
    :return: The label of the cluster that the seg is member of
    """
    for name, grp in segGrpHier[0][1]:
        if seg in (s for t, s in grp):
            return name.split(", ", 2)[-1]

    if isinstance(seg, Template):
        inGroup = None
        for bs in seg.baseSegments:
            for name, grp in segGrpHier[0][1]:
                if bs in (s for t, s in grp):
                    if inGroup is None or inGroup == name:
                        inGroup = name
                    else:
                        return "[mixed]"
        if inGroup is not None:
            return inGroup.split(", ", 2)[-1]
        else:
            return "[unknown]"

    return False


def writePerformanceStatistics(specimens, clusterer, algos,
                               segmentationTime, dist_calc_segmentsTime, dist_calc_messagesTime,
                               cluster_params_autoconfTime, cluster_messagesTime, align_messagesTime):
    fileNameS = "NEMETYL-performance-statistcs"
    csvpath = os.path.join(reportFolder, fileNameS + '.csv')
    csvWriteHead = False if os.path.exists(csvpath) else True

    print('Write performance statistcs to {}...'.format(csvpath))
    with open(csvpath, 'a') as csvfile:
        statisticscsv = csv.writer(csvfile)
        if csvWriteHead:
            statisticscsv.writerow([
                'script', 'pcap', 'parameters', 'algos',
                'segmentation', 'dist-calc segments', 'dist-calc messages',
                'cluster-params autoconf', 'cluster messages', 'align messages'
            ])
        from sklearn.cluster import DBSCAN
        # noinspection PyUnresolvedReferences,PyPackageRequirements
        import __main__ as main
        statisticscsv.writerow([
            os.path.basename(main.__file__), os.path.basename(specimens.pcapFileName),
            "{} eps {:.3f} ms {}".format(type(clusterer).__name__, clusterer.eps, clusterer.min_samples)
                if isinstance(clusterer, DBSCAN)
                else "{} mcs {} ms {}".format(type(clusterer).__name__, clusterer.min_cluster_size, clusterer.min_samples),
            algos,
            segmentationTime, dist_calc_segmentsTime, dist_calc_messagesTime, cluster_params_autoconfTime,
            cluster_messagesTime, align_messagesTime
            ])


def printClusterMergeConditions(clunuAB, alignedFieldClasses, matchingConditions, dc, diff=True):
    from inference.templates import Template
    from tabulate import tabulate

    cluTable = [(clunu, *[fv.bytes.hex() if isinstance(fv, MessageSegment) else
                          fv.bytes.decode() if isinstance(fv, Template) else fv for fv in fvals])
                for clunu, fvals in zip(clunuAB, alignedFieldClasses[clunuAB])]
    cluTable.extend(zip(*matchingConditions[clunuAB]))

    # distance to medoid for DYN-STA mixes
    # DYN-STA / STA-DYN : medoid to static distance
    dynstamixdists = [ # afcB.distToNearest(afcA, dc)
        segA.distToNearest(segB, dc)
                 if isinstance(segA, Template) and isinstance(segB, MessageSegment)
             else segB.distToNearest(segA, dc)
                 if isinstance(segB, Template) and isinstance(segA, MessageSegment)
             else None
             for segA, segB in zip(*alignedFieldClasses[clunuAB])]
    # STA-STA : static distance
    stastamixdists = [
        dc.pairDistance(segA, segB)
                 if isinstance(segA, MessageSegment) and isinstance(segB, MessageSegment)
             else None
             for segA, segB in zip(*alignedFieldClasses[clunuAB])]
    # DYN-STA / STA-DYN : medoid to static distance compared to maxDistToMedoid in DYN : good if > -.1 ?
    medstamixdists = [
        (segA.maxDistToMedoid(dc) + (1-segA.maxDistToMedoid(dc))*.33 - dc.pairDistance(segA.medoid, segB))
                 if isinstance(segA, Template) and isinstance(segB, MessageSegment)
             else (segB.maxDistToMedoid(dc) + (1-segB.maxDistToMedoid(dc))*.33 - dc.pairDistance(segB.medoid, segA))
                 if isinstance(segB, Template) and isinstance(segA, MessageSegment)
             else None
             for segA, segB in zip(*alignedFieldClasses[clunuAB])]

    # TODO clean up test code
    medstamixdistsblu = [
        segA.maxDistToMedoid(dc)
                 if isinstance(segA, Template) and isinstance(segB, MessageSegment)
             else segB.maxDistToMedoid(dc)
                 if isinstance(segB, Template) and isinstance(segA, MessageSegment)
             else None
             for segA, segB in zip(*alignedFieldClasses[clunuAB])]
    medstamixdistsbbb = [
        dc.pairDistance(segA.medoid, segB)
                 if isinstance(segA, Template) and isinstance(segB, MessageSegment)
             else dc.pairDistance(segB.medoid, segA)
                 if isinstance(segB, Template) and isinstance(segA, MessageSegment)
             else None
             for segA, segB in zip(*alignedFieldClasses[clunuAB])]

    cluTable += [tuple(["DSdist"] + ["{:.3f}".format(val) if isinstance(val, float) else val for val in dynstamixdists])]
    cluTable += [tuple(["SSdist"] + ["{:.3f}".format(val) if isinstance(val, float) else val for val in stastamixdists])]
    cluTable += [tuple(["MSdist"] + ["{:.3f}".format(val) if isinstance(val, float) else val for val in medstamixdists])]
    # TODO clean up test code
    # cluTable += [
    #     tuple(["MSdist"] + ["{:.3f}".format(val) if isinstance(val, float) else val for val in medstamixdistsblu])]
    # cluTable += [
    #     tuple(["MSdist"] + ["{:.3f}".format(val) if isinstance(val, float) else val for val in medstamixdistsbbb])]

    fNums = []
    for fNum, (cluA, cluB) in enumerate(zip(cluTable[0], cluTable[1])):
        if not cluA == cluB:
            fNums.append(fNum)
    if diff:  # only field diff
        cluDiff = [[col for colNum, col in enumerate(line) if colNum in fNums] for line in cluTable]
        print(tabulate(cluDiff, headers=fNums, disable_numparse=True))
    else:
        # complete field table
        print(tabulate(cluTable, disable_numparse=True))
    print()


def searchSeqOfSeg(sequence: Sequence[Union[MessageSegment, Sequence[MessageSegment]]], pattern: bytes):
    assert isinstance(pattern, bytes)

    if isinstance(sequence[0], Sequence):
        return [msg for msg in sequence if any(pattern in seg.bytes for seg in msg if isinstance(seg, MessageSegment))]
    else:
        return [seg for seg in sequence if pattern in seg.bytes]


def calcHexDist(hexA, hexB):
    from netzob.Model.Vocabulary.Messages.RawMessage import RawMessage
    from inference.analyzers import Value
    from inference.segments import MessageSegment
    from inference.templates import DistanceCalculator

    bytedata = [bytes.fromhex(hexA),bytes.fromhex(hexB)]
    messages = [RawMessage(bd) for bd in bytedata]
    analyzers = [Value(message) for message in messages]
    segments = [MessageSegment(analyzer, 0, len(analyzer.message.data)) for analyzer in analyzers]
    dc = DistanceCalculator(segments)
    return dc.pairDistance(*segments)


def cacheAndLoadDC(pcapfilename: str, analysisTitle: str, tokenizer: str, debug: bool,
                   analyzerType: type, analysisArgs: Tuple=None, sigma: float=None, filterTrivial=False,
                   refinementCallback:Union[Callable, None] = refinements,
                   disableCache=False) \
        -> Tuple[SpecimenLoader, MessageComparator, List[Tuple[MessageSegment]], DistanceCalculator,
        float, float]:
    """
    cache or load the DistanceCalculator to or from the filesystem


    :param filterTrivial: Filter out **one-byte** segments and such just consisting of **zeros**.
    :param disableCache: When experimenting with distances manipulation, deactivate caching!
    :return:
    """
    pcapbasename = os.path.basename(pcapfilename)
    # if refinementCallback == pcaMocoRefinements:
    #     sigma = pcamocoSigmapertrace[pcapbasename] if not sigma and pcapbasename in pcamocoSigmapertrace else \
    #         0.9 if not sigma else sigma
    # else:
    sigma = sigmapertrace[pcapbasename] if not sigma and pcapbasename in sigmapertrace else \
        0.9 if not sigma else sigma
    pcapName = os.path.splitext(pcapbasename)[0]
    # noinspection PyUnboundLocalVariable
    tokenparm = tokenizer if tokenizer != "nemesys" else \
        "{}{:.0f}".format(tokenizer, sigma * 10)
    dccachefn = os.path.join(cacheFolder, 'cache-dc-{}-{}-{}-{}-{}.{}'.format(
        analysisTitle, tokenparm, "filtered" if filterTrivial else "all",
        refinementCallback.__name__ if refinementCallback is not None else "raw",
        pcapName, 'ddc'))
    # dccachefn = 'cache-dc-{}-{}-{}.{}'.format(analysisTitle, tokenizer, pcapName, 'dc')
    if disableCache or not os.path.exists(dccachefn):
        # dissect and label messages
        print("Load messages from {}...".format(pcapName))
        specimens = SpecimenLoader(pcapfilename, 2, True)
        comparator = MessageComparator(specimens, 2, True, debug=debug)

        print("Segmenting messages...", end=' ')
        segmentationTime = time.time()
        # select tokenizer by command line parameter
        if tokenizer == "tshark":
            # 1. segment messages according to true fields from the labels
            segmentedMessages = annotateFieldTypes(analyzerType, analysisArgs, comparator)
        elif tokenizer == "4bytesfixed":
            # 2. segment messages into fixed size chunks for testing
            segmentedMessages = segmentsFixed(4, comparator, analyzerType, analysisArgs)
        elif tokenizer == "nemesys":
            # 3. segment messages by NEMESYS
            segmentsPerMsg = bcDeltaGaussMessageSegmentation(specimens, sigma)

            # get analyzer requested by analyzerType/analysisArgs
            segmentedMessages = [[
                MessageSegment(MessageAnalyzer.findExistingAnalysis(
                    analyzerType, MessageAnalyzer.U_BYTE, seg.message, analysisArgs), seg.offset, seg.length)
                for seg in msg] for msg in segmentsPerMsg]

            if refinementCallback is not None:
                if refinementCallback.__code__.co_argcount > 1:
                    # assume the second argument is expected to be a distance calculator
                    chainedSegments = list(chain.from_iterable(segmentedMessages))
                    print("Refinement: Calculate distance for {} segments...".format(len(chainedSegments)))
                    if len(chainedSegments)**2 > MemmapDC.maxMemMatrix:
                        refinementDC = MemmapDC(chainedSegments)
                    else:
                        refinementDC = DelegatingDC(chainedSegments)
                    segmentedMessages = refinementCallback(segmentedMessages, refinementDC)
                else:
                    segmentedMessages = refinementCallback(segmentedMessages)

            # segments = list(chain.from_iterable(segmentedMessages))

        segmentationTime = time.time() - segmentationTime
        print("done.")

        if filterTrivial:
            # noinspection PyUnboundLocalVariable
            chainedSegments = [seg for seg in chain.from_iterable(segmentedMessages) if
                        seg.length > 1 and set(seg.values) != {0}]
        else:
            # noinspection PyUnboundLocalVariable
            chainedSegments = list(chain.from_iterable(segmentedMessages))

        print("Calculate distance for {} segments...".format(len(chainedSegments)))
        # dc = DistanceCalculator(chainedSegments, reliefFactor=0.33)  # Pairwise similarity of segments: dc.distanceMatrix
        dist_calc_segmentsTime = time.time()
        if len(chainedSegments) ** 2 > MemmapDC.maxMemMatrix:
            dc = MemmapDC(chainedSegments)
        else:
            dc = DelegatingDC(chainedSegments)
        assert chainedSegments == dc.rawSegments
        dist_calc_segmentsTime = time.time() - dist_calc_segmentsTime
        try:
            with open(dccachefn, 'wb') as f:
                pickle.dump((segmentedMessages, comparator, dc), f, pickle.HIGHEST_PROTOCOL)
        except MemoryError as e:
            print("DC could not be cached due to a MemoryError. Removing", dccachefn, "and continuing.")
            os.remove(dccachefn)
    else:
        print("Load distances from cache file {}".format(dccachefn))
        with open(dccachefn, 'rb') as f:
            segmentedMessages, comparator, dc = pickle.load(f)
        if not (isinstance(comparator, MessageComparator)
                and isinstance(dc, DistanceCalculator)):
            print('Loading of cached distances failed.')
            exit(10)
        specimens = comparator.specimens
        # chainedSegments = list(chain.from_iterable(segmentedMessages))
        segmentationTime, dist_calc_segmentsTime = None, None

    return specimens, comparator, segmentedMessages, dc, segmentationTime, dist_calc_segmentsTime


def resolveTemplates2Segments(segments: Iterable[AbstractSegment]):
    """
    Resolve a (mixed) list of segments and templates into a list of single segments.
    :param segments: (mixed) list of segments and templates
    :return: list of single segments with all given segments and the base segments of the templates.
    """
    resolvedSegments = list()
    for seg in segments:
        if isinstance(seg, Template):
            resolvedSegments.extend(seg.baseSegments)
        else:
            resolvedSegments.append(seg)
    return resolvedSegments



