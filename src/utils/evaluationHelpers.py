"""
Module encapsulating evaluation parameters and helper functions to validate aspects of the
NEMESYS and NEMETYL approaches.
"""
from typing import Union, Tuple, List, TypeVar, Hashable, Sequence
from netzob.all import RawMessage
import os, csv

from utils.loader import SpecimenLoader
from validation.dissectorMatcher import MessageComparator
from inference.analyzers import *
from inference.segmentHandler import segmentsFromLabels
from inference.segments import MessageAnalyzer, TypedSegment, MessageSegment
from visualization.multiPlotter import MultiMessagePlotter



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

sigmapertrace = {
    "dhcp_SMIA2011101X_deduped-100.pcap" : 0.6,
    # "nbns_SMIA20111010-one_deduped-100.pcap" : 1.8,
    "smb_SMIA20111010-one_deduped-100.pcap" : 1.2,
    "dns_ictf2010_deduped-100.pcap" : 0.6,
    "ntp_SMIA-20111010_deduped-100.pcap" : 1.2,
    "dhcp_SMIA2011101X_deduped-1000.pcap": 0.6,
    # "nbns_SMIA20111010-one_deduped-1000.pcap": 2.4,
    "smb_SMIA20111010-one_deduped-1000.pcap": 1.2,
    "dns_ictf2010_deduped-982-1000.pcap" : 0.6,
    "ntp_SMIA-20111010_deduped-1000.pcap": 1.2
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
clStatsFile = os.path.join(reportFolder, 'messagetype-cluster-statistics.csv')




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


# Element = TypeVar('Element')
def writeMessageClusteringStaticstics(
        clusters: Dict[Hashable, List[Tuple[MessageSegment]]], groundtruth: Dict[RawMessage, str],
        runtitle: str, comparator: MessageComparator):
    """
    calculate conciseness, correctness = precision, and recall

    """
    from collections import Counter

    print('Write message cluster statistics to {}...'.format(clStatsFile))

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
        typeFrequency = Counter([groundtruth[comparator.messages[element[0].message]] for element in cluster])
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
        noiseTypes = {groundtruth[comparator.messages[element[0].message]] for element in noise}

    csvWriteHead = False if os.path.exists(clStatsFile) else True
    with open(clStatsFile, 'a') as csvfile:
        clStatscsv = csv.writer(csvfile)  # type: csv.writer
        if csvWriteHead:
            # in "pagetitle": "seg_length", "analysis", "dist_measure", 'min_cluster_size'
            clStatscsv.writerow([
                'run_title', 'trace', 'conciseness', 'cluster_label', 'most_freq_type', 'precision', 'recall', 'cluster_size'])
        if noise:
            # noinspection PyUnboundLocalVariable
            clStatscsv.writerow([
                runtitle, comparator.specimens.pcapFileName, conciseness, 'NOISE', '', str(noiseTypes), ratioNoise, numNoise])
        clStatscsv.writerows([
            (runtitle, comparator.specimens.pcapFileName, conciseness, *pr) for pr in prList if pr is not None
        ])

    return prList, conciseness



def plotMultiSegmentLines(segmentGroups: List[Tuple[str, List[Tuple[str, TypedSegment]]]],
                          specimens: SpecimenLoader, pagetitle=None, colorPerLabel=False, typeDict = None,
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
    mmp = MultiMessagePlotter(specimens, pagetitle, len(segmentGroups), isInteractive=isInteractive)
    mmp.plotMultiSegmentLines(segmentGroups, colorPerLabel)
    numSegs = 0
    if typeDict:  # calculate conciseness, correctness = precision, and recall
        clusters = [segList for label, segList in segmentGroups]
        prList = []
        noise = None
        if 'Noise' in segmentGroups[0][0]:
            noise, *clusters = clusters  # remove the noise
            prList.append(None)

        numClusters = len(clusters)
        numFtypes = len(typeDict)
        conciseness = numClusters / numFtypes

        from collections import Counter
        for clusterSegs in clusters:
            typeFrequency = Counter([ft for ft, seg in clusterSegs])
            mostFreqentType, numMFTinCluster = typeFrequency.most_common(1)[0]
            numMFToverall = len(typeDict[mostFreqentType.split(':', 2)[0]])
            numSegsinCuster = len(clusterSegs)
            numSegs += numSegsinCuster

            precision = numMFTinCluster / numSegsinCuster
            recall = numMFTinCluster / numMFToverall

            prList.append((mostFreqentType, precision, recall, numSegsinCuster))

        mmp.textInEachAx(["precision = {:.2f}\n"  # correctness
                          "recall = {:.2f}".format(pr[1], pr[2]) if pr else None for pr in prList])

        # noise statistics
        if noise:
            numNoise = len(noise)
            numSegs += numNoise
            ratioNoise = numNoise / numSegs
            noiseTypes = {ft for ft, seg in noise}

        import os, csv
        clStatsFile = os.path.join('reports/', 'clusterStatisticsHDBSCAN.csv')
        csvWriteHead = False if os.path.exists(clStatsFile) else True
        with open(clStatsFile, 'a') as csvfile:
            clStatscsv = csv.writer(csvfile)  # type: csv.writer
            if csvWriteHead:
                # in "pagetitle": "seg_length", "analysis", "dist_measure", 'min_cluster_size'
                clStatscsv.writerow(['run_title', 'trace', 'conciseness', 'most_freq_type', 'precision', 'recall', 'cluster_size'])
            if noise:
                # noinspection PyUnboundLocalVariable
                clStatscsv.writerow([pagetitle, specimens.pcapFileName, conciseness, 'NOISE', str(noiseTypes), ratioNoise, numNoise])
            clStatscsv.writerows([
                (pagetitle, specimens.pcapFileName, conciseness, *pr) for pr in prList if pr is not None
            ])

    mmp.writeOrShowFigure()
    del mmp


def labelForSegment(segGrpHier: List[Tuple[str, List[Tuple[str, List[Tuple[str, TypedSegment]]]]]],
                    seg: MessageSegment) -> Union[str, bool]:
    """
    Determine group label of an segment from deep hierarchy of segment clusters/groups.

    see segments2clusteredTypes()

    :param segGrpHier:
    :param seg:
    :return:
    """
    for name, grp in segGrpHier[0][1]:
        if seg in (s for t, s in grp):
            return name.split(", ", 2)[-1]
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


def segmentInfo(comparator: MessageComparator, segment: MessageSegment):
    pm = comparator.parsedMessages[comparator.messages[segment.message]]
    print(pm.messagetype)

    fs = pm.getFieldSequence()
    fsnum = 0
    offset = 0
    while offset < segment.offset:
        offset += fs[fsnum][1]
        fsnum += 1
    print(fs[fsnum][0])
    print(pm.getTypeSequence()[fsnum][0])
    print(segment.bytes)
    print(segment.bytes.hex())


def printClusterMergeConditions(clunuAB, alignedFieldClasses, matchingConditions, dc, diff=True):
    from inference.templates import Template
    from tabulate import tabulate

    cluTable = [(clunu, *[fv.bytes.hex() if isinstance(fv, MessageSegment) else
                          fv.bytes.decode() if isinstance(fv, Template) else fv for fv in fvals])
                for clunu, fvals in zip(clunuAB, alignedFieldClasses[clunuAB])] + \
               list(zip(*matchingConditions[clunuAB]))

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



