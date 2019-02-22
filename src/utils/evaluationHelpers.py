"""
Module encapsulating evaluation parameters and helper functions to validate aspects of the
NEMESYS and NEMETYL approaches.
"""
from typing import Union, Tuple, List
from utils.loader import SpecimenLoader

from inference.analyzers import *
from inference.segmentHandler import segmentsFromLabels
from inference.segments import MessageAnalyzer, TypedSegment




# available analysis methods
from visualization.multiPlotter import MultiMessagePlotter

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

epsdefault = 2.4




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


def plotMultiSegmentLines(segmentGroups: List[Tuple[str, List[Tuple[str, TypedSegment]]]],
                          specimens: SpecimenLoader, pagetitle=None, colorPerLabel=False, typeDict = None,
                          isInteractive=False):
    """
    This is a not awfully important helper function saving the writing of a few lines code.

    :param segmentGroups:
    :param pagetitle:
    :param colorPerLabel:
    :param typeDict: dict of types (str-keys: list of segments) present in the segmentGroups
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


