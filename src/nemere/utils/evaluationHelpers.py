"""
Module encapsulating evaluation parameters and helper functions to validate aspects of the
NEMESYS and NEMETYL approaches.
"""
from collections import defaultdict, Counter
from typing import TypeVar, Sequence, Callable, Iterable, Optional

from itertools import chain
import os, csv, pickle, time
from os.path import join, splitext, isfile, isdir, basename, exists, abspath

from tabulate import tabulate

from nemere.utils.loader import SpecimenLoader
from nemere.validation.dissectorMatcher import MessageComparator, BaseComparator
from nemere.visualization.simplePrint import inferred4segment, markSegNearMatch
from nemere.inference.analyzers import *
from nemere.inference.formatRefinement import isOverlapping, BlendZeroSlices, CropChars
from nemere.inference.segmentHandler import segmentsFromLabels, bcDeltaGaussMessageSegmentation, refinements, \
    fixedlengthSegmenter, bcDeltaGaussMessageSegmentationLE
from nemere.inference.segments import MessageAnalyzer, TypedSegment, MessageSegment, AbstractSegment
from nemere.inference.templates import DistanceCalculator, DelegatingDC, Template, MemmapDC, TypedTemplate

Element = TypeVar('Element')


# available analysis methods
analyses = {
    'bcpnm': BitCongruenceNgramMean,
    'bc':    BitCongruence,
    'bcg':   BitCongruenceGauss,
    'bcd':   BitCongruenceDelta,
    'bcdg':  BitCongruenceDeltaGauss,
    'hbcg':  HorizonBitcongruenceGauss,
    'sbcdg': SlidingNbcDeltaGauss,
    'pivot': PivotBitCongruence,

    'variance': ValueVariance,  # Note: VARIANCE is the inverse of the removed PROGDIFF (ValueProgressionDelta)
    'value':    Value,
    'ntropy':   EntropyWithinNgrams,
    'stropy':   Entropy,  # TODO check applicability of (cosine) distance calculation to this feature
}


sigmapertrace = {
    "dhcp_SMIA2011101X-filtered_maxdiff-1000.pcap": 0.4,
    "dns_ictf2010-new_maxdiff-1000.pcap": 0.9,
    "nbns_SMIA20111010-one_maxdiff-1000.pcap": 0.4,
    "ntp_SMIA-20111010_maxdiff-1000.pcap": 1.3,
    "smb_SMIA20111010-one-rigid1_maxdiff-1000.pcap": 0.4,

    # assumptions derived from 1000s traces
    "dhcp_SMIA2011101X-filtered_maxdiff-100.pcap": 0.4,
    "dns_ictf2010-new_maxdiff-100.pcap": 0.9,
    "nbns_SMIA20111010-one_maxdiff-100.pcap": 0.4,
    "ntp_SMIA-20111010_maxdiff-100.pcap": 1.3,
    "smb_SMIA20111010-one-rigid1_maxdiff-100.pcap": 0.4,

    "dhcp_SMIA2011101X_deduped-10000.pcap": 0.4,
    "dns_ictf2010-new-deduped-10000.pcap": 0.9,
    "nbns_SMIA20111010-one_deduped-10000.pcap": 0.4,
    "ntp_SMIA-20111010_deduped-9995-10000.pcap": 1.3,
    "smb_SMIA20111010-one_deduped-10000.pcap": 0.4,
}

# FMS based - cft-130 + cft-132
pcamocoSigmapertrace = {
    "dhcp_SMIA2011101X_deduped-100.pcap"        : 0.6,
    "dns_ictf2010_deduped-100.pcap"             : 0.9,
    "dns_ictf2010-new-deduped-100.pcap"         : 1.0,
    "nbns_SMIA20111010-one_deduped-100.pcap"    : 0.8,
    "ntp_SMIA-20111010_deduped-100.pcap"        : 1.2,
    "smb_SMIA20111010-one_deduped-100.pcap"     : 0.9,
    "dhcp_SMIA2011101X_deduped-1000.pcap"       : 0.6,
    "dns_ictf2010_deduped-982-1000.pcap"        : 0.9,
    "dns_ictf2010-new-deduped-1000.pcap"        : 1.0,
    "nbns_SMIA20111010-one_deduped-1000.pcap"   : 0.9,
    "ntp_SMIA-20111010_deduped-1000.pcap"       : 1.2,
    "smb_SMIA20111010-one_deduped-1000.pcap"    : 0.8,
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

def labelForSegment(segGrpHier: List[Tuple[str, List[Tuple[str, List[Tuple[str, TypedSegment]]]]]],
                    seg: AbstractSegment) -> Union[str, None]:
    """
    Determine group label of an segment from deep hierarchy of segment clusters/groups.

    A more advanced variant of `numpy.array([seg.fieldtype for seg in tg.segments])`

    see #segments2clusteredTypes()

    deprecated see cauldron.label4Segment

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
            return unknown

    return None

def consolidateLabels(labels: numpy.ndarray, trigger = "$d_{max}$=0.000", maxLabels=20):
    """
    Replace the labels in the input, in-place, if the trigger is contained in a label.
    If after this procedure still more than maxLabels distinct labels are remaining, only the 20 largest are retained.
    """
    zeroDmaxUnique = {c for c in labels if isinstance(labels, Iterable) and trigger in c}
    zeroDcount = len(zeroDmaxUnique)
    commonLabel = f"one of {zeroDcount} clusters with $d_{{max}}$=0.000"
    for ci in range(labels.size):
        if trigger in labels[ci]:
            labels[ci] = commonLabel
    lCounter = Counter(labels)
    if len(lCounter) > maxLabels:
        print("Still too many cluster labels! Merging the smallest ones, retaining 20 clusters.")
        leastCommon = list(zip(*list(lCounter.most_common())[20:]))[0]
        lcAmount = len(leastCommon)
        if commonLabel in leastCommon:
            lcAmount += zeroDcount
        lcLabel = f"one of the {lcAmount} smallest clusters"
        for ci in range(labels.size):
            if labels[ci] in leastCommon:
                labels[ci] = lcLabel
    return labels


def writePerformanceStatistics(specimens, clusterer, algos,
                               segmentationTime, dist_calc_segmentsTime, dist_calc_messagesTime,
                               cluster_params_autoconfTime, cluster_messagesTime, align_messagesTime):
    fileNameS = "NEMETYL-performance-statistics"
    csvpath = os.path.join(reportFolder, fileNameS + '.csv')
    csvWriteHead = False if os.path.exists(csvpath) else True

    print('Write performance statistics to {}...'.format(csvpath))
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
    from nemere.inference.templates import Template
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
    from nemere.inference.analyzers import Value
    from nemere.inference.segments import MessageSegment
    from nemere.inference.templates import DistanceCalculator

    bytedata = [bytes.fromhex(hexA),bytes.fromhex(hexB)]
    messages = [RawMessage(bd) for bd in bytedata]
    analyzers = [Value(message) for message in messages]
    segments = [MessageSegment(analyzer, 0, len(analyzer.message.data)) for analyzer in analyzers]
    dc = DistanceCalculator(segments)
    return dc.pairDistance(*segments)


class CachedDistances(object):

    # tokenizers to select from
    tokenizers = ('nemesys', 'zeros')  # zeroslices + CropChars

    # refinement methods
    refinementMethods = [
        "none",
        "original",  # WOOT2018 paper
        "base",  # ConsecutiveChars+moco
        "nemetyl",  # INFOCOM2020 paper: ConsecutiveChars+moco+splitfirstseg
        "PCA1",  # PCA 1-pass | applicable to nemesys and zeros
        "PCAmoco",  # PCA+moco
        "zero",  # zeroslices+base
        "zeroPCA",  # zero+base + 2-pass PCA
        "PCAmocoSF",  # PCA+moco+SF (v2) | applicable to zeros
        "zerocharPCAmocoSF",  # with split fixed (v2)
        "emzcPCAmocoSF",  # zerocharPCAmocoSF + entropy based merging
    ]

    def __init__(self, pcapfilename: str, analysisTitle: str, layer=2, relativeToIP=True):
        """
        Cache or load the DistanceCalculator to or from the filesystem

        :param pcapfilename: File name string of the PCAP to load.
        :param analysisTitle: Text string to use as title/label for this analysis.
        :param layer: Protocol layer to extract from the encapsulation. 1 for raw frame (with set relativeToIP = False).
        :param relativeToIP: Interpret the layer relative to the IP layer (True) or absolute (False).
        """
        self.pcapfilename = pcapfilename  # type: str
        self.pcapbasename = os.path.basename(pcapfilename)
        self.pcapName = os.path.splitext(self.pcapbasename)[0]

        self.layer = layer  # type: int
        self.relativeToIP = relativeToIP  # type: bool
        self.analysisTitle = analysisTitle  # type: str
        self.analyzerType = analyses[analysisTitle]  # type: Type[MessageAnalyzer]
        self.analysisArgs = None  # type: Union[None, Tuple]

        self.tokenizer = None  # type: Union[None, str]
        self.sigma = None  # type: Union[None, float]
        self.filter = False  # type: bool
        self.refinementCallback = None  # type: Union[Callable, None]
        self.refinementArgs = None  # type: Union[None, Dict]
        """kwargs for the refinement function, e. g., reportFolder or collectedSubclusters"""
        self.forwardComparator = False  # type: bool

        self.dissectGroundtruth = True
        """Expect tshark to know a dissector for the protocol and initialize a MessageComparator for it. 
        Set to False for an unknown protocol!"""
        self.disableCache = False  # type: bool
        """When experimenting with distances manipulation, deactivate caching by setting disableCache to True!"""
        self.debug = False  # type: bool
        """Set debug flag in MessageComparator."""

        self.dccachefn = None  # type: Union[None, str]
        self.isLoaded = False

        self.specimens = None  # type: Union[None, SpecimenLoader]
        self.comparator = None  # type: Union[None, BaseComparator]
        self.segmentedMessages = None  # type: Union[None, List[Tuple[MessageSegment]]]
        self.rawSegmentedMessages = None  # type: Union[None, List[Tuple[MessageSegment]]]
        self.dc = None  # type: Union[None, DistanceCalculator]
        self.segmentationTime = None  # type: Union[None, float]
        self.dist_calc_segmentsTime = None  # type: Union[None, float]

    def configureAnalysis(self, *analysisArgs):
        """optional"""
        self.analysisArgs = analysisArgs

    def configureTokenizer(self, tokenizer: str, sigma: float=None, filtering=False):
        """
        mandatory (but may be called without parameters)

        :param tokenizer: The tokenizer to use. One of: "tshark", "4bytesfixed", "nemesys"
        :param sigma: Required only for nemesys: The sigma value to use. If not set,
            the value in sigmapertrace in this module is looked up and if the trace is not known there, use 0.9.
        :param filtering: Filter out **one-byte** segments and such, just consisting of **zeros**.
        """
        # if refinementCallback == pcaMocoRefinements:
        #     sigma = pcamocoSigmapertrace[pcapbasename] if not sigma and pcapbasename in pcamocoSigmapertrace else \
        #         0.9 if not sigma else sigma
        # else:
        self.sigma = sigmapertrace[self.pcapbasename] if not sigma and self.pcapbasename in sigmapertrace else \
            0.9 if not sigma else sigma
        self.tokenizer = tokenizer
        self.filter = filtering

    def configureRefinement(self, refinementCallback:Union[Callable, None] = refinements, forwardComparator=False,
                            **refinementArgs):
        """
        optional

        :param refinementCallback: The function to use for refinement.
            Existing refinements can be found in segmentHandler.
        :param forwardComparator: If True, the comparator instance is passed to the refinementCallback
            as additional keyword argument with the key "comparator".
        :param refinementArgs: kwargs for the refinement function, e. g., reportFolder or collectedSubclusters
        """
        self.refinementCallback = refinementCallback
        self.refinementArgs = refinementArgs
        self.forwardComparator = forwardComparator
        if forwardComparator and not self.dissectGroundtruth:
            raise ValueError("A comparator can only be forwarded to the refinement for evaluation if ground truth "
                             "(see CachedDistances.dissectGroundtruth) is available.")

    def _callRefinement(self):
        if self.refinementCallback is not None:
            self.rawSegmentedMessages = self.segmentedMessages
            if self.forwardComparator:
                if isinstance(self.refinementArgs, dict):
                    self.refinementArgs["comparator"] = self.comparator
                else:
                    self.refinementArgs = {"comparator": self.comparator}
            # noinspection PyUnresolvedReferences
            if self.refinementCallback.__code__.co_argcount > 1:  # not counting kwargs!
                # assume the second argument is expected to be a distance calculator
                chainedSegments = list(chain.from_iterable(self.segmentedMessages))
                print("Refinement: Calculate distance for {} segments...".format(len(chainedSegments)))
                if len(chainedSegments) ** 2 > MemmapDC.maxMemMatrix:
                    refinementDC = MemmapDC(chainedSegments)
                else:
                    refinementDC = DelegatingDC(chainedSegments)
                self.segmentedMessages = self.refinementCallback(self.segmentedMessages, refinementDC,
                                                                 **self.refinementArgs)
            else:
                self.segmentedMessages = self.refinementCallback(self.segmentedMessages, **self.refinementArgs)

    def _calc(self):
        """
        dissect and label messages
        """
        print("Load messages from {}...".format(self.pcapName))
        self.specimens = SpecimenLoader(self.pcapfilename, self.layer, relativeToIP=self.relativeToIP)
        if self.dissectGroundtruth:
            self.comparator = MessageComparator(
                self.specimens, layer=self.layer, relativeToIP=self.relativeToIP, debug=self.debug)
        else:
            self.comparator = BaseComparator(
                self.specimens, layer=self.layer, relativeToIP=self.relativeToIP, debug=self.debug)

        print("Segmenting messages...", end=' ')
        littleEndian = self.tokenizer[-2:] == "le"
        segmentationTime = time.time()
        # select tokenizer by command line parameter
        if self.tokenizer == "tshark":
            if isinstance(self.comparator, MessageComparator):
                # 1. segment messages according to true fields from the labels
                self.segmentedMessages = annotateFieldTypes(self.analyzerType, self.analysisArgs, self.comparator)
            else:
                raise ValueError("tshark tokenizer can only be used with existing (Wireshark) dissector "
                                 "and CachedDistances.dissectGroundtruth set to True.")
        elif self.tokenizer == "4bytesfixed":
            # 2. segment messages into fixed size chunks for testing
            self.segmentedMessages = fixedlengthSegmenter(4, self.specimens, self.analyzerType, self.analysisArgs)
        elif self.tokenizer in ["nemesys", "nemesysle"]:
            # 3. segment messages by NEMESYS
            if self.tokenizer == "nemesysle":  # little endian version
                segmentsPerMsg = bcDeltaGaussMessageSegmentationLE(self.specimens, self.sigma)
            else:
                segmentsPerMsg = bcDeltaGaussMessageSegmentation(self.specimens, self.sigma)

            # get analyzer requested by analyzerType/analysisArgs
            self.segmentedMessages = MessageAnalyzer.convertAnalyzers(
                segmentsPerMsg, self.analyzerType, self.analysisArgs)
            self._callRefinement()
        elif self.tokenizer in ["zeros", "zerosle"]:
            # 4. segment messages into their zero/non-zero parts
            segmentsPerMsg = [(MessageSegment(Value(msg), 0, len(msg.data)),)
                              for msg in self.specimens.messagePool.keys()]
            # .blend(ignoreSingleZeros=True) to omit single zeros
            zeroSlicedMessages = [BlendZeroSlices(list(msg)).blend(littleEndian=littleEndian)
                                  for msg in segmentsPerMsg]
            self.segmentedMessages = [CropChars(segs).split() for segs in zeroSlicedMessages]
            self._callRefinement()
        else:
            raise ValueError(f"tokenizer {self.tokenizer} is unknown")

        self.segmentationTime = time.time() - segmentationTime
        print("done.")

        if self.filter:
            chainedSegments = [seg for seg in chain.from_iterable(self.segmentedMessages) if
                        seg.length > 1 and set(seg.values) != {0}]
        else:
            chainedSegments = list(chain.from_iterable(self.segmentedMessages))

        print("Calculate distance for {} segments...".format(len(chainedSegments)))
        # dc = DistanceCalculator(chainedSegments, reliefFactor=0.33)  # Pairwise similarity of segments: dc.distanceMatrix
        dist_calc_segmentsTime = time.time()
        if len(chainedSegments) ** 2 > MemmapDC.maxMemMatrix:
            self.dc = MemmapDC(chainedSegments)
        else:
            self.dc = DelegatingDC(chainedSegments)
        self.dist_calc_segmentsTime = time.time() - dist_calc_segmentsTime
        try:
            with open(self.dccachefn, 'wb') as f:
                pickle.dump((self.segmentedMessages, self.comparator, self.dc), f, pickle.HIGHEST_PROTOCOL)
            print("Write distances to cache file {}".format(self.dccachefn))
        except MemoryError:
            print("DC could not be cached due to a MemoryError. Removing", self.dccachefn, "and continuing.")
            os.remove(self.dccachefn)

    def _load(self):
        print("Load distances from cache file {}".format(self.dccachefn))
        with open(self.dccachefn, 'rb') as f:
            self.segmentedMessages, self.comparator, self.dc = pickle.load(f)
        if not (isinstance(self.comparator, BaseComparator)
                and isinstance(self.dc, DistanceCalculator)):
            print('Loading of cached distances failed.')
            exit(10)
        if not isinstance(self.comparator, MessageComparator):
            print("Loaded without ground truth from dissector.")
        self.specimens = self.comparator.specimens
        self.segmentationTime, self.dist_calc_segmentsTime = None, None
        self.isLoaded = True

    def get(self):
        assert self.analysisTitle is not None
        assert self.tokenizer is not None
        assert self.pcapName is not None
        assert self.tokenizer != "nemesys" or self.sigma is not None

        tokenparm = self.tokenizer if self.tokenizer[:7] != "nemesys" else \
            "{}{:.0f}".format(self.tokenizer, self.sigma * 10)
        refine = (self.refinementCallback.__name__ if self.refinementCallback is not None else "raw") \
            + ("le" if self.refinementArgs is not None
                       and "littleEndian" in self.refinementArgs and self.refinementArgs["littleEndian"] else "")
        fnprefix = "cache-dc"
        if isinstance(self.comparator, MessageComparator):
            fnprefix += "-nogt"
        self.dccachefn = os.path.join(cacheFolder, '{}-{}-{}-{}-{}-{}-{}.{}'.format(
            fnprefix, self.analysisTitle, tokenparm, "filtered" if self.filter else "all",
            refine,
            self.pcapName,
            "" if self.layer == 2 and self.relativeToIP == True
                else str(self.layer) + "reltoIP" if self.relativeToIP else "",
            'ddc'))
        if self.disableCache or not os.path.exists(self.dccachefn):
            self._calc()
        else:
            self._load()

def cacheAndLoadDC(pcapfilename: str, analysisTitle: str, tokenizer: str, debug: bool,
                   analyzerType: type, analysisArgs: Tuple=None, sigma: float=None, filtering=False,
                   refinementCallback: Union[Callable, None] = refinements,
                   disableCache=False, layer=2, relativeToIP=True) \
        -> Tuple[SpecimenLoader, BaseComparator, List[Tuple[MessageSegment]], DistanceCalculator, Optional[float],
                 Optional[float]]:
    """
    Legacy, **deprecated**:
    Wrapper around class CachedDistances for backwards compatibility:
        cache or load the DistanceCalculator to or from the filesystem

    >>> from nemere.utils.baseAlgorithms import generateTestSegments
    >>> segments = generateTestSegments()
    >>> dc = DistanceCalculator(segments)
    Calculated distances for 37 segment pairs in ... seconds.
    >>> chainedSegments = dc.rawSegments

    :param analysisArgs: Optional arguments to use for instantiating for the given analysis type class.
    :param debug: Set debug flag in MessageComparator.
    :param pcapfilename: File name string of the PCAP to load.
    :param analysisTitle: Text string to use as title/label for this analysis.
    :param layer: Protocol layer to extract from the encapsulation. 1 for raw frame (with set relativeToIP = False).
    :param relativeToIP: Interpret the layer relative to the IP layer (True) or absolute (False).
    :param tokenizer: The tokenizer to use. One of: "tshark", "4bytesfixed", "nemesys"
    :param sigma: Required only for nemesys: The sigma value to use. If not set,
        the value in sigmapertrace in this module is looked up and if the trace is not known there, use 0.9.
    :param analyzerType: Unused
    :param filtering: Filter out **one-byte** segments and such, just consisting of **zeros**.
    :param disableCache: When experimenting with distances manipulation, deactivate caching!
    :param refinementCallback: The function to use for refinement.
        Existing refinements can be found in segmentHandler.
    :return:
    """
    fromCache = CachedDistances(pcapfilename, analysisTitle, layer, relativeToIP)
    fromCache.disableCache = disableCache
    fromCache.debug = debug
    fromCache.configureAnalysis(*analysisArgs)
    fromCache.configureTokenizer(tokenizer, sigma, filtering)
    fromCache.configureRefinement(refinementCallback)
    fromCache.get()

    assert fromCache.specimens is not None
    assert fromCache.comparator is not None
    assert fromCache.segmentedMessages is not None
    assert fromCache.dc is not None

    return fromCache.specimens, fromCache.comparator, fromCache.segmentedMessages, fromCache.dc, \
           fromCache.segmentationTime, fromCache.dist_calc_segmentsTime




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



class StartupFilecheck(object):
    def __init__(self, pcapfilename: str, reportFullPath: str=None):
        if not isfile(pcapfilename):
            print('File not found:', pcapfilename)
            exit(1)
        self.pcapfilename = pcapfilename
        self.pcapbasename = basename(pcapfilename)
        self.pcapstrippedname = splitext(self.pcapbasename)[0]
        print("\n\nTrace:", self.pcapbasename)

        if reportFullPath is None:
            self.reportFullPath = join(reportFolder, self.pcapstrippedname)
            """A path name that is inside the report folder and reflects the pcap base name without extension."""
        else:
            self.reportFullPath = reportFullPath
            """A path name that is inside the report folder and reflects the pcap base name without extension."""
        try:
            os.makedirs(self.reportFullPath)
        except FileExistsError:
            if isdir(self.reportFullPath):
                print("Using existing ", self.reportFullPath, " as report folder.")
            else:
                print("Path that should be used as report folder is an existing file. Aborting.")
                exit(1)

        self.timestamp = time.time()
        self.timeformated = time.strftime("%Y%m%d-%H%M%S", time.gmtime(self.timestamp))

    def reportWithTimestamp(self, inferenceTitle: str=None):
        if inferenceTitle is None:
            reportPathTS = join(self.reportFullPath, self.timeformated)
        else:
            reportPathTS = join(self.reportFullPath, "{}_{}".format(inferenceTitle, self.timeformated))
        os.makedirs(reportPathTS, exist_ok=True)
        return reportPathTS

    def writeReportMetadata(self, dcCacheFile: str=None, scriptRuntime: float=None):
        import sys, git
        if not exists(self.reportFullPath):
            raise FileNotFoundError("Report folder must be existing. It does not.")
        repo = git.Repo(search_parent_directories=True)
        timeformat = "%d.%m.%Y %H:%M:%S %Z"

        lines = {
            "fullCommandLine": " ".join(sys.argv),
            "absolutepcapfilename": abspath(self.pcapfilename),
            "dcCacheFile": "n/a" if dcCacheFile is None else dcCacheFile,
            "gitCommit": repo.head.object.hexsha + f" ({repo.active_branch})",
            "currentTime": time.strftime(timeformat),
            "scriptRuntime": "{:.3f} s".format(time.time() - self.timestamp
                                               if scriptRuntime is None else scriptRuntime),
            "host": os.uname().nodename
        }

        with open(join(self.reportFullPath, "run-metadata.md"), "a") as md:
            md.write("# Report Metadata\n\n")
            md.writelines(f"{k}: {v}\n\n" for k, v in lines.items())


class TrueOverlays(object):
    """
    Count and the amount of (falsely) inferred boundaries in the scope of each true field.
    """
    def __init__(self, trueSegments: Dict[str, Sequence[MessageSegment]],
                 inferredSegments: List[Sequence[MessageSegment]], comparator: MessageComparator, minLen=3):
        self.trueSegments = trueSegments
        self.comparator = comparator
        self.inferredSegments = inferredSegments
        # at least minLen bytes long
        self.keys4longer = [k for k, segs in self.trueSegments.items() if any(len(s) >= minLen for s in segs)]
        self.trueNamesOverlay = defaultdict(defaultdict)
        self._classifyTrueNamesOverlays()
        self.trueNamesOverlayCounters = self._trueOverlayCounters()
        # maxoverlaysegcount = max(cnt.keys() for cnt in self.trueNamesOverlayCounters.values())

    def _classifyTrueNamesOverlays(self):
        # sort the inferred segments per true segment by type of overlapping
        for k4l in self.keys4longer:
            for seg in self.trueSegments[k4l]:
                inf4msg = inferred4segment(seg, self.inferredSegments)
                # inferred field overlapping
                overlapping = [i4m for i4m in inf4msg if isOverlapping(seg, i4m)]
                # true and inferred fields match exactly
                # .. t .. t ..
                # .. i .. i ..
                if len(overlapping) == 1 and \
                        seg.offset == overlapping[0].offset and seg.nextOffset == overlapping[0].nextOffset:
                    self.trueNamesOverlay[k4l][seg] = 0
                # overspecific inference: true field overlaps by multiple inferred segments
                # .. t ......... t ..
                #   .. i .. i .. i ..
                elif len(overlapping) > 1:
                    self.trueNamesOverlay[k4l][seg] = len(overlapping)
                # underspecific inference: true field is only substring of an inferred
                #   .. t .. t ..
                # .. i ......... i ..
                else:  # len(overlapping) == 1 and (seg.offset < overlapping[0].offset or seg.nextOffset > overlapping[0].nextOffset)
                    self.trueNamesOverlay[k4l][seg] = -1

    def _trueOverlayCounters(self):
        # amount of (falsely) inferred boundaries in the scope of each true field per true field name.
        return {fname: Counter(segcnt.values()) for fname, segcnt in
                                    self.trueNamesOverlay.items()}  # type: Dict[str, Counter]

    _cutoff = 10
    _cntheaders = ["Field name  /  inf per true", "min len", "max len", "sum", "all nulls",
                                           "underspecific", "exact"] \
                  + list(range(2, _cutoff)) + [f"> {_cutoff-1} (segments too many)"]

    def _cnttable(self):
         return [[fname,
                     min(len(s) for s in self.trueSegments[fname]), max(len(s) for s in self.trueSegments[fname]),
                     len(self.trueSegments[fname]),
                     sum(set(s.values) == {0} for s in self.trueSegments[fname])
                     ] + [
            cnt[c] if c in cnt else None for c in [-1, 0] + list(range(2, TrueOverlays._cutoff))
        ] + [sum(c for k, c in cnt.items() if k >= TrueOverlays._cutoff)]
                    for fname, cnt in self.trueNamesOverlayCounters.items()]

    def __repr__(self):
        """
        Arrange the data of trueNamesOverlayCounters in a table like this:

        # Field name  /  inf per true | underspecific | exact | 1 | 2 | 3 | 4 | ... | > 10 | (segments too many)
        # wlan.fixed.ftm.param.delim2 |       50      |  21   | ...
        # wlan.tag.oui                |               |   3   | ...
        # wlan.fixed.ftm_toa          | ...

        :return: Visual table
        """
        cnttable = self._cnttable()
        return tabulate(cnttable, headers=TrueOverlays._cntheaders)

    def toCSV(self, folder: str):
        import csv
        csvPath = join(folder, type(self).__name__ + ".csv")
        if exists(csvPath):
            raise FileExistsError("Will not overwrite existing file " + csvPath)
        with open(csvPath, 'w') as csvFile:
            cntcsv = csv.writer(csvFile)  # type: csv.writer
            cntcsv.writerow(self._cntheaders)
            cntcsv.writerows(self._cnttable())

    @staticmethod
    def uniqSort(someSegs: Dict[str, Sequence[MessageSegment]]):
        # remove double values by adding into dicts
        uniqSegs = {fname: {s.values: s for s in segs} for fname, segs in someSegs.items()}
        sortedSegs = {fname: sorted(segs.values(), key=lambda s: s.values) for fname, segs in uniqSegs.items()}
        return sortedSegs

    def filterUnderspecific(self):
        """
        true segments that are not all nulls and underspecific / sorted by segment value
        :return:
        """
        filteredSegs = {fname: (s for s, c in segolc.items() if set(s.values) != {0} and c < 0)
                        for fname, segolc in self.trueNamesOverlay.items()}
        return TrueOverlays.uniqSort(filteredSegs)

    def filterOverspecific(self, segCnt: int=3):
        # true segments that are not all nulls and (+2/+3/*) inferred
        filteredSegs = {fname: (s for s, c in segolc.items() if set(s.values) != {0} and c == segCnt)
                        for fname, segolc in self.trueNamesOverlay.items()}
        return TrueOverlays.uniqSort(filteredSegs)

    def printSegmentContexts(self, trueSegments: Dict[str, Sequence[MessageSegment]], maxlines=10):
        """
        print the selected fields for reference
        :param trueSegments:
        :param maxlines: Limit output per field category to this number, no limit if <= 0
        """
        for lab, segs in trueSegments.items():
            if len(segs) > 0:
                print("\n" "# #", lab)
                if maxlines > 0:
                    truncatedSegs = segs[:maxlines]
                else:
                    truncatedSegs = segs
                markSegNearMatch(truncatedSegs, self.inferredSegments, self.comparator, 3)
            # for seg in segs:
            #     # inf4msg = inferred4segment(seg, self.inferredSegments)
            #     # overlapping = [i4m for i4m in inf4msg if isOverlapping(seg, i4m)]
            #     # # print(overlapping)
            #     # if len(overlapping) == 1:
            #     #     # print("match or inferred larger. continuing")
            #     #     continue
            #     markSegNearMatch(seg, self.inferredSegments, 3)


class TrueDataTypeOverlays(TrueOverlays):
    def __init__(self, trueSegmentedMessages: Dict[AbstractMessage, Tuple[TypedSegment]],
                 inferredSegments: List[Sequence[MessageSegment]], comparator: MessageComparator, minLen: int = 3):
        # all true fields of one data type
        trueDataTypes = defaultdict(list)
        for seg in (seg for msgsegs in trueSegmentedMessages.values() for seg in msgsegs):
            trueDataTypes[seg.fieldtype].append(seg)
        super().__init__(trueDataTypes, inferredSegments, comparator, minLen)


class TrueFieldNameOverlays(TrueOverlays):
    def __init__(self, trueSegmentedMessages: Dict[AbstractMessage, Tuple[TypedSegment]],
                 inferredSegments: List[Sequence[MessageSegment]], comparator: MessageComparator, minLen: int = 3):
        # all true fields of one field type (tshark name)
        trueFieldNames = defaultdict(list)
        for absmsg, msgsegs in trueSegmentedMessages.items():
            pm = comparator.parsedMessages[comparator.specimens.messagePool[absmsg]]
            fnames = pm.getFieldNames()
            # here we assume that the fnames and msgsegs are in the same order (and have the same amount),
            # which should be the case if ParsedMessage works correctly and trueSegmentedMessages was not tampered with.
            assert len(msgsegs) == len(fnames)
            for seg, fna in zip(msgsegs, fnames):
                trueFieldNames[fna].append(seg)
        super().__init__(trueFieldNames, inferredSegments, comparator, minLen)


class TitleBuilder(object):
    """Builds readable strings from the configuration parameters of the analysis."""
    def __init__(self, tokenizer, refinement = None, sigma = None, clusterer = None):
        self.tokenizer = tokenizer
        self.refinement = refinement
        self._clusterer = clusterer
        self.postProcess = None  # multiple. to be adjusted dynamically

        self.sigma = sigma

    @property
    def tokenParams(self):
        return f"sigma {self.sigma}" if self.tokenizer[:7] == "nemesys" else None

    @property
    def clusterer(self):
        return type(self._clusterer).__name__

    @clusterer.setter
    def clusterer(self, val):
        self._clusterer = val

    @property
    def clusterParams(self):
        from nemere.inference.templates import DBSCANsegmentClusterer, HDBSCANsegmentClusterer, OPTICSsegmentClusterer
        from sklearn.cluster import DBSCAN, OPTICS
        from hdbscan import HDBSCAN
        if isinstance(self._clusterer, (DBSCANsegmentClusterer, DBSCAN)):
            return f"eps {self._clusterer.eps:.3f} ms {self._clusterer.min_samples}"
        elif isinstance(self._clusterer, (HDBSCANsegmentClusterer, HDBSCAN)):
            return f"mcs {self._clusterer.min_cluster_size} ms {self._clusterer.min_samples}"
        elif isinstance(self._clusterer, (OPTICSsegmentClusterer, OPTICS)):
            return f"ms {self._clusterer.min_samples} maxeps {self._clusterer.max_eps}"

    @property
    def plotTitle(self):
        plotTitle = self.tokenizer
        if self.tokenParams is not None: plotTitle += "-" + self.tokenParams
        if self.refinement is not None: plotTitle += "-" + self.refinement
        plotTitle += " " + self.clusterer + " " + self.clusterParams
        if self.postProcess is not None: plotTitle += " " + self.postProcess
        return plotTitle

    @property
    def dict(self):
        return {
            "tokenizer": self.tokenizer,
            "tokenParams": self.tokenParams,
            "refinement": self.refinement,
            "clusterer": self.clusterer,
            "clusterParams": self.clusterParams,
            "postProcess": self.postProcess
            }


class TitleBuilderSens(TitleBuilder):
    """include Sensitivitiy from clusterer in title"""
    @property
    def clusterParams(self):
        from nemere.inference.templates import DBSCANsegmentClusterer
        if isinstance(self._clusterer, DBSCANsegmentClusterer):
            return f"S {self._clusterer.S:.1f} eps {self._clusterer.eps:.3f} ms {self._clusterer.min_samples}"
        else:
            return super().clusterParams


def segIsTyped(someSegment):
    return isinstance(someSegment, (TypedTemplate, TypedSegment))


uulmColors = {
    "uulm"       : "#7D9AAA",  # blue-gray
    "uulm-akzent": "#A9A28D",  # beige
    "uulm-in"    : "#A32638",  # magenta
    "uulm-med"   : "#26247C",  # bluish
    "uulm-mawi"  : "#56AA1C",  # green
    "uulm-nawi"  : "#BD6005"   # orange
}
