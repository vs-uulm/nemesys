"""
Use groundtruth about field segmentation by dissectors and align segments
on the similarity of their feature "personality".

Takes a PCAP trace of a known protocol, dissects each message into their fields, and yields segments from each of them.
These segments get analyzed by the given analysis method which is used as feature to determine their similarity.
Similar fields are then aligned.
"""

import argparse, IPython, itertools, pickle, csv
import time
from math import ceil
from os.path import isfile, splitext, basename, exists, join
from tabulate import tabulate

from alignment.alignMessages import SegmentedMessages
from inference.segments import AbstractSegment
from inference.templates import DistanceCalculator, DelegatingDC, Template
from alignment.hirschbergAlignSegments import HirschbergOnSegmentSimilarity
from inference.analyzers import *
from utils.evaluationHelpers import annotateFieldTypes, writeMessageClusteringStaticstics, writePerformanceStatistics
from validation.dissectorMatcher import MessageComparator
from utils.loader import SpecimenLoader
from characterize_fieldtypes import analyses
from visualization.multiPlotter import MultiMessagePlotter

debug = False

analysis_method = 'value'
distance_method = 'canberra'
tokenizer = 'tshark'  # (, '4bytesfixed', 'nemesys')


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # Evaluation helpers  # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


def delign(segseqA):
    return [s for s in segseqA if s >= -1]

# realign
def relign(segseqA, segseqB):
    hirsch = HirschbergOnSegmentSimilarity(dc.similarityMatrix())
    return hirsch.align(dc.segments2index([s for s in segseqA if isinstance(s, MessageSegment)]),
                        dc.segments2index([s for s in segseqB if isinstance(s, MessageSegment)]))

def resolveIdx2Seg(segseq):
    """

    :param segseq: list of segment indices (from raw segment list) per message
    :return:
    """
    print(tabulate([[dc.segments[s].bytes.hex() if s != -1 else None for s in m]
                        for m in segseq], disable_numparse=True, headers=range(len(segseq[0]))))

def columnOfAlignment(alignedSegments: List[List[MessageSegment]], colnum: int):
    return [msg[colnum] for msg in alignedSegments]

def column2first(dc: DistanceCalculator, alignedSegments: List[List[MessageSegment]], colnum: int):
    """
    Similarities of entries 1 to n of one column to its first (not None) entry.

    :param dc:
    :param alignedSegments:
    :param colnum:
    :return:
    """
    column = [msg[colnum] for msg in alignedSegments] #columnOfAlignment(alignedSegments, colnum)

    # strip Nones
    nonepos = [idx for idx, seg in enumerate(column) if seg is None]
    stripedcol = [seg for seg in column if seg is not None]

    dists2first = ["- (reference)"] + dc.distancesSubset(stripedcol[0:1], stripedcol[1:]).tolist()[0]

    # re-insert Nones
    for idx in nonepos:
        dists2first.insert(idx, None)

    # transpose
    d2ft = list(map(list, zip(column, dists2first)))
    return d2ft

def printSegDist(d2ft: List[Tuple[MessageSegment, float]]):
    print(tabulate([(s.bytes.hex() if isinstance(s, MessageSegment) else "-", d) for s, d in d2ft],
                   headers=['Seg (hex)', 'Distance'], floatfmt=".4f"))

def seg2seg(dc: DistanceCalculator, alignedSegments: List[List[MessageSegment]],
            coordA: Tuple[int, int], coordB: Tuple[int, int]):
    """
    Distance between segments that are selected by coordinates in an alignment

    :param dc: DistanceCalculator holding the segment distances.
    :param alignedSegments: 2-dimensional list holding the alignment
    :param coordA: Coordinates of segment A within the alignment
    :param coordB: Coordinates of segment B within the alignment
    :return:
    """
    segA = alignedSegments[coordA[0]][coordA[1]]
    print(segA)
    segB = alignedSegments[coordB[0]][coordB[1]]
    print(segB)
    return dc.pairDistance(segA, segB)

def quicksegmentTuple(dc: DistanceCalculator, segment: MessageSegment):
    return dc.segments2index([segment])[0], segment.length, tuple(segment.values)

def epsautoconfeval(epsilon):
    """
    # investigate distance properties for clustering autoconfiguration
    # plots of k-nearest-neighbor distance histogram and "knee"

    See SegmentedMessages#autoconfigureDBSCAN

    :param epsilon The manually determined "best" epsilon for comparison
    :return:
    """

    # # distribution of all distances in matrix
    # hstplt = SingleMessagePlotter(specimens, tokenizer+'-distance-distribution-histo', args.interactive)
    # hstplt.histogram(tril(sm.distances), bins=[x / 50 for x in range(50)])
    # plt.axvline(epsilon, label="manually determined eps={:0.2f}".format(epsilon), c="red")
    # hstplt.text('max {:.3f}, mean {:.3f}'.format(sm.distances.max(), sm.distances.mean()))
    # hstplt.writeOrShowFigure()
    # del hstplt

    neighbors = sm.neighbors()  # list of tuples: (index from sm.distances, distance) sorted by distance

    mmp = MultiMessagePlotter(specimens, tokenizer + "-k-nearest-neighbor-distance-distribution", 1, 2,
                              isInteractive=args.interactive)
    mmp.axes[0].axhline(epsilon, label="manually determined eps={:0.2f}".format(epsilon), c="red")
    mmp.axes[1].axhline(epsilon, label="manually determined eps={:0.2f}".format(epsilon), c="red")

    krange = (0, 16, 1)

    for k in range(*krange):
        knearest = sorted([nfori[k][1] for nfori in neighbors])
        mmp.plotToSubfig(1, knearest, alpha=.4, label="k={}".format(k))

    # # kneedle approach
    # # unusable results. does not find a knee!
    # from kneed import KneeLocator
    # kneeK = dict()
    # for k in range(*krange):
    #     knearest = sorted([nfori[k][1] for nfori in neighbors])
    #     mmp.plotToSubfig(0, knearest, alpha=.1)  # , label="k={}".format(k)
    #
    #     kneel = KneeLocator(range(len(knearest)), knearest, curve='convex', direction='increasing')
    #     kneeX = kneel.knee
    #     kneeK[knearest[kneeX]] = k
    #
    # kneeDists = list(kneeK.keys())
    # bestEpsMatchIdx = numpy.argmin([abs(kneeDist-epsilon) for kneeDist in kneeDists])
    # bestEpsMatchDist = kneeDists[bestEpsMatchIdx]

    # # range of distances for a k-neighborhood, determines the maximum range of distances
    # # results do not correlate with a suitable choice of k compared to manual introspection.
    # distrange4k = list()
    # for k in range(*krange):
    #     knearest = sorted([nfori[k][1] for nfori in neighbors])
    #     distrange4k.append((k, max(knearest) - min(knearest)))
    # maxdran = max([dran[1] for dran in distrange4k])
    # maxkran = [dran[0] for dran in distrange4k if dran[1] == maxdran][0]
    # mmp.textInEachAx(["max range {:.2f} at k={}".format(maxdran, maxkran)])
    # mmp.axes[0].axhline(bestEpsMatchDist, linestyle='dashed', color='blue', alpha=.4,
    #                     label="kneedle {:.2f} of k={}".format(bestEpsMatchDist, kneeK[bestEpsMatchDist]))

    # smoothing approach
    from scipy.ndimage.filters import gaussian_filter1d
    from math import log

    sigma = log(len(neighbors))
    knearest = dict()
    smoothknearest = dict()
    seconddiff = dict()
    seconddiffMax = (0, 0, 0)

    ksteepeststats = list()
    for k in range(0, len(neighbors) // 10):  # round(2*log(len(neighbors)))
        knearest[k] = sorted([nfori[k][1] for nfori in neighbors])
        smoothknearest[k] = gaussian_filter1d(knearest[k], sigma)
        # max of second difference (maximum upwards curvature) as knee
        seconddiff[k] = numpy.diff(smoothknearest[k], 2)
        seconddiffargmax = seconddiff[k].argmax()
        diffrelmax = seconddiff[k].max() / smoothknearest[k][seconddiffargmax]
        if sigma < seconddiffargmax < len(neighbors) - sigma and diffrelmax > seconddiffMax[2]:
            seconddiffMax = (k, seconddiffargmax, diffrelmax)
        ksteepeststats.append((k, seconddiff[k].max(), diffrelmax))
    print(tabulate(ksteepeststats, headers=("k", "max(f'')", "max(f'')/f")))

    # prepare to plot the smoothed nearest neigbor distribution and its second derivative
    k = seconddiffMax[0]
    x = seconddiffMax[1] + 1

    mmp.plotToSubfig(0, smoothknearest[k], label="smooth k={}, sigma={:.2f}".format(k, sigma), alpha=.4)
    mmp.plotToSubfig(1, smoothknearest[k], label="smooth k={}, sigma={:.2f}".format(k, sigma), alpha=1, color='blue')
    mmp.plotToSubfig(0, knearest[k], alpha=.4)

    ax0twin = mmp.axes[0].twinx()
    # mmp.plotToSubfig(ax0twin, seconddiff[k], linestyle='dotted', color='cyan', alpha=.4)
    mmp.plotToSubfig(ax0twin, [None] + seconddiff[k].tolist(), linestyle='dotted',
                     color='magenta', alpha=.4)

    # epsilon = knearest[k][x]
    epsilon = smoothknearest[k][x]

    mmp.axes[0].axhline(epsilon, linestyle='dashed', color='blue', alpha=.4,
                        label="curvature max {:.2f} of k={}".format(
                            epsilon, k))
    mmp.axes[0].axvline(x, linestyle='dashed', color='blue', alpha=.4)


    mmp.writeOrShowFigure()
    del mmp

    # if args.interactive:
    #     from tabulate import tabulate
    #     IPython.embed()
    # exit(0)

    return epsilon



# # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # END : Evaluation helpers  # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # #














if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Analyze fields as segments of messages and align on the similarity of their feature "personality".')
    parser.add_argument('pcapfilename', help='Filename of the PCAP to load.')
    parser.add_argument('-i', '--interactive', help='Open ipython prompt after finishing the analysis.',
                        action="store_true")
    # TODO select tokenizer by command line parameter
    # TODO toggle filtering on/off
    args = parser.parse_args()

    if not isfile(args.pcapfilename):
        print('File not found: ' + args.pcapfilename)
        exit(1)
    analyzerType = analyses[analysis_method]
    analysisArgs = None
    analysisTitle = analysis_method

    # cache the DistanceCalculator to the filesystem
    pcapName = splitext(basename(args.pcapfilename))[0]
    dccachefn = 'cache-dc-{}-{}-{}.{}'.format(analysisTitle, tokenizer, pcapName, 'ddc')
    # dccachefn = 'cache-dc-{}-{}-{}.{}'.format(analysisTitle, tokenizer, pcapName, 'dc')
    if not exists(dccachefn):
        # dissect and label messages
        print("Load messages...")
        specimens = SpecimenLoader(args.pcapfilename, 2, True)
        comparator = MessageComparator(specimens, 2, True, debug=debug)

        # TODO select tokenizer by command line parameter
        print("Segmenting messages...", end=' ')
        segmentationTime = time.time()
        # 1. segment messages according to true fields from the labels
        segmentedMessages = annotateFieldTypes(analyzerType, analysisArgs, comparator)
        # # 2. segment messages into fixed size chunks for testing
        # segmentedMessages = segmentsFixed(4, comparator, analyzerType, analysisArgs)
        # # 3. segment messages by NEMESYS
        # segmentedMessages = ...
        segmentationTime = time.time() - segmentationTime
        print("done.")

        chainedSegments = list(itertools.chain.from_iterable(segmentedMessages))

        # TODO filter segments to reduce similarity calculation load - use command line parameter to toggle filtering on/off
        # filteredSegments = filterSegments(chainedSegments)
        # if length < 3:
        #     pass  # Handle short segments

        print("Calculate distance for {} segments...".format(len(chainedSegments)))
        # dc = DistanceCalculator(chainedSegments, reliefFactor=0.33)  # Pairwise similarity of segments: dc.distanceMatrix
        dist_calc_segmentsTime = time.time()
        dc = DelegatingDC(chainedSegments)
        dist_calc_segmentsTime = time.time() - dist_calc_segmentsTime
        with open(dccachefn, 'wb') as f:
            pickle.dump((segmentedMessages, comparator, dc), f, pickle.HIGHEST_PROTOCOL)
    else:
        print("Load distances from cache file {}".format(dccachefn))
        segmentedMessages, comparator, dc = pickle.load(open(dccachefn, 'rb'))
        if not (isinstance(comparator, MessageComparator)
                and isinstance(dc, DistanceCalculator)):
            print('Loading of cached distances failed.')
            exit(10)
        specimens = comparator.specimens
        chainedSegments = list(itertools.chain.from_iterable(segmentedMessages))
        segmentationTime, dist_calc_segmentsTime = None, None

    print("Calculate distance for {} messages...".format(len(segmentedMessages)))
    dist_calc_messagesTime = time.time()
    sm = SegmentedMessages(dc, segmentedMessages)
    dist_calc_messagesTime = time.time() - dist_calc_messagesTime

    cluster_params_autoconfTime = time.time()
    eps, min_samples = sm.autoconfigureDBSCAN()
    cluster_params_autoconfTime = time.time() - cluster_params_autoconfTime


    # # DEBUG and TESTING
    # #
    # # retrieve manually determined epsilon value
    # pcapbasename = basename(args.pcapfilename)
    # epsilon = message_epspertrace[pcapbasename] if pcapbasename in message_epspertrace else 0.15
    eps = epsautoconfeval(eps)
    # #
    # # DEBUG and TESTING


    # cluster and align messages and calculate statistics of it
    print('Clustering messages...')
    cluster_messagesTime = time.time()
    messageClusters, labels, clusterer = sm.clusterMessageTypesDBSCAN(eps=eps, min_samples=3)
    cluster_messagesTime = time.time() - cluster_messagesTime
    plotTitle = "{}-{} eps {:.3f} ms {}".format(
        tokenizer, type(clusterer).__name__, clusterer.eps, clusterer.min_samples)
    # messageClusters, labels, clusterer = sm.clusterMessageTypesHDBSCAN()
    # plotTitle = "{}-{} mcs {} ms {}".format(
    #     tokenizer, type(clusterer).__name__, clusterer.min_cluster_size, clusterer.min_samples)




    # write message clustering statistics to csv
    groundtruth = {msg: pm.messagetype for msg, pm in comparator.parsedMessages.items()}
    clusterStats, conciseness = writeMessageClusteringStaticstics(messageClusters, groundtruth, "{}-{}-eps={:.2f}-min_samples={}".format(
        tokenizer, type(clusterer).__name__, clusterer.eps, clusterer.min_samples), comparator)
    for msg, mtype in groundtruth.items():
        msg.messageType = mtype

    # plot distances and message clusters
    print("Plot distances...")
    from visualization.distancesPlotter import DistancesPlotter
    dp = DistancesPlotter(specimens, 'message-distances-' + plotTitle, False)
    dp.plotManifoldDistances(
        [specimens.messagePool[seglist[0].message] for seglist in segmentedMessages],
        sm.distances, labels)  # segmentedMessages
    dp.writeOrShowFigure()

    # align cluster members
    align_messagesTime = time.time()
    alignedClusters = dict()
    print("Align each cluster...")
    for clunu, msgcluster in messageClusters.items():  # type: int, List[Tuple[MessageSegment]]
        clusteralignment, alignedsegments = sm.alignMessageType(msgcluster)
        alignedClusters[clunu] = alignedsegments

        # get gaps at the corresponding positions
        # print('Cluster', clunu)
        hexalnseg = [[s.bytes.hex() if s is not None else None for s in m] for m in alignedsegments]
    print()
    align_messagesTime = time.time() - align_messagesTime

















    # check for cluster merge candidates
    print("Check for cluster merge candidates...")
    alignedFields = {clunu: [field for field in zip(*cluelms)] for clunu, cluelms in alignedClusters.items() if clunu > -1}
    FC_DYN = b"DYNAMIC"
    FC_GAP = "GAP"
    statDynFields = dict()
    for clunu, alfi in alignedFields.items():
        statDynFields[clunu] = list()
        for fvals in alfi:
            fvalsGapless = [val for val in fvals if val is not None]
            # if len(fvalsGapless) < len(fvals): there are GAPs in there
            if all([val.bytes == fvalsGapless[0].bytes for val in fvalsGapless]):
                # This leaves only an example in the field list:
                # The association to the original message is lost for the other segments in this field!
                statDynFields[clunu].append(fvalsGapless[0])
            else:
                statDynFields[clunu].append(Template(list(FC_DYN), fvalsGapless))
    # generate a similarity matrix for field-classes (static-dynamic)
    statDynValues = list(set(itertools.chain.from_iterable(statDynFields.values())))
    statDynValuesMap = {sdv: idx for idx, sdv in enumerate(statDynValues)}
    statDynIndices = {clunu: [statDynValuesMap[fc] for fc in sdf] for clunu, sdf in statDynFields.items()}
    fcSimMatrix = numpy.array([[0 if fcL.bytes != fcK.bytes else 0.5 if isinstance(fcL, Template) else 1
                                for fcL in statDynValues] for fcK in statDynValues])
    clusterpairs = list(itertools.combinations(statDynFields.keys(), 2))
    fclassHirsch = HirschbergOnSegmentSimilarity(fcSimMatrix)
    alignedFCIndices = {(clunuA, clunuB): fclassHirsch.align(statDynIndices[clunuA], statDynIndices[clunuB])
        for clunuA, clunuB in clusterpairs}
    alignedFieldClasses = {clupa: ([statDynValues[a] if a > -1 else FC_GAP for a in afciA],
                                   [statDynValues[b] if b > -1 else FC_GAP for b in afciB])
                           for clupa, (afciA, afciB) in alignedFCIndices.items()}
    mergingThreshold = 0  # allow merge of clusters if at most this number of field do not match
    # matchingClusters = [ (clunuA, clunuB) for clunuA, clunuB in clusterpairs if mergingThreshold >= len(
    #         [ False for afcA, afcB in zip(*alignedFieldClasses[(clunuA, clunuB)])
    #           if not (afcA == FC_GAP or afcB == FC_GAP or afcA.bytes == afcB.bytes
    #                      or isinstance(afcA, Template) and isinstance(afcB, MessageSegment) and set(afcB.bytes) == {0}
    #                      or isinstance(afcB, Template) and isinstance(afcA, MessageSegment) and set(afcA.bytes) == {0}
    #                      or isinstance(afcA, Template) and afcB.bytes in [bs.bytes for bs in afcA.baseSegments]
    #                      or isinstance(afcB, Template) and afcA.bytes in [bs.bytes for bs in afcB.baseSegments])
    #           ]
    #                     ) ]
    # noinspection PyTypeChecker
    matchingConditions = { (clunuA, clunuB): [("Agap","Bgap","equal","Azero","Bzero","BinA","AinB")] + [  # ,"ALL" # "AdynB0","BdynA0"
        (afcA == FC_GAP, afcB == FC_GAP,
         isinstance(afcA, (MessageSegment, Template)) and isinstance(afcB, (MessageSegment, Template)) and afcA.bytes == afcB.bytes,
         isinstance(afcA, MessageSegment) and set(afcA.bytes) == {0}, isinstance(afcB, MessageSegment) and set(afcB.bytes) == {0},
         # isinstance(afcA, Template) and isinstance(afcB, MessageSegment) and set(afcB.bytes) == {0},
         # isinstance(afcB, Template) and isinstance(afcA, MessageSegment) and set(afcA.bytes) == {0},
         isinstance(afcA, Template) and isinstance(afcB, MessageSegment) and afcB.bytes in [bs.bytes for bs in afcA.baseSegments],
         isinstance(afcB, Template) and isinstance(afcA, MessageSegment) and afcA.bytes in [bs.bytes for bs in afcB.baseSegments],
         # afcA == FC_GAP or afcB == FC_GAP or afcA.bytes == afcB.bytes
         # or isinstance(afcA, Template) and isinstance(afcB, MessageSegment) and set(afcB.bytes) == {0}
         # or isinstance(afcB, Template) and isinstance(afcA, MessageSegment) and set(afcA.bytes) == {0}
         # or isinstance(afcA, Template) and afcB.bytes in [bs.bytes for bs in afcA.baseSegments]
         # or isinstance(afcB, Template) and afcA.bytes in [bs.bytes for bs in afcB.baseSegments]
         )
         for afcA, afcB in zip(*alignedFieldClasses[(clunuA, clunuB)])
    ] for clunuA, clunuB in clusterpairs }
    matchingClusters = [(clunuA, clunuB) for clunuA, clunuB in clusterpairs if mergingThreshold >= len(
        [ False for condResult in matchingConditions[(clunuA, clunuB)][1:] if not any(condResult) ]) and
                        len([True for c in matchingConditions[(clunuA, clunuB)][1:] if c[5] or c[6]]) <=
                        ceil(.1*len(matchingConditions[(clunuA, clunuB)][1:])) ]
    # dynStaPairs may not exceed 10% of fields (ceiling) to match
    # [(clupair, len([a[6] for a in matchingConditions[clupair] if a[5] == True or a[6] == True]),
    #   len(matchingConditions[clupair]) - 1) for clupair in matchingClusters]


    # search in filteredMatches for STATIC - DYNAMIC - STATIC with different static values and remove from matchingClusters
    # : the matches on grounds of the STA value in DYN condition, with the DYN role(s) in a set in the first element of each tuple
    dynStaPairs = list()
    for clunuPair in matchingClusters:
        dynRole = [ clunuPair[0] if fieldCond[-2] else clunuPair[1] for fieldCond in matchingConditions[clunuPair][1:]
            if not any(fieldCond[:-2]) and (fieldCond[-2] or fieldCond[-1]) ]
        if dynRole:
            dynStaPairs.append((set(dynRole), clunuPair))
    dynRoles = set(itertools.chain.from_iterable([dynRole for dynRole, clunuPair in dynStaPairs]))
    # List of STA roles for each DYN role
    staRoles = {dynRole: [clunuPair[0] if clunuPair[1] in dr else clunuPair[1] for dr, clunuPair in dynStaPairs
                 if dynRole in dr] for dynRole in dynRoles}
    removeFromMatchingClusters = list()
    for dynRole, staRoleList in staRoles.items():
        try:
            staMismatch = False
            staValues = None
            clunuPairs = list()
            for staRole in staRoleList:
                clunuPair = (dynRole, staRole) if (dynRole, staRole) in matchingConditions else (staRole, dynRole) \
                    if (staRole, dynRole) in matchingConditions else None
                if clunuPair is None:
                    print("Skipping ({}, {})".format(staRole, dynRole))
                    continue
                clunuPairs.append(clunuPair)
                cluPairCond = matchingConditions[clunuPair]
                fieldMatches = [fieldNum for fieldNum, fieldCond in enumerate(cluPairCond[1:])
                              if not any(fieldCond[:-2]) and (fieldCond[-2] or fieldCond[-1])]
                curStaValues = [alignedFieldClasses[clunuPair][0][fieldMatch]
                                if clunuPair[0] == staRole else alignedFieldClasses[clunuPair][1][fieldMatch]
                            for fieldMatch in fieldMatches]
                if staValues is None:  # set the static values list to the currently determined STA field values for the DYN-STA pair clunuPair
                    staValues = curStaValues
                elif sorted([sv.bytes for sv in staValues]) != sorted([csv.bytes for csv in curStaValues]):
                    staMismatch = True
                    break
            if staMismatch:
                # mask to remove the clunuPairs of all combinations with this dynRole
                removeFromMatchingClusters.extend([
                    (dynRole, staRole) if (dynRole, staRole) in matchingConditions else (staRole, dynRole)
                    for staRole in staRoleList
                ])
        except KeyError as e:
            print(e)
            IPython.embed()
            raise e
    # if a chain of matches would merge more than .5 of all clusters, remove that chain
    from networkx import Graph
    from networkx.algorithms.components.connected import connected_components
    dracula = Graph()
    dracula.add_edges_from(set(matchingClusters) - set(removeFromMatchingClusters))
    connectedDracula = list(connected_components(dracula))
    for chain in connectedDracula:
        if len(chain) > .5 * len(alignedClusters):
            for clunu in chain:
                for remainingPair in set(matchingClusters) - set(removeFromMatchingClusters):
                    if clunu in remainingPair:
                        removeFromMatchingClusters.append(remainingPair)

    remainingClusters = set(matchingClusters) - set(removeFromMatchingClusters)
    if len(remainingClusters) > 0:
        print("Clusters could be merged:")
        for clunuAB in remainingClusters:
            cluTable = [(clunu, *[fv.bytes.hex() if isinstance(fv, MessageSegment) else
                       fv.bytes.decode() if isinstance(fv, Template) else fv for fv in fvals])
             for clunu, fvals in zip(clunuAB, alignedFieldClasses[clunuAB])] + list(zip(*matchingConditions[clunuAB]))
            fNums = []
            for fNum, (cluA, cluB) in enumerate(zip(cluTable[0], cluTable[1])):
                if not cluA == cluB:
                    fNums.append(fNum)
            cluDiff = [[col for colNum, col in enumerate(line) if colNum in fNums] for line in cluTable]
            print(tabulate(cluDiff, headers=fNums))
            print()

    print("remove:", removeFromMatchingClusters)

    print("remain:", remainingClusters)
    chainedRemains = Graph()
    chainedRemains.add_edges_from(remainingClusters)
    connectedClusters = list(connected_components(dracula))
    print("connected:", connectedClusters)

    print("should merge:")
    # identify over-specified clusters and list which ones ideally would be merged:
    clusterMostFreqStats = {stats[1]: [] for stats in clusterStats if stats is not None}
    # 'cluster_label', 'most_freq_type', 'precision', 'recall', 'cluster_size'
    for stats in clusterStats:
        if stats is None:
            continue
        cluster_label, most_freq_type, precision, recall, cluster_size = stats
        clusterMostFreqStats[most_freq_type].append((cluster_label, precision))
    overspecificClusters = {mft: stat for mft, stat in clusterMostFreqStats.items() if len(stat) > 1}
    for mft, stat in overspecificClusters.items():
        superCluster = [str(label) if precision > 0.5 else "[{}]".format(label) for label, precision in stat]
        print("    \"{}\"  ({})".format(mft, ", ".join(superCluster)))
    # END # of # check for cluster merge candidates #



















    # write alignments to csv
    reportFolder = "reports"
    fileNameS = "NEMETYL-symbols-" + plotTitle + "-" + pcapName
    csvpath = join(reportFolder, fileNameS + '.csv')
    if not exists(csvpath):
        print('Write alignments to {}...'.format(csvpath))
        with open(csvpath, 'w') as csvfile:
            symbolcsv = csv.writer(csvfile)
            for clunu, clusg in alignedClusters.items():
                symbolcsv.writerow(["# Cluster", clunu, "- Fields -", "- Alignment -"])
                symbolcsv.writerows([sg.bytes.hex() if sg is not None else '' for sg in msg] for msg in clusg)
                symbolcsv.writerow(["---"] * 5)
    else:
        print("Symbols not saved. File {} already exists.".format(csvpath))
        if not args.interactive:
            IPython.embed()

    writePerformanceStatistics(
        specimens, clusterer,
        "{} {} {}".format(tokenizer, analysis_method, distance_method),
        segmentationTime, dist_calc_segmentsTime, dist_calc_messagesTime,
        cluster_params_autoconfTime, cluster_messagesTime, align_messagesTime
    )


    if args.interactive:
        from tabulate import tabulate
        IPython.embed()




