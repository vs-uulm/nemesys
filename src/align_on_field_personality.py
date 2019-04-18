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

from alignment.alignMessages import SegmentedMessages, alignFieldClasses, FC_GAP
from inference.segmentHandler import segmentsFixed
from inference.segments import AbstractSegment
from inference.templates import DistanceCalculator, DelegatingDC, Template
from alignment.hirschbergAlignSegments import HirschbergOnSegmentSimilarity, NWonSegmentSimilarity
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

    # ksteepeststats = list()

    # can we omit k = 0 ?
    # No - recall and even more so precision deteriorates for dns and dhcp (1000s)
    for k in range(0, len(neighbors) // 10):  # round(2*log(len(neighbors)))
        knearest[k] = sorted([nfori[k][1] for nfori in neighbors])
        smoothknearest[k] = gaussian_filter1d(knearest[k], sigma)
        # max of second difference (maximum upwards curvature) as knee
        seconddiff[k] = numpy.diff(smoothknearest[k], 2)
        seconddiffargmax = seconddiff[k].argmax()
        diffrelmax = seconddiff[k].max() / smoothknearest[k][seconddiffargmax]
        if sigma < seconddiffargmax < len(neighbors) - sigma and diffrelmax > seconddiffMax[2]:
            seconddiffMax = (k, seconddiffargmax, diffrelmax)

        # ksteepeststats.append((k, seconddiff[k].max(), diffrelmax))
    # print(tabulate(ksteepeststats, headers=("k", "max(f'')", "max(f'')/f")))

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
    smcachefn = 'cache-sm-{}-{}-{}.{}'.format(analysisTitle, tokenizer, pcapName, 'sm')
    # dccachefn = 'cache-dc-{}-{}-{}.{}'.format(analysisTitle, tokenizer, pcapName, 'dc')
    if not exists(dccachefn):
        # dissect and label messages
        print("Load messages...")
        specimens = SpecimenLoader(args.pcapfilename, 2, True)
        comparator = MessageComparator(specimens, 2, True, debug=debug)

        # TODO select tokenizer by command line parameter
        print("Segmenting messages...", end=' ')
        segmentationTime = time.time()
        if tokenizer == "tshark":
            # 1. segment messages according to true fields from the labels
            segmentedMessages = annotateFieldTypes(analyzerType, analysisArgs, comparator)
        elif tokenizer == "4bytesfixed":
            # 2. segment messages into fixed size chunks for testing
            segmentedMessages = segmentsFixed(4, comparator, analyzerType, analysisArgs)
        elif tokenizer == "nemesys":
            # 3. segment messages by NEMESYS
            segmentedMessages = None
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

    # if not exists(smcachefn):
    print("Calculate distance for {} messages...".format(len(segmentedMessages)))
    dist_calc_messagesTime = time.time()
    sm = SegmentedMessages(dc, segmentedMessages)
    dist_calc_messagesTime = time.time() - dist_calc_messagesTime
    #     with open(smcachefn, 'wb') as f:
    #         pickle.dump(sm, f, pickle.HIGHEST_PROTOCOL)
    # else:
    #     print("Load distances from cache file {}".format(smcachefn))
    #     sm = pickle.load(open(smcachefn, 'rb'))
    #     if not isinstance(sm, SegmentedMessages):
    #         print('Loading of cached message distances failed.')
    #         exit(11)

    cluster_params_autoconfTime = time.time()
    eps, min_samples = sm.autoconfigureDBSCAN()
    cluster_params_autoconfTime = time.time() - cluster_params_autoconfTime


    # # DEBUG and TESTING
    # #
    # # retrieve manually determined epsilon value
    # pcapbasename = basename(args.pcapfilename)
    # epsilon = message_epspertrace[pcapbasename] if pcapbasename in message_epspertrace else 0.15
    # eps = epsautoconfeval(eps)
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

    # # plot distances and message clusters
    # print("Plot distances...")
    # from visualization.distancesPlotter import DistancesPlotter
    # dp = DistancesPlotter(specimens, 'message-distances-' + plotTitle, False)
    # dp.plotManifoldDistances(
    #     [specimens.messagePool[seglist[0].message] for seglist in segmentedMessages],
    #     sm.distances, labels)  # segmentedMessages
    # dp.writeOrShowFigure()

    # align cluster members
    align_messagesTime = time.time()
    alignedClusters = dict()
    alignedClustersHex = dict()
    print("Align each cluster...")
    for clunu, msgcluster in messageClusters.items():  # type: int, List[Tuple[MessageSegment]]
        clusteralignment, alignedsegments = sm.alignMessageType(msgcluster)
        alignedClusters[clunu] = alignedsegments

        # get gaps at the corresponding positions
        # print('Cluster', clunu)
        alignedClustersHex[clunu] = [[s.bytes.hex() if s is not None else None for s in m] for m in alignedsegments]
    print()
    align_messagesTime = time.time() - align_messagesTime

















    # check for cluster merge candidates
    print("Check for cluster merge candidates...")
    from alignment.alignMessages import mergeClusters
    from utils.evaluationHelpers import printClusterMergeConditions
    def lenAndTrue(boolist, length=2, truths=0):
        return len(boolist) <= length and len([a for a in boolist if a]) > truths

    clusterpairs, alignedFieldClasses = alignFieldClasses(alignedClusters, dc, (0,-1,5))
    globals().update(locals())
    #                                            0      1       2       3       4      5      6      7          8
    # noinspection PyTypeChecker
    matchingConditions = { (clunuA, clunuB): [("Agap","Bgap","equal","Azero","Bzero","BinA","AinB","DSdist","SSdist")] + [
            (afcA == FC_GAP, afcB == FC_GAP,
             isinstance(afcA, (MessageSegment, Template)) and isinstance(afcB, (MessageSegment, Template)) and afcA.bytes == afcB.bytes,
             isinstance(afcA, MessageSegment) and set(afcA.bytes) == {0}, isinstance(afcB, MessageSegment) and set(afcB.bytes) == {0},
             isinstance(afcA, Template) and isinstance(afcB, MessageSegment) and afcB.bytes in [bs.bytes for bs in afcA.baseSegments],
             isinstance(afcB, Template) and isinstance(afcA, MessageSegment) and afcA.bytes in [bs.bytes for bs in afcB.baseSegments],
             0.7 > afcA.distToNearest(afcB, dc)  # 0.2 dhcp-1000: +2merges // 9,30
                 if isinstance(afcA, Template) and isinstance(afcB, MessageSegment)
                 else 0.7 > afcB.distToNearest(afcA, dc)
                 if isinstance(afcB, Template) and isinstance(afcA, MessageSegment)
                 else False,
             0.2 > dc.pairDistance(afcA, afcB)
                 if isinstance(afcA, MessageSegment) and isinstance(afcB, MessageSegment)
                 else False
             )
             for afcA, afcB in zip(*alignedFieldClasses[(clunuA, clunuB)])
        ] for clunuA, clunuB in clusterpairs }
    globals().update(locals())
    matchingClusters = [
        (clunuA, clunuB) for clunuA, clunuB in clusterpairs
            # any of "Agap","Bgap","equal","Azero","Bzero","BinA","AinB" are true
            if (all([any(condResult[:7]) for condResult in matchingConditions[(clunuA, clunuB)][1:]])
                # dynStaPairs may not exceed 10% of fields (ceiling) to match
                and len([True for c in matchingConditions[(clunuA, clunuB)][1:] if c[5] or c[6]])
                    <= ceil(.1 * len(matchingConditions[(clunuA, clunuB)][1:])))
            # if merging is based solely on SSdist for any field, allow only one other "if not equal"
            or (all([any(condResult) for condResult in matchingConditions[(clunuA, clunuB)][1:]])
                and lenAndTrue( [not any(condResult[:7]) and condResult[8]  # True if solely SSdist for field
                    for condResult in matchingConditions[(clunuA, clunuB)][1:] if not condResult[2]]
                )
                # and not any([True for condResult in matchingConditions[(clunuA, clunuB)][1:]
                #           if not condResult[2] and any(condResult[5:8])])  # prevents ntp merging of (1, 6) solely on ntp.stratum STA-STA
            or all([condResult[7]                                              # ! this is the key for DHCP
                 for condResult in matchingConditions[(clunuA, clunuB)][1:] if not any(condResult[:7])])
                # condResult[2:7] lets (ALL) queries be merged for DNS / removes a number of merges in DHCP /
                # replaces one valid merge for NTP by two invalid ones
                )
        ]
            # if mergingThreshold >= len(
            #     [ False for condResult in matchingConditions[(clunuA, clunuB)][1:]
            #       if not any(condResult) ]) and

            #     ]
    globals().update(locals())
    # [(clupair, len([a[6] for a in matchingConditions[clupair] if a[5] == True or a[6] == True]),
    #   len(matchingConditions[clupair]) - 1) for clupair in matchingClusters]

    mergedClusters, missedmergepairs = mergeClusters(
        messageClusters, clusterStats, alignedClusters, alignedFieldClasses,
                  clusterpairs, matchingClusters, matchingConditions, dc)
    globals().update(locals())
    mergedClusterStats, mergedConciseness = writeMessageClusteringStaticstics(
        mergedClusters, groundtruth,
        "merged-{}-{}-eps={:.2f}-min_samples={}".format(
            tokenizer, type(clusterer).__name__, clusterer.eps, clusterer.min_samples),
        comparator)

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
        # globals().update(locals())




