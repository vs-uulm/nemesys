"""
Use groundtruth about field segmentation by dissectors and align segments
on the similarity of their feature "personality".

Takes a PCAP trace of a known protocol, dissects each message into their fields, and yields segments from each of them.
These segments get analyzed by the given analysis method which is used as feature to determine their similarity.
Similar fields are then aligned.
"""

import argparse, IPython, pickle, csv
import time
from math import ceil
from os.path import isfile, splitext, basename, exists, join
from itertools import chain
from typing import List, Any, Union

from tabulate import tabulate

# # change drawing toolkit for matplotlib (to be set before the pyplot import)
# import matplotlib as mpl
# mpl.use('Agg')

from alignment.alignMessages import SegmentedMessages
from inference.segmentHandler import segmentsFixed, bcDeltaGaussMessageSegmentation, refinements
from inference.templates import DistanceCalculator, DelegatingDC, Template
from alignment.hirschbergAlignSegments import HirschbergOnSegmentSimilarity, NWonSegmentSimilarity
from utils.evaluationHelpers import *
from validation.dissectorMatcher import MessageComparator
from utils.loader import SpecimenLoader
from characterize_fieldtypes import analyses
from visualization.multiPlotter import MultiMessagePlotter
from alignment.clusterMerging import ClusterMerger, ClusterClusterer
from utils.baseAlgorithms import ecdf

debug = False

analysis_method = 'value'
distance_method = 'canberra'
tokenizers = ('tshark', '4bytesfixed', 'nemesys')
roundingprecision = 10**8


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

def columnOfAlignment(alignedSegments: List[List[MessageSegment]], colnum: int):
    return [msg[colnum] for msg in alignedSegments]


# noinspection PyShadowingNames
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

    dists2first = ["- (reference)"] + list(dc.distancesSubset(stripedcol[0:1], stripedcol[1:]).tolist())[0]  # type: List[Union[str, None]]

    # re-insert Nones
    for idx in nonepos:
        dists2first.insert(idx, None)

    # transpose
    d2ft = list(map(list, zip(column, dists2first)))
    return d2ft

def printSegDist(d2ft: List[Tuple[MessageSegment, float]]):
    print(tabulate([(s.bytes.hex() if isinstance(s, MessageSegment) else "-", d) for s, d in d2ft],
                   headers=['Seg (hex)', 'Distance'], floatfmt=".4f"))


# noinspection PyShadowingNames
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


# noinspection PyShadowingNames
def quicksegmentTuple(dc: DistanceCalculator, segment: MessageSegment):
    return dc.segments2index([segment])[0], segment.length, tuple(segment.values)


def epsautoconfcdf(epsilon):
    """
    According to the introspection of the plots, there is no usable epsilon detected by kneedle.
    TODO Probably using the raw maximum curvature by own calculation could be an alternative if necessary.

    :param epsilon:
    :return:
    """
    neighbors = sm.neighbors()  # list of tuples: (index from sm.distances, distance) sorted by distance
    from math import log
    sigma = log(len(neighbors))

    mmp = MultiMessagePlotter(specimens, tokenizer + "-knn-cdf-distance-funtion", 1, 2,
                              isInteractive=False)
    mmp.axes[0].axvline(epsilon, label="manually determined eps={:0.2f}".format(epsilon), c="red")
    mmp.axes[1].axvline(epsilon, label="manually determined eps={:0.2f}".format(epsilon), c="red")

    krange = (0, 16, 1)

    # kneedle approach
    from kneed import KneeLocator
    kneeK = dict()
    for k in range(*krange):
        knearest = ecdf([nfori[k][1] for nfori in neighbors], False)
        # smoothknn = gaussian_filter1d(knearest[k], sigma)

        # knArray = numpy.array([a for a in zip(*knearest)])
        # IPython.embed()
        # mmp.plotToSubfig(1, knearest, alpha=.4, label="k={}".format(k))
        mmp.axes[1].plot(*knearest, alpha=.4, label="k={}".format(k))

        kneel = KneeLocator(*knearest, curve='convex', direction='increasing')
        kneeX = kneel.knee
        kneeK[k] = kneeX
        mmp.axes[0].axvline(kneeX, linestyle='dashed', color='blue', alpha=.4,
                            label="kneedle {:.2f} of k={}".format(kneeX, k))
        if k == round(sigma):
            mmp.axes[1].axvline(kneeX, linestyle='dashed', color='blue', alpha=.4,
                                label="kneedle {:.2f} of k={}".format(kneeX, k))
            mmp.axes[0].plot(*knearest, alpha=.4)

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

    mmp.writeOrShowFigure()

    return None

def epsautoconfeval(epsilon):
    """
    investigate distance properties for clustering autoconfiguration
    plots of k-nearest-neighbor distance histogram and "knee"

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

    mmp = MultiMessagePlotter(specimens, tokenizer + "-knn-distance-funtion", 1, 2,
                              isInteractive=False)
    mmp.axes[0].axhline(epsilon, label="manually determined eps={:0.2f}".format(epsilon), c="red")
    mmp.axes[1].axhline(epsilon, label="manually determined eps={:0.2f}".format(epsilon), c="red")

    krange = (0, 16, 1)

    for k in range(*krange):
        knearest = sorted([nfori[k][1] for nfori in neighbors])
        mmp.plotToSubfig(1, knearest, alpha=.4, label="k={}".format(k))

    # # kneedle approach: yields unusable results. does not find a knee!


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
    #   --> No - recall and even more so precision deteriorates for dns and dhcp (1000s)
    for k in range(0, len(neighbors) // 10):  # round(2*log(len(neighbors)))
        knearest[k] = sorted([nfori[k][1] for nfori in neighbors])
        smoothknearest[k] = gaussian_filter1d(knearest[k], sigma)
        # max of second difference (maximum upwards curvature) as knee
        seconddiff[k] = numpy.diff(smoothknearest[k], 2)
        seconddiffargmax = seconddiff[k].argmax()
        diffrelmax = seconddiff[k].max() / smoothknearest[k][seconddiffargmax]
        if 2*sigma < seconddiffargmax < len(neighbors) - 2*sigma and diffrelmax > seconddiffMax[2]:
            seconddiffMax = (k, seconddiffargmax, diffrelmax)

        # ksteepeststats.append((k, seconddiff[k].max(), diffrelmax))
    # print(tabulate(ksteepeststats, headers=("k", "max(f'')", "max(f'')/f")))

    # prepare to plot the smoothed nearest neighbor distribution and its second derivative
    k = seconddiffMax[0]
    x = seconddiffMax[1] + 1

    # # calc mean of first derivative to estimate the noisiness (closer to 1 is worse)
    # firstdiff = numpy.diff(smoothknearest[k], 1)
    # # alt: integral
    # diag = numpy.empty_like(smoothknearest[k])
    # for i in range(diag.shape[0]):
    #     diag[i] = smoothknearest[k][0] + i*(smoothknearest[k][-1] - smoothknearest[k][0])/smoothknearest[k][-1]
    # belowdiag = diag - smoothknearest[k]
    # print("f' median={:.2f}".format(numpy.median(firstdiff)))
    # print("diag-f={:.2f}".format(sum(belowdiag)))

    mmp.plotToSubfig(0, smoothknearest[k], label="smooth k={}, sigma={:.2f}".format(k, sigma), alpha=.4)
    mmp.plotToSubfig(1, smoothknearest[k], label="smooth k={}, sigma={:.2f}".format(k, sigma), alpha=1, color='blue')
    mmp.plotToSubfig(0, knearest[k], alpha=.4)

    ax0twin = mmp.axes[0].twinx()
    # mmp.plotToSubfig(ax0twin, seconddiff[k], linestyle='dotted', color='cyan', alpha=.4)
    mmp.plotToSubfig(ax0twin, [None] + list(seconddiff[k].tolist()), linestyle='dotted',
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
    parser.add_argument('-t', '--tokenizer', help='Select the tokenizer for this analysis run.', default="tshark")
    parser.add_argument('-s', '--sigma', type=float, help='Only NEMESYS: sigma for noise reduction (gauss filter),'
                                                          'default: 0.9')
    parser.add_argument('--split', help='Use old split-clusters implementation.',
                        action="store_true")
    parser.add_argument('-p', '--with-plots',
                        help='Generate plots.',
                        action="store_true")
    args = parser.parse_args()
    withplots = args.with_plots

    print("\n\n")

    if not isfile(args.pcapfilename):
        print('File not found: ' + args.pcapfilename)
        exit(1)
    pcapbasename = basename(args.pcapfilename)

    analyzerType = analyses[analysis_method]
    analysisArgs = None
    analysisTitle = analysis_method

    if args.tokenizer in tokenizers:
        tokenizer = args.tokenizer
    else:
        print("Unsupported tokenizer:", args.tokenizer, "allowed values are:", tokenizers)
        exit(2)

    # # # # # # # # # # # # # # # # # # # # # # # #
    # cache/load the DistanceCalculator to the filesystem
    # # # # # # # # # # # # # # # # # # # # # # # #
    # TODO when manipulating distances, deactivate caching! by adding "True"
    # noinspection PyUnboundLocalVariable
    specimens, comparator, segmentedMessages, dc, segmentationTime, dist_calc_segmentsTime = cacheAndLoadDC(
        args.pcapfilename, analysisTitle, tokenizer, debug, analyzerType, analysisArgs, args.sigma   # , True
    )
    chainedSegments = dc.rawSegments



    # # # # # # # # # # # # # # # # # # # # # # # #
    # triangle empirics: segments
    # # # # # # # # # # # # # # # # # # # # # # # #
    # from scipy.special import binom
    # print("count of segment triples", 6 * binom(dc.distanceMatrix.shape[0], 3))
    # from itertools import permutations
    # triangles = list()
    # falseTriangleCount = 0
    # chunkIndex = 0
    # for a, b, c in permutations(range(dc.distanceMatrix.shape[0]), 3):
    #     triangles.append((dc.distanceMatrix[a,b] + dc.distanceMatrix[b,c], dc.distanceMatrix[a,c]))
    #     if chunkIndex > 100000:
    #         triround = numpy.array(triangles).round(6)
    #         triineq = numpy.greater_equal(triround[:, 0], triround[:, 1])
    #         falseTriangleCount += len(triineq[triineq == False])
    #         del triround, triineq
    #         triangles = list()
    #         chunkIndex = -1
    #     chunkIndex += 1
    # triround = numpy.array(triangles).round(6)
    # triineq = numpy.greater_equal(triround[:, 0], triround[:, 1])
    # falseTriangleCount += len(triineq[triineq == False])
    # del triround, triineq
    # print(falseTriangleCount, "segment distance triangles false of", 6*binom(dc.distanceMatrix.shape[0], 3))
    # del triangles
    # # # # # # # # # # # # # # # # # # # # # # # #



    # # # # # # # # # # # # # # # # # # # # # # # #
    # if not exists(smcachefn):
    print("Calculate distance for {} messages...".format(len(segmentedMessages)))
    dist_calc_messagesTime = time.time()
    sm = SegmentedMessages(dc, segmentedMessages)
    dist_calc_messagesTime = time.time() - dist_calc_messagesTime
    # smcachefn = 'cache-sm-{}-{}-{}.{}'.format(analysisTitle, tokenparm, pcapName, 'sm')
    #     with open(smcachefn, 'wb') as f:
    #         pickle.dump(sm, f, pickle.HIGHEST_PROTOCOL)
    # else:
    #     print("Load distances from cache file {}".format(smcachefn))
    #     sm = pickle.load(open(smcachefn, 'rb'))
    #     if not isinstance(sm, SegmentedMessages):
    #         print('Loading of cached message distances failed.')
    #         exit(11)
    # # # # # # # # # # # # # # # # # # # # # # # #
    cluster_params_autoconfTime = time.time()
    eps, min_samples = sm.autoconfigureDBSCAN()
    cluster_params_autoconfTime = time.time() - cluster_params_autoconfTime
    # # # # # # # # # # # # # # # # # # # # # # # #


    # # # # # # # # # # # # # # # # # # # # # # # #
    # triangle empirics: messages
    # # # # # # # # # # # # # # # # # # # # # # # #
    # from itertools import permutations
    # from scipy.special import binom
    # triangles = list()
    # falseTriangleCount = 0
    # chunkIndex = 0
    # for a, b, c in permutations(range(sm.distances.shape[0]), 3):
    #     triangles.append((sm.distances[a,b] + sm.distances[b,c], sm.distances[a,c]))
    #     if chunkIndex > 100000:
    #         triround = numpy.array(triangles).round(6)
    #         triineq = numpy.greater_equal(triround[:, 0], triround[:, 1])
    #         falseTriangleCount += len(triineq[triineq == False])
    #         del triround, triineq
    #         triangles = list()
    #         chunkIndex = -1
    #     chunkIndex += 1
    # triround = numpy.array(triangles).round(6)
    # triineq = numpy.greater_equal(triround[:, 0], triround[:, 1])
    # falseTriangleCount += len(triineq[triineq == False])
    # del triround, triineq
    # print(falseTriangleCount, "message distance triangles false of", 6*binom(sm.distances.shape[0], 3))
    # del triangles
    # # # # # # # # # # # # # # # # # # # # # # # #


    # # # # # # # # # # # # # # # # # # # # # # # #
    # DEBUG and TESTING
    # # # # # # # # # # # # # # # # # # # # # # # #
    # retrieve manually determined epsilon value
    # epsilon = message_epspertrace[pcapbasename] if pcapbasename in message_epspertrace else 0.15
    if tokenizer == "nemesys":
        eps = eps * .8
    if withplots:
        epsConfirm = epsautoconfeval(eps)
        # epsautoconfcdf(eps)  # CDF°!°
        # exit()
    # DEBUG and TESTING
    # # # # # # # # # # # # # # # # # # # # # # # #


    # # # # # # # # # # # # # # # # # # # # # # # #
    # cluster and align messages and calculate statistics of it
    # # # # # # # # # # # # # # # # # # # # # # # #
    print('Clustering messages...')
    cluster_messagesTime = time.time()
    messageClusters, labels, clusterer = sm.clusterMessageTypesDBSCAN(eps=eps, min_samples=3)
    cluster_messagesTime = time.time() - cluster_messagesTime
    plotTitle = "{}-{} eps {:.3f} ms {}".format(
        tokenizer, type(clusterer).__name__, clusterer.eps, clusterer.min_samples)
    # messageClusters, labels, clusterer = sm.clusterMessageTypesHDBSCAN()
    # plotTitle = "{}-{} mcs {} ms {}".format(
    #     tokenizer, type(clusterer).__name__, clusterer.min_cluster_size, clusterer.min_samples)
    # # # # # # # # # # # # # # # # # # # # # # # #


    groundtruth = {msg: pm.messagetype for msg, pm in comparator.parsedMessages.items()}
    for msg, mtype in groundtruth.items():
        msg.messageType = mtype

    minCsize = numpy.log(len(segmentedMessages))


    # # # # # # # # # # # # # # # # # # # # # # # #
    # write message clustering statistics to csv
    # # # # # # # # # # # # # # # # # # # # # # # #
    clusterStats, conciseness = writeMessageClusteringStaticstics(
        messageClusters, groundtruth, "{}-{}-eps={:.2f}-min_samples={}".format(
        tokenizer, type(clusterer).__name__, clusterer.eps, clusterer.min_samples), comparator)
    # # # # # # # #
    writeCollectiveClusteringStaticstics(
        messageClusters, groundtruth, "{}-{}-eps={:.2f}-min_samples={}".format(
        tokenizer, type(clusterer).__name__, clusterer.eps, clusterer.min_samples), comparator)
    # # # # # # # # min cluster size
    noisekey = 'Noise' if 'Noise' in messageClusters else -1
    filteredClusters = {k: v for k, v in messageClusters.items() if len(v) >= minCsize}
    filteredClusters[noisekey] = list() if not noisekey in filteredClusters else filteredClusters[noisekey].copy()
    filteredClusters[noisekey].extend(s for k, v in messageClusters.items()
                                      if len(v) < minCsize for s in v)
    writeCollectiveClusteringStaticstics(
        filteredClusters, groundtruth, "{}-{}-eps={:.2f}-min_samples={}-minCsize".format(
        tokenizer, type(clusterer).__name__, clusterer.eps, clusterer.min_samples), comparator)
    # # # # # # # # # # # # # # # # # # # # # # # #


    if withplots:
        # plot distances and message clusters
        print("Plot distances...")
        from visualization.distancesPlotter import DistancesPlotter
        dp = DistancesPlotter(specimens, 'message-distances-' + plotTitle, False)
        dp.plotManifoldDistances(
            [specimens.messagePool[seglist[0].message] for seglist in segmentedMessages],
            sm.distances, labels)  # segmentedMessages
        dp.writeOrShowFigure()


    # # # # # # # # # # # # # # # # # # # # # # # #
    # align cluster members
    # # # # # # # # # # # # # # # # # # # # # # # #
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
    # # # # # # # # # # # # # # # # # # # # # # # #










    # # # # # # # # # # # # # # # # # # # # # # # #
    # split clusters based on fields without rare values
    # # # # # # # # # # # # # # # # # # # # # # # #
    if not args.split:
        from alignment.clusterSplitting import *

        cSplitter = RelaxedExoticClusterSplitter(6 if not tokenizer == "tshark" else 3,
                                    alignedClusters, messageClusters, sm)
        cSplitter.activateCVSout("{}-{}-eps={:.2f}-min_samples={}".format(
            tokenizer, type(clusterer).__name__, clusterer.eps, clusterer.min_samples),
            comparator.specimens.pcapFileName, {cs[0]: cs[2] for cs in clusterStats if cs is not None})
        # in-place split of clusters in alignedClusters and messageClusters
        cSplitter.split()
        labels = cSplitter.labels

        # # # # # # # # # # # # # # # # # # # # # # # #
        if withplots:
            # plot distances and message clusters
            print("Plot distances...")
            from visualization.distancesPlotter import DistancesPlotter

            dp = DistancesPlotter(specimens, 'message-distances-' + plotTitle + '-split', False)
            dp.plotManifoldDistances(
                [specimens.messagePool[seglist[0].message] for seglist in segmentedMessages],
                sm.distances, labels)  # segmentedMessages
            dp.writeOrShowFigure()

        # # # # # # # # # # # # # # # # # # # # # # # #
        # clusterStats for merger
        # # # # # # # # # # # # # # # # # # # # # # # #
        # write message clustering statistics to csv
        # # # # # # # # # # # # # # # # # # # # # # # #
        clusterStats, conciseness = writeMessageClusteringStaticstics(
            messageClusters, groundtruth, "{}-{}-eps={:.2f}-min_samples={}-split".format(
            tokenizer, type(clusterer).__name__, clusterer.eps, clusterer.min_samples), comparator)
        # # # # # # # #
        writeCollectiveClusteringStaticstics(
            messageClusters, groundtruth, "{}-{}-eps={:.2f}-min_samples={}-split".format(
            tokenizer, type(clusterer).__name__, clusterer.eps, clusterer.min_samples), comparator)
        # # # # # # # # min cluster size

        noisekey = 'Noise' if 'Noise' in messageClusters else -1
        filteredClusters = {k: v for k, v in messageClusters.items()
                            if len(v) >= minCsize }
        filteredClusters[noisekey] = list() if not noisekey in filteredClusters else filteredClusters[noisekey].copy()
        filteredClusters[noisekey].extend(s for k, v in messageClusters.items()
                                          if len(v) < minCsize for s in v)
        writeCollectiveClusteringStaticstics(
            filteredClusters, groundtruth, "{}-{}-eps={:.2f}-min_samples={}-split-minCsize".format(
            tokenizer, type(clusterer).__name__, clusterer.eps, clusterer.min_samples), comparator)
        # # # # # # # # # # # # # # # # # # # # # # # #



        # # # # # # # # # # # # # # # # # # # # # # # #
    else:
        # old implementation
        from collections import Counter

        exoticValueStats = "reports/exotic-values-statistics.csv"
        fieldLenThresh = 6 if not tokenizer == "tshark" else 3

        clusterReplaceMap = dict()
        for aNum, aClu in alignedClusters.items():
            if aNum == -1:
                continue
            cPrec = next(cs[2] for cs in clusterStats if cs is not None and cs[0] == aNum)
            freqThresh = numpy.floor(numpy.log(len(aClu)))  # numpy.round(numpy.log(len(aClu)))
            fields = [fld for fld in zip(*aClu)]  # type: List[List[MessageSegment]]
            distinctVals4fields = [{tuple(val.values) for val in fld if val is not None} for fld in fields]
            # amount of distinct values per field
            valAmount4fields = [len(valSet) for valSet in distinctVals4fields]

            print("\nCluster", aNum, "of size", len(aClu),
                  "- threshold", numpy.log(len(aClu)))
            print("Cluster should", "" if cPrec < 1 else "not", "be split. Precision is", cPrec)

            valCounts4fields = {fidx: Counter(tuple(seg.values) for seg in segs if seg is not None)
                                for fidx, segs in enumerate(fields)}
            pivotFieldIds = [fidx for fidx, vCnt in enumerate(valAmount4fields)
                 if 1 < vCnt <= freqThresh  # knee
                 and len([True for val in fields[fidx] if val is None]) <= freqThresh  # omit fields that have many gaps
                 and not any(val.length > fieldLenThresh for val in fields[fidx] if val is not None)  # omit fields longer than 3/4
                 and not any(set(val.values) == {0} for val in fields[fidx] if val is not None)  # omit fields that have zeros
                 and not any(cnt <= freqThresh
                             for cnt in valCounts4fields[fidx].values())]  # remove fields with exotic values

            preExotic = [fidx for fidx, vCnt in enumerate(valAmount4fields)
                 if 1 < vCnt <= freqThresh  # knee
                 and len([True for val in fields[fidx] if val is None]) <= freqThresh  # omit fields that have many gaps
                 and not any([val.length > fieldLenThresh for val in fields[fidx] if val is not None])  # omit fields longer than 3/4
                 and not any([set(val.values) == {0} for val in fields[fidx] if val is not None])  # omit fields that have zeros
                         ]

            for fidx in preExotic:
                scnt = sorted(valCounts4fields[fidx].values())
                diffmax = (numpy.diff(scnt).argmax()+1) if len(scnt) > 1 else "-"
                csvWriteHead = False if exists(exoticValueStats) else True
                with open(exoticValueStats, 'a') as csvfile:
                    exoticcsv = csv.writer(csvfile)  # type: csv.writer
                    if csvWriteHead:
                        exoticcsv.writerow([
                            'run_title', 'trace', 'cluster_label', 'precision', 'cluster_size', 'field',
                            'num_vals',
                            'maxdiff_n', 'maxdiff_v', 'sum<n', 'sum>=n', 'mean<n', 'mean>=n',
                            'stdev<n', 'stdev>=n', 'median<n', 'median>=n'
                        ])
                    fieldParameters = [ "{}-{}-eps={:.2f}-min_samples={}".format(
                                tokenizer, type(clusterer).__name__, clusterer.eps, clusterer.min_samples),
                            comparator.specimens.pcapFileName,
                            aNum, cPrec, len(aClu), fidx, len(scnt)]
                    if len(scnt) > 1:
                        exoticcsv.writerow([
                            *fieldParameters, diffmax, scnt[diffmax],
                            sum(scnt[:diffmax]), sum(scnt[diffmax:]),
                            numpy.mean(scnt[:diffmax]), numpy.mean(scnt[diffmax:]),
                            numpy.std(scnt[:diffmax]), numpy.std(scnt[diffmax:]),
                            numpy.median(scnt[:diffmax]), numpy.median(scnt[diffmax:])
                        ])
                    else:
                        exoticcsv.writerow(fieldParameters + [""] * 10)



            newExotic = list()
            for fidx in preExotic:
                scnt = sorted(valCounts4fields[fidx].values())
                if len(scnt) > 1:
                    if scnt[0] > freqThresh >= len(scnt):
                        newExotic.append(fidx)
                        continue

                    # the pivot index and value to split the sorted list of type amounts
                    # iVal, pVal = next((i, cnt) for i, cnt in enumerate(scnt) if cnt > freqThresh)
                    iVal = numpy.diff(scnt).argmax() + 1
                    # the special case of any(cnt <= freqThresh for cnt in scnt) is relaxedly included here
                    numValues_u = len(scnt) - iVal
                    # if there are no or only one frequent value, do not split
                    if numValues_u > 1:
                        pVal = scnt[iVal]
                        mean_u = numpy.mean(scnt[iVal:])
                        halfsRatio =  sum(scnt[:iVal]) / sum(scnt[iVal:])
                        if halfsRatio < 0.1 and mean_u > 2 * len(aClu) / numpy.log(len(aClu)):
                            newExotic.append(fidx)


            addExotics = set(newExotic) - set(pivotFieldIds)
            remExotics = set(pivotFieldIds) - set(newExotic)
            if not len(addExotics) == len(remExotics) == 0:
                print("pivot field changes due to new exotics:")
                print("  add fields:", addExotics)
                print("  remove fields:", remExotics)
            # print info only if we do not split
            if len(newExotic) == 0:
                print("no pivot fields left")
                # exoticCondition = [(fidx, any(cnt <= freqThresh for cnt in valCounts4fields[fidx].values()))
                #                    for fidx, vCnt in enumerate(valAmount4fields)]
                continue  # conditions not met for splitting: next cluster
            elif len(newExotic) > 2:
                print("too many pivot fields:", len(newExotic))
                continue  # conditions not met for splitting: next cluster


            print(newExotic)
            # split clusters
            clusterSplits = dict()  # type: Dict[Union[Tuple, None], List[Tuple[MessageSegment]]]
            for msgsegs in aClu:
                # concatenate multiple distinct field combinations
                pivotVals = tuple([(pId, *msgsegs[pId].values) if msgsegs[pId] is not None else None
                                   for pId in newExotic])
                if pivotVals not in clusterSplits:
                    clusterSplits[pivotVals] = list()
                clusterSplits[pivotVals].append(msgsegs)
            clusterReplaceMap[aNum] = clusterSplits
            print("replace cluster", aNum, "by")
            print(tabulate((clusterSplits.keys())))


        # replace clusters by their splits
        for aNum, clusterSplits in clusterReplaceMap.items():
            for nci, cluSpl in enumerate(clusterSplits.values()):  # type: int, List[Tuple[MessageSegment]]
                # newCluLabel = (aNum+1) * 100 + nci
                newCluLabel = "{}s{}".format(aNum, nci)

                msgs = [next(seg for seg in msgsegs if seg is not None).message for msgsegs in cluSpl]
                messageClusters[newCluLabel] = [msgsegs for msgsegs in messageClusters[aNum]
                                                if msgsegs[0].message in msgs]

                clusteralignment, alignedsegments = sm.alignMessageType(messageClusters[newCluLabel])
                alignedClusters[newCluLabel] = alignedsegments

            del alignedClusters[aNum]
            del messageClusters[aNum]

        # labels for distance plot
        msgLabelMap = {tuple(msgsegs): clunu for clunu, msgs in messageClusters.items() for msgsegs in msgs}
        labels = numpy.array([msgLabelMap[tuple(seglist)] for seglist in segmentedMessages])

        # # # # # # # # # # # # # # # # # # # # # # # #
        if withplots:
            # plot distances and message clusters
            print("Plot distances...")
            from visualization.distancesPlotter import DistancesPlotter

            dp = DistancesPlotter(specimens, 'message-distances-' + plotTitle + '-split', False)
            dp.plotManifoldDistances(
                [specimens.messagePool[seglist[0].message] for seglist in segmentedMessages],
                sm.distances, labels)  # segmentedMessages
            dp.writeOrShowFigure()

        # # # # # # # # # # # # # # # # # # # # # # # #
        # clusterStats for merger
        # # # # # # # # # # # # # # # # # # # # # # # #
        # write message clustering statistics to csv
        # # # # # # # # # # # # # # # # # # # # # # # #
        clusterStats, conciseness = writeMessageClusteringStaticstics(
            messageClusters, groundtruth, "{}-{}-eps={:.2f}-min_samples={}-split".format(
            tokenizer, type(clusterer).__name__, clusterer.eps, clusterer.min_samples), comparator)
        # # # # # # # #
        writeCollectiveClusteringStaticstics(
            messageClusters, groundtruth, "{}-{}-eps={:.2f}-min_samples={}-split".format(
            tokenizer, type(clusterer).__name__, clusterer.eps, clusterer.min_samples), comparator)
        # # # # # # # # min cluster size

        noisekey = 'Noise' if 'Noise' in messageClusters else -1
        filteredClusters = {k: v for k, v in messageClusters.items()
                            if len(v) >= minCsize }
        filteredClusters[noisekey] = list() if not noisekey in filteredClusters else filteredClusters[noisekey].copy()
        filteredClusters[noisekey].extend(s for k, v in messageClusters.items()
                                          if len(v) < minCsize for s in v)
        writeCollectiveClusteringStaticstics(
            filteredClusters, groundtruth, "{}-{}-eps={:.2f}-min_samples={}-split-minCsize".format(
            tokenizer, type(clusterer).__name__, clusterer.eps, clusterer.min_samples), comparator)
        # # # # # # # # # # # # # # # # # # # # # # # #






    # # # # # # # # # # # # # # # # # # # # # # # #
    # check for cluster merge candidates
    # # # # # # # # # # # # # # # # # # # # # # # #
    # TODO fully integrate into/encapsulate in ClusterMerger class
    print("Check for cluster merge candidates...")
    from utils.evaluationHelpers import printClusterMergeConditions
    # noinspection PyUnreachableCode
    if True:
        # ClusterMerger
        clustermerger = ClusterMerger(alignedClusters, dc)

        alignedFieldClasses = clustermerger.alignFieldClasses((0, -1, 5))  # TODO alt1
        # alignedFieldClasses = clustermerger.alignFieldClasses((0, -5, 5))  # TODO alt2
        if tokenizer == "nemesys":
            alignedFieldClasses = clustermerger.gapMerging4nemesys(alignedFieldClasses)
        matchingConditions = clustermerger.generateMatchingConditions(alignedFieldClasses)
        matchingClusters = ClusterMerger.selectMatchingClusters(alignedFieldClasses, matchingConditions)
        mergedClusters = clustermerger.mergeClusters(
            messageClusters, clusterStats, alignedFieldClasses, matchingClusters, matchingConditions)
        mergedClusterStats, mergedConciseness = writeMessageClusteringStaticstics(
            mergedClusters, groundtruth,
            "merged-{}-{}-eps={:.2f}-min_samples={}".format(
                tokenizer, type(clusterer).__name__, clusterer.eps, clusterer.min_samples),
            comparator)
        # # # # # # # #
        writeCollectiveClusteringStaticstics(
            mergedClusters, groundtruth,
            "merged-{}-{}-eps={:.2f}-min_samples={}".format(
                tokenizer, type(clusterer).__name__, clusterer.eps, clusterer.min_samples),
            comparator)
        # # # # # # # # min cluster size
        noisekey = 'Noise' if 'Noise' in mergedClusters else -1
        filteredMerged = {k: v for k, v in mergedClusters.items() if len(v) >= minCsize}
        filteredMerged[noisekey] = list() if not noisekey in filteredMerged else filteredMerged[noisekey].copy()
        filteredMerged[noisekey].extend(s for k, v in mergedClusters.items()
                                          if len(v) < minCsize for s in v)
        writeCollectiveClusteringStaticstics(
            filteredMerged, groundtruth,
            "merged-{}-{}-eps={:.2f}-min_samples={}-minCsize".format(
                tokenizer, type(clusterer).__name__, clusterer.eps, clusterer.min_samples),
            comparator)
    else:
        # # # # # # # # # # # # # # # # # # # # #
        # alternative idea of clustering clusters: does not improve merging - perhaps the similarity matrix is not good enough?!
        # ClusterClusterer
        clusterclusterer = ClusterClusterer(alignedClusters, dc)
        # clusterDists = clusterclusterer.calcClusterDistances()

        mergeEps, mergeMpts = clusterclusterer.autoconfigureDBSCAN()

        clusterClusters, labels, mergeclusterer = clusterclusterer.clusterMessageTypesDBSCAN(mergeEps, min_samples=2)
        clusterClustersNoiseless = {k: v for k, v in clusterClusters.items() if k > -1}
        mergedClusters = ClusterClusterer.mergeClusteredClusters(clusterClustersNoiseless, messageClusters)
        ClusterClusterer.printShouldMerge(list(clusterClustersNoiseless.values()), clusterStats)

        mergedClusterStats, mergedConciseness = writeMessageClusteringStaticstics(
            mergedClusters, groundtruth,
            "merged-{}-{}-eps={:.2f}-min_samples={}".format(
                tokenizer, type(mergeclusterer).__name__, mergeclusterer.eps, mergeclusterer.min_samples),
            comparator)

        from netzob.Model.Vocabulary.Messages.RawMessage import RawMessage
        from visualization.distancesPlotter import DistancesPlotter
        typedClusterDummys = list()
        for clunu in clusterclusterer.clusterOrder:
            clusta = None
            for stats in clusterStats:
                if stats is not None and stats[0] == clunu:
                    clusta = stats[1] if stats[2] == 1.0 else "({})".format(stats[1])
                    break
            msgdum = RawMessage(messageType=clusta)
            typedClusterDummys.append(msgdum)

        dp = DistancesPlotter(specimens, "cluster-clustering-" + plotTitle, False)
        dp.plotManifoldDistances(typedClusterDummys, clusterclusterer.distances, labels)
        dp.writeOrShowFigure()

    # # overwrite existing variables
    # # # # # # # # # # # # # # # # # # # # # # # #
    # messageClusters = mergedClusters
    #
    # # align clusters that have been merged
    # mergedAligned = dict()
    # for cLabel, clusterMerges in messageClusters.items():  # type: Union[int, str], List[Tuple[MessageSegment]]
    #     if cLabel not in alignedClusters:
    #         clusteralignment, alignedsegments = sm.alignMessageType(clusterMerges)
    #         mergedAligned[cLabel] = alignedsegments
    #     else:
    #         mergedAligned[cLabel] = alignedClusters[cLabel]
    # alignedClusters = mergedAligned
    # del mergedAligned
    #
    # # labels for distance plot
    # msgLabelMap = {tuple(msgsegs): clunu for clunu, msgs in messageClusters.items() for msgsegs in msgs}
    # labels = numpy.array([msgLabelMap[tuple(seglist)] for seglist in segmentedMessages])

    # END # of # check for cluster merge candidates #
    # # # # # # # # # # # # # # # # # # # # # # # #















    # # # # # # # # # # # # # # # # # # # # # # # #
    # write alignments to csv
    # # # # # # # # # # # # # # # # # # # # # # # #
    reportFolder = "reports"
    pcapName = splitext(pcapbasename)[0]
    fileNameS = "NEMETYL-symbols-" + plotTitle + "-" + pcapName
    csvpath = join(reportFolder, fileNameS + '.csv')
    if not exists(csvpath):
        print('Write alignments to {}...'.format(csvpath))
        with open(csvpath, 'w') as csvfile:
            symbolcsv = csv.writer(csvfile)
            for clunu, clusg in alignedClusters.items():
                symbolcsv.writerow(["# Cluster", clunu, "- Fields -", "- Alignment -"])
                symbolcsv.writerows(
                    [groundtruth[comparator.messages[next(seg for seg in msg if seg is not None).message]]]
                    + [sg.bytes.hex() if sg is not None else '' for sg in msg] for msg in clusg
                )
                symbolcsv.writerow(["---"] * 5)
    else:
        print("Symbols not saved. File {} already exists.".format(csvpath))
        if not args.interactive:
            IPython.embed()
    # # # # # # # # # # # # # # # # # # # # # # # #

    writePerformanceStatistics(
        specimens, clusterer,
        "{} {} {}".format(tokenizer, analysis_method, distance_method),
        segmentationTime, dist_calc_segmentsTime, dist_calc_messagesTime,
        cluster_params_autoconfTime, cluster_messagesTime, align_messagesTime
    )


    if args.interactive:
        from tabulate import tabulate
        # globals().update(locals())
        IPython.embed()









