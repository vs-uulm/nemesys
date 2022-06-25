"""
Evaluation implementation for NEMETYL, the NEtwork MEssage TYpe identification by aLignment.
INFOCOM 2020.

Use different segmenters to tokenize messages and align segments on the similarity of their "personality"
derived from the segments' features.
This script uses groundtruth about field segmentation by tshark dissectors to evaluate the quality.
Thus, it takes a PCAP trace of a known protocol, dissects each message into their fields and compares the results to
the selected heuristic segmenter.

The selected segmenter yields segments from each message. These segments are analyzed by the given analysis method
which is used as feature to determine their similarity. Similar fields are then aligned.
"""

import argparse

from nemere.alignment.alignMessages import TypeIdentificationByAlignment
from nemere.inference.segmentHandler import originalRefinements, baseRefinements, nemetylRefinements
from nemere.alignment.hirschbergAlignSegments import HirschbergOnSegmentSimilarity
from nemere.inference.templates import ClusterAutoconfException
from nemere.utils.evaluationHelpers import *
from nemere.utils.reportWriter import IndividualClusterReport, CombinatorialClustersReport
from nemere.visualization.multiPlotter import MultiMessagePlotter
from nemere.alignment.clusterMerging import ClusterClusterer

# https://stackoverflow.com/questions/15639779/why-does-multiprocessing-use-only-a-single-core-after-i-import-numpy
os.system("taskset -p 0xffffffffffffffffff %d" % os.getpid())

debug = False

# fix the analysis method to VALUE
analysis_method = 'value'
# fix the distance method to canberra
distance_method = 'canberra'
# tokenizers to select from
tokenizers = ('tshark', '4bytesfixed', 'nemesys')
roundingprecision = 10**8
# refinement methods
refinementMethods = [
    "none",
    "original", # WOOT2018 paper
    "base",     # ConsecutiveChars+moco
    "nemetyl",  # ConsecutiveChars+moco+splitfirstseg
    ]


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

    # noinspection PyTypeChecker
    disulist = list(dc.distancesSubset(stripedcol[0:1], stripedcol[1:]).tolist())  # type: list
    dists2first = ["- (reference)"] + disulist[0]  # type: List[Union[str, None]]

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

def epsautoconfeval(epsilon, plotTitle):
    """
    investigate distance properties for clustering autoconfiguration
    plots of k-nearest-neighbor distance histogram and "knee"

    See SegmentedMessages#autoconfigureDBSCAN

    :param plotTitle: Part of plot's filename and header
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

    neighbors = tyl.sm.neighbors()  # list of tuples: (index from sm.distances, distance) sorted by distance

    mmp = MultiMessagePlotter(specimens, "knn-distance-funtion_" + plotTitle, 1, 2,
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
    seconddiff = dict()  # type: Dict[int, numpy.ndarray]
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
        # noinspection PyArgumentList
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
    # noinspection PyTypeChecker
    mmp.plotToSubfig(ax0twin, [None] + seconddiff[k].tolist(), linestyle='dotted',
                     color='magenta', alpha=.4)

    # epsilon = knearest[k][x]
    epsilon = smoothknearest[k][x]

    mmp.axes[0].axhline(epsilon, linestyle='dashed', color='blue', alpha=.4,
                        label="curvature max {:.2f} of k={}".format(
                            epsilon, k))
    mmp.axes[0].axvline(x, linestyle='dashed', color='blue', alpha=.4)


    mmp.writeOrShowFigure(filechecker.reportFullPath)
    del mmp

    # if args.interactive:
    #     from tabulate import tabulate
    #     IPython.embed()
    # exit(0)

    return epsilon

def clusterClusters():
    """
    alternative idea of merging clusters by clustering them:
        does not improve merging - perhaps the similarity matrix is not good enough?!
    :return:
    """
    # ClusterClusterer
    clusterclusterer = ClusterClusterer(tyl.alignedClusters, dc)
    # clusterDists = clusterclusterer.calcClusterDistances()

    mergeEps, mergeMpts = clusterclusterer.autoconfigureDBSCAN()

    cluclu, labels, mergeclusterer = clusterclusterer.clusterMessageTypesDBSCAN(mergeEps, min_samples=2)
    clusterClustersNoiseless = {k: v for k, v in cluclu.items() if k > -1}
    mergedClusters = ClusterClusterer.mergeClusteredClusters(clusterClustersNoiseless, tyl.messageObjClusters)
    ClusterClusterer.printShouldMerge(list(clusterClustersNoiseless.values()), splitClusterReport.precisionRecallList)
    mergedObjClusters = {lab: [comparator.messages[element[0].message] for element in segseq]
                         for lab, segseq in mergedClusters.items()}

    inferenceParams.postProcess += "split+mergedAlt-{}-eps={:.2f}-min_samples={}".format(
        type(mergeclusterer).__name__, mergeclusterer.eps, mergeclusterer.min_samples)
    clusteredClusterReport = IndividualClusterReport(groundtruth, filechecker)
    clusteredClusterReport.write(mergedObjClusters, inferenceParams.dict)

    from netzob.Model.Vocabulary.Messages.RawMessage import RawMessage
    from nemere.visualization.distancesPlotter import DistancesPlotter
    typedClusterDummys = list()
    for clun in clusterclusterer.clusterOrder:
        clusta = None
        for stats in clusteredClusterReport.precisionRecallList:
            if stats is not None and stats[0] == clun:
                clusta = stats[1] if stats[2] == 1.0 else "({})".format(stats[1])
                break
        msgdum = RawMessage(messageType=clusta)
        typedClusterDummys.append(msgdum)

    dipl = DistancesPlotter(specimens, "cluster-clustering-" + inferenceParams.plotTitle, False)
    dipl.plotManifoldDistances(typedClusterDummys, clusterclusterer.distances, labels)
    dipl.writeOrShowFigure(filechecker.reportFullPath)

def printm4c(clusters, selected, printer):
    """
    Print messages for clusters.

    :param clusters:
    :param selected:
    :param printer: callable with one parameter for the message
    :return:
    """
    for lab, msgs in clusters.items():
        if lab in selected:
            print("\n\nCluster", lab, "\n")
            for msg in msgs:
                printer(msg)

def printVals4Field(clusters, selected, field):
    """all values for the given field name in the cluster"""
    printm4c(clusters, selected,
             lambda msg: print(comparator.parsedMessages[specimens.messagePool[msg[0].message]].getValuesByName(field)))

def printSpecClu(clusters, selected):
    printm4c(clusters, selected,
             lambda msg: comparator.pprint2Interleaved(msg, messageSlice=(None, 100)))

def countVals4Field(clusters, selected, field):
    valCounter = dict()
    for lab, msgs in clusters.items():
        if lab in selected:
            print("\n\nCluster", lab, "\n")
            valCounter[lab] = Counter(chain.from_iterable(
                comparator.parsedMessages[specimens.messagePool[msg[0].message]].getValuesByName(
                    field) for msg in msgs))
            print(valCounter[lab])
    # psftck = [set(tc.keys()) for tc in valCounter.values()]
    # # these tag values are present in all psf messages
    # psfcommon = psftck[0].intersection(*psftck)
    # [[b for b in a if b not in psfcommon] for a in psftck]
    return valCounter

def singularFields(clusters, cmp, select):
    fields = defaultdict(lambda: defaultdict(set))
    for lab, msgs in clusters.items():
        if lab in select:
            for msg in msgs:
                pm = cmp.parsedMessages[cmp.specimens.messagePool[msg[0].message]]
                for fn in pm.getFieldNames():
                    fields[lab][fn].update(pm.getValuesByName(fn))
    # find fields that have a single value within the cluster
    single = {lab: {fn: next(iter(val)) for fn, val in fnval.items() if len(val) == 1}
                 for lab, fnval in fields.items()}
    return fields, single

def discriminators(single, selectcount):
    # find singular fields (e.g., from mifsingle) in other clusters that have different singular values there: discriminator
    elgnis = defaultdict(dict)
    for lab, fnval in single.items():
        for fn, val in fnval.items():
            elgnis[fn][lab] = val
    return elgnis, {fn: {lab: val for lab, val in labval.items()} for fn, labval in elgnis.items()
                     if selectcount * 0.5 == len(set(labval.values()))}  # to allow for overspecific clusters (above groundtruth types)

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # END : Evaluation helpers  # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # #











if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Analyze fields as segments of messages and align on the similarity of their feature "personality".')
    parser.add_argument('pcapfilename', help='Filename of the PCAP to load.')
    parser.add_argument('-i', '--interactive', help='Open ipython prompt after finishing the analysis.',
                        action="store_true")
    parser.add_argument('-l', '--layer', type=int, default=2,
                        help='Protocol layer relative to IP to consider. Default is 2 layers above IP '
                             '(typically the payload of a transport protocol).')
    parser.add_argument('-r', '--relativeToIP', default=False, action='store_true')
    parser.add_argument('-t', '--tokenizer', help='Select the tokenizer for this analysis run.',
                        choices=tokenizers, default="tshark")
    parser.add_argument('-e', '--littleendian', help='Toggle presumed endianness to little.', action="store_true")
    parser.add_argument('-s', '--sigma', type=float, help='Only NEMESYS: sigma for noise reduction (gauss filter),'
                                                          'default: 0.9')
    parser.add_argument('-f', '--refinement', help='Select segment refinement method.', choices=refinementMethods,
                        default=refinementMethods[-1])
    parser.add_argument('-p', '--with-plots',
                        help='Generate plots.',
                        action="store_true")
    args = parser.parse_args()

    filechecker = StartupFilecheck(args.pcapfilename)
    withplots = args.with_plots
    littleendian = args.littleendian == True
    analyzerType = analyses[analysis_method]
    analysisArgs = None
    analysisTitle = analysis_method
    tokenizer = args.tokenizer
    if littleendian:
        tokenizer += "le"

    # # # # # # # # # # # # # # # # # # # # # # # #
    # cache/load the DistanceCalculator to the filesystem
    #
    fromCache = CachedDistances(args.pcapfilename, analysisTitle, args.layer, args.relativeToIP)
    # Note!  When manipulating distances calculation, deactivate caching by uncommenting the following assignment.
    # fromCache.disableCache = True
    fromCache.debug = debug
    if analysisArgs is not None:
        # noinspection PyArgumentList
        fromCache.configureAnalysis(*analysisArgs)
    fromCache.configureTokenizer(tokenizer, args.sigma)

    if tokenizer[:7] == "nemesys":
        if args.refinement == "original":
            fromCache.configureRefinement(originalRefinements)
        elif args.refinement == "base":
            fromCache.configureRefinement(baseRefinements)
        elif args.refinement == "nemetyl":
            fromCache.configureRefinement(nemetylRefinements)
        else:
            print("No refinement selected. Performing raw segmentation.")
    try:
        fromCache.get()
    except ClusterAutoconfException as e:
        print("Initial clustering of the segments in the trace failed. The protocol in this trace cannot be inferred. "
              "The original exception message was:\n", e)
        exit(10)
    segmentedMessages = fromCache.segmentedMessages
    specimens, comparator, dc = fromCache.specimens, fromCache.comparator, fromCache.dc
    segmentationTime, dist_calc_segmentsTime = fromCache.segmentationTime, fromCache.dist_calc_segmentsTime

    # # # # # # # # # # # # # # # # # # # # # # # #
    # Start the NEMETYL inference process
    # # # # # # # # # # # # # # # # # # # # # # # #
    tyl = TypeIdentificationByAlignment(dc, segmentedMessages, tokenizer, specimens.messagePool)
    # # # # # # # # # # # # # # # # # # # # # # # #

    # # # # # # # # # # # # # # # # # # # # # # # #
    # if not exists(smcachefn):
    #
    # # # # # # # # # # # # # # # # # # # # # # # #
    # Calculate Alignment-Score and CLUSTER messages
    # # # # # # # # # # # # # # # # # # # # # # # #
    tyl.clusterMessages()
    # Prepare basic information about the inference run for the report
    inferenceParams = TitleBuilder(tokenizer, args.refinement, args.sigma, tyl.clusterer)
    #
    # smcachefn = 'cache-sm-{}-{}-{}.{}'.format(analysisTitle, tokenparm, filechecker.pcapstrippedname, 'sm')
    #     with open(smcachefn, 'wb') as f:
    #         pickle.dump(sm, f, pickle.HIGHEST_PROTOCOL)
    # else:
    #     print("Load distances from cache file {}".format(smcachefn))
    #     sm = pickle.load(open(smcachefn, 'rb'))
    #     if not isinstance(sm, SegmentedMessages):
    #         print('Loading of cached message distances failed.')
    #         exit(11)
    # # # # # # # # # # # # # # # # # # # # # # # #
    if withplots:
        # plot message distances and clusters
        print("Plot distances...")
        from nemere.visualization.distancesPlotter import DistancesPlotter
        dp = DistancesPlotter(specimens, 'message-distances_' + inferenceParams.plotTitle, False)
        dp.plotManifoldDistances(
            [specimens.messagePool[seglist[0].message] for seglist in segmentedMessages],
            tyl.sm.distances, tyl.labels)  # segmentedMessages
        dp.writeOrShowFigure(filechecker.reportFullPath)


    # # # # # # # # # # # # # # # # # # # # # # # #
    # DEBUG and TESTING
    # # # # # # # # # # # # # # # # # # # # # # # #
    # retrieve manually determined epsilon value
    # epsilon = message_epspertrace[filechecker.pcapbasename]
    #               if filechecker.pcapbasename in message_epspertrace else 0.15
    if withplots:
        epsConfirm = epsautoconfeval(tyl.eps, tokenizer +
                                     (f"-s{args.sigma}-{args.refinement}" if tokenizer[:7] == "nemesys" else "") )
    # # # # # # # # # # # # # # # # # # # # # # # #
    # DEBUG and TESTING
    # # # # # # # # # # # # # # # # # # # # # # # #

    groundtruth = {msg: pm.messagetype for msg, pm in comparator.parsedMessages.items()}
    for msg, mtype in groundtruth.items():
        msg.messageType = mtype
    minCsize = numpy.log(len(segmentedMessages))

    # # # # # # # # # # # # # # # # # # # # # # # #
    # write message clustering statistics to csv
    # # # # # # # # # # # # # # # # # # # # # # # #
    fullClusterReport = IndividualClusterReport(groundtruth, filechecker)
    fullClusterReport.write(tyl.messageObjClusters, inferenceParams.dict)
    # # # # # # # #
    fullCombinReport = CombinatorialClustersReport(groundtruth, filechecker)
    fullCombinReport.write(tyl.messageObjClusters, inferenceParams.dict)

    # # min cluster size # # # # # #
    inferenceParams.postProcess = "minCsize"
    noisekey = 'Noise' if 'Noise' in tyl.messageObjClusters else -1
    filteredClusters = {k: v for k, v in tyl.messageObjClusters.items() if len(v) >= minCsize}
    filteredClusters[noisekey] = list() if not noisekey in filteredClusters else filteredClusters[noisekey].copy()
    filteredClusters[noisekey].extend(s for k, v in tyl.messageObjClusters.items()
                                      if len(v) < minCsize for s in v)
    filteredCombinReport = CombinatorialClustersReport(groundtruth, filechecker)
    filteredCombinReport.write(filteredClusters, inferenceParams.dict)



    # # # # # # # # # # # # # # # # # # # # # # # #
    # ALIGN cluster members
    # # # # # # # # # # # # # # # # # # # # # # # #
    tyl.alignClusterMembers()
    # # # # # # # # # # # # # # # # # # # # # # # #



    # # # # # # # # # # # # # # # # # # # # # # # #
    # SPLIT clusters based on fields without rare values
    # # # # # # # # # # # # # # # # # # # # # # # #
    inferenceParams.postProcess = tyl.splitClusters(  # activateCVSout by the following kwargs
        runtitle = inferenceParams.dict,
        trace = filechecker.pcapstrippedname,
        clusterPrecisions = {cs[0]: cs[2] for cs in fullClusterReport.precisionRecallList if cs is not None})
    # # # # # # # # # # # # # # # # # # # # # # # #
    if withplots:
        # plot distances and message clusters
        print("Plot distances...")
        from nemere.visualization.distancesPlotter import DistancesPlotter
        dp = DistancesPlotter(specimens, 'message-distances_' + inferenceParams.plotTitle, False)
        dp.plotManifoldDistances(
            [specimens.messagePool[seglist[0].message] for seglist in segmentedMessages],
            tyl.sm.distances, tyl.labels)  # segmentedMessages
        dp.writeOrShowFigure(filechecker.reportFullPath)

    # # # # # # # # # # # # # # # # # # # # # # # #
    # clusterStats for splitter
    # # # # # # # # # # # # # # # # # # # # # # # #
    # write message clustering statistics to csv
    # # # # # # # # # # # # # # # # # # # # # # # #
    splitClusterReport = IndividualClusterReport(groundtruth, filechecker)
    splitClusterReport.write(tyl.messageObjClusters, inferenceParams.dict)
    # # # # # # # #
    splitCombinReport = CombinatorialClustersReport(groundtruth, filechecker)
    splitCombinReport.write(tyl.messageObjClusters, inferenceParams.dict)

    # # # # # # # # min cluster size
    inferenceParams.postProcess += "-minCsize"
    noisekey = 'Noise' if 'Noise' in tyl.messageObjClusters else -1
    filteredClusters = {k: v for k, v in tyl.messageObjClusters.items()
                        if len(v) >= minCsize }
    filteredClusters[noisekey] = list() if not noisekey in filteredClusters else filteredClusters[noisekey].copy()
    filteredClusters[noisekey].extend(s for k, v in tyl.messageObjClusters.items()
                                      if len(v) < minCsize for s in v)
    filteredSplitCombinReport = CombinatorialClustersReport(groundtruth, filechecker)
    filteredSplitCombinReport.write(filteredClusters, inferenceParams.dict)
    # # # # # # # # # # # # # # # # # # # # # # # #



    # # # # # # # # # # # # # # # # # # # # # # # #
    # Check for cluster MERGE candidates
    # # # # # # # # # # # # # # # # # # # # # # # #
    inferenceParams.postProcess = tyl.mergeClusters()

    # # # # # # # # # # # # # # # # # # # # # # # #
    # clusterStats for merger
    # # # # # # # # # # # # # # # # # # # # # # # #
    # write message clustering statistics to csv
    # # # # # # # # # # # # # # # # # # # # # # # #
    mergedClusterReport = IndividualClusterReport(groundtruth, filechecker)
    mergedClusterReport.write(tyl.messageObjClusters, inferenceParams.dict)
    # # # # # # # #
    mergedCombinReport = CombinatorialClustersReport(groundtruth, filechecker)
    mergedCombinReport.write(tyl.messageObjClusters, inferenceParams.dict)

    # # # # # # # # min cluster size
    inferenceParams.postProcess += "-minCsize"
    noisekey = 'Noise' if 'Noise' in tyl.messageObjClusters else -1
    filteredMerged = {k: v for k, v in tyl.messageObjClusters.items() if len(v) >= minCsize}
    filteredMerged[noisekey] = list() if not noisekey in filteredMerged else filteredMerged[noisekey].copy()
    filteredMerged[noisekey].extend(s for k, v in tyl.messageObjClusters.items()
                                      if len(v) < minCsize for s in v)
    filteredSplitCombinReport = CombinatorialClustersReport(groundtruth, filechecker)
    filteredSplitCombinReport.write(filteredClusters, inferenceParams.dict)

    # # Alternative approach to ClusterMerger, discarded.
    # clusterClusters()

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
    #
    # END # of # check for cluster merge candidates #
    # # # # # # # # # # # # # # # # # # # # # # # #







    writePerformanceStatistics(
        specimens, tyl.clusterer, inferenceParams.plotTitle,
        segmentationTime, dist_calc_segmentsTime,
        tyl.dist_calc_messagesTime, tyl.cluster_params_autoconfTime, tyl.cluster_messagesTime, tyl.align_messagesTime
    )
    filechecker.writeReportMetadata(fromCache.dccachefn if fromCache.isLoaded else None)

    # # # # # # # # # # # # # # # # # # # # # # # #
    # write alignments to csv
    # # # # # # # # # # # # # # # # # # # # # # # #
    # TODO split clusters are internally re-aligned, but NOT merged clusters. Can this lead to an inconsistency?
    csvpath = join(filechecker.reportFullPath,
                   f"NEMETYL-symbols-{inferenceParams.plotTitle}-{filechecker.pcapstrippedname}.csv")
    if not exists(csvpath):
        print('Write alignments to {}...'.format(csvpath))
        with open(csvpath, 'w') as csvfile:
            symbolcsv = csv.writer(csvfile)
            symbolcsv.writerow(["Cluster", "Type", "frame.time_epoch", "Field", "Alignment"])
            for clunu, clusg in tyl.alignedClusters.items():
                symbolcsv.writerows(
                    [clunu,  # cluster label
                     groundtruth[comparator.messages[next(seg for seg in msg if seg is not None).message]],  # message type string from gt
                     next(seg for seg in msg if seg is not None).message.date]  # frame.time_epoch
                    + [sg.bytes.hex() if sg is not None else '' for sg in msg] for msg in clusg
                )
    else:
        print("Symbols not saved. File {} already exists.".format(csvpath))
        if not args.interactive:
            IPython.embed()
    # # # # # # # # # # # # # # # # # # # # # # # #





    if args.interactive:
        import numpy
        from tabulate import tabulate

        # globals().update(locals())
        IPython.embed()









