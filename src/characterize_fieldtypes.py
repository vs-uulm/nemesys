"""
Use groundtruth about field segmentation by dissectors and apply field type identification to them.

Takes a PCAP trace of a known protocol, dissects each message into their fields, and yields segments from each of them.
These segments get analyzed by the given analysis method which is used as feature to determine their similarity.
Similar fields are then clustered by DBSCAN and for comparison plotted in groups of their real field types.
In addition, a MDS projection into a 2D plane for visualization of the relative distances of the features is plotted.
"""

import argparse, IPython
from os.path import isfile, basename
from itertools import chain

from utils.evaluationHelpers import epspertrace, epsdefault, analyses, annotateFieldTypes
from inference.templates import TemplateGenerator, DistanceCalculator
from inference.segments import TypedSegment
from inference.analyzers import *
from inference.segmentHandler import groupByLength, segments2types, segments2clusteredTypes, \
    filterSegments
from validation.dissectorMatcher import MessageComparator
from utils.loader import SpecimenLoader
from visualization.multiPlotter import MultiMessagePlotter
from visualization.distancesPlotter import DistancesPlotter


debug = False


def segments2typedClusters(segments: List[TypedSegment], analysisTitle) \
        -> List[Tuple[str, List[Tuple[str, TypedSegment]]]]:
    """
    Cluster segments and arrange them into groups of types.

    :param segments:
    :param analysisTitle:
    :return:
    """

    typegroups = segments2types(segments)

    segmentGroups = list()  # type: List[Tuple[str, List[Tuple[str, TypedSegment]]]
    # one plot per type with clusters
    for ftype, segs in typegroups.items():  # [label, segment]
        tg = TemplateGenerator(segs)
        noise, *clusters = tg.clusterSimilarSegments(False)
        print("{} clusters generated from {} segments".format(len(clusters), len(segs)))

        segmentClusters = ("{}: {}, {} bytes".format(
            analysisTitle, ftype,
            clusters[0][0].length if clusters else noise[0].length), list())

        if len(noise) > 0:
            segmentClusters[1].append(('Noise: {} Seg.s'.format(len(noise)),
                                       [('', cseg) for cseg in noise]))

        if len(clusters) > 0:
            for clusternum, clustersegs in enumerate(clusters):
                segmentClusters[1].append(('Cluster #{}: {} Seg.s'.format(clusternum, len(clustersegs)),
                                           [('', cseg) for cseg in clustersegs]))
            segmentGroups.append(segmentClusters)
    return segmentGroups


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


def plotMultiSegmentLines(segmentGroups: List[Tuple[str, List[Tuple[str, TypedSegment]]]],
                          pagetitle=None, colorPerLabel=False, typeDict = None):
    """
    This is a not awfully important helper function saving the writing of a few lines code.

    :param segmentGroups:
    :param pagetitle:
    :param colorPerLabel:
    :param typeDict: dict of types (str-keys: list of segments) present in the segmentGroups
    :return:
    """
    mmp = MultiMessagePlotter(specimens, pagetitle, len(segmentGroups), isInteractive=args.interactive)
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



def evaluateFieldTypeClusteringWithIsolatedLengths():
    segsByLen = groupByLength(segmentedMessages)
    print("done.")
    for length, segments in segsByLen.items():  # type: int, List[MessageSegment]
        filteredSegments = filterSegments(segments)

        # if length == 4: # > 8:  # != 4:
        #     continue
        if length < 3:
            continue
        if len(filteredSegments) < 16:
            print("Too few relevant segments for length {} after Filtering. {} segments remaining:".format(
                length, len(filteredSegments)
            ))
            for each in filteredSegments:
                print("   ", each)
            print()
            continue

        typeDict = segments2types(filteredSegments)

        # iterate distance_methods
        for distance_method in ['canberra']:  # [ 'cosine' , 'euclidean' , 'canberra' , 'correlation' ]:  #
            print("Calculate distances...")
            # ftype = 'id'
            # segments = [seg for seg in segsByLen[4] if seg.fieldtype == ftype]
            # distance_method = 'canberra' # 'cosine' | 'euclidean' | 'canberra' | 'correlation'
            tg = TemplateGenerator(filteredSegments, distance_method)

            # iterate clusterer parameters
            for mcs in range(3, 15):  # [ 0 ]: # range(3, 15):
                print("Clustering...")
                # typeGroups = segments2typedClusters(segments,analysisTitle)
                segmentGroups = segments2clusteredTypes(tg, analysisTitle, min_cluster_size=mcs)
                # re-extract cluster labels for segments
                labels = numpy.array([
                    labelForSegment(segmentGroups, seg) for seg in tg.segments
                ])

                # # if args.distances:
                print("Plot distances...")
                sdp = DistancesPlotter(specimens, 'distances-{}-{}-{}-{}'.format(
                    length, analysisTitle, distance_method, tg.clusterer if tg.clusterer else 'n/a'), args.interactive)
                # sdp.plotSegmentDistances(tg, numpy.array([seg.fieldtype for seg in tg.segments]))
                sdp.plotSegmentDistances(tg, labels)
                sdp.writeOrShowFigure()

                print("Prepare output...")
                for pagetitle, segmentClusters in segmentGroups:
                    plotMultiSegmentLines(segmentClusters, "{} ({}, {}-{})".format(pagetitle, distance_method,
                                                                                   tg.thresholdFunction.__name__,
                                                                                   tg.thresholdArgs), True, typeDict)




def evaluateFieldTypeClustering(filteredSegments, eps, thresholdFunction, thresholdArgs):
    print("Calculate distances...")
    tg = TemplateGenerator(filteredSegments, distance_method, thresholdFunction, thresholdArgs)

    print("Clustering...")
    # # use HDBSCAN
    # segmentGroups = segments2clusteredTypes(tg, analysisTitle, min_cluster_size=15)
    # use DBSCAN
    segmentGroups = segments2clusteredTypes(tg, analysisTitle,
                                            clustererClass=TemplateGenerator.DBSCAN, epsilon=eps, minpts=20)
    # re-extract cluster labels for segments
    labels = numpy.array([
        labelForSegment(segmentGroups, seg) for seg in tg.segments
    ])

    typeDict = segments2types(filteredSegments)

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # # Testing for mixed-length field type identification efficacy
    # lenMasks = {}
    # for idx, seg in enumerate(filteredSegments):
    #     if not seg.length in lenMasks:
    #         lenMasks[seg.length] = [False] * len(filteredSegments)
    #     lenMasks[seg.length][idx] = True
    #
    from tabulate import tabulate
    from inference.templates import Template
    from utils.baseAlgorithms import tril
    from visualization.singlePlotter import SingleMessagePlotter
    # # field type change for labels
    # segFieldtypes = [seg.fieldtype if pseg.fieldtype != seg.fieldtype else '' for seg, pseg in
    #                  zip(filteredSegments, filteredSegments[:1] + filteredSegments)]
    # # # equal lengths
    # # l4 = tg.distanceMatrix[lenMasks[4]][:,numpy.array(lenMasks[4])]
    # # # unequal lengths
    # # l4 = tg.distanceMatrix[lenMasks[4]][:,~numpy.array(lenMasks[4])]
    # # # Structure of segmentGroups
    # # segmentGroups[0]  # page (here: all)
    # # segmentGroups[0][0]  # pagetitle
    clusters = segmentGroups[0][1]  # clusters
    # # segmentGroups[0][1][0]  # cluster 0
    # # segmentGroups[0][1][0][0]  # cluster 0: title
    # # segmentGroups[0][1][0][1]  # cluster 0: elements
    # # segmentGroups[0][1][0][1][0]  # cluster 0: element 0
    # # segmentGroups[0][1][0][1][0][0]  # cluster 0: element 0: str(type, sum type elements)
    # # segmentGroups[0][1][0][1][0][1]  # cluster 0: element 0: MessageSegment

    DistanceCalculator.debug = False

    segsByLen = dict()
    clusterStats = list()
    for ctitle, typedelements in clusters:
        # basics (type, lengths)
        celements = [s for t, s in typedelements]
        segsByLen[ctitle] = groupByLength([celements])
        cntPerLen = [(l, len(segs)) for l, segs in segsByLen[ctitle].items()]
        majOfLen = (-1, -1)  # (length, count)
        for l, c in cntPerLen:
            if c > majOfLen[1]:
                majOfLen = (l, c)
        majSegs = segsByLen[ctitle][majOfLen[0]]


        # "center" of cluster
        majLenValues = numpy.array([seg.values for seg in majSegs])
        center = majLenValues.mean(0)  # mean per component on axis 0
        # alternative: best fit for all mixed length segments, requires the offset for each shorter segment
        # (thats a lot, if the majority is shorter than other contained segments as for ntp100-timestamp+checksum)
        #
        # calc distances of all segments of this cluster to the center
        majTem = Template(center, segsByLen[ctitle][majOfLen[0]])
        majCentDist = majTem.distancesTo().max()
        # make unequal length segments comparable
        nemaj = [s for l, grp in segsByLen[ctitle].items() if l != majOfLen[0] for s in grp]
        nemajTem = Template(center, nemaj)
        # (embedding into/of center, resulting distance, retain offset: -n for center, +n for segment)
        nemajCentOffsList = [o for d, o in nemajTem.distancesToMixedLength()]
        nemajCentDistList = [d for d, o in nemajTem.distancesToMixedLength()]
        nemajCentDist = numpy.array(nemajCentDistList).min() if len(nemajCentDistList) > 0 else None

        # max of distances between equal size segments
        majEqDist = tril(tg.distancesSubset(segsByLen[ctitle][majOfLen[0]]))
        majEqDistMax = majEqDist.max() if majEqDist.size > 0 else -1
        majNeqDist = tg.distancesSubset(
            [seg for l, seglist in segsByLen[ctitle].items() for seg in seglist if l != majOfLen[0]],
            segsByLen[ctitle][majOfLen[0]])
        # min of distances between unequal size segments
        majNeqDistMin = majNeqDist.min() if len(majNeqDist) > 0 else None

        # write cluster statistics
        clusterStats.append((
            ctitle, cntPerLen, majOfLen[0], majEqDistMax, majNeqDistMin, center, majCentDist, nemajCentDist
        ))
        # overlay plot of mixed length matches
        if majNeqDistMin:
            minoffset = min(nemajCentOffsList)
            maxoffset = max(nemajCentOffsList)
            majValuesList = majLenValues.tolist()
            if minoffset < 0:
                majValuesList = [[numpy.nan]*abs(minoffset) + v for v in majValuesList]
                nemajCentOffsList = [o - minoffset for o in nemajCentOffsList]
            nemajMaxLength = max([s.length for s in nemaj])
            maxOffsetLength = maxoffset + nemajMaxLength if maxoffset > 0 else abs(minoffset) + nemajMaxLength
            nemajValuesList = [
                [numpy.nan] * o + s.values +
                [numpy.nan] * (maxOffsetLength - o - s.length)
                for s, o in zip(nemaj, nemajCentOffsList)]

            # # plot overlay of most similar segments of unequal length
            # mmp = MultiMessagePlotter(specimens, "mixedneighbors-" + ctitle, len(nemaj[:50]), isInteractive=False)
            # # TODO plot multiple pages if more than 50 subfigs are needed
            # for eIdx, nemEl in enumerate(nemaj[:50]):
            #     nemNeighbors = tg.neigbors(nemEl, majSegs)
            #     nearestNeighbor = majSegs[nemNeighbors[0][0]]
            #     alpha = .8
            #     for nearestValues in [majValuesList[nemNeighbors[i][0]] for i in range(min(8, len(nemNeighbors)))]:
            #         mmp.plotToSubfig(eIdx, nearestValues, color='r', alpha=alpha)
            #         alpha -= .8/8
            #     mmp.plotToSubfig(eIdx, nemajValuesList[eIdx], color='b')
            #     mmp.textInEachAx([None] * eIdx +
            #                      ["$d_{{min}}$ = {:.3f}".format(nemNeighbors[0][1])])
            # if len(nemaj) > 0:
            #     mmp.writeOrShowFigure()
            # else:
            #     print(mmp.title, "has no neighbor entries.")
            # del mmp

            # # plot allover alignment of all unequal length segments with all equal ones
            # plt.plot(numpy.array(majValuesList).transpose(), color='b', alpha=.1)
            # plt.plot(numpy.array(nemajValuesList).transpose(),
            #          color='r', alpha=.8)
            # plt.show()

            # # distribution of distances of majority length and not-equal to majority lengthy segments
            # pwcd = sorted(majEqDist)[-20:] + sorted(majNeqDist.flat)[:20]
            # sns.barplot(list(range(len(pwcd))), pwcd)
            # pwcd = sorted(majEqDist)[-20:] + sorted(majNeqDist.flat)[:20]


    print(tabulate(clusterStats,
                   headers=["ctitle", "cntPerLen", "majOfLen", "majEqDistMax", "majNeqDistMin",
                            "center", "majCentDist", "nemajCentDist"]))



    # IPython.embed()  # TODO here I am!
    # # # distance to one field candidate
    # # sns.set(font_scale=.6)
    # # sns.barplot(list(range(tg.distanceMatrix.shape[0])), tg.distanceMatrix[180,])
    # # plt.show()
    # # # distance matrix heatmap (sorted above for this plot)
    # # xymax = 160
    # # sns.heatmap(tg.distanceMatrix[:xymax,:xymax], xticklabels=segFieldtypes[:xymax], yticklabels=segFieldtypes[:xymax])
    # # plt.show()

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    titleFormat = "{} ({}, {}-{})".format(
        segmentGroups[0][0], distance_method, tg.thresholdFunction.__name__,
        "".join([str(k) + str(v) for k, v in tg.thresholdArgs.items()]) if tg.thresholdArgs else '')

    print("Plot distances...")
    sdp = DistancesPlotter(specimens, 'distances-' + titleFormat,
                           args.interactive)
    # old: 'mixedlength', analysisTitle, distance_method, tg.clusterer if tg.clusterer else 'n/a'
    # sdp.plotSegmentDistances(tg, numpy.array([seg.fieldtype for seg in tg.segments]))
    sdp.plotSegmentDistances(tg, labels)
    sdp.writeOrShowFigure()
    del sdp

    hstplt = SingleMessagePlotter(specimens, 'histo-distance-' + titleFormat, args.interactive)
    hstplt.histogram(tril(tg.distanceMatrix), bins=[x/50 for x in range(50)])
    hstplt.writeOrShowFigure()
    del hstplt

    print("Prepare output...")
    for pagetitle, segmentClusters in segmentGroups:
        plotMultiSegmentLines(segmentClusters, titleFormat,
                              True, typeDict)


def iterateDBSCANParameters():
    filteredSegments = filterSegments(chain.from_iterable(segmentedMessages))

    print("done.")

    for cnt in (180, 200, 220, 240, 260): # (170, 180, 190, 200, 210):# (100, 120, 130, 150, 160):
        eps = cnt * .01
        for threshFunc, threshArgs in (
                (TemplateGenerator.neutralThreshold, {}),
                # (TemplateGenerator.sigmoidThreshold, {'shift': .2}),
                # (TemplateGenerator.sigmoidThreshold, {'shift': .4}),
                # (TemplateGenerator.sigmoidThreshold, {'shift': .5}),
                # (TemplateGenerator.sigmoidThreshold, {'shift': .6}),
                # (TemplateGenerator.sigmoidThreshold, {'shift': .8}),
        ):
            evaluateFieldTypeClustering(filteredSegments, eps, threshFunc, threshArgs)








if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Analyze fields as segments of messages and plot field type identification quality.')
    parser.add_argument('pcapfilename', help='Filename of the PCAP to load.')
    parser.add_argument('-i', '--interactive', help='Show interactive plot instead of writing output to file and '
                                                    'open ipython prompt after finishing the analysis.',
                        action="store_true")
    # parser.add_argument('analysis', type=str,
    #                     help='The kind of analysis to apply on the messages. Available methods are: '
    #                     + ', '.join(analyses.keys()) + '.')
    # parser.add_argument('--parameters', '-p', help='Parameters for the analysis.')
    parser.add_argument('--isolengths', help='Cluster fields of same size isolatedly.', action="store_true")
    parser.add_argument('--iterate', help='Iterate over DBSCAN parameters to select valid eps and threshold-shift.',
                        action="store_true")

    parser.add_argument('--epsilon', '-e', help='Parameter epsilon for the DBSCAN clusterer.', type=float, default=epsdefault)
    args = parser.parse_args()

    if not isfile(args.pcapfilename):
        print('File not found: ' + args.pcapfilename)
        exit(1)
    if args.isolengths and args.iterate:
        print('Iterating clustering parameters over isolated-lengths fields is not implemented.')
        exit(2)
    if args.epsilon != parser.get_default('epsilon') and (args.isolengths or args.iterate):
        print('Setting epsilon is not supported for clustering over isolated-lengths fields and parameter iteration.')
        exit(2)

    # fix the analysis method to VALUE
    analysisTitle = 'value'
    analyzerType = analyses[analysisTitle]
    analysisArgs = None
    # if args.analysis not in analyses:
    #     print('Analysis {} unknown. Available methods are:\n' + ', '.join(analyses.keys()) + '.')
    #     exit(2)
    # analyzerType = analyses[args.analysis]
    # analysisArgs = args.parameters
    # analysisTitle = "{}{}".format(args.analysis, "" if not analysisArgs else " ({})".format(analysisArgs))

    # fix the distance method to canberra
    distance_method = 'canberra'

    # dissect and label messages
    print("Load messages...")
    specimens = SpecimenLoader(args.pcapfilename, 2, True)
    comparator = MessageComparator(specimens, 2, True, debug=debug)

    # segment messages according to true fields from the labels
    print("Segmenting messages...")
    segmentedMessages = annotateFieldTypes(analyzerType, analysisArgs, comparator)


    if not args.isolengths:  # select between old, field-size dependent and new, mixed-length similarity
        if args.iterate:
            iterateDBSCANParameters()
        else:
            filteredSegments = filterSegments(chain.from_iterable(segmentedMessages))

            # fixed values based on evaluation from Jan 18-22, 2019 - evaluation in nemesys-reports commit be95f9c
            # epsion should be 1.2 (default)
            # SUPERSEDED!

            pcapbasename = basename(specimens.pcapFileName)
            print("Trace:", pcapbasename)
            epsilon = args.epsilon  # TODO make "not set" epsilon and "default" distinguishable
            if args.epsilon == epsdefault and pcapbasename in epspertrace:
                epsilon = epspertrace[pcapbasename]
            evaluateFieldTypeClustering(filteredSegments, epsilon, TemplateGenerator.neutralThreshold, {})
                                                     # TemplateGenerator.sigmoidThreshold, {'shift': .6})
    else:
        evaluateFieldTypeClusteringWithIsolatedLengths()


    # TODO More Hypotheses to find "rules" for:
    #  small values at fieldend: int
    #  all 0 values with variance vector starting with -255: 0-pad (validate what's the predecessor-field?)
    #  len > 16: chars (pad if all 0)
    # For a new hypothesis: what are the longest seen ints?
    #


    # typeGrp = [(t, [('', s) for s in typeDict[t]]) for t in typeDict.keys()]
    # plotMultiSegmentLines(typeGrp, "{} ({} bytes) fieldtypes".format(analysisTitle, length), True)









    if args.interactive:
        IPython.embed()




