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

from utils.evaluationHelpers import epspertrace, epsdefault, analyses, annotateFieldTypes, plotMultiSegmentLines, \
    labelForSegment
from inference.templates import TemplateGenerator, DistanceCalculator, DBSCANsegmentClusterer, HDBSCANsegmentClusterer, DelegatingDC
from inference.segments import TypedSegment
from inference.analyzers import *
from inference.segmentHandler import groupByLength, segments2types, segments2clusteredTypes, \
    filterSegments
from validation.dissectorMatcher import MessageComparator
from utils.loader import SpecimenLoader
from visualization.distancesPlotter import DistancesPlotter


debug = False

# fix the analysis method to VALUE
analysisTitle = 'value'
# fix the distance method to canberra
distance_method = 'canberra'




def evaluateFieldTypeClusteringWithIsolatedLengths():
    segsByLen = groupByLength(segmentedMessages)
    print("done.")
    for length, segments in segsByLen.items():  # type: int, List[MessageSegment]
        filteredSegments = filterSegments(segments)  # type: List[TypedSegment]

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
            dc = DistanceCalculator(filteredSegments, distance_method)

            # iterate clusterer parameters
            for mcs in range(3, 15):  # [ 0 ]: # range(3, 15):
                print("Clustering...")
                clusterer = HDBSCANsegmentClusterer(dc, min_cluster_size=mcs)
                segmentGroups = segments2clusteredTypes(clusterer, analysisTitle)
                # re-extract cluster labels for segments
                labels = numpy.array([
                    labelForSegment(segmentGroups, seg) for seg in dc.segments
                ])

                # # if args.distances:
                print("Plot distances...")
                sdp = DistancesPlotter(specimens, 'distances-{}-{}-{}-{}'.format(
                    length, analysisTitle, distance_method, clusterer), args.interactive)
                # sdp.plotSegmentDistances(tg, numpy.array([seg.fieldtype for seg in tg.segments]))
                sdp.plotSegmentDistances(dc, labels)
                sdp.writeOrShowFigure()

                print("Prepare output...")
                for pagetitle, segmentClusters in segmentGroups:
                    plotMultiSegmentLines(segmentClusters, specimens, "{} ({}, {}-{})".format(
                        pagetitle, distance_method, dc.thresholdFunction.__name__, dc.thresholdArgs),
                                          True, typeDict, args.interactive)




def evaluateFieldTypeClustering(filteredSegments, eps, thresholdFunction, thresholdArgs):
    print("Calculate distances...")
    dc = DistanceCalculator(filteredSegments, distance_method, thresholdFunction, thresholdArgs)
    # dc = DelegatingDC(filteredSegments)

    print("Clustering...")
    # # use HDBSCAN
    # segmentGroups = segments2clusteredTypes(tg, analysisTitle, min_cluster_size=15)
    # use DBSCAN
    clusterer = DBSCANsegmentClusterer(dc, eps=eps, min_samples=20)
    segmentGroups = segments2clusteredTypes(clusterer, analysisTitle)
    # re-extract cluster labels for segments
    labels = numpy.array([
        labelForSegment(segmentGroups, seg) for seg in dc.segments
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
        majEqDist = tril(dc.distancesSubset(segsByLen[ctitle][majOfLen[0]]))
        majEqDistMax = majEqDist.max() if majEqDist.size > 0 else -1
        majNeqDist = dc.distancesSubset(
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
        segmentGroups[0][0], distance_method, dc.thresholdFunction.__name__,
        "".join([str(k) + str(v) for k, v in dc.thresholdArgs.items()]) if dc.thresholdArgs else '')

    print("Plot distances...")
    sdp = DistancesPlotter(specimens, 'distances-' + titleFormat,
                           args.interactive)
    # old: 'mixedlength', analysisTitle, distance_method, tg.clusterer if tg.clusterer else 'n/a'
    # sdp.plotSegmentDistances(tg, numpy.array([seg.fieldtype for seg in tg.segments]))
    sdp.plotSegmentDistances(dc, labels)
    sdp.writeOrShowFigure()
    del sdp

    hstplt = SingleMessagePlotter(specimens, 'histo-distance-' + titleFormat, args.interactive)
    hstplt.histogram(tril(dc.distanceMatrix), bins=[x/50 for x in range(50)])
    hstplt.writeOrShowFigure()
    del hstplt

    print("Prepare output...")
    for pagetitle, segmentClusters in segmentGroups:
        plotMultiSegmentLines(segmentClusters, specimens, titleFormat,
                              True, typeDict, args.interactive)


def iterateDBSCANParameters():
    filteredSegments = filterSegments(chain.from_iterable(segmentedMessages))

    print("done.")

    for cnt in (180, 200, 220, 240, 260): # (170, 180, 190, 200, 210):# (100, 120, 130, 150, 160):
        eps = cnt * .01
        for threshFunc, threshArgs in (
                (DistanceCalculator.neutralThreshold, {}),
                # (DistanceCalculator.sigmoidThreshold, {'shift': .2}),
                # (DistanceCalculator.sigmoidThreshold, {'shift': .4}),
                # (DistanceCalculator.sigmoidThreshold, {'shift': .5}),
                # (DistanceCalculator.sigmoidThreshold, {'shift': .6}),
                # (DistanceCalculator.sigmoidThreshold, {'shift': .8}),
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

    analyzerType = analyses[analysisTitle]
    analysisArgs = None
    # if args.analysis not in analyses:
    #     print('Analysis {} unknown. Available methods are:\n' + ', '.join(analyses.keys()) + '.')
    #     exit(2)
    # analyzerType = analyses[args.analysis]
    # analysisArgs = args.parameters
    # analysisTitle = "{}{}".format(args.analysis, "" if not analysisArgs else " ({})".format(analysisArgs))

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
            # filteredSegments = filterSegments(chain.from_iterable(segmentedMessages))
            filteredSegments = list(chain.from_iterable(segmentedMessages))

            # fixed values based on evaluation from Jan 18-22, 2019 - evaluation in nemesys-reports commit be95f9c
            # epsion should be 1.2 (default)
            # SUPERSEDED!

            pcapbasename = basename(specimens.pcapFileName)
            print("Trace:", pcapbasename)
            epsilon = args.epsilon  # TODO make "not set" epsilon and "default" distinguishable
            if args.epsilon == epsdefault and pcapbasename in epspertrace:
                epsilon = epspertrace[pcapbasename]
            evaluateFieldTypeClustering(filteredSegments, epsilon, DistanceCalculator.neutralThreshold, {})
                                                     # DistanceCalculator.sigmoidThreshold, {'shift': .6})
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




