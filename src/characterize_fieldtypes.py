"""
Use groundtruth about field segmentation by dissectors and apply field type identification to them.

Takes a PCAP trace of a known protocol, dissects each message into their fields, and yields segments from each of them.
These segments get analyzed by the given analysis method which is used as feature to determine their similarity.
Similar fields are then clustered by DBSCAN and for comparison plotted in groups of their real field types.
In addition, a MDS projection into a 2D plane for visualization of the relative distances of the features is plotted.
"""

import argparse, IPython
from os.path import isfile

from inference.templates import TemplateGenerator
from inference.segments import TypedSegment
from inference.analyzers import *
from inference.segmentHandler import annotateFieldTypes, groupByLength, segments2types
from validation.dissectorMatcher import MessageComparator
from utils.loader import SpecimenLoader
from visualization.multiPlotter import MultiMessagePlotter
from visualization.distancesPlotter import DistancesPlotter


debug = False



def filterSegments(segments):
    """
    Filter input segment for only those segments that are adding relevant information for further analysis.

    :param segments:
    :return:
    """
    # filter out segments shorter than 3 bytes
    filteredSegments = [t for t in segments if t.length > 2]

    # filter out segments that contain no relevant byte data, i. e., all-zero byte sequences
    filteredSegments = [t for t in filteredSegments if t.bytes.count(b'\x00') != len(t.bytes)]

    # filter out segments that resulted in no relevant feature data, i. e.,
    # (0, .., 0) | (nan, .., nan) | or a mixture of both
    filteredSegments = [s for s in filteredSegments if
                        numpy.count_nonzero(s.values) - numpy.count_nonzero(numpy.isnan(s.values)) > 0]

    # filter out identical segments
    uniqueFeatures = set()
    fS = filteredSegments
    filteredSegments = list()
    for s in fS:
        svt = tuple(s.values)
        if svt not in uniqueFeatures:
            uniqueFeatures.add(svt)
            filteredSegments.append(s)

    return filteredSegments


def segments2typedClusters(segments: List[TypedSegment], analysisTitle) \
        -> List[Tuple[str, List[Tuple[str, TypedSegment]]]]:
    typegroups = segments2types(segments)
    # plotLinesSegmentValues([(typelabel, typ) for typelabel, tgroup in typegroups.items() for typ in tgroup])

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
            # plotLinesSegmentValues([(str(clusternum) + ftype + str(len(clustersegs)), seg)
            #                     for clusternum, clustersegs in enumerate(clusters) for seg in clustersegs])
            for clusternum, clustersegs in enumerate(clusters):
                segmentClusters[1].append(('Cluster #{}: {} Seg.s'.format(clusternum, len(clustersegs)),
                                           [('', cseg) for cseg in clustersegs]))
            segmentGroups.append(segmentClusters)
    return segmentGroups


def segments2clusteredTypes(tg : TemplateGenerator, analysisTitle: str, **kwargs) \
        -> List[Tuple[str, List[Tuple[str, List[Tuple[str, TypedSegment]]]]]]:
    """
    Cluster segments according to the distance of their feature vectors.
    Keep and label segments classified as noise.

    :param tg:
    :param analysisTitle:
    :param kwargs: arguments for the clusterer
    :return: List/Tuple structure of annotated analyses, clusters, and segments.
        List [ of
            Tuples (
                 "analysis label",
                 List [ of cluster
                    Tuples (
                        "cluster label",
                        List [ of segment
                            Tuples (
                                "segment label (e. g. field type)",
                                MessageSegment object
                            )
                        ]
                    )
                ]
            )
        ]
    """
    print("Clustering segments...")
    if not kwargs:
        noise, *clusters = tg.clusterSimilarSegments(False)
    else:
        noise, *clusters = tg.clusterSimilarSegments(False, **kwargs)
    print("{} clusters generated from {} segments".format(len(clusters), len(tg.segments)))

    segmentClusters = list()
    segLengths = set()
    numNoise = len(noise)
    if numNoise > 0:
        noiseSegLengths = {seg.length for seg in noise}
        segLengths.update(noiseSegLengths)
        noisetypes = {t: len(s) for t, s in segments2types(noise).items()}
        segmentClusters.append(('{} ({} bytes), Noise: {} Seg.s'.format(
            analysisTitle, " ".join([str(slen) for slen in noiseSegLengths]), numNoise),
                                   [("{}: {} Seg.s".format(cseg.fieldtype, noisetypes[cseg.fieldtype]), cseg)
                                    for cseg in noise] )) # ''
    for cnum, segs in enumerate(clusters):
        clusterDists = tg.pairwiseDistance(segs, segs)
        typegroups = segments2types(segs)
        clusterSegLengths = {seg.length for seg in segs}
        segLengths.update(clusterSegLengths)

        segmentGroups = ('{} ({} bytes), Cluster #{}: {} Seg.s ($d_{{max}}$={:.3f})'.format(
            analysisTitle, " ".join([str(slen) for slen in clusterSegLengths]),
            cnum, len(segs), clusterDists.max()), list())
        for ftype, tsegs in typegroups.items():  # [label, segment]
            segmentGroups[1].extend([("{}: {} Seg.s".format(ftype, len(tsegs)), tseg) for tseg in tsegs])
        segmentClusters.append(segmentGroups)

    # print(len(clusters), len(noise))

    segmentClusters = [ ( '{} ({} bytes) {}'.format(analysisTitle,
                                                    " ".join([str(slen) for slen in segLengths]),
                                                   tg.clusterer if tg.clusterer else 'n/a'),
                          segmentClusters) ]
    return segmentClusters


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
    :return:
    """
    mmp = MultiMessagePlotter(specimens, pagetitle, len(segmentGroups) + 1, isInteractive=args.interactive)
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

        mmp.axes[-1].text(0, 0, "conciseness = {:.2f}".format(conciseness))

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
            ratioNoise = numNoise / numSegs
            noiseTypes = {ft for ft, seg in noise}

        import os, csv
        clStatsFile = os.path.join('reports/', 'clusterStatisticsHDBSCAN.csv')
        csvWriteHead = False if os.path.exists(clStatsFile) else True
        with open(clStatsFile, 'a') as csvfile:
            clStatscsv = csv.writer(csvfile)  # type: csv.writer
            if csvWriteHead:
                # in "pagetitle": "seg_length", "analysis", "dist_measure", 'min_cluster_size'
                clStatscsv.writerow(['run_title', 'conciseness', 'most_freq_type', 'precision', 'recall', 'cluster_size'])
            if noise:
                # noinspection PyUnboundLocalVariable
                clStatscsv.writerow([pagetitle, conciseness, 'NOISE', str(noiseTypes), ratioNoise, numNoise])
            clStatscsv.writerows([
                (pagetitle, conciseness, *pr) for pr in prList if pr is not None
            ])

    mmp.writeOrShowFigure()
    del mmp



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





if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Analyze fields as segments of messages and plot field type identification quality.')
    parser.add_argument('pcapfilename', help='Filename of the PCAP to load.')
    parser.add_argument('-i', '--interactive', help='Show interactive plot instead of writing output to file and '
                                                    'open ipython prompt after finishing the analysis.',
                        action="store_true")
    parser.add_argument('analysis', type=str,
                        help='The kind of analysis to apply on the messages. Available methods are: '
                        + ', '.join(analyses.keys()) + '.')
    parser.add_argument('--parameters', '-p', help='Parameters for the analysis.')
    parser.add_argument('--epsilon', '-e', help='Parameter epsilon for the DBSCAN clusterer.', type=float, default=1.0)
    args = parser.parse_args()

    if not isfile(args.pcapfilename):
        print('File not found: ' + args.pcapfilename)
        exit(1)
    if args.analysis not in analyses:
        print('Analysis {} unknown. Available methods are:\n' + ', '.join(analyses.keys()) + '.')
        exit(2)
    analyzerType = analyses[args.analysis]
    analysisArgs = args.parameters
    analysisTitle = "{}{}".format(args.analysis, "" if not analysisArgs else " ({})".format(analysisArgs))

    # dissect and label messages
    print("Load messages...")
    specimens = SpecimenLoader(args.pcapfilename, 2, True)
    comparator = MessageComparator(specimens, 2, True, debug=debug)

    # segment messages according to true fields from the labels
    print("Segmenting messages...", end=' ')
    segmentedMessages = annotateFieldTypes(analyzerType, analysisArgs, comparator)

    splitFieldLength = False
    if not splitFieldLength:  # dummy to select between old, field-size dependent and new, general similarity
        distance_method = 'canberra'
        mcs = 25
        minlength = 3

        from itertools import chain, compress
        filteredSegments = filterSegments(chain.from_iterable(segmentedMessages)) # TODO: del len < minlength
        filteredSegments = sorted(filteredSegments, key=lambda x: x.length)  # sorted only for visual representation in heatmap below
        print("done.")

        print("Calculate distances...")
        tg = TemplateGenerator(filteredSegments, distance_method)
        print("Clustering...")
        segmentGroups = segments2clusteredTypes(tg, analysisTitle, min_cluster_size=mcs) # epsilon=2.5)
        # re-extract cluster labels for segments
        labels = numpy.array([
            labelForSegment(segmentGroups, seg) for seg in tg.segments
        ])

        typeDict = segments2types(filteredSegments)

        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        # Testing for mixed-length field type identification efficacy
        lenMasks = {}
        for idx, seg in enumerate(filteredSegments):
            if not seg.length in lenMasks:
                lenMasks[seg.length] = [False] * len(filteredSegments)
            lenMasks[seg.length][idx] = True

        import seaborn as sns
        import matplotlib.pyplot as plt
        segFieldtypes = [seg.fieldtype if pseg.fieldtype != seg.fieldtype else '' for seg, pseg in
                         zip(filteredSegments, filteredSegments[:1] + filteredSegments)]
        # # equal lengths
        # l4 = tg.distanceMatrix[lenMasks[4]][:,numpy.array(lenMasks[4])]
        # # unequal lengths
        # l4 = tg.distanceMatrix[lenMasks[4]][:,~numpy.array(lenMasks[4])]
        IPython.embed()  # TODO here I am!
        # # distance to one field candidate
        # sns.set(font_scale=.6)
        # sns.barplot(list(range(tg.distanceMatrix.shape[0])), tg.distanceMatrix[180,])
        # plt.show()
        # # distance matrix heatmap (sorted above for this plot)
        # xymax = 160
        # sns.heatmap(tg.distanceMatrix[:xymax,:xymax], xticklabels=segFieldtypes[:xymax], yticklabels=segFieldtypes[:xymax])
        # plt.show()
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

        print("Plot distances...")
        sdp = DistancesPlotter(specimens, 'distances-{} ({})'.format(
            segmentGroups[0][0], distance_method), args.interactive)
        # old: 'mixedlength', analysisTitle, distance_method, tg.clusterer if tg.clusterer else 'n/a'
        # sdp.plotDistances(tg, numpy.array([seg.fieldtype for seg in tg.segments]))
        sdp.plotDistances(tg, labels)
        sdp.writeOrShowFigure()

        print("Prepare output...")
        for pagetitle, segmentClusters in segmentGroups:
            plotMultiSegmentLines(segmentClusters, "{} ({})".format(pagetitle, distance_method), True, typeDict)


    else:
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
                    # sdp.plotDistances(tg, numpy.array([seg.fieldtype for seg in tg.segments]))
                    sdp.plotDistances(tg, labels)
                    sdp.writeOrShowFigure()

                    print("Prepare output...")
                    for pagetitle, segmentClusters in segmentGroups:
                        plotMultiSegmentLines(segmentClusters, "{} - {}".format(pagetitle, distance_method), True, typeDict)



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




