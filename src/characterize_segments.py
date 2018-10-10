"""
Characterize subsequences of messages by multiple metrics and visualize them.

To find valid features for identifying field boundaries and to distinguish between field types.
"""
import argparse

import IPython, numpy, math
from os.path import isfile, splitext, basename
from itertools import chain

import inference.segmentHandler as sh
from utils.loader import SpecimenLoader
from inference.analyzers import *
from inference.segments import MessageSegment, TypedSegment
from inference.templates import TemplateGenerator
from validation.dissectorMatcher import MessageComparator
from visualization.multiPlotter import MultiMessagePlotter
from visualization.distancesPlotter import DistancesPlotter




def segmentsFromLabels(analyzer, labels) -> List[TypedSegment]:
    """
    Segment messages according to true fields from the labels
    and mark each segment with its true type.

    :param analyzer: An Analyzer for/with a message
    :param labels: The labels of the true format
    :return: Segments of the analyzer's message according to the true format
    """
    segments = list()
    offset = 0
    for ftype, flen in labels:
        segments.append(TypedSegment(analyzer, offset, flen, ftype))
        offset += flen

    return segments


def removeIdenticalLabels(plt):
    handles, labels = plt.gca().get_legend_handles_labels()
    newLabels, newHandles = [], []
    for handle, label in zip(handles, labels):
        if label not in newLabels:
            newLabels.append(label)
            newHandles.append(handle)
    plt.legend(newHandles, newLabels)


def plotLinesSegmentValues(segments: List[Tuple[str, MessageSegment]]):
    import matplotlib.cm
    import matplotlib.pyplot as plt

    for lab, seg in segments:
        color = hash(lab) % 8  # TODO make more collision resistant
        plt.plot(seg.values, c=matplotlib.cm.Set1(color + 1), alpha=0.4, label=lab)

    removeIdenticalLabels(plt)

    # plt.legend(scatterpoints=1, loc='best', shadow=False)
    plt.show()
    plt.clf()


def plotMultiSegmentLines(segmentGroups: List[Tuple[str, List[Tuple[str, TypedSegment]]]], pagetitle=None,
                              colorPerLabel=False):
    ncols = 5 if len(segmentGroups) > 4 else math.ceil(len(segmentGroups) / 2)
    nrows = (len(segmentGroups) // ncols)
    if not len(segmentGroups) % ncols == 0:
        nrows += 1
    if nrows < 3:
        nrows = 3

    mmp = MultiMessagePlotter(specimens, pagetitle, nrows, ncols, args.interactive)
    mmp.plotMultiSegmentLines(segmentGroups, colorPerLabel)
    mmp.writeOrShowFigure()
    mmp = None


def segments2types(segments: List[TypedSegment]):
    typegroups = dict()
    for seg in segments:
        if seg.fieldtype in typegroups:
            typegroups[seg.fieldtype].append(seg)
        else:
            typegroups[seg.fieldtype] = [seg]
    return typegroups


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


def segments2clusteredTypes(segments: List[TypedSegment], analysisTitle) \
        -> List[Tuple[str, List[Tuple[str, TypedSegment]]]]:
    """
    Cluster segments according to the distance of their feature vectors.
    Keep and label segments classified as noise.

    :param segments:
    :param analysisTitle:
    :return:
    """
    print("Calculate distances...")
    tg = TemplateGenerator(segments)
    print("Clustering segments...")
    noise, *clusters = tg.clusterSimilarSegments(False)
    print("{} clusters generated from {} segments".format(len(clusters), len(segments)))

    segmentClusters = list()
    if len(noise) > 0:
        segmentClusters.append(('{} ({} bytes), Noise: {} Seg.s'.format(
            analysisTitle, noise[0].length, len(noise)),
                                   [('', cseg) for cseg in noise] ))
    for cnum, segs in enumerate(clusters):
        typegroups = segments2types(segs)

        segmentGroups = ('{} ({} bytes), Cluster #{}: {} Seg.s'.format(
            analysisTitle, segs[0].length, cnum, len(segs)), list())
        for ftype, tsegs in typegroups.items():  # [label, segment]
            segmentGroups[1].extend([("{}: {} Seg.s".format(ftype, len(tsegs)), tseg) for tseg in tsegs])
        segmentClusters.append(segmentGroups)

    segmentClusters = [ ( '{} ({} bytes), '.format(analysisTitle,
                                                   clusters[0][0].length if clusters else noise[0].length),
                          segmentClusters) ]
    return segmentClusters






# available analysis methods
analyses = {
    'bcpnm': BitCongruenceNgramMean,
    # 'bcpnv': BitCongruenceNgramStd,  in branch inference-experiments
    'bc': BitCongruence,
    'bcd': BitCongruenceDelta,
    'bcdg': BitCongruenceDeltaGauss,
    'mbhbv': HorizonBitcongruence,

    'variance': ValueVariance,
    'progdiff': ValueProgressionDelta,
    'progcumudq': CumulatedProgressionDelta,
}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Characterize subsequences of messages by multiple metrics and visualize them.')
    parser.add_argument('pcapfilename', help='pcapfilename') # , nargs='+')
    parser.add_argument('-i', '--interactive', help='show interactive plot instead of writing output to file.',
                        action="store_true")
    parser.add_argument('analysis', type=str,
                        help='The kind of analysis to apply on the messages. Available methods are: '
                        + ', '.join(analyses.keys()))
    parser.add_argument('--parameters', '-p', help='Parameters for the analysis.')
    parser.add_argument('--distances', '-d', help='Plot distances instead of features.',
                        action="store_true")
    parser.add_argument('--count', '-c', help='Count common values of features.',
                        action="store_true")
    args = parser.parse_args()
    #for pcap in args.pcapfilename:
    if not isfile(args.pcapfilename):
        print('File not found: ' + args.pcapfilename)
        exit(1)

    analyzerType = analyses[args.analysis]
    analysisArgs = args.parameters

    # dissect and label messages
    print("Loading messages...")
    specimens = SpecimenLoader(args.pcapfilename, layer=2, relativeToIP=True)
    comparator = MessageComparator(specimens, layer=2, relativeToIP=True)
    # segment messages according to true fields from the labels
    print("Segmenting messages...", end=' ')
    segmentedMessages = [segmentsFromLabels(
        MessageAnalyzer.findExistingAnalysis(analyzerType, MessageAnalyzer.U_BYTE,
                                             l4msg, analysisArgs), comparator.dissections[rmsg])
        for l4msg, rmsg in specimens.messagePool.items()]
    print("done.")

    # groupbylength
    segsByLen = dict()
    for seg in chain.from_iterable(segmentedMessages):
        seglen = len(seg.values)
        if seglen not in segsByLen:
            segsByLen[seglen] = list()
        segsByLen[seglen].append(seg)

    from tabulate import tabulate

    analysisTitle = "{}{}".format(args.analysis, "" if not analysisArgs else " ({})".format(analysisArgs))
    for length, segments in segsByLen.items():  # type: int, List[MessageSegment]

        #############################
        if args.count:
            # count common values:
            cntr = dict()
            for seg in segments:
                vkey = tuple(seg.values)
                if vkey in cntr:
                    cntr[vkey].append(seg)
                else:
                    cntr[vkey] = [seg]

            vlmt = [(vlct, len(elmt)) for vlct,elmt in cntr.items()]

            print('Segement length:', length)
            moCoTen = sorted(vlmt, key=lambda v: v[1])[:-11:-1]
            print(tabulate([(
                mct, featvect,
                set([(cv.bytes.hex(), cv.fieldtype) for cv in cntr[featvect]])
            ) for featvect, mct in moCoTen],
                headers= ['Count', 'Feature', 'Fields']))

            # TODO place labels with most common feature and/or byte values in distance graph

            # [seg.fieldtype for seg in segments]
            # IPython.embed()
        #############################
        else:
            # if length < 3 or len(segments) < 10:
            #     continue
            if length != 4:  # TODO only 4 byte fields for now!
                continue

            # TODO filter segments that contain no relevant feature data, i. e.,
            # (0, .., 0)
            # (nan, .., nan)
            filteredSegments = [t for t in segments if t.bytes.count(b'\x00') != len(t.bytes)]
            filteredSegments = [s for s in filteredSegments if
                                numpy.count_nonzero(s.values) - numpy.count_nonzero(numpy.isnan(s.values)) > 0 ]

            # More Hypotheses:
            #  small values at fieldend: int
            #  all 0 values with variance vector starting with -255: 0-pad (validate what's the predecessor-field?)
            #  len > 16: chars (pad if all 0)
            # For a new hypothesis: what are the longest seen ints?
            #


            if args.distances:
                print("Calculate distances...")
                # ftype = 'id'
                # segments = [seg for seg in segsByLen[4] if seg.fieldtype == ftype]
                tg = TemplateGenerator(filteredSegments)
                print("Plot distances...")
                sdp = DistancesPlotter(specimens, 'distances-{}-{}'.format(length ,analysisTitle), args.interactive)
                sdp.plotDistances(tg, numpy.array([seg.fieldtype for seg in tg.segments]))
                sdp.writeOrShowFigure()

                # IPython.embed()

            else:
                # segmentGroups = segments2typedClusters(segments,analysisTitle)
                segmentGroups = segments2clusteredTypes(filteredSegments, analysisTitle)

                print("Prepare output...")
                for pagetitle, segmentClusters in segmentGroups:
                    plotMultiSegmentLines(segmentClusters, pagetitle, True)

    exit()





    # # TODO cluster by template generation
    # print("Templating segments...")
    # tg = TemplateGenerator(list(chain.from_iterable(segmentedMessages)))
    # clusters = tg.clusterSimilarSegments()


    # #
    # # TODO for each cluster: count the number of fields per type (histograms)
    # for cluster in clusters:
    #     cntr = Counter()
    #     cntr.update([segment.fieldtype for segment in cluster])
    #     print("Cluster info: ", len(cluster), " segments of length", cluster[0].length)
    #     pprint(cntr)
    #     print()


    # TODO: test (differential) value progression for field type separation

    # TODO: Entropy rate (to identify non-inferable segments)

    # TODO: Value domain per byte/nibble (for chars, flags,...)

    # print("Plotting templates...")
    # import analysis_message_segments as ams
    # ams.plotDistances(tg)

    # templates = tg.generateTemplates()



    exit()

    # statistics per segment
    meanSegments = sh.segmentMeans(segmentedMessages)
    varSegments = sh.segmentStdevs(segmentedMessages)

    # plot
    mmp = MultiMessagePlotter(specimens, args.analysis, 5, 8, args.interactive)
    # mmp.plotSegmentsInEachAx(segmentedMessages)
    # mmp.plotSegmentsInEachAx(meanSegments)
    # TODO use transported ftype to color the segments in the plot afterwards.
    mmp.plotSegmentsInEachAx(varSegments)
    mmp.writeOrShowFigure()

    # IPython.embed()























