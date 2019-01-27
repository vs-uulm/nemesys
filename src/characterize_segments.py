"""
Characterize subsequences of messages by multiple metrics and visualize them.

To find valid features for identifying field boundaries and to distinguish between field types.
"""
import argparse

from os.path import isfile

import inference.segmentHandler as sh
from utils.loader import SpecimenLoader
from inference.analyzers import *
from inference.segments import MessageSegment, TypedSegment
from inference.templates import TemplateGenerator
from validation.dissectorMatcher import MessageComparator
from visualization.multiPlotter import MultiMessagePlotter
from visualization.distancesPlotter import DistancesPlotter

from characterize_fieldtypes import analyses, plotMultiSegmentLines, labelForSegment
from inference.segmentHandler import segments2clusteredTypes, filterSegments


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
    segmentedMessages = sh.annotateFieldTypes(analyzerType, analysisArgs, comparator)
    segsByLen = sh.groupByLength(segmentedMessages)
    print("done.")



    from tabulate import tabulate

    analysisTitle = "{}{}".format(args.analysis, "" if not analysisArgs else " ({})".format(analysisArgs))
    for length, segments in segsByLen.items():  # type: int, List[MessageSegment]

        #############################
        if args.count:
            # count common values:
            cntr = dict()  # type: Dict[Tuple[float], List[TypedSegment]]
            for seg in segments:  # type: TypedSegment
                vkey = tuple(seg.values)
                if vkey in cntr:
                    cntr[vkey].append(seg)
                else:
                    cntr[vkey] = [seg]

            vlmt = [(vlct, len(elmt)) for vlct,elmt in cntr.items()]

            print('Segment length:', length)
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
            # filter out segments that contain no relevant feature byte/data
            filteredSegments = filterSegments(segments)

            if length < 3:
                continue
            if len(filteredSegments) < 16:
                print("Too few relevant segments after Filtering for length {}. {} segments remaining:".format(
                    length, len(filteredSegments)
                ))
                for each in filteredSegments:
                    print("   ", each)
                print()
                continue
            # if length > 9:
            #     continue

            # TODO More Hypotheses:
            #  small values at fieldend: int
            #  all 0 values with variance vector starting with -255: 0-pad (validate what's the predecessor-field?)
            #  len > 16: chars (pad if all 0)
            # For a new hypothesis: what are the longest seen ints?
            #

            print("Calculate distances...")
            # ftype = 'id'
            # segments = [seg for seg in segsByLen[4] if seg.fieldtype == ftype]
            tg = TemplateGenerator(filteredSegments)

            print("Clustering...")
            # typeGroups = segments2typedClusters(segments,analysisTitle)
            segmentGroups = segments2clusteredTypes(tg, analysisTitle)
            # re-extract cluster labels for segments
            labels = numpy.array([
                labelForSegment(segmentGroups, seg) for seg in tg.segments
            ])

            # # if args.distances:
            print("Plot distances...")
            sdp = DistancesPlotter(specimens, 'distances-{}-{}-DBSCANe{}'.format(
                length, analysisTitle, tg.clusterer.epsilon if tg.clusterer else 'n/a'), args.interactive)
            # sdp.plotDistances(tg, numpy.array([seg.fieldtype for seg in tg.segments]))
            sdp.plotDistances(tg, labels)
            sdp.writeOrShowFigure()


            print("Prepare output...")
            for pagetitle, segmentClusters in segmentGroups:
                plotMultiSegmentLines(segmentClusters, pagetitle, True)

            typeDict = sh.segments2types(filteredSegments)
            typeGrp = [(t, [('', s) for s in typeDict[t]]) for t in typeDict.keys()]
            plotMultiSegmentLines(typeGrp, "{} ({} bytes) fieldtypes".format(analysisTitle, length), True)

            # IPython.embed()

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























