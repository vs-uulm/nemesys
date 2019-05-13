"""
Use ground truth about field borders and their data types generate MessageSegments representing the field values.

Takes a PCAP trace of a known protocol, dissects each message into their fields, and yields segments from each of them.
These segments get analyzed by the given analysis method which is used as feature.
Charateristic feature abstraction (mean and stdev) is generated as templates for specific field data types.

"""
import argparse, IPython
from os.path import isfile, basename
from itertools import chain

from utils.evaluationHelpers import epspertrace, epsdefault, analyses, annotateFieldTypes, plotMultiSegmentLines, \
    labelForSegment
from inference.templates import TemplateGenerator, DistanceCalculator, DBSCANsegmentClusterer, HDBSCANsegmentClusterer, \
    DelegatingDC
from inference.segments import TypedSegment
from inference.analyzers import *
from inference.segmentHandler import groupByLength, segments2types, segments2clusteredTypes, \
    filterSegments
from validation.dissectorMatcher import MessageComparator
from utils.loader import SpecimenLoader
from visualization.distancesPlotter import DistancesPlotter
from visualization.multiPlotter import MultiMessagePlotter, PlotGroups



debug = False

# fix the analysis method to VALUE
analysisTitle = 'value'
# fix the distance method to canberra
distance_method = 'canberra'



def types2segmentGroups(segmentGroups: Dict[str, List[TypedSegment]]) \
        -> List[Tuple[str, List[Tuple[str, TypedSegment]]]]:
    """

    :param analysisTitle: label for the outermost structure label
    :param segmentGroups: A dict of
        fieldtype (str) : segments of this type (list)

    :return: List/Tuple structure of annotated analyses, clusters, and segments.
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
    """
    plotGroups = PlotGroups("")
    segLengths = set()

    for sType, tSegs in segmentGroups.items():
        # handle lengths for adding them to the labels
        groupSegLengths = {seg.length for seg in tSegs}
        outputLengths = [str(slen) for slen in groupSegLengths]
        if len(outputLengths) > 5:
            outputLengths = outputLengths[:2] + ["..."] + outputLengths[-2:]
        segLengths.update(groupSegLengths)

        plotGroups.appendPlot(0, '{} ({} bytes): {} Seg.s'.format(sType, " ".join(outputLengths), len(tSegs)),
                                    [("", tseg) for tseg in tSegs])
    return plotGroups.plotsList(0)


def segments2typedClusters(segments: List[TypedSegment]) \
        -> List[Tuple[str, List[Tuple[str, List[Tuple[str, TypedSegment]]]]]]:
    """
    Cluster segments and arrange them into groups of types.

    :param segments:
    :return:
    """
    typegroups = segments2types(segments)
    plotGroups = PlotGroups()

    # one plot per type with clusters
    for ftype, segs in typegroups.items():  # [label, segment]
        dc = DistanceCalculator(segs)
        clusterer = DBSCANsegmentClusterer(dc)
        noise, *clusters = clusterer.clusterSimilarSegments(False)
        print("{} clusters generated from {} segments".format(len(clusters), len(segs)))

        cid = plotGroups.appendCanvas("{}, {} bytes".format(
            ftype,
            clusters[0][0].length if clusters else noise[0].length)
        )

        if len(noise) > 0:
            plotGroups.appendPlot(cid, 'Noise: {} Seg.s'.format(len(noise)),
                                  [('', cseg) for cseg in noise])
        if len(clusters) > 0:
            for clusternum, clustersegs in enumerate(clusters):
                plotGroups.appendPlot(cid, 'Cluster #{}: {} Seg.s'.format(clusternum, len(clustersegs)),
                                           [('', cseg) for cseg in clustersegs])
    return plotGroups.canvasList




if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Analyze fields as segments of messages and plot field type identification quality.')
    parser.add_argument('pcapfilename', help='Filename of the PCAP to load.')
    parser.add_argument('-i', '--interactive', help='Show interactive plot instead of writing output to file and '
                                                    'open ipython prompt after finishing the analysis.',
                        action="store_true")
    parser.add_argument('-c', '--clusters-per-type',
                        help='Cluster each true field type and plot one page for each type.',
                        action="store_true")
    args = parser.parse_args()

    if not isfile(args.pcapfilename):
        print('File not found: ' + args.pcapfilename)
        exit(1)

    analyzerType = analyses[analysisTitle]
    analysisArgs = None

    # dissect and label messages
    print("Load messages...")
    specimens = SpecimenLoader(args.pcapfilename, 2, True)
    comparator = MessageComparator(specimens, 2, True, debug=debug)

    # segment messages according to true fields from the labels
    print("Segmenting messages...")
    segmentedMessages = annotateFieldTypes(analyzerType, analysisArgs, comparator)
    segments = list(chain.from_iterable(segmentedMessages))





    print("Prepare output...")
    pagetitle = "true-field-types_{}-{}".format(analysisTitle, distance_method)

    justtypes = not args.clusters_per_type
    if justtypes:
        typegroups = segments2types(segments)
        groupStructure = types2segmentGroups(typegroups)

        mmp = MultiMessagePlotter(specimens, pagetitle, len(groupStructure), isInteractive=False)
        mmp.plotMultiSegmentLines(groupStructure, True)
        mmp.writeOrShowFigure()
        del mmp
    else:
        groupStructure = segments2typedClusters(segments)

        for pnum, (ptitle, page) in enumerate(groupStructure):
            mmp = MultiMessagePlotter(specimens, "{}_{}_{}".format(pagetitle, pnum, ptitle),
                                      len(page), isInteractive=False)
            mmp.plotMultiSegmentLines(page, True)
            mmp.writeOrShowFigure()
            del mmp

    if args.interactive:
        IPython.embed()
