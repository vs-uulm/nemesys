"""
NEMEFTR-full mode 2, step 1:
Generate FieldTypeTemplates representing data types.
For segmentation, it uses ground truth about field borders and field data types.

Takes a PCAP trace of a known protocol, dissects each message into their fields, and yields segments from each of them.
These segments get analyzed by the given analysis method which is used as feature.
For abstraction from individual field values to the charateristic feature (mean and covariance)
FieldTypeTemplates are generated to persist the specific field type characteristics.

"""
import argparse, IPython
from os.path import isfile, basename
from itertools import chain

from inference.templates import DBSCANsegmentClusterer, DelegatingDC, DistanceCalculator, FieldTypeTemplate
from inference.fieldTypes import FieldTypeMemento
from inference.segments import TypedSegment, HelperSegment
from inference.analyzers import *
from inference.segmentHandler import groupByLength, segments2types, \
    filterSegments
from validation.dissectorMatcher import MessageComparator
from utils.loader import SpecimenLoader
from visualization.distancesPlotter import DistancesPlotter
from visualization.multiPlotter import MultiMessagePlotter, PlotGroups
from utils.evaluationHelpers import *


debug = False

# fix the analysis method to VALUE
analysisTitle = 'value'
# fix the distance method to canberra
distance_method = 'canberra'



def types2segmentGroups(segmentGroups: Dict[str, List[TypedSegment]]) \
        -> List[Tuple[str, List[Tuple[str, TypedSegment]]]]:
    """
    Converts lists of segments grouped in a dict by field type into the list structure of plotable features

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


def segments2typedClusters(segments: List[TypedSegment], withPlots=True) \
        -> List[Tuple[str, List[Tuple[str, List[Tuple[str, TypedSegment]]]]]]:
    """
    Cluster segments and arrange them into groups of types.
    The autodetected eps is increased by factor 2.0 to add some margin.

    On the way, plot distances of segments and clusters in a 2-dimensional projection.

    :param withPlots: Generate distance plots for the clusters while creating them.
    :param segments: The segments to determine true types for and cluster these.
    :return: A group structure of types, clusters, and segments for plotting with
        MultiMessagePlotter#plotMultiSegmentLines
    """
    typegroups = segments2types(segments)
    plotGroups = PlotGroups()

    # one plot per type with clusters
    for ftype, segs in typegroups.items():  # [label, segment]
        # # raw segments
        # dc = DistanceCalculator(segs)
        # deduplicated segments
        dc = DelegatingDC(segs)
        clusterer = DBSCANsegmentClusterer(dc)
        # autoconf is working correct, but for pre-sorted field types we need higher threshold
        clusterer.eps = clusterer.eps * 2.0  # we can add some more margin to the eps
        noise, *clusters = clusterer.clusterSimilarSegments(False)
        print("Type {}: {} clusters generated from {} segments".format(ftype, len(clusters), len(segs)))

        cid = plotGroups.appendCanvas("{}, {} Seg.s in {} clusters".format(
            ftype, len(segs), len(clusters))
        )

        if len(noise) > 0:
            plotGroups.appendPlot(cid, 'Noise: {} Seg.s'.format(len(noise)),
                                  [('', cseg) for cseg in noise])
        if len(clusters) > 0:
            for clusternum, clustersegs in enumerate(clusters):
                plotGroups.appendPlot(cid, 'Cluster #{}: {} Seg.s'.format(clusternum, len(clustersegs)),
                                           [('', cseg) for cseg in clustersegs])

        if withPlots:
            # print("Plot distances...")
            sdp = DistancesPlotter(specimens,
                                   'distances-' + "tft_{}_DBSCAN-eps{:0.3f}-ms{}".format(
                                       plotGroups.canvasList[cid][0], clusterer.eps, clusterer.min_samples), False)
            clustermask = {segid: cluN for cluN, segL in enumerate(clusters) for segid in dc.segments2index(segL)}
            clustermask.update({segid: "Noise" for segid in dc.segments2index(noise)})
            sdp.plotSegmentDistances(dc, numpy.array([clustermask[segid] for segid in range(len(dc.segments))]))
            sdp.writeOrShowFigure()
            del sdp

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
    parser.add_argument('-p', '--with-plots',
                        help='Generate plots of true field types and their distances.',
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

    withPlots = args.with_plots



    print("Prepare output...")
    pagetitle = "true-field-types_{}-{}".format(analysisTitle, distance_method)

    justtypes = not args.clusters_per_type
    if justtypes:
        typegroups = segments2types(segments)
        groupStructure = types2segmentGroups(typegroups)

        mmp = MultiMessagePlotter(specimens, pagetitle, len(groupStructure), isInteractive=False)
        mmp.plotMultiSegmentLines(groupStructure, True)
        mmp.writeOrShowFigure()
    else:
        groupStructure = segments2typedClusters(segments, withPlots)

        # FieldTypeTemplates with mean, and stdev
        fieldtypeCandidates = list()
        for ptitle, page in groupStructure:
            # names of iterated local vars page/plot are according to the groupStructure
            currPage = list()
            for plotTitle, plotData in page:
                currPlot = FieldTypeTemplate([pd for s, pd in plotData])
                currPage.append(currPlot)
            fieldtypeCandidates.append(currPage)
        # lookup of typeIDs
        ftMap = {plot.typeID: plot for page in fieldtypeCandidates for plot in page}  # type: Dict[str, FieldTypeTemplate]

        if withPlots:
            for (ptitle, page), ftTemplates in zip(groupStructure, fieldtypeCandidates):
                mmp = MultiMessagePlotter(specimens, "{}_fieldtype_{}".format(pagetitle, ptitle),
                                          len(page), isInteractive=False)
                mmp.plotMultiSegmentLines(page, True)

                groupStats = (list(), list(), list())
                for ftTempl in ftTemplates:
                    # for each vector component plot mean, mean - stdev, mean + stdev in mmp
                    groupStats[0].append(ftTempl.mean)
                    groupStats[1].append(ftTempl.upper)
                    groupStats[2].append(ftTempl.lower)
                mmp.plotInEachAx(groupStats[0], {'c': 'black'})
                mmp.plotInEachAx(groupStats[1], {'c': 'green'})
                mmp.plotInEachAx(groupStats[2], {'c': 'red'})
                mmp.textInEachAx([ftTempl.typeID for ftTempl in ftTemplates])

                mmp.writeOrShowFigure()
                del mmp

        # binaryProtocols = "binaryprotocols_merged_500.pcap"
        binaryProtocols = "binaryprotocols_maxdiff-fromOrig-500.pcap"

        if basename(specimens.pcapFileName) != binaryProtocols:
            print("Run the script with input file", binaryProtocols, " to access selected field type templates.")
        else:
            try:
                # # # # # # # # # # # # # # # # # # # # # # # # # # # #
                # select promising field type templates from binaryprotocols_merged_500.pcap
                fieldtypeTemplates = dict()

                # for ipv4: use the complete group
                fieldtypeTemplates["ipv4"] = [
                    # FieldTypeTemplate([ segs[1] for ptitle, page in groupStructure if ptitle.startswith("ipv4")
                    #   for cluster in page for segs in cluster[1] ])
                    # FTT from selected cluster: e.g. combine clusters n to m
                    FieldTypeTemplate(chain.from_iterable([ftMap[tid].baseSegments for tid in
                                                           ["ce412960", "60b04ad2", "c9f10baa", "f4891db9"]])),
                    ftMap["88357dbb"],
                    FieldTypeTemplate(chain.from_iterable([ftMap[tid].baseSegments for tid in
                                                           ["3235ee72", "b2eded77"]])),
                ]
                for ftt in fieldtypeTemplates["ipv4"]:
                    ftt.fieldtype = "ipv4"

                # for macaddr: use the complete group
                fieldtypeTemplates["macaddr"] = [FieldTypeTemplate(
                    # [ segs[1] for ptitle, page in groupStructure if ptitle.startswith("macaddr")
                    #   for cluster in page for segs in cluster[1] ]
                    chain.from_iterable([ftMap[tid].baseSegments for tid in
                                         ["7a7d3084", "f7dc5c07"]])
                )]

                # for id: test whether this works
                fieldtypeTemplates["id"] = [FieldTypeTemplate(chain.from_iterable([ftMap[tid].baseSegments for tid in
                                                           ["5470aaf1", "e5fe2f7a"]])
                )]
                # fieldtypeTemplates["id"][0].fieldtype = "id"

                # # for int: use the complete group AND refine
                # intHelpers = list()
                # # modify values to raise the first bytes from zero to get something like floatMean = [1, 4, 22, 117]
                # int_rand_lower = numpy.array([0, 0, 0, 0]) # , 14, 44]   # [1, 4, 0, 0]
                # int_rand_upper = numpy.array([2, 6, 0, 0]) # , 31, 190]
                # for tid in ["bdee139a"]:
                #     for bs in ftMap[tid].baseSegments:  # type; AbstractSegment
                # # for bs in (segs[1] for ptitle, page in groupStructure if ptitle.startswith("int")
                # #              for cluster in page for segs in cluster[1] if segs[1].length == 4):  # only of length 4
                #         ih = HelperSegment(bs.analyzer, 0, bs.length)
                #         ih.values = bs.values + \
                #                     (int_rand_upper - int_rand_lower) * numpy.random.rand(4) + int_rand_lower
                #         intHelpers.append(ih)
                fieldtypeTemplates["int"] = [ftMap["bdee139a"]]  # FieldTypeTemplate(intHelpers)
                # fieldtypeTemplates["int"][0].fieldtype = "int"

                # for timestamp: use the complete group
                fieldtypeTemplates["timestamp"] = [ FieldTypeTemplate(
                    [ segs[1] for ptitle, page in groupStructure if ptitle.startswith("timestamp")
                      for cluster in page for segs in cluster[1] if segs[1].length == 8 ]
                ) ]

                # for checksum: use only 8 byte fields
                fieldtypeTemplates["checksum"] = [ftMap["06bbd67e"]]

                # # TODO for int: test two byte template
                # fieldtypeTemplates["int"] = [FieldTypeTemplate(
                #     ftMap["4e923fd9"].baseSegments
                # )]

                # Python code representation to persist the fieldtypeTemplates
                fieldtypeMementos = list()
                print("[")
                for ftype, templateList in fieldtypeTemplates.items():
                    print("#", ftype)
                    for ftt in templateList:
                        ftm = FieldTypeMemento.fromTemplate(ftt)
                        print(ftm.codePersist, ",")
                        fieldtypeMementos.append(ftm)
                print("]")
                # # # # # # # # # # # # # # # # # # # # # # # # # # # #

                # # # # # # # # # # # # # # # # # # # # # # # # # # # #
                # # extract some suitable example messages for testing (nearest, farthest for template)
                #
                # def nf4ftt(ftt: FieldTypeTemplate):
                #     maha = [ftt.mahalanobis(bs.values) for bs in ftt.baseSegments]
                #     nea = ftt.baseSegments[numpy.argmin(maha)]
                #     far = ftt.baseSegments[numpy.argmax(maha)]
                #     return nea.analyzer.message, far.analyzer.message
                #
                #
                # for ftt in [fieldtypeTemplates["float"][0], fieldtypeTemplates["ipv4"][0]]:
                #     near, far = nf4ftt(ftt)
                #     nid = [idx for idx, absmsg in enumerate(specimens.messagePool.keys()) if absmsg == near][0]
                #     fid = [idx for idx, absmsg in enumerate(specimens.messagePool.keys()) if absmsg == far][0]
                #
                #     print("near and far messages for", ftt.fieldtype, nid, fid)
                # # # # # # # # # # # # # # # # # # # # # # # # # # # #



                # for ftype, templateList in fieldtypeTemplates.items():
                #     print("#", ftype, "\n[")
                #     for ftt in templateList:
                #         print("std: ", ftt.stdev)
                #         print("cov: ", numpy.sqrt(ftt.cov.diagonal()) )

            except KeyError as e:
                # "binaryprotocols_merged_500.pcap at commit f442b9d. "

                print("There seems to have been a change since the (manual) selection of the templates. "
                      "Templates have been selected from clustering of tshark segments from "
                      "binaryprotocols_maxdiff-fromOrig-500.pcap at commit 73f19ba. "
                      "You most probably need to select new cluster IDs that are suitable as templates for types from "
                      "the plots and replace the invalid IDs in this script with your new ones.")



    if args.interactive:
        IPython.embed()




