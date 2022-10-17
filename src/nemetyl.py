"""
Reference implementation for calling NEMETYL, the NEtwork MEssage TYpe identification by aLignment
with an unknown protocol.
INFOCOM 2020.

Use different segmenters to tokenize messages and align segments on the similarity of their "personality"
derived from the segments' features. The selected segmenter yields segments from each message. These segments are
analyzed by the given analysis method which is used as feature to determine their similarity.
Similar fields are then aligned to determine a score that is used as affinity value (dissimilarities) of messages
for clustering. The clusters are refined by splitting and merging on heuristics.
"""

import argparse
import csv
import os

import IPython

from nemere.alignment.alignMessages import TypeIdentificationByAlignment
from nemere.inference.segmentHandler import originalRefinements, nemetylRefinements, zerocharPCAmocoSFrefinements, \
    pcaMocoSFrefinements
from nemere.inference.templates import ClusterAutoconfException
from nemere.utils.evaluationHelpers import StartupFilecheck, CachedDistances, TitleBuilder, writePerformanceStatistics

# https://stackoverflow.com/questions/15639779/why-does-multiprocessing-use-only-a-single-core-after-i-import-numpy
os.system("taskset -p 0xffffffffffffffffff %d" % os.getpid())

debug = False

# fix the analysis method to VALUE
analysis_method = 'value'
# fix the distance method to canberra
distance_method = 'canberra'
# tokenizers to select from
tokenizers = ('4bytesfixed', 'nemesys', 'zeros')
roundingprecision = 10**8
# refinement methods
refinementMethods = [
    "none",
    "original", # WOOT2018 paper
    "nemetyl",  # INFOCOM2020 paper: ConsecutiveChars+moco+splitfirstseg
    "PCAmocoSF",  # PCA+moco+SF (v2) | applicable to zeros
    "zerocharPCAmocoSF"  # with split fixed (v2)
    ]

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Analyze fields as segments of messages and align on the similarity of their "personality" '
                    'derived from the segments\' features.')
    parser.add_argument('pcapfilename', help='Filename of the PCAP to load.')
    parser.add_argument('-i', '--interactive', help='Open ipython prompt after finishing the analysis.',
                        action="store_true")
    parser.add_argument('-l', '--layer', type=int, default=2,
                        help='Protocol layer to consider. Default is layer 2. Use --relativeToIP '
                             'to use a layer relative to IP layer.')
    parser.add_argument('-r', '--relativeToIP', default=False, action='store_true',
                        help='Consider a layer relative to the IP layer (see also --layer flag)')
    parser.add_argument('-t', '--tokenizer', help='Select the tokenizer for this analysis run.',
                        choices=tokenizers, default="nemesys")
    parser.add_argument('-e', '--littleendian', help='Toggle presumed endianness to little.', action="store_true")
    parser.add_argument('-s', '--sigma', type=float,
                        help='Only NEMESYS: sigma for noise reduction (gauss filter), default: 0.9')
    parser.add_argument('-f', '--refinement', help='Select segment refinement method.', choices=refinementMethods,
                        default=refinementMethods[-1])
    parser.add_argument('-p', '--with-plots', help='Generate plots.', action="store_true")
    args = parser.parse_args()

    filechecker = StartupFilecheck(args.pcapfilename)
    withplots = args.with_plots
    littleendian = args.littleendian == True
    tokenizer = args.tokenizer
    if littleendian:
        tokenizer += "le"

    # # # # # # # # # # # # # # # # # # # # # # # #
    # Cache/load the segmentation and segment dissimilarities
    # to/from the filesystem to improve performance of repeated analyses of the same trace.
    # # # # # # # # # # # # # # # # # # # # # # # #
    fromCache = CachedDistances(args.pcapfilename, analysis_method, args.layer, args.relativeToIP)
    # Note!  When manipulating distances calculation, deactivate caching by uncommenting the following assignment.
    # fromCache.disableCache = True
    fromCache.debug = debug
    # As we analyze a truly unknown protocol, tell CachedDistances that it should not try to use tshark to obtain
    # a dissection. The switch may be set to true for evaluating the approach with a known protocol.
    # see src/nemetyl_align-segments.py
    fromCache.dissectGroundtruth = False
    fromCache.configureTokenizer(tokenizer, args.sigma)
    if tokenizer[:7] == "nemesys":
        if args.refinement == "original":
            fromCache.configureRefinement(originalRefinements)
        elif args.refinement == "nemetyl":
            fromCache.configureRefinement(nemetylRefinements)
        elif args.refinement == "zerocharPCAmocoSF":
            fromCache.configureRefinement(zerocharPCAmocoSFrefinements, littleEndian=littleendian)
            if littleendian:
                refinement = args.refinement + "le"
        elif args.refinement is None or args.refinement == "none":
            print("No refinement selected. Performing raw segmentation.")
        else:
            print(f"The refinement {args.refinement} is not supported with this tokenizer. Abort.")
            exit(2)
    elif tokenizer[:5] == "zeros":
        if args.refinement == "PCAmocoSF":
            fromCache.configureRefinement(pcaMocoSFrefinements, littleEndian=littleendian)
            if littleendian:
                refinement = args.refinement + "le"
        elif args.refinement is None or args.refinement == "none":
            print("No refinement selected. Performing zeros segmentation with CropChars.")
        else:
            print(f"The refinement {args.refinement} is not supported with this tokenizer. Abort.")
            exit(2)
    try:
        fromCache.get()
    except ClusterAutoconfException as e:
        print("Initial clustering of the segments in the trace failed. The protocol in this trace cannot be inferred. "
              "The original exception message was:\n", e)
        exit(10)
    segmentedMessages = fromCache.segmentedMessages
    specimens, _, dc = fromCache.specimens, fromCache.comparator, fromCache.dc
    segments = dc.rawSegments
    segmentationTime, dist_calc_segmentsTime = fromCache.segmentationTime, fromCache.dist_calc_segmentsTime


    # # # # # # # # # # # # # # # # # # # # # # # #
    # Start the NEMETYL inference process
    tyl = TypeIdentificationByAlignment(dc, segmentedMessages, tokenizer, specimens.messagePool)
    # # # # # # # # # # # # # # # # # # # # # # # #


    # # # # # # # # # # # # # # # # # # # # # # # #
    # Calculate Alignment-Score and CLUSTER messages
    tyl.clusterMessages()
    # Prepare basic information about the inference run for the report
    inferenceParams = TitleBuilder(tokenizer, args.refinement, args.sigma, tyl.clusterer)
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
    # ALIGN cluster members
    tyl.alignClusterMembers()
    # # # # # # # # # # # # # # # # # # # # # # # #


    # # # # # # # # # # # # # # # # # # # # # # # #
    # SPLIT clusters based on fields without rare values
    inferenceParams.postProcess = tyl.splitClusters()
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
    # Check for cluster MERGE candidates
    inferenceParams.postProcess = tyl.mergeClusters()
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
    # Write results to report
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
    csvpath = os.path.join(filechecker.reportFullPath,
                   f"NEMETYL-symbols-{inferenceParams.plotTitle}-{filechecker.pcapstrippedname}.csv")
    if not os.path.exists(csvpath):
        print('Write alignments to {}...'.format(csvpath))
        with open(csvpath, 'w') as csvfile:
            symbolcsv = csv.writer(csvfile)
            symbolcsv.writerow(["Cluster", "Type", "frame.time_epoch", "Field", "Alignment"])
            for clunu, clusg in tyl.alignedClusters.items():
                symbolcsv.writerows(
                    [clunu, "unknown"]  # cluster label  # message type string from gt
                    + [next(seg for seg in msg if seg is not None).message.date]  # frame.time_epoch
                    + [sg.bytes.hex() if sg is not None else '' for sg in msg] for msg in clusg
                )
    else:
        print("Symbols not saved. File {} already exists.".format(csvpath))
        if not args.interactive:
            IPython.embed()
    # # # # # # # # # # # # # # # # # # # # # # # #


    if args.interactive:
        # noinspection PyUnresolvedReferences
        from tabulate import tabulate

        IPython.embed()
