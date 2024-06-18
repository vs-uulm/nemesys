"""
Reference implementation for calling NEMEPCA, the
**PCA-refined NEMESYS: NEtwork MEssage Syntax analysYS**
message format inference method with an unknown protocol.



Infer messages from PCAPs by the NEMESYS approach (BCDG-segmentation)
and write inference result to the terminal. Finally drop to an IPython console
and expose API to interact with the result.
"""

import argparse
from itertools import islice

from nemere.inference.templates import FieldTypeContext, ClusterAutoconfException
from nemere.inference.segmentHandler import isExtendedCharSeq, \
    originalRefinements, baseRefinements, pcaRefinements, pcaPcaRefinements, zeroBaseRefinements, nemetylRefinements, \
    zerocharPCAmocoSFrefinements, pcaMocoRefinements, zerocharPCAmocoRefinements, \
    entropymergeZeroCharPCAmocoSFrefinements
from nemere.inference.formatRefinement import RelocatePCA
from nemere.utils.evaluationHelpers import *
from nemere.visualization.simplePrint import SegmentPrinter

debug = False

# fix the analysis method to VALUE
analysisTitle = 'value'
# fix the distance method to canberra
distance_method = 'canberra'
# use NEMESYS segments
tokenizer = 'nemesys'
# tokenizer = '4bytesfixed' # "tshark"

refinementMethods = [
    "original", # WOOT2018 paper, merge/resplit chars
    "base",     # merge/resplit chars + moco + cumuchars
    "nemetyl",  # INFOCOM2020 paper, base + splitfirstseg
    "PCA",      # 2-pass PCA
    "PCA1",     # 1-pass PCA
    "PCAmoco",  # 2-pass PCA + moco
    "zeroPCA",  # zero+base + 1-pass PCA
    "zero",     # zero+base
    "zerocharPCAmoco",  # PCA1 + moco + crop char
    "zerocharPCAmocoSF",  # zerocharPCAmoco + split fixed (v2)
    "emzcPCAmocoSF", # zerocharPCAmocoSF + entropy based merging
    ]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Cluster NEMESYS segments of messages according to similarity.')
    parser.add_argument('pcapfilename', help='Filename of the PCAP to load.')
    parser.add_argument('-i', '--interactive',
                        help='Show interactive plot instead of writing output to file and '
                             'open ipython prompt after finishing the analysis.',
                        action="store_true")
    parser.add_argument('-s', '--sigma', type=float,
                        help='Sigma for noise reduction (gauss filter) in NEMESYS, default: 1.2', default=1.2)
    parser.add_argument('-l', '--layer', type=int, default=2,
                        help='Protocol layer relative to IP to consider. Default is 2 layers above IP '
                             '(typically the payload of a transport protocol).')
    parser.add_argument('-r', '--relativeToIP', default=False, action='store_true',
                        help='Consider a layer relative to the IP layer (see also --layer flag)')
    parser.add_argument('-f', '--refinement', help='Select segment refinement method.',
                        choices=refinementMethods, default=refinementMethods[-1])
    parser.add_argument('-e', '--littleendian', help='Toggle presumed endianness to little.',
                        action="store_true")
    args = parser.parse_args()

    filechecker = StartupFilecheck(args.pcapfilename)
    analyzerType = analyses[analysisTitle]
    analysisArgs = None
    if tokenizer == "nemesys" and args.littleendian:
        tokenizer = "nemesysle"

    # # # # # # # # # # # # # # # # # # # # # # # #
    # for PCA refinement evaluation
    collectedSubclusters = list()  # type: List[Union[RelocatePCA, FieldTypeContext]]

    # # # # # # # # # # # # # # # # # # # # # # # #
    # cache/load the DistanceCalculator to the filesystem
    #
    fromCache = CachedDistances(args.pcapfilename, analysisTitle, args.layer, args.relativeToIP)
    # Note!  When manipulating distances calculation, deactivate caching by uncommenting the following assignment.
    # fromCache.disableCache = True
    fromCache.debug = debug
    # As we analyze a truly unknown protocol, tell CachedDistances that it should not try to use tshark to obtain
    # a dissection. The switch may be set to true for evaluating the approach with a known protocol.
    # see src/nemetyl_align-segments.py
    fromCache.dissectGroundtruth = False
    if analysisArgs is not None:
        # noinspection PyArgumentList
        fromCache.configureAnalysis(*analysisArgs)
    fromCache.configureTokenizer(tokenizer, args.sigma)

    # # # # # # # # # # # # # # # # # # # # # # # #
    # some display values
    inferenceTitle = f"{tokenizer}-s{fromCache.sigma:.1f}"
    refinement = args.refinement if args.refinement is not None else "un"
    refinementTitle = inferenceTitle + "_" + refinement + '-refined'
    # # # # # # # # # # # # # # # # # # # # # # # #

    if args.refinement == "original":
        fromCache.configureRefinement(originalRefinements)
    elif args.refinement == "base":
        fromCache.configureRefinement(baseRefinements)
    elif args.refinement == "nemetyl":
        fromCache.configureRefinement(nemetylRefinements)
    elif args.refinement == "zeroPCA":
        fromCache.configureRefinement(lambda i: pcaRefinements(zeroBaseRefinements(i), littleEndian=args.littleendian,
                                      reportFolder=filechecker.reportWithTimestamp(refinementTitle),
                                      collectedSubclusters=collectedSubclusters))
    elif args.refinement == "zero":
        fromCache.configureRefinement(zeroBaseRefinements)
    elif args.refinement == "PCA1":
        fromCache.configureRefinement(pcaRefinements, littleEndian=args.littleendian,
                                      reportFolder=filechecker.reportWithTimestamp(refinementTitle),
                                      collectedSubclusters=collectedSubclusters)
    elif args.refinement == "PCA":
        fromCache.configureRefinement(pcaPcaRefinements, littleEndian=args.littleendian,
                                      reportFolder=filechecker.reportWithTimestamp(refinementTitle),
                                      collectedSubclusters=collectedSubclusters)
    elif args.refinement == "PCAmoco":
        fromCache.configureRefinement(pcaMocoRefinements, littleEndian=args.littleendian,
                                      reportFolder=filechecker.reportWithTimestamp(refinementTitle),
                                      collectedSubclusters=collectedSubclusters)
    elif args.refinement == "zerocharPCAmoco":
        fromCache.configureRefinement(zerocharPCAmocoRefinements,
                                      littleEndian=args.littleendian,
                                      reportFolder=filechecker.reportWithTimestamp(refinementTitle),
                                      collectedSubclusters=collectedSubclusters, trace=filechecker.pcapstrippedname)
    elif args.refinement == "zerocharPCAmocoSF":
        fromCache.configureRefinement(zerocharPCAmocoSFrefinements,
                                      littleEndian=args.littleendian,
                                      reportFolder=filechecker.reportWithTimestamp(refinementTitle),
                                      collectedSubclusters=collectedSubclusters, trace=filechecker.pcapstrippedname)
    elif args.refinement == "emzcPCAmocoSF":
        fromCache.configureRefinement(entropymergeZeroCharPCAmocoSFrefinements,
                                      littleEndian=args.littleendian,
                                      reportFolder=filechecker.reportWithTimestamp(refinementTitle),
                                      collectedSubclusters=collectedSubclusters, trace=filechecker.pcapstrippedname)
    else:
        print("No refinement selected. Performing raw segmentation.")

    startRefinement = time.time()
    try:
        fromCache.get()
    except ClusterAutoconfException as e:
        print("Initial clustering of the segments in the trace failed. The protocol in this trace cannot be inferred. "
              "The original exception message was:\n", e)
        exit(10)
    refinedSegmentedMessages = fromCache.segmentedMessages
    inferredSegmentedMessages = fromCache.rawSegmentedMessages
    specimens, comparator, dc = fromCache.specimens, fromCache.comparator, fromCache.dc
    segmentationTime, dist_calc_segmentsTime = fromCache.segmentationTime, fromCache.dist_calc_segmentsTime
    runtimeRefinement = time.time() - startRefinement
    print('Segmented and refined in {:.3f}s'.format(time.time() - runtimeRefinement))

    # # # # # # # # # # # # # # # # # # # # # # # #
    #
    # for msgsegs in inferredSegmentedMessages:
    #     comparator.pprint2Interleaved(msgsegs[0].message, [infs.nextOffset for infs in msgsegs])
    # exit()
    # # # # # # # # # # # # # # # # # # # # # # # #

    # # # # # # # # # # # # # # # # # # # # # # # #
    relevantSubclusters, eigenVnV, screeKnees = \
        RelocatePCA.filterRelevantClusters(
            [a.similarSegments for a in collectedSubclusters if isinstance(a, RelocatePCA)])
    # # # # # # # # # # # # # # # # # # # # # # # # #

    # output visualization of at most 100 messages on terminal and all into file
    segprint = SegmentPrinter(refinedSegmentedMessages)
    segprint.toConsole(islice(specimens.messagePool.keys(),100))
    # segprint.toTikzFile()
    # omit messages longer than 200 bytes (and not more than 100 messages)
    segprint.toTikzFile(islice((msg for msg in specimens.messagePool.keys() if len(msg.data) < 200), 100))

    # # # # # # # # # # # # # # # # # # # # # # # # #
    # write statistics ...
    scoFile = "subcluster-overview.csv"
    scoHeader = ["trace", "cluster label", "cluster size", "max segment length",
                 "interesting", "length diff", "# unique", "is char",
                 "min dissimilarity", "max dissimilarity", "mean dissimilarity"]

    #   ... for all subclusters, including the ones filtered out, for confirmation.
    for cid, sc in enumerate(pcaClusters for pcaClusters in collectedSubclusters
                             if isinstance(pcaClusters, RelocatePCA)):
        # if cid not in relevantSubclusters:
        #     print("Cluster filtered out: " + sc.similarSegments.fieldtype)
        #     for bs in sc.similarSegments.baseSegments:
        #         markSegNearMatch(bs)

        bslen = {bs.length for bs in sc.similarSegments.baseSegments}
        lendiff = max(bslen) - min(bslen)

        uniqvals = {bs.bytes for bs in sc.similarSegments.baseSegments}
        internDis = [dis for dis, idx in sc.similarSegments.distancesToMixedLength(dc)]
        ischar = sum([isExtendedCharSeq(seg.bytes)
                      for seg in sc.similarSegments.baseSegments]) > .5 * len(sc.similarSegments.baseSegments)

        fn = join(filechecker.reportWithTimestamp(refinementTitle), scoFile)
        writeheader = not exists(fn)
        with open(fn, "a") as segfile:
            segcsv = csv.writer(segfile)
            if writeheader:
                segcsv.writerow(scoHeader)
            segcsv.writerow([
                filechecker.pcapstrippedname, sc.similarSegments.fieldtype, len(sc.similarSegments.baseSegments),
                sc.similarSegments.length,
                repr(cid in relevantSubclusters), lendiff, len(uniqvals), ischar,
                min(internDis), max(internDis), numpy.mean(internDis)
            ])
    # # # # # # # # # # # # # # # # # # # # # # # # #
    segFn = "segmentclusters-" + filechecker.pcapstrippedname + ".csv"
    with open(join(filechecker.reportWithTimestamp(refinementTitle), segFn), "a") as segfile:
        segcsv = csv.writer(segfile)
        for sc in collectedSubclusters:  # type: Union[RelocatePCA, FieldTypeContext]
            if isinstance(sc, RelocatePCA):
                segcsv.writerows([
                    [],
                    ["# Cluster", sc.similarSegments.fieldtype, "Segments", len(sc.similarSegments.baseSegments)],
                    ["-"*10]*4,
                ])
                segcsv.writerows(
                    {(seg.bytes.hex(), seg.bytes) for seg in sc.similarSegments.baseSegments}
                )
            else:
                segcsv.writerows([
                    [],
                    ["# Cluster", sc.fieldtype, "Segments", len(sc.baseSegments)],
                    ["-"*10]*4,
                ])
                segcsv.writerows(
                    {(seg.bytes.hex(), seg.bytes) for seg in sc.baseSegments}
                )
    # # #

    filechecker.writeReportMetadata(fromCache.dccachefn if fromCache.isLoaded else None)

    if args.interactive:
        from tabulate import tabulate
        # globals().update(locals())
        IPython.embed()

        # interesting vars:
        # refinedSM
        # inferredSegmentedMessages



