"""
Implementation for evaluating NEMEPCA, the
**PCA-refined NEMESYS: NEtwork MEssage Syntax analysYS**
message format inference method with a KNOWN protocol.



Infer messages from PCAPs by the NEMESYS approach (BCDG-segmentation)
and write inference result to the terminal. Finally drop to an IPython console
and expose API to interact with the result.
"""

import argparse, IPython
from typing import Any
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.colors as colors

from nemere.inference.templates import FieldTypeTemplate, FieldTypeContext, ClusterAutoconfException, \
    DBSCANsegmentClusterer
from nemere.inference.segmentHandler import symbolsFromSegments, wobbleSegmentInMessage, isExtendedCharSeq, \
    originalRefinements, baseRefinements, pcaRefinements, pcaPcaRefinements, zeroBaseRefinements, nemetylRefinements, \
    zerocharPCAmocoSFrefinements, pcaMocoRefinements, zerocharPCAmocoRefinements, \
    entropymergeZeroCharPCAmocoSFrefinements
from nemere.inference.formatRefinement import RelocatePCA, entropyOfXor, entropyOfBytes, EntropyMerger
from nemere.utils import reportWriter
from nemere.validation.dissectorMatcher import DissectorMatcher
from nemere.visualization.distancesPlotter import DistancesPlotter
from nemere.utils.evaluationHelpers import *
from nemere.visualization.simplePrint import markSegNearMatch, inferred4segment, ComparingPrinter

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


def inferredFEs4segment(segment: MessageSegment, segmentedMessages: List[Sequence[MessageSegment]]) -> List[int]:
    """
    Determine all the field ends from segmentedMessages in the message of the one given segment.

    :param segment: The input segment.
    :param segmentedMessages: List of segmented messages to search in.
    :return: List of boundary offsets.
    """
    return [infs.nextOffset for infs in inferred4segment(segment, segmentedMessages)]

def plotComponentAnalysis(pcaRelocator: RelocatePCA, eigValnVec: Tuple[numpy.ndarray], relocateFromEnd: List[int]):
    # # # # # # # # # # # # # # # # # # # # # # # #
    # Count true boundaries for the segments' relative positions
    trueOffsets = list()
    for bs in pcaRelocator.similarSegments.baseSegments:
        fe = comparator.fieldEndsPerMessage(bs.analyzer.message)
        offs, nxtOffs = pcaRelocator.similarSegments.paddedPosition(bs)
        trueOffsets.extend(o - offs for o in fe if offs <= o <= nxtOffs)
    truOffCnt = Counter(trueOffsets)

    # # # # # # # # # # # # # # # # # # # # # # # #
    # plotting stuff
    compFig = plt.figure()
    compAx = compFig.add_subplot(1, 1, 1)
    try:
        compAx.axhline(0, color="black", linewidth=1, alpha=.4)
        if not any(pcaRelocator.principalComponents):
            # if there is no principal component, plot all vectors for investigation
            principalComponents = numpy.array([True] * eigValnVec[0].shape[0])

            eigVsorted = list(reversed(sorted([(val, vec) for val, vec in zip(eigValnVec[0], eigValnVec[1].T)],
                                              key=lambda x: x[0])))
            eigVecS = numpy.array([colVec[1] for colVec in eigVsorted]).T

            noPrinComps = True
        else:
            eigVsorted = list(reversed(sorted([(val, vec) for val, vec in zip(
                eigValnVec[0][pcaRelocator.principalComponents], pcaRelocator.contribution.T)],
                                              key=lambda x: x[0])))
            eigVecS = numpy.array([colVec[1] for colVec in eigVsorted]).T

            # lines = compAx.plot(pcaRelocator.contribution)  # type: List[plt.Line2D]
            # for i, (l, ev) in enumerate(zip(lines, eigValnVec[0][pcaRelocator.principalComponents])):
            #     l.set_label(repr(i + 1) + ", $\lambda=$" + "{:.1f}".format(ev))
            noPrinComps = False
        # IPython.embed()
        lines = compAx.plot(eigVecS)  # type: List[plt.Line2D]
        for i, (l, ev) in enumerate(zip(lines, [eigVal[0] for eigVal in eigVsorted])):
            l.set_label(repr(i + 1) + ", $\lambda=$" + "{:.1f}".format(ev))

        # mark correct boundaries with the intensity (alpha) of the relative offset
        for offs, cnt in truOffCnt.most_common():
            compAx.axvline(offs - 0.5, color="black", linewidth=.5, alpha=cnt / max(truOffCnt.values()))
        # mark reallocations
        for rfe in relocateFromEnd:
            compAx.scatter(rfe - 0.5, 0, c="red")
        compAx.xaxis.set_major_locator(ticker.MultipleLocator(1))
        compAx.legend()
        if noPrinComps:
            compFig.suptitle("All Eigenvectors (NO Eigenvalue is > {:.0f}) - Cluster ".format(
                pcaRelocator.screeThresh) + pcaRelocator.similarSegments.fieldtype)
        else:
            compFig.suptitle("Principal Eigenvectors (with Eigenvalue > {:.0f}) - Cluster ".format(
                pcaRelocator.screeThresh) + pcaRelocator.similarSegments.fieldtype)
        compFig.savefig(join(filechecker.reportFullPath, "component-analysis_{}.pdf".format(pcaRelocator.similarSegments.fieldtype)))
    except Exception as e:
        print("\n\n### {} ###\n\n".format(e))
        IPython.embed()
    plt.close(compFig)

    # # # # # # # # # # # # # # # # # # # # # # # #
    # plot heatmaps of covariance matrices "cov-test-heatmap"
    heatFig = plt.figure()
    heatAx = heatFig.add_subplot(1, 1, 1)

    # make the colormap symmetric around zero
    vmax = max(abs(pcaRelocator.similarSegments.cov.min()), abs(pcaRelocator.similarSegments.cov.max()))
    heatMap = heatAx.imshow(pcaRelocator.similarSegments.cov, cmap='PuOr', norm=colors.Normalize(-vmax, vmax))
    # logarithmic scale hides the interesting variance differences. For reference on how to do it:
    #   norm=colors.SymLogNorm(100, vmin=-vmax, vmax=vmax)

    # mark correct boundaries with the intensity (alpha) of the relative offset
    for offs, cnt in truOffCnt.most_common():
        heatAx.axvline(offs - 0.5, color="white", linewidth=1.5,
                       alpha=cnt / max(truOffCnt.values()))
        heatAx.axhline(offs - 0.5, color="white", linewidth=1.5,
                       alpha=cnt / max(truOffCnt.values()))
    heatAx.xaxis.set_major_locator(ticker.MultipleLocator(1))
    heatAx.yaxis.set_major_locator(ticker.MultipleLocator(1))
    heatFig.colorbar(heatMap)
    heatFig.savefig(join(filechecker.reportFullPath, "cov-test-heatmap_cluster{}.pdf".format(pcaRelocator.similarSegments.fieldtype)))
    plt.close(heatFig)

def plotRecursiveComponentAnalysis(relocateFromEnd: Union[Dict, Any]):
    """
    Support for subclustering, where the method returns a dict of relevant information about each subcluster

    :param relocateFromEnd:
    :return: True if recursion occured, False otherwise
    """
    if isinstance(relocateFromEnd, dict):
        # print("\n\nSubclustering result in relocate...\n\n")
        # IPython.embed()
        for sc in relocateFromEnd.values():
            if not plotRecursiveComponentAnalysis(sc[0]):
                if withPlots:
                    plotComponentAnalysis(sc[3], sc[1], sc[0])
        return True
    else:
        return False

def factorAnalysis(fTypeContexts: Iterable[FieldTypeContext]):
    """
    Evaluation of Factor Analysis, see: https://www.datacamp.com/community/tutorials/introduction-factor-analysis
            ==> Factor Analysis not feasible for our use case

    :param fTypeContexts:
    :return:
    """
    from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity, calculate_kmo
    for ftc in fTypeContexts:
        print("# # Cluster", ftc.fieldtype)

        # test sphericity (Bartlett’s test)
        paddedValues = ftc.paddedValues()
        nonConstPVals = paddedValues[:,ftc.stdev > 0]
        if nonConstPVals.shape[1] > 1:
            chi_square_value, p_value = calculate_bartlett_sphericity(nonConstPVals)
            print("p-Value (Bartlett’s test): {:.6f} "
                  "(should be very close to zero to pass the significance check)".format(p_value))
            # Kaiser-Meyer-Olkin (KMO) Test
            kmo_all, kmo_model = calculate_kmo(nonConstPVals)
            print("KMO model: {:.3f} (should be at least above .6, better above .8)".format(kmo_model))
        else:
            print("Bartlett’s and KMO tests impossible: only one factor is non-constant.")

        import sklearn
        fa = sklearn.decomposition.FactorAnalysis(2)
        fa.fit(ftc.paddedValues())
        plt.plot(abs(fa.components_.T))
        plt.show()
        plt.imshow(abs(fa.components_.T))
        plt.show()
    # # # # # # # # # # # # # # # # # # # # # # # #

def wobble(interestingClusters):
    for cluster in interestingClusters:  # type: RelocatePCA
        print("Cluster", cluster.similarSegments.fieldtype)
        # # # # # # # # # # # # # # # # # # # # # # # #
        # wobble segments in one cluster
        # start test with only one cluster: this is thought to be ntp.c1

        # templates must be resolved into segments before wobbling
        wobbles = list()
        for element in interestingClusters:
            if isinstance(element, Template):
                segments = element.baseSegments
            else:
                segments = [element]
            for seg in segments:
                wobbles.extend(wobbleSegmentInMessage(seg))
        wobDC = DelegatingDC(wobbles)
        wobClusterer = DBSCANsegmentClusterer(wobDC)
        # wobClusterer.autoconfigureEvaluation(join(filechecker.reportFullPath, "wobble-epsautoconfig.pdf"))
        wobClusterer.eps = wobClusterer.eps * 0.8
        wobNoise, *wobClusters = wobClusterer.clusterSimilarSegments(False)

        from nemere.utils.baseAlgorithms import tril
        print("Wobbled cluster distances:")
        print(tabulate([(clunu, wobDC.distancesSubset(wobclu).max(), tril(wobDC.distancesSubset(wobclu)).mean())
                        for clunu, wobclu in enumerate(wobClusters)],
                       headers=["Cluster", "max", "mean"]))
        # nothing very interesting in here. Wobble-clusters of true boundaries are not denser or smaller than others.

        for clunu, wobclu in enumerate(wobClusters):
            print("\n# # Wobbled Cluster", clunu)
            for seg in wobclu:
                markSegNearMatch(seg, refinedSM, comparator)

        # interestingWobbles = [0, 1]  # in ntp.c1
        interestingWobbles = [0, 1]  # in ntp.c0
        wc = dict()
        wceig = dict()
        for iW in interestingWobbles:
            wc[iW] = FieldTypeTemplate(wobClusters[iW])
            wceig[iW] = numpy.linalg.eigh(wc[iW].cov)
            print("Eigenvalues wc {}".format(iW))
            print(tabulate([wceig[iW][0]]))

        if withPlots:
            # # "component-analysis"
            import matplotlib.pyplot as plt
            # import matplotlib.colors as colors
            #
            # the principal components (i. e. with Eigenvalue > 1) of the covariance matrix need to be towards the end
            # for floats. The component is near 1 or -1 in the Eigenvector of the respective Eigenvalue.
            # TODO Does this make sense mathematically?
            fig, ax = plt.subplots(1, len(interestingWobbles))  # type: plt.Figure, Sequence[plt.Axes]
            for axI, iW in enumerate(interestingWobbles):
                principalComponents = wceig[iW][0] > 1
                lines = ax[axI].plot(wceig[iW][1][:, principalComponents])  # type: List[plt.Line2D]
                for i, l in enumerate(lines):
                    l.set_label(repr(i + 1))
                ax[axI].legend()
                ax[axI].set_title("Wobbled Cluster " + repr(iW))
            fig.suptitle("Principal (with Eigenvalue > 1) Eigenvectors - Cluster " + repr(cluster.similarSegments.fieldtype))
            plt.savefig(join(filechecker.reportFullPath, "component-analysis_cluster{}.pdf".format(cluster.similarSegments.fieldtype)))
            plt.close('all')

        if withPlots:
            # TODO plot heatmaps for covariance matrices of all wobclusters and clusters: mark correct boundaries
            # plot heatmaps of cov matrices "cov-test-heatmap"
            fig, ax = plt.subplots(1, len(interestingWobbles))  # type: plt.Figure, Sequence[plt.Axes]
            # im = list()
            for axI, iW in enumerate(interestingWobbles):
                # logcolors = colors.SymLogNorm(min(wc00.cov.min(), wc01.cov.min()), max(wc00.cov.max(), wc01.cov.max()))
                ax[axI].imshow(wc[iW].cov)  # , norm=logcolors)  im[axI]
                # ax[axI].pcolormesh()
            # fig.colorbar(im[0], ax=ax[1], extend='max')
            plt.savefig(join(filechecker.reportFullPath, "cov-test-heatmap_cluster{}.pdf".format(cluster.similarSegments.fieldtype)))
            plt.close('all')

        if withPlots:
            print("Plot distances...")
            sdp = DistancesPlotter(specimens,
                                   'distances-wobble-' + "nemesys-segments_DBSCAN-eps{:0.3f}-ms{:d}-cluster{}".format(
                                       wobClusterer.eps, int(wobClusterer.min_samples), cluster.similarSegments.fieldtype), False)

            clustermask = {segid: cluN for cluN, segL in enumerate(wobClusters) for segid in wobDC.segments2index(segL)}
            clustermask.update({segid: "Noise" for segid in wobDC.segments2index(wobNoise)})
            labels = numpy.array([clustermask[segid] for segid in range(len(wobDC.segments))])
            # plotManifoldDistances(dc.segments, dc.distanceMatrix, labels)
            sdp.plotSegmentDistances(wobDC, labels)
            sdp.writeOrShowFigure()
            del sdp


def commonBoundsIrrelevant():
    # # # # # # # # # # # # # # # # # # # # # # # #
    # Common Bounds refinement for NON-interestingClusters and NON-relevantSubclusters
    print("# "*20)
    for noninCluster in (sc for cid, sc in enumerate(collectedSubclusters) if cid not in relevantSubclusters):
        baseOffs = {bs: noninCluster.similarSegments.baseOffset(bs) for bs in noninCluster.similarSegments.baseSegments}
        fromEnd = {bs: noninCluster.similarSegments.maxLen - noninCluster.similarSegments.baseOffset(bs) - bs.length
                   for bs in noninCluster.similarSegments.baseSegments}
        if len(set(baseOffs.values())) > 1 or len(set(fromEnd.values())) > 1:
            print("#", noninCluster.similarSegments.fieldtype, "# "*10)
            for bs in noninCluster.similarSegments.baseSegments:
                markSegNearMatch(bs, refinedSM, comparator)
            print(tabulate(noninCluster.similarSegments.paddedValues(), showindex=
                           [(baseOffs[bs], fromEnd[bs]) for bs in noninCluster.similarSegments.baseSegments]))
    print("# " * 20)
    # Considering the probable impact, not worth the effort.
    # # # # # # # # # # # # # # # # # # # # # # # #


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Cluster NEMESYS segments of messages according to similarity.')
    parser.add_argument('pcapfilename', help='Filename of the PCAP to load.')
    parser.add_argument('-i', '--interactive', help='Show interactive plot instead of writing output to file and '
                                                    'open ipython prompt after finishing the analysis.',
                        action="store_true")
    parser.add_argument('-s', '--sigma', type=float, help='Sigma for noise reduction (gauss filter) in NEMESYS,'
                                                          'default: 0.9')
    parser.add_argument('-p', '--with-plots',
                        help='Generate plots of true field types and their distances.',
                        action="store_true")
    parser.add_argument('-l', '--layer', type=int, default=2,
                        help='Protocol layer relative to IP to consider. Default is 2 layers above IP '
                             '(typically the payload of a transport protocol).')
    parser.add_argument('-r', '--relativeToIP', default=False, action='store_true',
                        help='Consider a layer relative to the IP layer (see also --layer flag)')
    parser.add_argument('-f', '--refinement', help='Select segment refinement method.', choices=refinementMethods)
    parser.add_argument('-e', '--littleendian', help='Toggle presumed endianness to little.', action="store_true")
    args = parser.parse_args()

    filechecker = StartupFilecheck(args.pcapfilename)
    withPlots = args.with_plots
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
        fromCache.configureRefinement(pcaRefinements, forwardComparator=True, littleEndian=args.littleendian,
                                      reportFolder=filechecker.reportWithTimestamp(refinementTitle),
                                      collectedSubclusters=collectedSubclusters)
    elif args.refinement == "PCA":
        fromCache.configureRefinement(pcaPcaRefinements, forwardComparator=True, littleEndian=args.littleendian,
                                      reportFolder=filechecker.reportWithTimestamp(refinementTitle),
                                      collectedSubclusters=collectedSubclusters)
    elif args.refinement == "PCAmoco":
        fromCache.configureRefinement(pcaMocoRefinements, forwardComparator=True, littleEndian=args.littleendian,
                                      reportFolder=filechecker.reportWithTimestamp(refinementTitle),
                                      collectedSubclusters=collectedSubclusters)
    elif args.refinement == "zerocharPCAmoco":
        fromCache.configureRefinement(zerocharPCAmocoRefinements, forwardComparator=True,
                                      littleEndian=args.littleendian,
                                      reportFolder=filechecker.reportWithTimestamp(refinementTitle),
                                      collectedSubclusters=collectedSubclusters, trace=filechecker.pcapstrippedname)
    elif args.refinement == "zerocharPCAmocoSF":
        fromCache.configureRefinement(zerocharPCAmocoSFrefinements, forwardComparator=True,
                                      littleEndian=args.littleendian,
                                      reportFolder=filechecker.reportWithTimestamp(refinementTitle),
                                      collectedSubclusters=collectedSubclusters, trace=filechecker.pcapstrippedname)
    elif args.refinement == "emzcPCAmocoSF":
        fromCache.configureRefinement(entropymergeZeroCharPCAmocoSFrefinements, forwardComparator=True,
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
    refinedSM = fromCache.segmentedMessages
    inferredSegmentedMessages = fromCache.rawSegmentedMessages
    specimens, comparator, dc = fromCache.specimens, fromCache.comparator, fromCache.dc
    segmentationTime, dist_calc_segmentsTime = fromCache.segmentationTime, fromCache.dist_calc_segmentsTime
    runtimeRefinement = time.time() - startRefinement
    # # # # # # # # # # # # # # # # # # # # # # # #
    #
    # for msgsegs in inferredSegmentedMessages:
    #     comparator.pprint2Interleaved(msgsegs[0].message, [infs.nextOffset for infs in msgsegs])
    # exit()
    # # # # # # # # # # # # # # # # # # # # # # # #
    trueSegmentedMessages = {msgseg[0].message: msgseg
                         for msgseg in annotateFieldTypes(analyzerType, analysisArgs, comparator)
                         }  # type: Dict[AbstractMessage, Tuple[TypedSegment]]
    # tabuSeqOfSeg(trueSegmentedMessages)
    # print(trueSegmentedMessages.values())

    cprinter = ComparingPrinter(comparator, refinedSM)
    cprinter.toConsole()

    # # # # # # # # # # # # # # # # # # # # # # # # #
    # # cluster segments to determine field types on commonality
    # try:
    #     clusterer = DBSCANsegmentClusterer(dc, S=kneedleSensitivity)
    # except ClusterAutoconfException as e:
    #     print("Initial clustering of the segments in the trace failed. The protocol in this trace cannot be inferred. "
    #           "The original exception message was:\n", e)
    #     exit(10)
    #
    # # noinspection PyUnboundLocalVariable
    # noise, *clusters = clusterer.clusterSimilarSegments(False)
    # # noise: List[MessageSegment]
    # # clusters: List[List[MessageSegment]]
    #
    # # extract "large" templates from noise that should rather be its own cluster
    # for idx, seg in reversed(list(enumerate(noise.copy()))):  # type: int, MessageSegment
    #     freqThresh = log(len(dc.rawSegments))
    #     if isinstance(seg, Template):
    #         if len(seg.baseSegments) > freqThresh:
    #             clusters.append(noise.pop(idx).baseSegments)
    #
    # print("{} clusters generated from {} distinct segments".format(len(clusters), len(dc.segments)))
    # # # # # # # # # # # # # # # # # # # # # # # # #


    # # # # # # # # # # # # # # # # # # # # # # # # #
    # fTypeTemplates = list()
    # fTypeContext = list()
    # for cLabel, segments in enumerate(clusters):
    #     # generate FieldTypeTemplates (padded nans)
    #     ftype = FieldTypeTemplate(segments)
    #     ftype.fieldtype = "tf{:02d}".format(cLabel)
    #     fTypeTemplates.append(ftype)
    #     with open(join(filechecker.reportWithTimestamp(refinementTitle), "segmentclusters-" + filechecker.pcapstrippedname + ".csv"), "a") as segfile:
    #         segcsv = csv.writer(segfile)
    #         segcsv.writerows([
    #             [],
    #             ["# Cluster", cLabel, "Segments", len(segments)],
    #             ["-"*10]*4,
    #         ])
    #         segcsv.writerows(
    #             {(seg.bytes.hex(), seg.bytes) for seg in segments}
    #         )
    #
    #     # generate FieldTypeContexts (padded values)
    #     resolvedSegments = list()
    #     for seg in segments:
    #         if isinstance(seg, Template):
    #             resolvedSegments.extend(seg.baseSegments)
    #         else:
    #             resolvedSegments.append(seg)
    #     fcontext = FieldTypeContext(resolvedSegments)
    #     fcontext.fieldtype = ftype.fieldtype
    #     fTypeContext.append(fcontext)
    # # # # # # # # # # # # # # # # # # # # # # # # #

    # interestingClusters = [1]  # this is thought to be ntp.c1 (4-byte floats)
    # interestingClusters = [0]  # this is thought to be ntp.c0
    # interestingClusters = range(len(clusters))  # all
    # interestingClusters = [cid for cid, clu in enumerate(clusters)
    #                        if any([isExtendedCharSeq(seg.bytes) for seg in clu])] # all not-chars
    # interestingClusters, eigenVnV, screeKnees = RelocatePCA.filterRelevantClusters(fTypeContext)
    # interestingClusters = RelocatePCA.filterForSubclustering(fTypeContext)

    # # # # # # # # # # # # # # # # # # # # # # # # #
    # # ==> Factor Analysis not feasible for our use case
    # factorAnalysis(fTypeContext[iC] for iC in interestingClusters)
    # # # # # # # # # # # # # # # # # # # # # # # # #


    # # # # # # # # # # # # # # # # # # # # # # # #
    relevantSubclusters, eigenVnV, screeKnees = \
        RelocatePCA.filterRelevantClusters(
            [a.similarSegments for a in collectedSubclusters if isinstance(a, RelocatePCA)])
    # # select one tf
    # tf02 = next(c for c in collectedSubclusters if c.similarSegments.fieldtype == "tf02")
    # # # # # # # # # # # # # # # # # # # # # # # # #




    # # # # # # # # # # # # # # # # # # # # # # # # # #
    # # single segment refinement component calls
    # #
    # # collect new bounds for each segment and group them by message
    # newBounds = dict()  # type: Dict[AbstractMessage, Dict[MessageSegment, List[int]]]
    # for cid, sc in enumerate(collectedSubclusters):  # type: int, RelocatePCA
    #     if cid in relevantSubclusters:
    #         clusterBounds = sc.relocateBoundaries(dc, kneedleSensitivity, comparator, filechecker.reportWithTimestamp(refinementTitle))
    #         for segment, bounds in clusterBounds.items():
    #             if segment.message not in newBounds:
    #                 newBounds[segment.message] = dict()
    #             elif segment in newBounds[segment.message]:
    #                 print("\nSame segment was refined (PCA) multiple times. Needs resolving. Segment is:\n", segment)
    #                 print()
    #                 IPython.embed()
    #             newBounds[segment.message][segment] = bounds
    #
    # compareBounds = {m: {s: b.copy() for s, b in sb.items()} for m, sb in newBounds.items()}
    #
    # # remove from newBounds, in place
    # RelocatePCA.removeSuperfluousBounds(newBounds)
    #
    # # apply refinement to segmented messages
    # refinedSegmentedMessages = RelocatePCA.refineSegmentedMessages(inferredSegmentedMessages, newBounds)

    # # make some development output about refinedSegmentedMessages
    # for msgsegs in refinedSM:
    #     infms = next(ms for ms in inferredSegmentedMessages if ms[0].message == msgsegs[0].message)
    #     if msgsegs != infms:
    #         # comparator.pprint2Interleaved(infms[0].message, [infs.nextOffset for infs in infms])
    #         comparator.pprint2Interleaved(msgsegs[0].message, [infs.nextOffset for infs in msgsegs])

    # #         newMsgBounds = sorted(chain(*newBounds[msgsegs[0].message].values()))
    # #         print(newMsgBounds)
    # #         missedBound = [nmb for nmb in newMsgBounds if nmb not in (ref.offset for ref in msgsegs)]
    # #         if len(missedBound) > 0:
    # #             print("missedBounds", missedBound)
    # #
    # #         segseq = [ref1 for ref1, ref2 in zip(msgsegs[:-1], msgsegs[1:]) if ref1.nextOffset != ref2.offset]
    # #         if len(segseq) > 0:
    # #             print("Segment sequence error!\n", segseq)
    # #         shoseg = [ref for ref in msgsegs if ref.offset > 0 and ref.length < 2 and ref not in infms]
    # #         if len(shoseg) > 0:
    # #             print("Short segments:\n", shoseg)
    # #         print()
    # # # # # # # # # # # # # # # # # # # # # # # # #
    #
    # print("\n# # # # # # # # # # # # # # # # # # # # # # # #")
    # if [[rsm.nextOffset for rsm in msg] for msg in refinedSM] \
    #         == [[rsm.nextOffset for rsm in msg] for msg in refinedSegmentedMessages]:
    #     print("Static wrapper function for PCA is valid!")
    # else:
    #     print("Static wrapper function for PCA is invalid!")
    # print("# # # # # # # # # # # # # # # # # # # # # # # #\n")
    # # # # # # # # # # # # # # # # # # # # # # # # #



    # # show position of each segment individually.
    # for seg in dc.segments:
    #     markSegNearMatch(seg)

    # comparator.pprint2Interleaved(dc.segments[6].message, inferredFEs4segment(dc.segments[6]),
    #                               (dc.segments[6].offset-2, dc.segments[6].nextOffset+1))











    # # # # # # # # # # # # # # # # # # # # # # # #
    if withPlots:
        print("Plot component analyses...")
        for cid, sc in enumerate(pcaClusters for pcaClusters in collectedSubclusters
                                 if isinstance(pcaClusters, RelocatePCA)):  # type: int, RelocatePCA
            if cid in relevantSubclusters:
                # print(sc.similarSegments.fieldtype, "*" if cid in relevantSubclusters else "")
                relocate = sc.relocateOffsets(filechecker.reportWithTimestamp(refinementTitle),
                                              filechecker.pcapstrippedname, comparator)
                plotComponentAnalysis(sc,
                                      eigenVnV[cid] if cid in eigenVnV else numpy.linalg.eigh(sc.similarSegments.cov),
                                      relocate)
                if debug:
                    print("Segments in subcluster", cid)
                    for bs in sc.similarSegments.baseSegments:
                        markSegNearMatch(bs, refinedSM, comparator)
        # # # # # # # # # # # # # # # # # # # # # # # # #
        # # deactivated due to performance impact and little use
        # print("Plot distances...")
        # sdp = DistancesPlotter(specimens,
        #                        'distances-' + "nemesys-segments_DBSCAN-eps{:0.3f}-ms{:d}".format(
        #                            clusterer.eps, int(clusterer.min_samples)), False)
        # clustermask = {segid: cluN for cluN, segL in enumerate(clusters) for segid in dc.segments2index(segL)}
        # clustermask.update({segid: "Noise" for segid in dc.segments2index(noise)})
        # labels = numpy.array([clustermask[segid] for segid in range(len(dc.segments))])
        # # plotManifoldDistances(dc.segments, dc.distanceMatrix, labels)
        # sdp.plotSegmentDistances(dc, labels)
        # sdp.writeOrShowFigure()
        # del sdp
    # # # # # # # # # # # # # # # # # # # # # # # #





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


    # # # # # # # # # # # # # # # # # # # # # # # #
    # Write FMS statistics
    print("Prepare writing of FMS statistics...")

    if inferredSegmentedMessages is not None:
        # # # # # # # # # # # # # # # # # # # # # # # #
        # Determine the amount of off-by-one errors in original segments
        message2quality = DissectorMatcher.symbolListFMS(comparator, symbolsFromSegments(inferredSegmentedMessages))
        reportWriter.writeReport(message2quality, runtimeRefinement, comparator, inferenceTitle,
                                 filechecker.reportWithTimestamp(inferenceTitle))
        exactcount, offbyonecount, offbymorecount = reportWriter.countMatches(message2quality.values())
        minmeanmax = reportWriter.getMinMeanMaxFMS([round(q.score, 3) for q in message2quality.values()])
        print("Format Match Scores")
        print("  (min, mean, max): ", *minmeanmax)
        print("near matches")
        print("  off-by-one:", offbyonecount)
        print("  off-by-more:", offbymorecount)
        print("exact matches:", exactcount)

    # # # # # # # # # # # # # # # # # # # # # # # #
    # Determine the amount of off-by-one errors in refined segmentation
    message2quality = DissectorMatcher.symbolListFMS(comparator, symbolsFromSegments(refinedSM))
    reportWriter.writeReport(message2quality, -1.0, comparator,
                             refinementTitle,
                             filechecker.reportWithTimestamp(refinementTitle))
    exactcount, offbyonecount, offbymorecount = reportWriter.countMatches(message2quality.values())
    minmeanmax = reportWriter.getMinMeanMaxFMS([round(q.score, 3) for q in message2quality.values()])
    print("Format Match Scores (refined)")
    print("  (min, mean, max): ", *minmeanmax)
    print("near matches")
    print("  off-by-one:", offbyonecount)
    print("  off-by-more:", offbymorecount)
    print("exact matches:", exactcount)
    # # # # # # # # # # # # # # # # # # # # # # # #
    filechecker.writeReportMetadata(fromCache.dccachefn if fromCache.isLoaded else None)


    trueFieldNames = TrueFieldNameOverlays(trueSegmentedMessages, refinedSM, comparator, 2)
    trueDataTypes = TrueDataTypeOverlays(trueSegmentedMessages, refinedSM, comparator, 2)

    # check the amount of (falsely) inferred boundaries in the scope of each true field.
    print("\n" * 2)
    print(trueFieldNames)
    # to write a csv
    trueFieldNames.toCSV(filechecker.reportWithTimestamp(refinementTitle))
    print("\n" * 2)
    print(trueDataTypes)
    # to write a csv
    trueDataTypes.toCSV(filechecker.reportWithTimestamp(refinementTitle))
    # # print the "best/worst"
    # print("\n# FieldNames underspecific")
    # trueFieldNames.printSegmentContexts(trueFieldNames.filterUnderspecific())
    # print("\n# FieldNames overspecific")
    # trueFieldNames.printSegmentContexts(trueFieldNames.filterOverspecific(2))
    # print("\n\n# DataTypes underspecific")
    # trueDataTypes.printSegmentContexts(trueDataTypes.filterUnderspecific())
    # print("\n# DataTypes overspecific")
    # trueDataTypes.printSegmentContexts(trueDataTypes.filterOverspecific(2))



    if args.interactive:
        from tabulate import tabulate
        # globals().update(locals())
        IPython.embed()

        # interesting vars:
        # trueSegmentedMessages
        # refinedSM
        # inferredSegmentedMessages



