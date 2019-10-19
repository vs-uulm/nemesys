"""
segment messages by NEMESYS and cluster
"""

import argparse, IPython
from os.path import isfile, basename, join, splitext, exists
from os import makedirs
from math import log
from typing import Union, Any
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.colors as colors
from collections import Counter

from inference.templates import DBSCANsegmentClusterer, FieldTypeTemplate, Template, TypedTemplate, FieldTypeContext, \
    ClusterAutoconfException
from inference.segmentHandler import symbolsFromSegments, wobbleSegmentInMessage, isExtendedCharSeq
from inference.formatRefinement import RelocatePCA
from validation import reportWriter
from visualization.distancesPlotter import DistancesPlotter
from visualization.simplePrint import *
from utils.evaluationHelpers import *
from validation.dissectorMatcher import FormatMatchScore


debug = False

# fix the analysis method to VALUE
analysisTitle = 'value'
# fix the distance method to canberra
distance_method = 'canberra'
# use NEMESYS segments
tokenizer = 'nemesys'



def markSegNearMatch(segment: Union[MessageSegment, Template]):
    """
    Print messages with the given segment in each message marked.
    Supports Templates by resolving them to their base segments.

    :param segment: list of segments that should be printed, i. e.,
        marked within the print of the message it is originated from.
    """
    if isinstance(segment, Template):
        segs = segment.baseSegments
    else:
        segs = [segment]

    print()  # one blank line for visual structure
    for seg in segs:
        inf4seg = inferred4segment(seg)
        comparator.pprint2Interleaved(seg.message, [infs.nextOffset for infs in inf4seg],
                                      (seg.offset, seg.nextOffset))

    # # a simpler approach
    # markSegmentInMessage(segment)

    # # get field number of next true field
    # tsm = trueSegmentedMessages[segment.message]  # type: List[MessageSegment]
    # fsnum, offset = 0, 0
    # while offset < segment.offset:
    #     offset += tsm[fsnum].offset
    #     fsnum += 1
    # markSegmentInMessage(trueSegmentedMessages[segment.message][fsnum])

    # # limit to immediate segment context
    # posSegMatch = None  # first segment that starts at or after the recognized field
    # for sid, seg in enumerate(trueSegmentedMessages[segment.message]):
    #     if seg.offset > segment.offset:
    #         posSegMatch = sid
    #         break
    # posSegEnd = None  # last segment that ends after the recognized field
    # for sid, seg in enumerate(trueSegmentedMessages[segment.message]):
    #     if seg.nextOffset > segment.nextOffset:
    #         posSegEnd = sid
    #         break
    # if posSegMatch is not None:
    #     contextStart = max(posSegMatch - 2, 0)
    #     if posSegEnd is None:
    #         posSegEnd = posSegMatch
    #     contextEnd = min(posSegEnd + 1, len(trueSegmentedMessages))

def inferred4segment(segment: MessageSegment) -> Sequence[MessageSegment]:
    """
    :param segment: The input segment.
    :return: All inferred segments for the message which the input segment is from.
    """
    return next(msegs for msegs in inferredSegmentedMessages if msegs[0].message == segment.message)

def inferredFEs4segment(segment: MessageSegment) -> List[int]:
    return [infs.nextOffset for infs in inferred4segment(segment)]

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
        compFig.savefig(join(reportFolder, "component-analysis_{}.pdf".format(pcaRelocator.similarSegments.fieldtype)))
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
    heatFig.savefig(join(reportFolder, "cov-test-heatmap_cluster{}.pdf".format(pcaRelocator.similarSegments.fieldtype)))
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




kneedleSensitivity=12.0




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
    args = parser.parse_args()

    if not isfile(args.pcapfilename):
        print('File not found: ' + args.pcapfilename)
        exit(1)
    pcapbasename = basename(args.pcapfilename)
    withPlots = args.with_plots

    analyzerType = analyses[analysisTitle]
    analysisArgs = None

    # # # # # # # # # # # # # # # # # # # # # # # #
    # cache/load the DistanceCalculator to the filesystem
    #
    # noinspection PyUnboundLocalVariable
    specimens, comparator, inferredSegmentedMessages, dc, segmentationTime, dist_calc_segmentsTime = cacheAndLoadDC(
        args.pcapfilename, analysisTitle, tokenizer, debug, analyzerType, analysisArgs, args.sigma, True#, True
    )  # Note!  When manipulating distances, deactivate caching by adding "True".
    # chainedSegments = dc.rawSegments
    # # # # # # # # # # # # # # # # # # # # # # # #
    trueSegmentedMessages = {msgseg[0].message: msgseg
                         for msgseg in annotateFieldTypes(analyzerType, analysisArgs, comparator)
                         }
    # tabuSeqOfSeg(trueSegmentedMessages)
    # print(trueSegmentedMessages.values())

    reportFolder = join(reportFolder, splitext(pcapbasename)[0])
    makedirs(reportFolder)

    # # # # # # # # # # # # # # # # # # # # # # # #
    # Determine the amount of off-by-one errors
    from validation.dissectorMatcher import DissectorMatcher
    message2quality = DissectorMatcher.symbolListFMS(comparator, symbolsFromSegments(inferredSegmentedMessages))
    exactcount = 0
    offbyonecount = 0
    offbymorecount = 0
    for fms in message2quality.values():  # type: FormatMatchScore
        exactcount += fms.exactCount
        offbyonecount += sum(1 for truf, inff in fms.nearMatches.items() if abs(truf - inff) == 1)
        offbymorecount += sum(1 for truf, inff in fms.nearMatches.items() if abs(truf - inff) > 1)
    minmeanmax = reportWriter.getMinMeanMaxFMS([round(q.score, 3) for q in message2quality.values()])
    print("Format Match Scores")
    print("  (min, mean, max): ", *minmeanmax)
    print("near matches")
    print("  off-by-one:", offbyonecount)
    print("  off-by-more:", offbymorecount)
    print("exact matches:", exactcount)
    # # # # # # # # # # # # # # # # # # # # # # # #


    # # # # # # # # # # # # # # # # # # # # # # # #
    # conduct PCA refinement
    try:
        refinedSegmentedMessages = RelocatePCA.refineSegments(dc, kneedleSensitivity)
        # TODO needs recalculation of segment distances
    except ClusterAutoconfException as e:
        print("Initial clustering of the segments in the trace failed. The protocol in this trace cannot be inferred. "
              "The original exception message was:\n", e)
        exit(10)













    # # # # # # # # # # # # # # # # # # # # # # # #
    # cluster segments to determine field types on commonality
    try:
        clusterer = DBSCANsegmentClusterer(dc, S=kneedleSensitivity)
    except ClusterAutoconfException as e:
        print("Initial clustering of the segments in the trace failed. The protocol in this trace cannot be inferred. "
              "The original exception message was:\n", e)
        exit(10)

    # noinspection PyUnboundLocalVariable
    noise, *clusters = clusterer.clusterSimilarSegments(False)
    # noise: List[MessageSegment]
    # clusters: List[List[MessageSegment]]

    # extract "large" templates from noise that should rather be its own cluster
    for idx, seg in reversed(list(enumerate(noise.copy()))):  # type: int, MessageSegment
        freqThresh = log(len(dc.rawSegments))
        if isinstance(seg, Template):
            if len(seg.baseSegments) > freqThresh:
                clusters.append(noise.pop(idx).baseSegments)

    print("{} clusters generated from {} distinct segments".format(len(clusters), len(dc.segments)))
    # # # # # # # # # # # # # # # # # # # # # # # #









    # # # # # # # # # # # # # # # # # # # # # # # #
    fTypeTemplates = list()
    fTypeContext = list()
    for cLabel, segments in enumerate(clusters):
        ftype = FieldTypeTemplate(segments)
        ftype.fieldtype = "tf{:02d}".format(cLabel)
        fTypeTemplates.append(ftype)
        with open(join(reportFolder, "segmentclusters-" + splitext(pcapbasename)[0] + ".csv"), "a") as segfile:
            segcsv = csv.writer(segfile)
            segcsv.writerows([
                [],
                ["# Cluster", cLabel, "Segments", len(segments)],
                ["-"*10]*4,
            ])
            segcsv.writerows(
                {(seg.bytes.hex(), seg.bytes) for seg in segments}
            )

        # instead of nan-padded offset alignment, fill shorter segments with the values of the message
        # at the respective position
        resolvedSegments = list()
        for seg in segments:
            if isinstance(seg, Template):
                resolvedSegments.extend(seg.baseSegments)
            else:
                resolvedSegments.append(seg)
        fcontext = FieldTypeContext(resolvedSegments)
        fcontext.fieldtype = ftype.fieldtype
        fTypeContext.append(fcontext)

        # print("\nCluster", cLabel, "Segments", len(segments))
        # print({seg.bytes for seg in segments})

        # for seg in segments:
        #     # # sometimes raises: ValueError: On entry to DLASCL parameter number 4 had an illegal value
        #     # try:
        #     #     confidence = float(ftype.confidence(numpy.array(seg.values))) if ftype.length == seg.length else 0.0
        #     # except ValueError as e:
        #     #     print(seg.values)
        #     #     raise e
        #     #
        #
        #     confidence = 0.0
        #     if isinstance(seg, Template):
        #         for bs in seg.baseSegments:
        #             recog = RecognizedVariableLengthField(bs.message, ftype, bs.offset, bs.nextOffset, confidence)
        #             printFieldContext(trueSegmentedMessages, recog)
        #     else:
        #         recog = RecognizedVariableLengthField(seg.message, ftype, seg.offset, seg.nextOffset, confidence)
        #         printFieldContext(trueSegmentedMessages, recog)
    # # # # # # # # # # # # # # # # # # # # # # # #





    doWobble = False

    trace = splitext(pcapbasename)[0]


    # interestingClusters = [1]  # this is thought to be ntp.c1 (4-byte floats)
    # interestingClusters = [0]  # this is thought to be ntp.c0
    # interestingClusters = range(len(clusters))  # all
    # interestingClusters = [cid for cid, clu in enumerate(clusters)
    #                        if any([isExtendedCharSeq(seg.bytes) for seg in clu])] # all not-chars
    # interestingClusters, eigenVnV, screeKnees = RelocatePCA.filterRelevantClusters(fTypeContext)
    interestingClusters = RelocatePCA.filterForSubclustering(fTypeContext)

    for iC in interestingClusters:
        print("# # Cluster", iC)

        # # # # # # # # # # # # # # # # # # # # # # # # #
        # # Evaluation of Factor Analysis, see: https://www.datacamp.com/community/tutorials/introduction-factor-analysis
        # from factor_analyzer import FactorAnalyzer
        # from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity, calculate_kmo
        # # test sphericity (Bartlett’s test)
        # paddedValues = fTypeContext[iC].paddedValues()
        # nonConstPVals = paddedValues[:,fTypeContext[iC].stdev > 0]
        # if nonConstPVals.shape[1] > 1:
        #     chi_square_value, p_value = calculate_bartlett_sphericity(nonConstPVals)
        #     print("p-Value (Bartlett’s test): {:.6f} "
        #           "(should be very close to zero to pass the significance check)".format(p_value))
        #     # Kaiser-Meyer-Olkin (KMO) Test
        #     kmo_all, kmo_model = calculate_kmo(nonConstPVals)
        #     print("KMO model: {:.3f} (should be at least above .6, better above .8)".format(kmo_model))
        # else:
        #     print("Bartlett’s and KMO tests impossible: only one factor is non-constant.")
        # # ==> Factor Analysis not feasible for our use case
        #
        # import sklearn
        # fa = sklearn.decomposition.FactorAnalysis(2)
        # fa.fit(fTypeContext[5].paddedValues())
        # plt.plot(abs(fa.components_.T))
        # plt.show()
        # plt.imshow(abs(fa.components_.T))
        # plt.show()
        # # # # # # # # # # # # # # # # # # # # # # # # #

        # # # # # # # # # # # # # # # # # # # # # # # #
        # determine cluster contents in message context
        # for bs in clusters[iC]:
        #     markSegNearMatch(bs)
        # # # # # # # # # # # # # # # # # # # # # # # #

    # TODO reduced from 12 to 6 due to large dissimilarities in clusters (more checks!)
    kneedleSensitivity = 6.0

    collectedSubclusters = list()
    for iC in interestingClusters:
        # # # # # # # # # # # # # # # # # # # # # # # #
        if doWobble:
            print("Cluster", iC)
            # # # # # # # # # # # # # # # # # # # # # # # #
            # wobble segments in one cluster
            # start test with only one cluster: this is thought to be ntp.c1

            # templates must be resolved into segments before wobbling
            wobbles = list()
            for element in clusters[iC]:
                if isinstance(element, Template):
                    segments = element.baseSegments
                else:
                    segments = [element]
                for seg in segments:
                    wobbles.extend(wobbleSegmentInMessage(seg))
            wobDC = DelegatingDC(wobbles)
            wobClusterer = DBSCANsegmentClusterer(wobDC)
            # wobClusterer.autoconfigureEvaluation(join(reportFolder, "wobble-epsautoconfig.pdf"))
            wobClusterer.eps = wobClusterer.eps * 0.8
            wobNoise, *wobClusters = wobClusterer.clusterSimilarSegments(False)

            from utils.baseAlgorithms import tril
            print("Wobbled cluster distances:")
            print(tabulate([(clunu, wobDC.distancesSubset(wobclu).max(), tril(wobDC.distancesSubset(wobclu)).mean())
                            for clunu, wobclu in enumerate(wobClusters)],
                           headers=["Cluster", "max", "mean"]))
            # nothing very interesting in here. Wobble-clusters of true boundaries are not denser or smaller than others.

            for clunu, wobclu in enumerate(wobClusters):
                print("\n# # Wobbled Cluster", clunu)
                for seg in wobclu:
                    markSegNearMatch(seg)


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
                fig.suptitle("Principal (with Eigenvalue > 1) Eigenvectors - Cluster " + repr(iC))
                plt.savefig(join(reportFolder, "component-analysis_cluster{}.pdf".format(iC)))
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
                plt.savefig(join(reportFolder, "cov-test-heatmap_cluster{}.pdf".format(iC)))
                plt.close('all')

            if withPlots:
                print("Plot distances...")
                sdp = DistancesPlotter(specimens,
                                       'distances-wobble-' + "nemesys-segments_DBSCAN-eps{:0.3f}-ms{:d}-cluster{}".format(
                                           wobClusterer.eps, int(wobClusterer.min_samples), iC), False)

                clustermask = {segid: cluN for cluN, segL in enumerate(wobClusters) for segid in wobDC.segments2index(segL)}
                clustermask.update({segid: "Noise" for segid in wobDC.segments2index(wobNoise)})
                labels = numpy.array([clustermask[segid] for segid in range(len(wobDC.segments))])
                # plotManifoldDistances(dc.segments, dc.distanceMatrix, labels)
                sdp.plotSegmentDistances(wobDC, labels)
                sdp.writeOrShowFigure()
                del sdp
        else:
            pcaRelocator = RelocatePCA(fTypeContext[iC])  # , eigenVnV[iC]
            collectedSubclusters.extend(pcaRelocator.getSubclusters(dc, kneedleSensitivity,
                                                                    reportFolder, trace))
    # plot stuff
    relevantSubclusters, eigenVnV, screeKnees = RelocatePCA.filterRelevantClusters(
        [a.similarSegments for a in collectedSubclusters])
    for cid, sc in enumerate(collectedSubclusters):  # type: int, RelocatePCA
        print(sc.similarSegments.fieldtype, "*" if cid in relevantSubclusters else "")
        if withPlots and cid in relevantSubclusters:
            relocate = sc.relocateOffsets(reportFolder, trace, comparator)
            plotComponentAnalysis(sc,
                                  eigenVnV[cid] if cid in eigenVnV else numpy.linalg.eigh(sc.similarSegments.cov),
                                  relocate)
            for bs in sc.similarSegments.baseSegments:
                markSegNearMatch(bs)

    # print clusters filtered out, for confirmation.
    scoFile = "subcluster-overview.csv"
    scoHeader = ["trace", "cluster label", "cluster size", "max segment length",
                 "interesting", "length diff", "# unique", "is char",
                 "min dissimilarity", "max dissimilarity", "mean dissimilarity"]
    for cid, ftc in enumerate(fTypeContext):
        bslen = {bs.length for bs in ftc.baseSegments}
        lendiff = max(bslen) - min(bslen)

        uniqvals = {bs.bytes for bs in ftc.baseSegments}
        internDis = [dis for dis, idx in ftc.distancesToMixedLength()]
        ischar = sum([isExtendedCharSeq(seg.bytes) for seg in ftc.baseSegments]) > .5 * len(ftc.baseSegments)

        # # # # # # # # # # # # # # # # # # # # # # # #
        # write statistics
        fn = join(reportFolder, scoFile)
        writeheader = not exists(fn)
        with open(fn, "a") as segfile:
            segcsv = csv.writer(segfile)
            if writeheader:
                segcsv.writerow(scoHeader)
            segcsv.writerow([
                trace, ftc.fieldtype, len(ftc.baseSegments), ftc.length,
                repr(cid in interestingClusters), lendiff, len(uniqvals), ischar,
                min(internDis), max(internDis), numpy.mean(internDis),
            ])
    #     if cid not in interestingClusters:
    #         print("Cluster filtered out early: " + ftc.fieldtype)
    #         for bs in ftc.baseSegments:
    #             markSegNearMatch(bs)
    for cid, sc in enumerate(collectedSubclusters):
        bslen = {bs.length for bs in sc.similarSegments.baseSegments}
        lendiff = max(bslen) - min(bslen)

        uniqvals = {bs.bytes for bs in sc.similarSegments.baseSegments}
        internDis = [dis for dis, idx in sc.similarSegments.distancesToMixedLength()]
        ischar = sum([isExtendedCharSeq(seg.bytes)
                      for seg in sc.similarSegments.baseSegments]) > .5 * len(sc.similarSegments.baseSegments)

        # # # # # # # # # # # # # # # # # # # # # # # #
        # write statistics
        fn = join(reportFolder, scoFile)
        writeheader = not exists(fn)
        with open(fn, "a") as segfile:
            segcsv = csv.writer(segfile)
            if writeheader:
                segcsv.writerow(scoHeader)
            segcsv.writerow([
                trace, sc.similarSegments.fieldtype, len(sc.similarSegments.baseSegments), sc.similarSegments.length,
                repr(cid in interestingClusters), lendiff, len(uniqvals), ischar,
                min(internDis), max(internDis), numpy.mean(internDis)
            ])
        # if cid not in relevantSubclusters:
        #     print("Cluster filtered out: " + sc.similarSegments.fieldtype)
        #     for bs in sc.similarSegments.baseSegments:
        #         markSegNearMatch(bs)


    # # # # # # # # # # # # # # # # # # # # # # # # #
    # TODO from here on the segment refinement happens

    # collect new bounds for each segment and group them by message
    newBounds = dict()  # type: Dict[AbstractMessage, Dict[MessageSegment, List[int]]]
    for cid, sc in enumerate(collectedSubclusters):  # type: int, RelocatePCA
        if cid in relevantSubclusters:
            clusterBounds = sc.relocateBoundaries(dc, kneedleSensitivity, comparator, reportFolder)
            for segment, bounds in clusterBounds.items():
                if segment.message not in newBounds:
                    newBounds[segment.message] = dict()
                elif segment in newBounds[segment.message]:
                    print("\nSame segment was refined (PCA) multiple times. Needs resolving. Segment is:\n", segment)
                    print()
                    IPython.embed()
                newBounds[segment.message][segment] = bounds

    compareBounds = {m: {s: b.copy() for s, b in sb.items()} for m, sb in newBounds.items()}

    def segs4bound(segs2bounds: Dict[MessageSegment, List[int]], bound: int) -> List[MessageSegment]:
        """
        helper for iterating bounds

        :param segs2bounds:
        :param bound:
        :return: Yields the segment that is a key in segs2bounds,
            if it has bound in its value list. Yields the same segment for each time bound is in its list.
        """
        for seg, bounds in segs2bounds.items():
            for b in bounds.copy():
                if b == bound:
                    yield seg

    # iterate new bounds per message, remove superfluous ones
    for message, segsbounds in newBounds.items():

        # # create lookup "bound position" : "segment list"
        # boundSegsLookup = dict()  # type: Dict[int, List[MessageSegment]]
        # for seg, bounds in segsbounds.items():
        #     for bound in bounds:
        #         if bound not in boundSegsLookup:
        #             boundSegsLookup[bound] = list()
        #         boundSegsLookup[bound].append(seg)

        # Below, by list(chain(...)) create a copy to iterate, so we can delete stuff in the original bound lists.

        # if bound in segment offsets or nextOffsets:
        #   remove bound (also resolves some off-by-one conflicts, giving precedence to moves)
        for bound in list(chain(*segsbounds.values())):
            if bound in (segA.offset for segA in segsbounds.keys()) \
                    or bound in (segA.nextOffset for segA in segsbounds.keys()) \
                    or 0 <= bound >= len(message.data):
                for lookedupSeg in segs4bound(segsbounds, bound):
                    segsbounds[lookedupSeg].remove(bound)  # calls remove as often as there is bound in the list
                    # while bound in segsbounds[lookedupSeg]:

        # if bound in other segment is as close as one position away: resolve
        flatSegsboundsCopy = [(seg, bound) for seg, bounds in segsbounds.items() for bound in bounds]
        for seg, bound in flatSegsboundsCopy:
            for neighbor in [bound - 1, bound + 1]:
                if neighbor in (b for segA, bounds in segsbounds.items() if segA != seg for b in bounds):
                    print("There are off-by-one neighbors:")
                    for lookedupSeg in segs4bound(segsbounds, bound):
                        markSegmentInMessage(lookedupSeg)
                        print("bound: {} - neighbor: {}".format(bound, neighbor))
                    print("Needs resolving!")
                    print()
                    IPython.embed()

        # if bound in scope of more than one segment: resolve
        for bound in list(chain(*segsbounds.values())):
            lookedupSegs = list(segs4bound(segsbounds, bound))
            if len(lookedupSegs) > 1:
                segsNotHavingBoundInScope = [seg for seg in lookedupSegs if seg.offset > bound or bound < seg.nextOffset]
                if len(lookedupSegs) - len(segsNotHavingBoundInScope) == 1:
                    for segOutScope in segsNotHavingBoundInScope:
                        while bound in segsbounds[segOutScope]:
                            segsbounds[segOutScope].remove(bound)
                elif len(lookedupSegs) - len(segsNotHavingBoundInScope) == 0:
                    # just leave one arbitrary reference to the bound.
                    for segOutScope in segsNotHavingBoundInScope[1:]:
                        while bound in segsbounds[segOutScope]:
                            segsbounds[segOutScope].remove(bound)
                else:
                    # multiple segments truly have bound in scope
                    print("Bound {} is in scope of multiple segments:".format(bound))
                    for lookedupSeg in segs4bound(segsbounds, bound):
                        markSegmentInMessage(lookedupSeg)
                    print("Needs resolving!")
                    print()
                    IPython.embed()


    # # make some development output
    # for msg, segs in newBounds.items():
    #     # [[ min(outbounds) < min(inbounds) < max(outbounds) or min(outbounds) < max(inbounds) < max(outbounds)
    #     #    for inbounds in segs.values()] for outbounds in segs.values()]
    #     minmaxBounds = [(min(bounds), max(bounds)) for bounds in segs.values() if len(bounds) > 0]
    #     if any(any(outbmin < inbmin < outbmax or outbmin < inbmax < outbmax
    #             for inbmin, inbmax in minmaxBounds) for outbmin, outbmax in minmaxBounds):
    #         print("#### Conflicting bounds!")
    #     msgSegs = next(msegs for msegs in inferredSegmentedMessages if msegs[0].message == msg)
    #     for seg, bounds in segs.items():
    #         markSegmentInMessage(seg)
    #         if len(bounds) > 0 and (min(bounds) < seg.offset or seg.nextOffset < max(bounds)):
    #             print("needs neigbor change.")
    #     for off in range(len(msg.data)):
    #         print("^ ", end="") if off in chain(*segs.values()) else print("  ", end="")
    #     print("\noriginal new bounds: ")
    #     # list(chain(*compareBounds[msg].values())))
    #     for off in range(len(msg.data)):
    #         print("^ ", end="") if off in chain(*compareBounds[msg].values()) else print("  ", end="")
    #     print()

    # IPython.embed()

    # iterate sorted message segments
    margin = 0
    refinedSegmentedMessages = list()  # type: List[List[MessageSegment]]
    for msgsegs in inferredSegmentedMessages:
        msg = msgsegs[0].message
        if msg not in newBounds:
            refinedSegmentedMessages.append(msgsegs)
            continue

        newMsgBounds = sorted(set(chain(*newBounds[msg].values())))
        lastBound = 0
        currentBoundID = 0
        refinedSegmentedMessages.append(list())
        for segInf in sorted(msgsegs, key=lambda x: x.offset):
            # add skipped bytes to next segment if next_segment.nextOffset < bound
            # create segment from last bound to min(next_segment.nextOffset, bound);
            if lastBound < segInf.offset:
                assert len(newMsgBounds) > 0  # should never happen otherwise
                if currentBoundID < len(newMsgBounds) and newMsgBounds[currentBoundID] <= segInf.nextOffset + margin:
                    nextOffset = newMsgBounds[currentBoundID]
                    currentBoundID += 1
                else:
                    nextOffset = segInf.nextOffset

                assert not len(refinedSegmentedMessages[-1]) > 0 or \
                       refinedSegmentedMessages[-1][-1].nextOffset == lastBound, \
                    "Segment sequence error: add skipped bytes"

                refinedSegmentedMessages[-1].append(
                    MessageSegment(segInf.analyzer, lastBound, nextOffset - lastBound)
                )
                lastBound = refinedSegmentedMessages[-1][-1].nextOffset
                if nextOffset > segInf.nextOffset:
                    continue

            # if no bounds in scope of segment: add old segment and continue
            if lastBound == segInf.offset and (
                    len(newMsgBounds) == 0 or currentBoundID >= len(newMsgBounds)
                    or newMsgBounds[currentBoundID] > segInf.nextOffset + margin):

                assert not len(refinedSegmentedMessages[-1]) > 0 \
                        or refinedSegmentedMessages[-1][-1].nextOffset == segInf.offset, \
                    "Segment sequence error: no bounds in scope of segment"

                refinedSegmentedMessages[-1].append(segInf)
                lastBound = segInf.nextOffset
                continue
            # if bound in scope of segment:
            #   create segment from segment offset or last bound (which is larger) to bound
            for bound in [nmb for nmb in newMsgBounds[currentBoundID:] if nmb < segInf.nextOffset]:
                newOffset = max(segInf.offset, lastBound)

                assert not len(refinedSegmentedMessages[-1]) > 0 or \
                        refinedSegmentedMessages[-1][-1].nextOffset == newOffset, \
                    "Segment sequence error: bound in scope of segment"

                refinedSegmentedMessages[-1].append(
                    MessageSegment(segInf.analyzer, newOffset, bound - newOffset)
                )
                lastBound = newMsgBounds[currentBoundID]
                currentBoundID += 1

            # no further bounds (at least until segment end)
            if segInf.nextOffset - lastBound <= margin and len(msg.data) - segInf.nextOffset > 0:
                continue

            # if no further bounds for message or bound > segment next offset+1 and resulting segment longer than 1:
            #   create segment from last bound to inferred segment's next offset;
            if currentBoundID >= len(newMsgBounds) or (
                    newMsgBounds[currentBoundID] > segInf.nextOffset + 1):

                assert not len(refinedSegmentedMessages[-1]) > 0 or \
                       refinedSegmentedMessages[-1][-1].nextOffset == lastBound, \
                    "Segment sequence error: if no further bounds"
                refinedSegmentedMessages[-1].append(
                    MessageSegment(segInf.analyzer, lastBound, segInf.nextOffset - lastBound)
                )
                lastBound = refinedSegmentedMessages[-1][-1].nextOffset
                # do not advance currentBoundID bound (in case there is another bound
                #   so we need to consider it in scope of a later segment)

            # bound == next offset+1 and resulting segment longer than 1: create segment from last bound to bound
            elif newMsgBounds[currentBoundID] == segInf.nextOffset + 1 and newMsgBounds[currentBoundID] - lastBound > 1:

                assert not len(refinedSegmentedMessages[-1]) > 0 or \
                       refinedSegmentedMessages[-1][-1].nextOffset == lastBound, \
                    "Segment sequence error: bound == next offset+1"

                refinedSegmentedMessages[-1].append(
                    MessageSegment(segInf.analyzer, lastBound, newMsgBounds[currentBoundID] - lastBound)
                )
                lastBound = refinedSegmentedMessages[-1][-1].nextOffset
                currentBoundID += 1

        msgbytes = b"".join([seg.bytes for seg in refinedSegmentedMessages[-1]])
        assert msgbytes == msg.data, "segment sequence does not match message bytes:"
            # print(msg.data.hex())
            # print(msgbytes.hex())
            # print(msgsegs)
            # IPython.embed()

    # for msgsegs in refinedSegmentedMessages:
    #     infms = next(ms for ms in inferredSegmentedMessages if ms[0].message == msgsegs[0].message)
    #     if msgsegs != infms:
    #         comparator.pprint2Interleaved(infms[0].message, [infs.nextOffset for infs in infms])
    #         comparator.pprint2Interleaved(msgsegs[0].message, [infs.nextOffset for infs in msgsegs])
    #         newMsgBounds = sorted(chain(*newBounds[msgsegs[0].message].values()))
    #         print(newMsgBounds)
    #         missedBound = [nmb for nmb in newMsgBounds if nmb not in (ref.offset for ref in msgsegs)]
    #         if len(missedBound) > 0:
    #             print("missedBounds", missedBound)
    #
    #         segseq = [ref1 for ref1, ref2 in zip(msgsegs[:-1], msgsegs[1:]) if ref1.nextOffset != ref2.offset]
    #         if len(segseq) > 0:
    #             print("Segment sequence error!\n", segseq)
    #         shoseg = [ref for ref in msgsegs if ref.offset > 0 and ref.length < 2 and ref not in infms]
    #         if len(shoseg) > 0:
    #             print("Short segments:\n", shoseg)
    #         print()

    # # final validity check
    # for msgsegs in refinedSegmentedMessages:
    #     msg = msgsegs[0].message
    #     msgbytes = b"".join([seg.bytes for seg in msgsegs])
    #     if not msgbytes == msg.data:
    #         print("segment sequence does not match message bytes:")
    #         print(msg.data.hex())
    #         print(msgbytes.hex())
    #         print(msgsegs)

    # # select one tf
    # tf02 = next(c for c in collectedSubclusters if c.similarSegments.fieldtype == "tf02")
    # # # # # # # # # # # # # # # # # # # # # # # # #


    # # # # # # # # # # # # # # # # # # # # # # # #
    # Determine the amount of off-by-one errors in refined segmentation

    from validation.dissectorMatcher import DissectorMatcher
    message2quality = DissectorMatcher.symbolListFMS(comparator, symbolsFromSegments(refinedSegmentedMessages))
    exactcount = 0
    offbyonecount = 0
    offbymorecount = 0
    for fms in message2quality.values():  # type: FormatMatchScore
        exactcount += fms.exactCount
        offbyonecount += sum(1 for truf, inff in fms.nearMatches.items() if abs(truf - inff) == 1)
        offbymorecount += sum(1 for truf, inff in fms.nearMatches.items() if abs(truf - inff) > 1)
    minmeanmax = reportWriter.getMinMeanMaxFMS([round(q.score, 3) for q in message2quality.values()])
    print("Format Match Scores (refined)")
    print("  (min, mean, max): ", *minmeanmax)
    print("near matches")
    print("  off-by-one:", offbyonecount)
    print("  off-by-more:", offbymorecount)
    print("exact matches:", exactcount)

    # # # # # # # # # # # # # # # # # # # # # # # #























    # # # # # # # # # # # # # # # # # # # # # # # # #
    # # TODO mark (at least) exact segment matches with the true data type
    # # currently marks only very few segments. Improve boundaries first!
    # typedMatchSegs = list()
    # for seg in dc.segments:
    #     if isinstance(seg, MessageSegment):
    #         typedMatchSegs.append(comparator.segment2typed(seg))
    #     elif isinstance(seg, Template):
    #         machingType = None
    #         typedBasesegments = list()
    #         for bs in seg.baseSegments:
    #             tempTyped = comparator.segment2typed(bs)
    #             if not isinstance(tempTyped, TypedSegment):
    #                 # print("At least one segment in template is not a field match.")
    #                 # markSegNearMatch(tempTyped)
    #                 machingType = None
    #                 break
    #             elif machingType == tempTyped.fieldtype or machingType is None:
    #                 machingType = tempTyped.fieldtype
    #                 typedBasesegments.append(tempTyped)
    #             else:
    #                 # print("Segment's matching field types are not the same in template, e. g., {} and {} ({})".format(
    #                 #     machingType, tempTyped.fieldtype, tempTyped.bytes.hex()
    #                 # ))
    #                 machingType = None
    #                 break
    #         if machingType is None:
    #             typedMatchSegs.append(seg)
    #         else:
    #             typedMatchSegs.append(TypedTemplate(seg.values, typedBasesegments, seg._method))
    # # # # # # # # # # # # # # # # # # # # # # # # #





    # # # # # # # # # # # # # # # # # # # # # # # #
    if withPlots:
        print("Plot distances...")
        sdp = DistancesPlotter(specimens,
                               'distances-' + "nemesys-segments_DBSCAN-eps{:0.3f}-ms{:d}".format(
                                   clusterer.eps, int(clusterer.min_samples)), False)

        clustermask = {segid: cluN for cluN, segL in enumerate(clusters) for segid in dc.segments2index(segL)}
        clustermask.update({segid: "Noise" for segid in dc.segments2index(noise)})
        labels = numpy.array([clustermask[segid] for segid in range(len(dc.segments))])
        # plotManifoldDistances(dc.segments, dc.distanceMatrix, labels)
        sdp.plotSegmentDistances(dc, labels)
        sdp.writeOrShowFigure()
        del sdp
    # # # # # # # # # # # # # # # # # # # # # # # #


    # # # # # # # # # # # # # # # # # # # # # # # #
    # The following is strictly speaking not clustering relevant:
    # # # # # # # # # # # # # # # # # # # # # # # #


    # # show position of each segment individually.
    # for seg in dc.segments:
    #     markSegNearMatch(seg)

    # comparator.pprint2Interleaved(dc.segments[6].message, inferredFEs4segment(dc.segments[6]),
    #                               (dc.segments[6].offset-2, dc.segments[6].nextOffset+1))


    # # # # # # # # # # # # # # # # # # # # # # # # #
    # # Count and compare the most common segment values
    # # # # # # # # # # # # # # # # # # # # # # # # #
    # from collections import Counter
    # infsegvalcnt = Counter(seg.bytes for seg in dc.rawSegments)
    # truesegvalcnt = Counter(seg.bytes for seg in chain.from_iterable(trueSegmentedMessages.values()))
    #
    # most10true = [(val.hex(), tcnt, infsegvalcnt[val]) for val, tcnt in truesegvalcnt.most_common(10)]
    # most10inf = [(val.hex(), truesegvalcnt[val], icnt) for val, icnt in infsegvalcnt.most_common(10)
    #                       if val.hex() not in (v for v, t, i in most10true)]
    # print(tabulate(most10true + most10inf, headers=["value and count for...", "true", "inferred"]))
    #
    # for mofreqhex, *cnts in most10inf:
    #     print("# "*10, mofreqhex, "# "*10)
    #     for m in specimens.messagePool.keys():
    #         pos = m.data.find(bytes.fromhex(mofreqhex))
    #         if pos > -1:
    #             comparator.pprint2Interleaved(m, [infs.nextOffset for infs in next(
    #                 msegs for msegs in inferredSegmentedMessages if msegs[0].message == m)],
    #                                           mark=(pos, pos+len(mofreqhex)//2))
    # # now see how well NEMESYS infers common fields (to use for correcting boundaries by replacing segments of less
    # #                                                frequent values with more of those of the most frequent values)
    # # # # # # # # # # # # # # # # # # # # # # # # #









    if args.interactive:
        from tabulate import tabulate
        # globals().update(locals())
        IPython.embed()










