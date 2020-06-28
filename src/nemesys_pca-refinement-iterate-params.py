"""
segment messages by NEMESYS and cluster
"""

import argparse, IPython
from os.path import isfile, basename, join, splitext
from os import makedirs
from typing import Any
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.colors as colors
from collections import Counter

import numpy

from nemere.inference.segments import MessageSegment
from nemere.inference.templates import DBSCANsegmentClusterer, FieldTypeTemplate, Template, FieldTypeContext, \
    ClusterAutoconfException, DelegatingDC
from nemere.inference.segmentHandler import symbolsFromSegments, wobbleSegmentInMessage, charRefinements
from nemere.inference.formatRefinement import RelocatePCA
from nemere.validation import reportWriter
from nemere.visualization.distancesPlotter import DistancesPlotter
from nemere.visualization.simplePrint import *
from nemere.utils.evaluationHelpers import *
from nemere.validation.dissectorMatcher import DissectorMatcher

debug = False

# fix the analysis method to VALUE
analysisTitle = 'value'
# fix the distance method to canberra
distance_method = 'canberra'
# use NEMESYS segments
# tokenizer = 'nemesys'
tokenizer = '4bytesfixed'



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
        # wobClusterer.autoconfigureEvaluation(join(reportFolder, "wobble-epsautoconfig.pdf"))
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
            fig.suptitle("Principal (with Eigenvalue > 1) Eigenvectors - Cluster " + repr(cluster.similarSegments.fieldtype))
            plt.savefig(join(reportFolder, "component-analysis_cluster{}.pdf".format(cluster.similarSegments.fieldtype)))
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
            plt.savefig(join(reportFolder, "cov-test-heatmap_cluster{}.pdf".format(cluster.similarSegments.fieldtype)))
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

def mostCommonValues():
    # # # # # # # # # # # # # # # # # # # # # # # #
    # Count and compare the most common segment values
    # # # # # # # # # # # # # # # # # # # # # # # #
    from collections import Counter
    infsegvalcnt = Counter(seg.bytes for seg in dc.rawSegments)
    truesegvalcnt = Counter(seg.bytes for seg in chain.from_iterable(trueSegmentedMessages.values()))

    most10true = [(val.hex(), tcnt, infsegvalcnt[val]) for val, tcnt in truesegvalcnt.most_common(10)]
    most10inf = [(val.hex(), truesegvalcnt[val], icnt) for val, icnt in infsegvalcnt.most_common(10)
                          if val.hex() not in (v for v, t, i in most10true)]
    print(tabulate(most10true + most10inf, headers=["value and count for...", "true", "inferred"]))

    for mofreqhex, *cnts in most10inf:
        print("# "*10, mofreqhex, "# "*10)
        for m in specimens.messagePool.keys():
            pos = m.data.find(bytes.fromhex(mofreqhex))
            if pos > -1:
                comparator.pprint2Interleaved(m, [infs.nextOffset for infs in next(
                    msegs for msegs in inferredSegmentedMessages if msegs[0].message == m)],
                                              mark=(pos, pos+len(mofreqhex)//2))
    # now see how well NEMESYS infers common fields (to use for correcting boundaries by replacing segments of less
    #                                                frequent values with more of those of the most frequent values)
    # # # # # # # # # # # # # # # # # # # # # # # #

def commonBoundsIrrelevant(relevantSubclusters):
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
                markSegNearMatch(bs)
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
    args = parser.parse_args()

    if not isfile(args.pcapfilename):
        print('File not found: ' + args.pcapfilename)
        exit(1)
    pcapbasename = basename(args.pcapfilename)
    withPlots = args.with_plots

    analyzerType = analyses[analysisTitle]
    analysisArgs = None

    reportFolder = join(reportFolder, splitext(pcapbasename)[0])
    makedirs(reportFolder)



    # for factor in [10, 5, 2, 1.6, 1.2, 1, 0.9, 0.7, 0.5, 0.2, 0.1]:  # 10 .. 0.1
    # for factor in [2, 1.6, 1.2, 1, 0.9, 0.7, 0.5]:  # 2 .. 0.5
    # for factor in [100, 60, 40, 20]:  # 100 .. 20
    # for factor in [1]:
    # for factor in [1.2, 1.1, 1.0, 0.9, 0.8, 0.7, 0.6]:  # 1.2 .. 0.6
    # for factor in [10, 5, 2, 1.8, 1.6, 1.4, 1.2, 1.1, 1, 0.9, 0.8]:  # 10 .. 0.8
    for factor in [1.20, 1.1, 1.08, 1.06, 1.04, 1.02, 1.01, 1, 0.99, 0.98, 0.96, 0.92, 0.9, 0.8]:  # 1.2 .. 0.8
        # sInit = 10.0*factor
        # evalLabel = "Sinit{:.3f}".format(sInit)
        # # initialKneedleSensitivity = sInit
        # sSC = 5.0*factor
        # evalLabel = "Ssubcluster{:.3f}".format(sSC)
        # # subclusterKneedleSensitivity = sSC

        # RelocatePCA.contributionRelevant = 0.1*factor
        # evalLabel = "contributionRelevant={:.3f}".format(RelocatePCA.contributionRelevant)

        # RelocatePCA.nearZero = 0.030*factor
        # evalLabel = "nearZero={:.3f}".format(RelocatePCA.nearZero)
        # RelocatePCA.notableContrib = 0.75*factor  # 0.66
        # evalLabel = "notableContrib={:.3f}".format(RelocatePCA.notableContrib)

        # RelocatePCA.relaxedNearZero = 0.05*factor
        # evalLabel = "relaxedNearZero={:.3f}".format(RelocatePCA.relaxedNearZero)
        # RelocatePCA.relaxedNZlength = 4*factor
        # evalLabel = "relaxedNZlength={:.3f}".format(RelocatePCA.relaxedNZlength)
        # RelocatePCA.relaxedNotableContrib = 0.005*factor
        # evalLabel = "relaxedNotableContrib={:.3f}".format(RelocatePCA.relaxedNotableContrib)
        # RelocatePCA.relaxedMaxContrib = 1.00*factor
        # evalLabel = "relaxedMaxContrib={:.3f}".format(RelocatePCA.relaxedMaxContrib)

        # RelocatePCA.CommonBoundUtil.uoboFreqThresh = 0.8*factor
        # evalLabel = "uoboFreqThresh={:.3f}".format(RelocatePCA.CommonBoundUtil.uoboFreqThresh)

        specimens, comparator, inferredSegmentedMessages, dc, segmentationTime, dist_calc_segmentsTime = cacheAndLoadDC(
            args.pcapfilename, analysisTitle, tokenizer, debug, analyzerType, analysisArgs, factor, True,
            refinementCallback=charRefinements,
            disableCache=True
        )  # Note!  When manipulating distances, deactivate caching by adding "True".
        trueSegmentedMessages = {msgseg[0].message: msgseg
                         for msgseg in annotateFieldTypes(analyzerType, analysisArgs, comparator)
                         }
        # evalLabel = "nemesysSigma={:.3f}".format(factor)
        evalLabel = "segments" # 4bytesfixed

        # IPython.embed()

        print(evalLabel)
        try:
            collectedSubclusters = list()  # type: List[RelocatePCA]
            startRefinement = time.time()
            refinedSM = RelocatePCA.refineSegments(inferredSegmentedMessages, dc,
                                                   comparator=comparator, reportFolder=reportFolder,
                                                   collectEvaluationData=collectedSubclusters)
            runtimeRefinement = time.time() - startRefinement
            # # # # # # # # # # # # # # # # # # # # # # # #
            # Write FMS statistics
            inferenceTitle = "4bytesfixed"
            # # # # # # # # # # # # # # # # # # # # # # # #
            # Determine the amount of off-by-one errors in refined segmentation
            message2quality = DissectorMatcher.symbolListFMS(comparator, symbolsFromSegments(refinedSM))
            reportWriter.writeReport(message2quality, -1.0, specimens, comparator,
                                     inferenceTitle + '_pcaRefined_' + evalLabel,
                                     reportFolder)
            exactcount, offbyonecount, offbymorecount = reportWriter.countMatches(message2quality.values())
            minmeanmax = reportWriter.getMinMeanMaxFMS([round(q.score, 3) for q in message2quality.values()])
            print("Format Match Scores (refined)")
            print("  (min, mean, max): ", *minmeanmax)
            print("near matches")
            print("  off-by-one:", offbyonecount)
            print("  off-by-more:", offbymorecount)
            print("exact matches:", exactcount)
            # # # # # # # # # # # # # # # # # # # # # # # #
        except ClusterAutoconfException as e:
            print("Initial clustering of the segments in the trace failed. "
                  "The protocol in this trace cannot be inferred. "
                  "The original exception message was:\n", e)




    # # # # # # # # # # # # # # # # # # # # # # # #
    # Write FMS statistics
    inferenceTitle = "4bytesfixed_raw_segments"
    # # # # # # # # # # # # # # # # # # # # # # # #
    # Determine the amount of off-by-one errors in original segments
    message2quality = DissectorMatcher.symbolListFMS(comparator, symbolsFromSegments(inferredSegmentedMessages))
    reportWriter.writeReport(message2quality, runtimeRefinement, specimens, comparator, inferenceTitle,
                             reportFolder)
    exactcount, offbyonecount, offbymorecount = reportWriter.countMatches(message2quality.values())
    minmeanmax = reportWriter.getMinMeanMaxFMS([round(q.score, 3) for q in message2quality.values()])
    print("Format Match Scores")
    print("  (min, mean, max): ", *minmeanmax)
    print("near matches")
    print("  off-by-one:", offbyonecount)
    print("  off-by-more:", offbymorecount)
    print("exact matches:", exactcount)









    if args.interactive:
        from tabulate import tabulate
        # globals().update(locals())
        IPython.embed()










