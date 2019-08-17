"""
segment messages by NEMESYS and cluster
"""

import argparse, IPython
from os.path import isfile, basename, join, splitext, exists
from math import log
from typing import Union
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.colors as colors
from collections import Counter

from inference.templates import DBSCANsegmentClusterer, FieldTypeTemplate, Template, TypedTemplate, FieldTypeContext
from inference.segmentHandler import symbolsFromSegments, wobbleSegmentInMessage, isExtendedCharSeq
from inference.formatRefinement import RelocatePCA
from visualization.distancesPlotter import DistancesPlotter
from visualization.simplePrint import *
from utils.evaluationHelpers import *


debug = False

# fix the analysis method to VALUE
analysisTitle = 'value'
# fix the distance method to canberra
distance_method = 'canberra'
# use NEMESYS segments
tokenizer = 'nemesys'



def markSegNearMatch(segment: Union[MessageSegment, Template]):

    # TODO support Templates
    if isinstance(segment, Template):
        segs = segment.baseSegments
    else:
        segs = [segment]

    print()  # one blank line for structure
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
    return next(msegs for msegs in inferredSegmentedMessages if msegs[0].message == segment.message)

def inferredFEs4segment(segment: MessageSegment) -> List[int]:
    return [infs.nextOffset for infs in inferred4segment(segment)]








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
        args.pcapfilename, analysisTitle, tokenizer, debug, analyzerType, analysisArgs, args.sigma, True, True
    )  # Note!  When manipulating distances, deactivate caching by adding "True".
    # chainedSegments = dc.rawSegments
    # # # # # # # # # # # # # # # # # # # # # # # #
    trueSegmentedMessages = {msgseg[0].message: msgseg
                         for msgseg in annotateFieldTypes(analyzerType, analysisArgs, comparator)
                         }
    # tabuSeqOfSeg(trueSegmentedMessages)
    # print(trueSegmentedMessages.values())


    # # # # # # # # # # # # # # # # # # # # # # # #
    # Determine the amount of off-by-one errors
    from validation.dissectorMatcher import DissectorMatcher
    message2quality = DissectorMatcher.symbolListFMS(comparator, symbolsFromSegments(inferredSegmentedMessages))
    offbyonecount = 0
    offbymorecount = 0
    for fms in message2quality.values():
        offbyonecount += sum(1 for truf, inff in fms.nearMatches.items() if abs(truf - inff) == 1)
        offbymorecount += sum(1 for truf, inff in fms.nearMatches.items() if abs(truf - inff) > 1)
    print("near matches")
    print("off-by-one:", offbyonecount)
    print("off-by-more:", offbymorecount)
    # # # # # # # # # # # # # # # # # # # # # # # #



    # # # # # # # # # # # # # # # # # # # # # # # #
    clusterer = DBSCANsegmentClusterer(dc)
    clusterer.eps = clusterer.eps * 0.8
    # noise: List[MessageSegment]
    # clusters: List[List[MessageSegment]]
    noise, *clusters = clusterer.clusterSimilarSegments(False)
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




    # interestingClusters = [1]  # this is thought to be ntp.c1 (4-byte floats)
    # interestingClusters = [0]  # this is thought to be ntp.c0
    # interestingClusters = range(len(clusters))  # all
    # interestingClusters = [cid for cid, clu in enumerate(clusters)
    #                        if any([isExtendedCharSeq(seg.bytes) for seg in clu])] # all not-chars
    interestingClusters, eigenVnV, screeKnees = RelocatePCA.filterRelevantClusters(fTypeContext)

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
        # # # # # # # # # # # # # # # # # # # # # # # # #

        # # # # # # # # # # # # # # # # # # # # # # # #
        # determine cluster contents in message context
        for bs in clusters[iC]:
            markSegNearMatch(bs)
        # # # # # # # # # # # # # # # # # # # # # # # #

    for iC in interestingClusters:
        # # # # # # # # # # # # # # # # # # # # # # # #
        if doWobble:
            print("# # Cluster", iC)
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
            if withPlots:
                pcaRelocator = RelocatePCA(fTypeContext[iC], eigenVnV[iC])
                relocateFromEnd = pcaRelocator.relocateBoundaries(dc, reportFolder, splitext(pcapbasename)[0])
                if isinstance(relocateFromEnd, dict):
                    # TODO add support for subclustering, where the method returns a dict of relecant information about each subcluster
                    continue

                # # # # # # # # # # # # # # # # # # # # # # # #
                # Count true boundaries for the segments' relative positions
                trueOffsets = list()
                for bs in fTypeContext[iC].baseSegments:
                    fe = comparator.fieldEndsPerMessage(bs.analyzer.message)
                    offs, nxtOffs = fTypeContext[iC].paddedPosition(bs)
                    trueOffsets.extend(o - offs for o in fe if offs <= o <= nxtOffs)
                truOffCnt = Counter(trueOffsets)

                # # # # # # # # # # # # # # # # # # # # # # # #
                # plotting stuff
                compFig = plt.figure()
                compAx = compFig.add_subplot(1,1,1)
                try:
                    if not any(pcaRelocator.principalComponents):
                        # if there is no principal component, plot all vectors for investigation
                        principalComponents = numpy.array([True]*eigenVnV[iC][0].shape[0])
                        noPrinComps = True
                    else:
                        noPrinComps = False
                    compAx.axhline(0, color="black", linewidth=1, alpha=.4)
                    lines = compAx.plot(pcaRelocator.contribution)  # type: List[plt.Line2D]
                    for i, (l, ev) in enumerate(zip(lines, eigenVnV[iC][0][pcaRelocator.principalComponents])):
                        l.set_label(repr(i + 1) + ", $\lambda=$" + "{:.1f}".format(ev))
                    # mark correct boundaries with the intensity (alpha) of the relative offset
                    for offs, cnt in truOffCnt.most_common():
                        compAx.axvline(offs - 0.5, color="black", linewidth=.5, alpha=cnt / max(truOffCnt.values()))
                    # mark reallocations
                    for rfe in relocateFromEnd:
                        compAx.scatter(rfe-0.5, 0, c="red")
                    compAx.xaxis.set_major_locator(ticker.MultipleLocator(1))
                    compAx.legend()
                    if noPrinComps:
                        compFig.suptitle("All Eigenvectors (NO Eigenvalue is > {:.0f}) - Cluster ".format(
                            pcaRelocator.screeThresh) + repr(iC))
                    else:
                        compFig.suptitle("Principal Eigenvectors (with Eigenvalue > {:.0f}) - Cluster ".format(
                            pcaRelocator.screeThresh) + repr(iC))
                    compFig.savefig(join(reportFolder, "component-analysis_cluster{}.pdf".format(iC)))
                except Exception as e:
                    print("\n\n### {} ###\n\n".format(e))
                    IPython.embed()
                plt.close(compFig)

                # # # # # # # # # # # # # # # # # # # # # # # #
                # plot heatmaps of covariance matrices "cov-test-heatmap"
                heatFig = plt.figure()
                heatAx = heatFig.add_subplot(1,1,1)

                # make the colormap symmetric around zero
                vmax = max(abs(fTypeContext[iC].cov.min()), abs(fTypeContext[iC].cov.max()))
                heatMap = heatAx.imshow(fTypeContext[iC].cov, cmap='PuOr', norm=colors.Normalize(-vmax, vmax))
                # logarithmic scale hides the interesting variance differences. For reference on how to do it:
                #   norm=colors.SymLogNorm(100, vmin=-vmax, vmax=vmax)

                # mark correct boundaries with the intensity (alpha) of the relative offset
                for offs, cnt in truOffCnt.most_common():
                    heatAx.axvline(offs - 0.5, color="white", linewidth=1.5,
                                alpha=cnt / max(truOffCnt.values()))
                    heatAx.axhline(offs - 0.5, color="white", linewidth=1.5,
                                alpha=cnt / max(truOffCnt.values()))
                heatAx.xaxis.set_major_locator(ticker.MultipleLocator(1))
                heatFig.colorbar(heatMap)
                heatFig.savefig(join(reportFolder, "cov-test-heatmap_cluster{}.pdf".format(iC)))
                plt.close(heatFig)

    # # # # # # # # # # # # # # # # # # # # # # # # #


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










