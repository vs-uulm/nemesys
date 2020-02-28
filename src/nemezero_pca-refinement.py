"""
segment messages by NEMESYS and cluster
"""

import argparse, IPython
from os.path import isfile, basename, join, splitext, exists
from os import makedirs
from typing import Union, Any
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.colors as colors
from collections import Counter

from inference.templates import DBSCANsegmentClusterer, FieldTypeTemplate, Template, TypedTemplate, FieldTypeContext, \
    ClusterAutoconfException
from inference.segmentHandler import symbolsFromSegments, wobbleSegmentInMessage, isExtendedCharSeq, \
    originalRefinements, baseRefinements, pcaRefinements, pcaPcaRefinements, zeroBaseRefinements
from inference.formatRefinement import RelocatePCA, CropDistinct, BlendZeroSlices, CropChars
from validation import reportWriter
from visualization.distancesPlotter import DistancesPlotter
from visualization.simplePrint import *
from utils.evaluationHelpers import *
from validation.dissectorMatcher import FormatMatchScore
from validation.dissectorMatcher import DissectorMatcher

debug = False

# fix the analysis method to VALUE
analysisTitle = 'value'
# fix the distance method to canberra
distance_method = 'canberra'
# use NEMESYS segments
tokenizer = 'zeros'

refinementMethods = [
    "base",     # moco+splitfirstseg
    "PCA",      # 2-pass PCA
    "PCA1",     # 1-pass PCA
    "PCAmoco",  # 2-pass PCA+moco
    ]


def markSegNearMatch(inferredSegmentedMessages: Sequence[Sequence[MessageSegment]],
                     segment: Union[MessageSegment, Template]):
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
        inf4seg = inferred4segment(inferredSegmentedMessages, seg)
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

def inferred4segment(inferredSegmentedMessages: Sequence[Sequence[MessageSegment]],
                     segment: MessageSegment) -> Sequence[MessageSegment]:
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

def mostCommonValues(inferredSegmentedMessages: Sequence[Sequence[MessageSegment]], dc: DistanceCalculator):
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
                markSegNearMatch(inferredSegmentedMessages, bs)
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
    parser.add_argument('-p', '--with-plots',
                        help='Generate plots of true field types and their distances.',
                        action="store_true")
    parser.add_argument('-r', '--refinement', help='Select segment refinement method.', choices=refinementMethods)
    args = parser.parse_args()

    if not isfile(args.pcapfilename):
        print('File not found: ' + args.pcapfilename)
        exit(1)
    pcapbasename = basename(args.pcapfilename)
    print("\nTrace:", pcapbasename)
    withPlots = args.with_plots
    analyzerType = analyses[analysisTitle]
    analysisArgs = None

    # # # # # # # # # # # # # # # # # # # # # # # #
    # local alternative to cache/load the DistanceCalculator to the filesystem
    #
    pcapName = os.path.splitext(pcapbasename)[0]
    # dissect and label messages
    print("Load messages from {}...".format(pcapName))
    specimens = SpecimenLoader(args.pcapfilename, 2, True)
    comparator = MessageComparator(specimens, 2, True, debug=debug)

    # zero/non-zero segments
    segmentsPerMsg = [(MessageSegment(Value(msg), 0, len(msg.data)),) for msg in specimens.messagePool.keys()]
    # blend: True to omit single zeros
    zeroSlicedMessages = [BlendZeroSlices(list(msg)).blend(True) for msg in segmentsPerMsg]
    inferredSegmentedMessages = [CropChars(segs).split() for segs in zeroSlicedMessages]

    # # # # # # # # # # # # # # # # # # # # # # # #
    trueSegmentedMessages = {msgseg[0].message: msgseg
                         for msgseg in annotateFieldTypes(analyzerType, analysisArgs, comparator)
                         }
    #
    # for msgsegs in inferredSegmentedMessages:
    #     comparator.pprint2Interleaved(msgsegs[0].message, [infs.nextOffset for infs in msgsegs])

    # # # # # # # # # # # # # # # # # # # # # # # #
    # conduct PCA refinement
    collectedSubclusters = list()  # type: List[RelocatePCA]
    pcaClusterer = list()  # type: List[DBSCANsegmentClusterer]

    startRefinement = time.time()
    if args.refinement == "base":
        refinedSM = baseRefinements(inferredSegmentedMessages)
    elif args.refinement == "PCA1":
        refinedSM = pcaRefinements(inferredSegmentedMessages,
                                   collectEvaluationData=collectedSubclusters, retClusterer=pcaClusterer)
    elif args.refinement == "PCA":
        refinedSM = pcaPcaRefinements(inferredSegmentedMessages,
                                      collectEvaluationData=collectedSubclusters, retClusterer=pcaClusterer)
    elif args.refinement == "PCAmoco":
        try:


            # # first perform most common values refinement
            # moco = CropDistinct.countCommonValues(inferredSegmentedMessages)
            # print([m.hex() for m in moco])
            # refinedSegmentedMessages = list()
            # for msg in inferredSegmentedMessages:
            #     croppedMsg = CropDistinct(msg, moco).split()
            #     refinedSegmentedMessages.append(croppedMsg)
            # refinedSM = RelocatePCA.refineSegments(refinedSegmentedMessages, dc,
            #                                        collectEvaluationData=collectedSubclusters)

            pcaRound = charRefinements(inferredSegmentedMessages)
            for i in range(2):
                refinementDC = DelegatingDC(list(chain.from_iterable(pcaRound)))
                pcaRound = RelocatePCA.refineSegments(pcaRound, refinementDC,
                                  collectEvaluationData=collectedSubclusters,retClusterer=pcaClusterer)

            # additionally perform most common values refinement
            moco = CropDistinct.countCommonValues(pcaRound)
            print([m.hex() for m in moco])
            refinedSM = list()
            for msg in pcaRound:
                croppedMsg = CropDistinct(msg, moco).split()
                refinedSM.append(croppedMsg)

            refinedSM = charRefinements(refinedSM)

            # refinedSM = refinedSegmentedMessages

            # TODO now needs recalculation of segment distances
        except ClusterAutoconfException as e:
            print("Initial clustering of the segments in the trace failed. The protocol in this trace cannot be inferred. "
                  "The original exception message was:\n", e)
            exit(10)
    runtimeRefinement = time.time() - startRefinement
    # # # # # # # # # # # # # # # # # # # # # # # #

    reportFolder = join(reportFolder, splitext(pcapbasename)[0])
    if not exists(reportFolder):
        makedirs(reportFolder)
    else:
        print("Adding report to existing folder:", reportFolder)

    # # # # # # # # # # # # # # # # # # # # # # # #
    # Distance Calculator
    print("Calculating Dissimilarities...")

    dc = MemmapDC(list(chain.from_iterable(refinedSM)))
    # # # # # # # # # # # # # # # # # # # # # # # #

    for msgsegs in refinedSM:
        comparator.pprint2Interleaved(msgsegs[0].message, [infs.nextOffset for infs in msgsegs])



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
    #     with open(join(reportFolder, "segmentclusters-" + splitext(pcapbasename)[0] + ".csv"), "a") as segfile:
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
        RelocatePCA.filterRelevantClusters([a.similarSegments for a in collectedSubclusters])
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
    #         clusterBounds = sc.relocateBoundaries(dc, kneedleSensitivity, comparator, reportFolder)
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
        for cid, sc in enumerate(collectedSubclusters):  # type: int, RelocatePCA
            if cid in relevantSubclusters:
                relocate = sc.relocateOffsets(reportFolder, pcapName, comparator)
                plotComponentAnalysis(sc,
                                      eigenVnV[cid] if cid in eigenVnV else numpy.linalg.eigh(sc.similarSegments.cov),
                                      relocate)
                # # print each segment marked in its message
                # print(sc.similarSegments.fieldtype, "*" if cid in relevantSubclusters else "")
                # for bs in sc.similarSegments.baseSegments:
                #     markSegNearMatch(inferredSegmentedMessages, bs)
        # # # # # # # # # # # # # # # # # # # # # # # # #
        print("Plot distances...")
        sdp = DistancesPlotter(specimens,
                               'distances-' + "nemezero-segments_DBSCAN-eps{:0.3f}-ms{:d}".format(
                                   pcaClusterer[0].eps, int(pcaClusterer[0].min_samples)),   # TODO: might be misleading for multiple iterations of PCA
                               False)
        clustermask = {segid: segL.similarSegments.fieldtype for segL in collectedSubclusters for segid in
                       dc.segments2index(bs for bs in segL.similarSegments.baseSegments if bs in dc.rawSegments)}
        # clustermask.update({segid: "Noise" for segid in dc.segments2index(noise)})
        clustermask.update({segid: "Noise" for segid in range(len(dc.segments)) if segid not in clustermask})  # TODO: not completely sure if that is the whole truth about the noise
        labels = numpy.array([clustermask[segid] for segid in range(len(dc.segments))])
        # plotManifoldDistances(dc.segments, dc.distanceMatrix, labels)
        sdp.plotSegmentDistances(dc, labels)
        sdp.writeOrShowFigure()
        del sdp
    # # # # # # # # # # # # # # # # # # # # # # # #





    # # # # # # # # # # # # # # # # # # # # # # # # #
    # write statistics ...
    scoFile = "subcluster-overview.csv"
    scoHeader = ["trace", "cluster label", "cluster size", "max segment length",
                 "interesting", "length diff", "# unique", "is char",
                 "min dissimilarity", "max dissimilarity", "mean dissimilarity"]

    #   ... for all subclusters, including the ones filtered out, for confirmation.
    for cid, sc in enumerate(collectedSubclusters):
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

        fn = join(reportFolder, scoFile)
        writeheader = not exists(fn)
        with open(fn, "a") as segfile:
            segcsv = csv.writer(segfile)
            if writeheader:
                segcsv.writerow(scoHeader)
            segcsv.writerow([
                pcapName, sc.similarSegments.fieldtype, len(sc.similarSegments.baseSegments),
                sc.similarSegments.length,
                repr(cid in relevantSubclusters), lendiff, len(uniqvals), ischar,
                min(internDis), max(internDis), numpy.mean(internDis)
            ])
    # # # # # # # # # # # # # # # # # # # # # # # # #
    segFn = "segmentclusters-" + splitext(pcapbasename)[0] + ".csv"
    with open(join(reportFolder, segFn), "a") as segfile:
        segcsv = csv.writer(segfile)
        for sc in collectedSubclusters:
            segcsv.writerows([
                [],
                ["# Cluster", sc.similarSegments.fieldtype, "Segments", len(sc.similarSegments.baseSegments)],
                ["-"*10]*4,
            ])
            segcsv.writerows(
                {(seg.bytes.hex(), seg.bytes) for seg in sc.similarSegments.baseSegments}
            )


    # # # # # # # # # # # # # # # # # # # # # # # #
    # Write FMS statistics
    inferenceTitle = tokenizer # "-".join([tokenizer, ])
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
    # # # # # # # # # # # # # # # # # # # # # # # #
    # Determine the amount of off-by-one errors in refined segmentation
    message2quality = DissectorMatcher.symbolListFMS(comparator, symbolsFromSegments(refinedSM))
    reportWriter.writeReport(message2quality, -1.0, specimens, comparator,
                             inferenceTitle + "_" + args.refinement + '-refined',
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








    if args.interactive:
        from tabulate import tabulate
        # globals().update(locals())
        IPython.embed()










