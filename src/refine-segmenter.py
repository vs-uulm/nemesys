import argparse
from os import makedirs
from os.path import isfile, basename, join, splitext, exists
from typing import List

import IPython

from inference.segmentHandler import originalRefinements, baseRefinements, pcaPcaRefinements, pcaMocoRefinements, \
    isExtendedCharSeq, symbolsFromSegments
from inference.segments import MessageAnalyzer, MessageSegment
from validation import reportWriter
from validation.dissectorMatcher import DissectorMatcher
from utils.evaluationHelpers import analyses, cacheAndLoadDC, annotateFieldTypes, reportFolder

debug = False

# fix the analysis method to VALUE
analysisTitle = 'value'
# fix the distance method to canberra
distance_method = 'canberra'
# use NEMESYS segments
tokenizer = 'nemesys'
# tokenizer = '4bytesfixed'

refinementMethods = [
    "raw",      # unrefined NEMESYS
    "original", # WOOT2018 paper
    "base",     # moco+splitfirstseg
    "PCA",      # 2-pass PCA
    "PCA1",     # 1-pass PCA
    "PCAmoco"   # 2-pass PCA+moco
    ]



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
    parser.add_argument('-r', '--refinement', help='Select segment refinement method.', choices=refinementMethods)
    args = parser.parse_args()

    if not isfile(args.pcapfilename):
        print('File not found: ' + args.pcapfilename)
        exit(1)
    pcapbasename = basename(args.pcapfilename)
    withPlots = args.with_plots

    analyzerType = analyses[analysisTitle]
    analysisArgs = None

    # # # # # # # # # # # # # # # # # # # # # # # #
    # cache/load the DistanceCalculator to/from the filesystem
    #

    doCache = True
    if args.refinement == "original":
        refinementCallback = originalRefinements
    elif args.refinement == "base":
        refinementCallback = baseRefinements
    elif args.refinement == "PCA":
        refinementCallback = pcaPcaRefinements
    elif args.refinement == "PCAmoco":
        refinementCallback = pcaMocoRefinements
    elif args.refinement == "raw":
        refinementCallback = None
    else:
        print("Unknown refinement", args.refinement, "\nAborting")
        exit(2)
    specimens, comparator, inferredSegmentedMessages, dc, segmentationTime, dist_calc_segmentsTime = cacheAndLoadDC(
        args.pcapfilename, analysisTitle, tokenizer, debug, analyzerType, analysisArgs, args.sigma, True,
        refinementCallback=refinementCallback, disableCache=not doCache
        )

    trueSegmentedMessages = {msgseg[0].message: msgseg
                             for msgseg in annotateFieldTypes(analyzerType, analysisArgs, comparator)}
    # # # # # # # # # # # # # # # # # # # # # # # #

    reportFolder = join(reportFolder, splitext(pcapbasename)[0])
    if not exists(reportFolder):
        makedirs(reportFolder)



    # Experiment: How to slice segments from between zero values.


    # mSlice = (0,40) # dhcp start
    # mSlice = (220, 1000)  # dhcp middle
    mSlice = (0, 1000)
    zeroSegments = list()
    combinedRefinedSegments = list()
    for msgsegs in inferredSegmentedMessages:
        zeroBounds = list()
        mdata = msgsegs[0].message.data  # type: bytes

        # all transitions from 0 to any and vice versa + message start and end
        for bi, bv in enumerate(mdata[1:], 1):
            if bv == 0 and mdata[bi-1] != 0 \
                    or bv != 0 and mdata[bi-1] == 0:
                zeroBounds.append(bi)
        zeroBounds = [0] + zeroBounds + [len(mdata)]

        # remove boundaries of short zero sequences to merge to previous or next non-zero segment
        minCharLen = 6  # 6
        zeroBounds = sorted(set(zeroBounds))
        zBCopy = zeroBounds.copy()
        for zi, zb in enumerate(zBCopy[:-1]):  # omit message end bound
            if mdata[zb] == 0:
                nzb = zBCopy[zi + 1]
                # ... there are only one or two zeros in a row ...
                if zb + 2 >= nzb:
                    # if chars are preceding, add zero to previous
                    if isExtendedCharSeq(mdata[max(0,zb-minCharLen):zb], minLen=minCharLen): # \
                            # or zb > 0 and MessageAnalyzer.nibblesFromBytes(mdata[zb-1:zb])[1] == 0:  # or the least significant nibble of the preceding byte is zero
                        zeroBounds.remove(zb)
                    # otherwise to next
                    elif zBCopy[zi+1] < len(mdata):
                        zeroBounds.remove(zBCopy[zi+1])

        # generate zero-bounded segments from bounds
        ms = list()
        for segStart, segEnd in zip(zeroBounds[:-1], zeroBounds[1:]):
            ms.append(MessageSegment(msgsegs[0].analyzer, segStart, segEnd - segStart))
        zeroSegments.append(ms)


        # integrate original inferred bounds with zero segments, zero bounds have precedence
        combinedMsg = list()  # type: List[MessageSegment]
        infMarginOffsets = [infs.nextOffset for infs in msgsegs
                            if infs.nextOffset - 1 not in zeroBounds and infs.nextOffset + 1 not in zeroBounds]
        remZeroBounds = [zb for zb in zeroBounds if zb not in infMarginOffsets]
        combinedBounds = sorted(infMarginOffsets + remZeroBounds)
        startEndMap = {(seg.offset, seg.nextOffset) : seg for seg in msgsegs}
        analyzer = msgsegs[0].analyzer
        for bS, bE in zip(combinedBounds[:-1], combinedBounds[1:]):
            # unchanged
            if (bS, bE) in startEndMap:
                combinedMsg.append(startEndMap[(bS, bE)])
            else:
                nseg = MessageSegment(analyzer, bS, bE-bS)
                combinedMsg.append(nseg)
        # final assertion of complete representation of message by the new segments
        msgbytes = b"".join([seg.bytes for seg in combinedMsg])
        assert msgbytes == mdata, "segment sequence does not match message bytes"
        combinedRefinedSegments.append(combinedMsg)

        # # compare zeros-slicing to nemesys segments:
        # comparator.pprint2Interleaved(msgsegs[0].message,
        #                               # combinedBounds,
        #                               [infs.nextOffset for infs in baseCombinedRefSegs[-1]],
        #                               # mark=betweenZeros,
        #                               messageSlice=mSlice)

    baseCombinedRefSegs = baseRefinements(combinedRefinedSegments)

    for msgsegs in baseCombinedRefSegs:
        comparator.pprint2Interleaved(msgsegs[0].message, [infs.nextOffset for infs in msgsegs])

    #
    # if its a single zero: add to previous slice if (n bytes) before are chars (extended definition),
    #   otherwise add to subsequent segment.

    # single non-zeros?

    # do nemesys on the resulting non-zero slices to create segments.

    # then refine by PCA, ...



    # write FMS
    symbols = symbolsFromSegments(baseCombinedRefSegs)
    message2quality = DissectorMatcher.symbolListFMS(comparator, symbols)
    exactcount, offbyonecount, offbymorecount = reportWriter.countMatches(message2quality.values())
    minmeanmax = reportWriter.getMinMeanMaxFMS([round(q.score, 3) for q in message2quality.values()])
    print("\nFormat Match Scores", specimens.pcapFileName)
    print("  (min, mean, max): ", *minmeanmax)
    print("near matches")
    print("  off-by-one:", offbyonecount)
    print("  off-by-more:", offbymorecount)
    print("exact matches:", exactcount)

    if args.interactive:
        from tabulate import tabulate
        # globals().update(locals())
        IPython.embed()




