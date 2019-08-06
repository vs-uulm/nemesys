"""
segment messages by NEMESYS and cluster
"""

import argparse, IPython
from os.path import isfile, basename, join, splitext
from math import log
from typing import Union

from inference.templates import DBSCANsegmentClusterer, FieldTypeTemplate, Template, TypedTemplate
from inference.fieldTypes import BaseTypeMemento, RecognizedField, RecognizedVariableLengthField
from inference.segments import TypedSegment, HelperSegment
from inference.segmentHandler import symbolsFromSegments
from inference.analyzers import *
from visualization.distancesPlotter import DistancesPlotter
from visualization.multiPlotter import MultiMessagePlotter, PlotGroups
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
    exit()
    # # # # # # # # # # # # # # # # # # # # # # # #



    # # # # # # # # # # # # # # # # # # # # # # # #
    clusterer = DBSCANsegmentClusterer(dc)
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




    # # # # # # # # # # # # # # # # # # # # # # # #
    # TODO mark (at least) exact segment matches with the true data type
    typedMatchSegs = list()
    for seg in dc.segments:
        if isinstance(seg, MessageSegment):
            typedMatchSegs.append(comparator.segment2typed(seg))
        elif isinstance(seg, Template):
            machingType = None
            typedBasesegments = list()
            for bs in seg.baseSegments:
                tempTyped = comparator.segment2typed(bs)
                if not isinstance(tempTyped, TypedSegment):
                    # print("At least one segment in template is not a field match.")
                    # markSegNearMatch(tempTyped)
                    machingType = None
                    break
                elif machingType == tempTyped.fieldtype or machingType is None:
                    machingType = tempTyped.fieldtype
                    typedBasesegments.append(tempTyped)
                else:
                    # print("Segment's matching field types are not the same in template, e. g., {} and {} ({})".format(
                    #     machingType, tempTyped.fieldtype, tempTyped.bytes.hex()
                    # ))
                    machingType = None
                    break
            if machingType is None:
                typedMatchSegs.append(seg)
            else:
                typedMatchSegs.append(TypedTemplate(seg.values, typedBasesegments, seg._method))
    # # # # # # # # # # # # # # # # # # # # # # # #




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










