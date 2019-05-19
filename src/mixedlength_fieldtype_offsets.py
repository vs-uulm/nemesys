"""
Use groundtruth about field segmentation by dissectors and apply field type identification to them.

Takes a PCAP trace of a known protocol, dissects each message into their fields, and yields segments from each of them.
These segments get analyzed by the "value" analysis method which is used as feature to determine their similarity.
Real field types are separated using ground truth and the quality of this separation is visualized.
"""

import argparse, IPython
from os.path import isfile, basename, splitext
from itertools import chain
from tabulate import tabulate
from math import ceil, floor

from inference.templates import DistanceCalculator
from inference.segments import TypedSegment
from inference.analyzers import *
from inference.segmentHandler import groupByLength, segments2types, segments2clusteredTypes, \
    filterSegments
from utils.evaluationHelpers import annotateFieldTypes
from validation.dissectorMatcher import MessageComparator
from utils.loader import SpecimenLoader
from visualization.multiPlotter import MultiMessagePlotter
from visualization.distancesPlotter import DistancesPlotter


debug = False

# fix the analysis method to VALUE
analysisTitle = 'value'
analyzerType = Value
analysisArgs = None
# fix the distance method to canberra
distance_method = 'canberra'
# tokenizer = 'tshark-unfiltered' # obscure parameters to DC
# tokenizer = 'tshark-deduplicated+dezeroed'  # with offsetCutoff = 6
tokenizer = 'tshark-deduplicated+dezeroed-nooffcut'





if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Analyze fields as segments of messages and plot field type identification quality.')
    parser.add_argument('pcapfilename', help='Filename of the PCAP to load.')
    parser.add_argument('-i', '--interactive', help='Show interactive plot instead of writing output to file and '
                                                    'open ipython prompt after finishing the analysis.',
                        action="store_true")
    args = parser.parse_args()
    if not isfile(args.pcapfilename):
        print('File not found: ' + args.pcapfilename)
        exit(1)

    try:
        segmentedMessages, comparator, dc = DistanceCalculator.loadCached(analysisTitle, tokenizer, args.pcapfilename)
        # chainedSegments = list(chain.from_iterable(segmentedMessages))
        chainedSegments = filterSegments(chain.from_iterable(segmentedMessages))
    except (TypeError, FileNotFoundError) as e:
        if isinstance(e, TypeError):
            print('Loading of cached distances failed. Continuing:')
        doCacheDC = isinstance(e, FileNotFoundError)
        segmentedMessages, comparator, dc = None, None, None

    if (segmentedMessages, comparator, dc) == (None, None, None):
        # dissect and label messages
        print("Load messages from {}...".format(args.pcapfilename))
        specimens = SpecimenLoader(args.pcapfilename, 2, True)
        comparator = MessageComparator(specimens, 2, True, debug=debug)

        # segment messages according to true fields from the labels
        print("Segmenting messages...")
        segmentedMessages = annotateFieldTypes(analyzerType, analysisArgs, comparator)
        # chainedSegments = list(chain.from_iterable(segmentedMessages))
        chainedSegments = filterSegments(chain.from_iterable(segmentedMessages))  # type: List[TypedSegment]
        print("Calculate distance for {} segments...".format(len(chainedSegments)))
        DistanceCalculator.offsetCutoff = None
        dc = DistanceCalculator(chainedSegments)
        # noinspection PyUnboundLocalVariable
        if doCacheDC:
            try:
                dc.saveCached(analysisTitle, tokenizer, comparator, segmentedMessages)
                print("Distances saved to cache file.")
            except FileExistsError as e:
                print(e)

    # noinspection PyUnboundLocalVariable
    typegroups = segments2types(chainedSegments)
    typelabels = list(typegroups.keys())

    stats2plot = list()
    maxpairs = 100
    # statistics per group:
    # determine relevant pairs by assuming (only) within one field type there are valid distances/offsets
    for typlab, typgrp in typegroups.items():
        segidcs = dc.segments2index(typgrp)
        segpairs = dc.offsets.keys()
        typpairs = [sp for sp in segpairs if sp[0] in segidcs and sp[1] in segidcs]
        pairoffsets = {pair: dc.offsets[pair] for pair in typpairs}
        pairsimi = {pair: dc.distanceMatrix[pair] for pair in typpairs}

        # "pair (indices)", "distance", "remaining trail", "innerlen", "outerlen"
        offsetpairs = dict()
        for pair, offset in pairoffsets.items():
            if offset not in offsetpairs:
                offsetpairs[offset] = list()
            innerlen = min(dc.segments[pair[0]].length, dc.segments[pair[1]].length)
            outerlen = max(dc.segments[pair[0]].length, dc.segments[pair[1]].length)
            offsetpairs[offset].append((
                pair, pairsimi[pair],
                innerlen + offset - outerlen, innerlen, outerlen
            ))

        print("\nType:", typlab)
        if len(offsetpairs) > 0:
            alloffsets = len(pairoffsets)
            # for offset, pairs in offsetpairs.items():
            #     print("Offset", offset, "({} pairs truncated to {})".format(len(offsetpairs), maxpairs)
            #         if maxpairs < len(offsetpairs) else "")
            #     print(tabulate(pairs[:100], headers=["pair", "dist", "trail", "innerlen", "outerlen"]))

            simimeans = sorted([(o, numpy.mean([s[1] for s in p])) for o, p in offsetpairs.items()], key=lambda k: k[0])
            simimins = sorted([(o, numpy.min([s[1] for s in p])) for o, p in offsetpairs.items()], key=lambda k: k[0])
            simimaxs = sorted([(o, numpy.max([s[1] for s in p])) for o, p in offsetpairs.items()], key=lambda k: k[0])
            # import matplotlib.pyplot as plt
            # plt.scatter([simi[0] for simi in simimeans], [simi[1] for simi in simimeans])
            # plt.scatter([simi[0] for simi in simimins], [simi[1] for simi in simimins], alpha=.3)
            # plt.scatter([simi[0] for simi in simimaxs], [simi[1] for simi in simimaxs], alpha=.3)
            # plt.savefig("offset2distance-{}-{}-.pdf".format(splitext(basename(args.pcapfilename))[0], typlab))
            stats2plot.append((
                typlab,
                ([simi[0] for simi in simimeans], [simi[1] for simi in simimeans]),
                ([simi[0] for simi in simimins], [simi[1] for simi in simimins]),
                ([simi[0] for simi in simimaxs], [simi[1] for simi in simimaxs])
            ))

            #        %/p    f  l
            count = { 25 : [0, 0],
                      10 : [0, 0],
                       6 : [0, 0],
                       4 : [0, 0],
                       2 : [0, 0]
                    }
            for offset, pairs in offsetpairs.items():
                for pair, dist, trail, ilen, olen in pairs:
                    #        %/p    first              last
                    verge = { 25 : [ceil(olen * 0.25), floor(olen * (1 - 0.25) - ilen)],  # the end of the inner segment should fall beyond the verge
                              10 : [ceil(olen * 0.1),  floor(olen * 0.9 - ilen)],  # the end of the inner segment should fall beyond the verge
                               6 : [6,                 olen - 6 - ilen],
                               4 : [4,                 olen - 4 - ilen],
                               2 : [2,                 olen - 2 - ilen]
                            }
                    for k in count.keys():
                        if offset <= verge[k][0]:
                            count[k][0] += 1
                        elif offset >= verge[k][1]:  # do not count double if inner segment is larger than outer - 2*verge
                            count[k][1] += 1
            percent = [(m, oc[0]/alloffsets, oc[1]/alloffsets) for m, oc in count.items()]
            print(typlab + ":")
            print(tabulate([(m, f, l, *count[m]) for m, f, l in percent], headers=[
                "verge", "%first", "%last", "#first", "#last"]))
            # print(tabulate([(m, *oc) for m, oc in count.items()] , headers=["verge", "#first", "#last"]))
            # print()
            # print(tabulate(percent, headers=["verge", "%first", "%last"]))

            # IPython.embed()
        else:
            print("No mixed-length segment pairs.")


    mmp = MultiMessagePlotter(comparator.specimens, "offset2distance", len(stats2plot))
    mmp.scatterInEachAx(([s2p[1] for s2p in stats2plot]), marker="o") # means
    mmp.scatterInEachAx(([s2p[2] for s2p in stats2plot]), marker="v")  # mins
    mmp.scatterInEachAx(([s2p[3] for s2p in stats2plot]), marker="^")  # maxs
    mmp.nameEachAx([s2p[0] for s2p in stats2plot]) # labels
    mmp.writeOrShowFigure()
    # TODO reintegrage statistics from across field types
    # IPython.embed()

    # # import matplotlib.pyplot as plt
    # mmp = MultiMessagePlotter(specimens, 'histo-templatecenters', len(templates))
    # for figIdx, (typlabl, typlate) in enumerate(zip(typelabels, templates)):
    #     # h_match histogram of distances to medoid for segments of typlate's type
    #     match = [di for di, of in typlate.distancesToMixedLength(tg)]
    #     # abuse template to get distances to non-matching field types
    #     filtermismatch = [typegroups[tl] for tl in typelabels if tl != typlabl]
    #     mismatchtemplate = Template(typlate.medoid, list(chain.from_iterable(filtermismatch)))
    #     # h_mimatch histogram of distances to medoid for segments that are not of typlate's type
    #     mismatch = [di for di, of in mismatchtemplate.distancesToMixedLength(tg)]
    #     # plot both histograms h overlapping (i.e. for each "bin" have two bars).
    #     # the bins denote ranges of distances from the medoid
    #     mmp.histoToSubfig(figIdx, [match, mismatch], bins=numpy.linspace(0, 1, 20), label=[typlabl, 'not ' + typlabl])
    # # plot in subfigures on one page
    # mmp.writeOrShowFigure()


    if args.interactive:
        IPython.embed()










