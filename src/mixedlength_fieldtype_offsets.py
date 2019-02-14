"""
Use groundtruth about field segmentation by dissectors and apply field type identification to them.

Takes a PCAP trace of a known protocol, dissects each message into their fields, and yields segments from each of them.
These segments get analyzed by the "value" analysis method which is used as feature to determine their similarity.
Real field types are separated using ground truth and the quality of this separation is visualized.
"""

import argparse, IPython
from os.path import isfile
from itertools import chain
from tabulate import tabulate

from inference.templates import DistanceCalculator
from inference.segments import TypedSegment
from inference.analyzers import *
from inference.segmentHandler import annotateFieldTypes, groupByLength, segments2types, segments2clusteredTypes, \
    filterSegments
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

    # dissect and label messages
    print("Load messages...")
    specimens = SpecimenLoader(args.pcapfilename, 2, True)
    comparator = MessageComparator(specimens, 2, True, debug=debug)

    # segment messages according to true fields from the labels
    print("Segmenting messages...", end=' ')
    segmentedMessages = annotateFieldTypes(analyzerType, analysisArgs, comparator)
    filteredSegments = list(chain.from_iterable(segmentedMessages))
    dc = DistanceCalculator(filteredSegments)

    typegroups = segments2types(filteredSegments)
    typelabels = list(typegroups.keys())

    maxpairs = 100
    for typlab, typgrp in typegroups.items():
        segidcs = dc.segments2index(typgrp)
        segpairs = dc.offsets.keys()
        typpairs = [sp for sp in segpairs if sp[0] in segidcs and sp[1] in segidcs]
        pairoffsets = {pair: dc.offsets[pair] for pair in typpairs}
        pairsimi = {pair: dc.distanceMatrix[pair] for pair in typpairs}

        offsetpairs = dict()
        for pair, offset in pairoffsets.items():
            if offset not in offsetpairs:
                offsetpairs[offset] = list()
            innerlen = min(dc.segments[pair[0]].length, dc.segments[pair[1]].length)
            outerlen = max(dc.segments[pair[0]].length, dc.segments[pair[1]].length)
            offsetpairs[offset].append((pair, pairsimi[pair], innerlen + offset - outerlen, innerlen, outerlen))

        print("\nType:", typlab)
        if len(offsetpairs) > 0:
            for offset, pairs in offsetpairs.items():
                print("Offset", offset, "({} pairs truncated to {})".format(len(offsetpairs), maxpairs)
                    if maxpairs < len(offsetpairs) else "")
                print(tabulate(pairs[:100], headers=["pair", "dist", "rest", "innerlen", "outerlen"]))

            simimeans = sorted([(o, numpy.mean([s[1] for s in p])) for o, p in offsetpairs.items()], key=lambda k: k[0])
            simimins = sorted([(o, numpy.min([s[1] for s in p])) for o, p in offsetpairs.items()], key=lambda k: k[0])
            simimaxs = sorted([(o, numpy.max([s[1] for s in p])) for o, p in offsetpairs.items()], key=lambda k: k[0])

            import matplotlib.pyplot as plt
            plt.plot([simi[0] for simi in simimeans], [simi[1] for simi in simimeans])
            plt.plot([simi[0] for simi in simimins], [simi[1] for simi in simimins], alpha=.3)
            plt.plot([simi[0] for simi in simimaxs], [simi[1] for simi in simimaxs], alpha=.3)
            plt.show()
        else:
            print("No mixed-length segment pairs.")

        IPython.embed()



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










