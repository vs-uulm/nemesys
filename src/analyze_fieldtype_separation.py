"""
Use groundtruth about field segmentation by dissectors and apply field type identification to them.

Takes a PCAP trace of a known protocol, dissects each message into their fields, and yields segments from each of them.
These segments get analyzed by the "value" analysis method which is used as feature to determine their similarity.
Real field types are separated using ground truth and the quality of this separation is visualized.
"""

import argparse, IPython
from os.path import isfile, basename
from itertools import chain
import numpy

from inference.templates import DistanceCalculator, Template, TemplateGenerator
from inference.segments import TypedSegment
from inference.analyzers import *
from inference.segmentHandler import groupByLength, segments2types, segments2clusteredTypes, \
    filterSegments
from utils.evaluationHelpers import annotateFieldTypes
from validation.dissectorMatcher import MessageComparator
from utils.loader import SpecimenLoader
from visualization.multiPlotter import MultiMessagePlotter
from visualization.singlePlotter import SingleMessagePlotter
from visualization.distancesPlotter import DistancesPlotter
from inference.formatRefinement import isPrintable, locateNonPrintable

debug = False

# fix the analysis method to VALUE
analysisTitle = 'value'
analyzerType = Value
analysisArgs = None
# fix the distance method to canberra
distance_method = 'canberra'




def filterChars(segments: List[MessageSegment], meanCorridor=(0x20, 0x7e), minLen=6):
    """
    Filter segments by some hypotheses about what might be a char sequence:
        1. All values are < 127 (0x7f)
        2. The sequence's values have a mean of between n and m, e. g. if 0x20 <= char <= 0x7e (printable chars)
        3. Segment length is >= 6/8/16 ?

    :param segments: List of segments to be filtered
    :param meanCorridor: Corridor of mean value that denotes a probable char sequence
    :param minLen: Minimum length of a segment to be condidered for hypothesis testing
    :return: Filtered segments: segments that hypothetically are chars
    """
    filtered = [seg for seg in segments
                if seg.length >= minLen
                and numpy.max(seg.values) < 0x7f
                and meanCorridor[0] <= numpy.mean([v for v in seg.values if v > 0x00]) <= meanCorridor[1]
                and 0.66 > len(locateNonPrintable(seg.bytes))/seg.length  # from smb one-char-many-zeros segments
                ]
    return filtered


def tPfPfN(hypothesis: List[MessageSegment], groundtruth: List[MessageSegment]):
    """

    results for 10000 msgs - meanCorridor=(0x20, 0x7e) - narrow meanCorridor=(50, 115):

    Trace: input/dns_ictf2010_deduped-9911-10000.pcap
    TP: 9367 FP: 0 FN 42
    narrow
    TP: 9367 FP: 0 FN 42

    Trace: input/nbns_SMIA20111010-one_deduped-1000.pcap
    TP: 1000 FP: 0 FN 55
    narrow
    TP: 1000 FP: 0 FN 55

    Trace: input/smb_SMIA20111010-one_deduped-10000.pcap
    TP: 2946 FP: 461 FN 1251
    narrow
    TP: 2946 FP: 454 FN 1251

    :param hypothesis:
    :param groundtruth:
    :return:
    """
    hypothesis = set(hypothesis)
    groundtruth = set(groundtruth)
    tp = hypothesis & groundtruth  # set intersection
    fp = hypothesis - groundtruth  # set difference
    fn = groundtruth - hypothesis
    return tp, fp, fn


def meanHisto(segments: List[MessageSegment]):
    segmentMeans = [numpy.mean([v for v in seg.values if v > 0x00]) for seg in segments ]

    smp = SingleMessagePlotter(specimens, "histo-charmeans")
    smp.histogram(segmentMeans, bins=numpy.linspace(0x20, 0x7f, 20))
    smp.text("min: {:.3f}\nmax: {:.3f}".format(numpy.min(segmentMeans), numpy.max(segmentMeans)))
    smp.writeOrShowFigure()
    # mmp = MultiMessagePlotter(specimens, 'histo-templatecenters', len(templates))

    return segmentMeans


def printChars(segments: List[MessageSegment]):
    from tabulate import tabulate
    table = [(chars.bytes.hex(), 100*len(locateNonPrintable(chars.bytes))/chars.length, chars.bytes.decode()) for chars in segments]
    print(tabulate(table))



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
    # print("Load messages...")
    specimens = SpecimenLoader(args.pcapfilename, 2, True)
    comparator = MessageComparator(specimens, 2, True, debug=debug)
    print("Trace:", specimens.pcapFileName)

    # segment messages according to true fields from the labels
    # print("Segmenting messages...")
    segmentedMessages = annotateFieldTypes(analyzerType, analysisArgs, comparator)
    segments = list(chain.from_iterable(segmentedMessages))
    # dc = DistanceCalculator(segments)

    typegroups = segments2types(segments)
    typelabels = list(typegroups.keys())

    filteredChars = filterChars(segments)
    # TODO make meancorridor narrower and compare 10000s tp, fp, fn
    filteredNarrow = filterChars(segments, meanCorridor=(50, 115))

    charskey = 'chars'

    if charskey in typelabels:
        # only non-null sequences quality as valid ground truth
        nnchars = [seg for seg in typegroups[charskey] if any(val > 0 for val in seg.values)]
        if len(nnchars) > 0:
            tp, fp, fn = tPfPfN(filteredChars, nnchars)
            print("TP:", len(tp), "FP:", len(fp), "FN", len(fn))
            ntp, nfp, nfn = tPfPfN(filteredNarrow, nnchars)
            print("narrow")
            print("TP:", len(ntp), "FP:", len(nfp), "FN", len(nfn))
            segmentMeans = meanHisto(nnchars)
            print("Means min: {:.3f} max: {:.3f}".format(numpy.min(segmentMeans), numpy.max(segmentMeans)))
        else:
            print("No non-zero char segments in trace. FP count:", len(filteredChars))
            if len(filteredChars) > 0:
                printChars(filteredChars)
                meanHisto(filteredChars)
    else:
        print("No char segments in trace. FP count:", len(filteredChars))
    print()

    # templates = TemplateGenerator.generateTemplatesForClusters(dc, [typegroups[ft] for ft in typelabels])

    # # labels of templates
    # labels = ['Noise'] * len(dc.segments)  # TODO check: list of segment indices (from raw segment list) per message
    # #                                           ^ here the question is, whether we like to print resolved segements or representatives
    # for l, t in zip(typelabels, templates):
    #     labels[dc.segments.index(t.medoid)] = l
    #
    # sdp = DistancesPlotter(specimens, 'distances-templatecenters', args.interactive)
    # sdp.plotSegmentDistances(dc, numpy.array(labels))
    # sdp.writeOrShowFigure()

    # # import matplotlib.pyplot as plt
    # mmp = MultiMessagePlotter(specimens, 'histo-templatecenters', len(templates))
    # for figIdx, (typlabl, typlate) in enumerate(zip(typelabels, templates)):
    #     # h_match histogram of distances to medoid for segments of typlate's type
    #     match = [di for di, of in typlate.distancesToMixedLength(dc)]
    #     # abuse template to get distances to non-matching field types
    #     filtermismatch = [typegroups[tl] for tl in typelabels if tl != typlabl]
    #     mismatchtemplate = Template(typlate.medoid, list(chain.from_iterable(filtermismatch)))
    #     # h_mimatch histogram of distances to medoid for segments that are not of typlate's type
    #     mismatch = [di for di, of in mismatchtemplate.distancesToMixedLength(dc)]
    #     # plot both histograms h overlapping (i.e. for each "bin" have two bars).
    #     # the bins denote ranges of distances from the medoid
    #     mmp.histoToSubfig(figIdx, [match, mismatch], bins=numpy.linspace(0, 1, 20), label=[typlabl, 'not ' + typlabl])
    # # plot in subfigures on one page
    # mmp.writeOrShowFigure()


    if args.interactive:
        IPython.embed()