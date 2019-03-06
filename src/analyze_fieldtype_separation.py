"""
Use groundtruth about field segmentation by dissectors and apply field type identification to them.

Takes a PCAP trace of a known protocol, dissects each message into their fields, and yields segments from each of them.
These segments get analyzed by the "value" analysis method which is used as feature to determine their similarity.
Real field types are separated using ground truth and the quality of this separation is visualized.
"""

import argparse, IPython
from os.path import isfile
from itertools import chain
import matplotlib.pyplot as plt
from tabulate import tabulate

from inference.analyzers import *
from inference.segmentHandler import segments2types, filterChars, searchSeqOfSeg
from inference.templates import DelegatingDC
from utils.baseAlgorithms import tril
from utils.evaluationHelpers import annotateFieldTypes
from validation.dissectorMatcher import MessageComparator
from utils.loader import SpecimenLoader
from visualization.singlePlotter import SingleMessagePlotter
from inference.formatRefinement import locateNonPrintable

debug = False

# fix the analysis method to VALUE
analysisTitle = 'value'
analyzerType = Value
analysisArgs = None
# fix the distance method to canberra
distance_method = 'canberra'

charskey = 'chars'



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
    table = [(chars.bytes.hex(), 100*len(locateNonPrintable(chars.bytes))/chars.length, chars.bytes.decode()) for chars in segments]
    print(tabulate(table))



def charsDistanceStatistics():
    # distances between all pairs from typegroups[charskey]
    matrix4chars = dc.representativesSubset(typegroups[charskey])[0]
    dist4chars = tril(matrix4chars)
    notChars = list(chain.from_iterable([group for tkey, group in typegroups.items() if tkey != charskey]))
    # and all pairs from typegroups[not charskey]
    matrixNotChars = dc.representativesSubset(notChars)[0]
    distNotChars = tril(matrixNotChars)
    # and all pairs between typegroups[charskey] and typegroups[not charskey]
    matrix4charsAndNot = dc.representativesSubset(typegroups[charskey], notChars)[0]
    dist4charsAndNot = tril(matrix4charsAndNot)

    # import IPython;IPython.embed()

    # make boxplots for all distance statistics on one page
    smp = SingleMessagePlotter(specimens, "distances-typegroups-chars")
    plt.title("Distances between all pairs...")
    plt.boxplot([dist4chars, distNotChars, dist4charsAndNot],
                labels=["... of chars", "... of NOT chars", "... between chars and not"])
    smp.writeOrShowFigure()


def charsValueMeanStatistics():
    # # make meancorridor narrower and compare 10000s tp, fp, fn
    # filteredNarrow = filterChars(segments, meanCorridor=(50, 115))

    if charskey in typelabels:
        # only non-null sequences quality as valid ground truth
        nnchars = [seg for seg in typegroups[charskey] if any(val > 0 for val in seg.values)]
        if len(nnchars) > 0:
            tp, fp, fn = tPfPfN(filteredChars, nnchars)
            print("TP:", len(tp), "FP:", len(fp), "FN", len(fn))
            # ntp, nfp, nfn = tPfPfN(filteredNarrow, nnchars)
            # print("narrow")
            # print("TP:", len(ntp), "FP:", len(nfp), "FN", len(nfn))
            segmentMeans = meanHisto(nnchars)
            print("Means min: {:.3f} max: {:.3f}".format(numpy.min(segmentMeans), numpy.max(segmentMeans)))
        else:
            print("No non-zero char segments in trace. FP count:", len(filteredChars))
            # if len(filteredChars) > 0:
            #     printChars(filteredChars)
            #     meanHisto(filteredChars)
    else:
        print("No char segments in trace. FP count:", len(filteredChars))
    print()


def ffffffff00():
    zeros = searchSeqOfSeg(segments, b"\x00")
    zero1b = [z for z in zeros if z.length == 1]
    ffffs = searchSeqOfSeg(segments, b"\xff\xff\xff")

    pairs00ff = dc.representativesSubset(zero1b, ffffs)
    # resulting distance 1.0
    print(pairs00ff)

    # find segment left and right of zeros/fffs
    leftz = {z: [segl for msg in segmentedMessages for segl, segr in zip(msg[:-1], msg[1:]) if segr == z][0] for z in
             zero1b if z.offset > 0}
    rightz = {z: [segr for msg in segmentedMessages for segl, segr in zip(msg[:-1], msg[1:]) if segl == z][0] for z in
             zero1b if z.offset < len(z.message.data) - z.length}
    leftf = {z: [segl for msg in segmentedMessages for segl, segr in zip(msg[:-1], msg[1:]) if segr == z][0] for z in
             ffffs if z.offset > 0}
    rightf = {z: [segr for msg in segmentedMessages for segl, segr in zip(msg[:-1], msg[1:]) if segl == z][0] for z in
             ffffs if z.offset < len(z.message.data) - z.length}

    pairs00left = dc.representativesSubset( *zip( *leftz.items() ) )
    pairs00right = dc.representativesSubset( *zip( *rightz.items() ) )
    pairsffleft = dc.representativesSubset( *zip( *leftf.items() ) )
    pairsffright = dc.representativesSubset( *zip( *rightf.items() ) )

    # means for smb-1000:
    print(pairs00left[0].mean())
    # 0.79837072649572638
    print(pairs00right[0].mean())
    # 0.62932046761634297
    print(pairsffleft[0].mean())
    # 0.84073738878912407
    print(pairsffright[0].mean())
    # 1.0

    plt.title("Distances between...")
    plt.boxplot([pairs00left[0][0], pairs00right[0][0], pairsffleft[0][0], pairsffright[0][0]],
                labels=["... 0x00 and its left",
                        "... 0x00 and its right",
                        "... 0xff*4 and its left",
                        "... 0x00*4 and its right"])
    plt.show()









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
    print("Trace:", specimens.pcapFileName)

    # # segment messages according to true fields from the labels
    # # print("Segmenting messages...")
    # segmentedMessages = annotateFieldTypes(analyzerType, analysisArgs, comparator)
    # segments = list(chain.from_iterable(segmentedMessages))
    #
    # # typegroups = segments2types(segments)
    # # typelabels = list(typegroups.keys())
    # # filteredChars = filterChars(segments)
    #
    # # # evaluate additional filtering criteria: mean of values
    # # charsValueMeanStatistics()
    #
    # dc = DelegatingDC(segments)
    #
    # # # boxplots for char distance statistics
    # # charsStatistics()



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

    from validation.messageParser import ParsedMessage

    pms = [pm for pm in comparator.parsedMessages.values()]  # type: List[ParsedMessage]
    msgtypes = {(pm.protocolname, pm.messagetype) for pm in pms}
    print(tabulate(msgtypes))

    if args.interactive:
        IPython.embed()




