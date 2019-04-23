"""
Characterize subsequences of messages by multiple metrics and visualize them.

To find valid features for identifying field boundaries and to distinguish between field types.
"""
import argparse

from os.path import isfile, basename
from itertools import chain
from collections import Counter
from tabulate import tabulate
import IPython

import inference.segmentHandler as sh
import utils.evaluationHelpers
from inference.formatRefinement import CropDistinct, CumulativeCharMerger
from utils.loader import SpecimenLoader
from inference.analyzers import *
from inference.segments import MessageSegment, TypedSegment
from inference.templates import DistanceCalculator, DBSCANsegmentClusterer
from validation.dissectorMatcher import MessageComparator
from visualization.multiPlotter import MultiMessagePlotter
from visualization.distancesPlotter import DistancesPlotter

from characterize_fieldtypes import analyses, labelForSegment
from utils.evaluationHelpers import plotMultiSegmentLines, sigmapertrace
from inference.segmentHandler import segments2clusteredTypes, filterSegments


def removeIdenticalLabels(plt):
    handles, labels = plt.gca().get_legend_handles_labels()
    newLabels, newHandles = [], []
    for handle, label in zip(handles, labels):
        if label not in newLabels:
            newLabels.append(label)
            newHandles.append(handle)
    plt.legend(newHandles, newLabels)


def plotLinesSegmentValues(segments: List[Tuple[str, MessageSegment]]):
    import matplotlib.cm
    import matplotlib.pyplot as plt

    for lab, seg in segments:
        color = hash(lab) % 8  # TODO make more collision resistant
        plt.plot(seg.values, c=matplotlib.cm.Set1(color + 1), alpha=0.4, label=lab)

    removeIdenticalLabels(plt)

    # plt.legend(scatterpoints=1, loc='best', shadow=False)
    plt.show()
    plt.clf()






if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Characterize subsequences of messages by multiple metrics and visualize them.')
    parser.add_argument('pcapfilename', help='pcapfilename') # , nargs='+')
    parser.add_argument('-i', '--interactive', help='show interactive plot instead of writing output to file.',
                        action="store_true")
    parser.add_argument('--analysis', type=str, default="value",
                        help='The kind of analysis to apply on the messages. Default is "value".'
                             ' Available methods are: ' + ', '.join(analyses.keys()))
    parser.add_argument('--parameters', '-p', help='Parameters for the analysis.')
    parser.add_argument('--distances', '-d', help='Plot distances instead of features.',
                        action="store_true")
    parser.add_argument('--count', '-c', help='Count common values of features.',
                        action="store_true")
    args = parser.parse_args()

    #for pcap in args.pcapfilename:
    if not isfile(args.pcapfilename):
        print('File not found: ' + args.pcapfilename)
        exit(1)
    analyzerType = analyses[args.analysis]
    analysisArgs = args.parameters

    pcapbasename = basename(args.pcapfilename)
    sigma = sigmapertrace[pcapbasename] if not args.parameters and pcapbasename in sigmapertrace else \
        0.9 if not args.parameters else args.parameters

    # dissect and label messages
    print("Loading messages...")
    specimens = SpecimenLoader(args.pcapfilename, layer=2, relativeToIP=True)
    comparator = MessageComparator(specimens, layer=2, relativeToIP=True)

    # segment messages according to true fields from the labels
    print("Segmenting messages...", end=' ')
    segmentsPerMsg = sh.bcDeltaGaussMessageSegmentation(specimens, sigma)
    segmentedMessages = sh.refinements(segmentsPerMsg)
    segments = list(chain.from_iterable(segmentedMessages))
    print("done.")

    analysisTitle = "{}{}".format(args.analysis, "" if not analysisArgs else " ({})".format(analysisArgs))



    # count common values
    if args.count:
        # segcnt = Counter([seg.bytes for seg in segments])
        # moco25 = segcnt.most_common(25)
        # print(tabulate([(featvect.hex(), cntr,
        #                  sum([msg.data.count(featvect) for msg in specimens.messagePool.keys()]),
        #                  sum([seg.bytes.count(featvect) for seg in segments])
        #                  ) for featvect, cntr in moco25],
        #     headers= ['Feature', 'Count (Inf)', 'Count (Msg)', 'Count (Seg)']))
        # # [sum([msg.data.count(moco) for msg in specimens.messagePool.keys()]) for moco in moco25]
        #
        # # # False Negatives - when only searching for full containment within segments
        # # featvect = bytes.fromhex("8182")
        # # inseg = [seg.message for seg in segments if seg.bytes.count(featvect) > 0]
        # # [msg.data.hex() for msg in set(inmsg) - set(inseg)]
        #
        #
        # # # determine drop (max gradient)
        # # mocoCnt = [cntr for featvect, cntr in segcnt.most_common()]
        # # diff2nd = numpy.diff(mocoCnt, 2)
        # # diff2ndmiX = diff2nd.argmax()
        # # steepdrop = mocoCnt[diff2ndmiX]
        # # # --> does not work at all for dns-1000: drop after the first entry, should be at 6 or so.
        # # from matplotlib import pyplot as plt
        # # plt.axhline(steepdrop)
        # # plt.plot(list(diff2nd[:25]))
        # # plt.plot([c for b, c in moco25])
        # # plt.text(0,0,"x {}; y {}".format(diff2ndmiX, steepdrop))
        # # plt.show()
        # # # so:
        # # # use message-amount-dependent threshold for frequency:
        # segFreq = segcnt.most_common()
        # freqThre = .2 * len(segmentedMessages)
        # thre = 0
        # while segFreq[thre][1] > freqThre:
        #     thre += 1
        # moco = [fv for fv, ct in segFreq[:thre] if set(fv) != {0}]  # omit \x00-sequences
        moco = CropDistinct.countCommonValues(segmentedMessages)

        # split segment to reinforce most common values
        newstuff = list()
        for msg in segmentedMessages:
            crop = CropDistinct(msg, moco)
            newmsg = crop.split()
            newstuff.append(newmsg)

            if newmsg != msg:
                if not b"".join([seg.bytes for seg in msg]) == b"".join([seg.bytes for seg in newmsg]):
                    print("\nINCORRECT SPLIT!\n")
                    print(" ".join([seg.bytes.hex() for seg in msg]))
                    print(" ".join([seg.bytes.hex() for seg in newmsg]))
                    print()

        newstuff2 = list()
        for msg in newstuff:
            charmerge = CumulativeCharMerger(msg)
            newmsg = charmerge.merge()
            newstuff2.append(newmsg)
            if newmsg != msg:
                if not b"".join([seg.bytes for seg in msg]) == b"".join([seg.bytes for seg in newmsg]):
                    print("\nINCORRECT SPLIT!\n")

                pm = comparator.parsedMessages[comparator.messages[msg[0].message]]
                print("DISSECT", " ".join([field for field in pm.getFieldValues()]))
                print("NEMESYS", " ".join([seg.bytes.hex() for seg in msg]))
                print("CHRMRGE", " ".join([seg.bytes.hex() for seg in newmsg]))
                print()
                # if newmsg[0].length > 2:
                #     from inference.segmentHandler import isExtendedCharSeq
                #     IPython.embed()


    if args.interactive:
        IPython.embed()

    exit()


    # # TODO for each cluster: count the number of fields per type (histograms)
    # for cluster in clusters:
    #     cntr = Counter()
    #     cntr.update([segment.fieldtype for segment in cluster])
    #     print("Cluster info: ", len(cluster), " segments of length", cluster[0].length)
    #     pprint(cntr)
    #     print()


    exit()

    # statistics per segment
    meanSegments = sh.segmentMeans(segmentedMessages)
    varSegments = sh.segmentStdevs(segmentedMessages)

    # plot
    mmp = MultiMessagePlotter(specimens, args.analysis, 5, 8, args.interactive)
    # TODO place labels with most common feature and/or byte values in distance graph

    # mmp.plotSegmentsInEachAx(segmentedMessages)
    # mmp.plotSegmentsInEachAx(meanSegments)
    # TODO use transported ftype to color the segments in the plot afterwards.
    mmp.plotSegmentsInEachAx(varSegments)
    mmp.writeOrShowFigure()

    # IPython.embed()























