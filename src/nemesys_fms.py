"""
Infer messages from PCAPs by the NEMESYS approach (BCDG-segmentation)
and write FMS and other evaluation data to report.

Usenix WOOT 2018.
"""

import argparse, time
from os.path import isfile, join, splitext, basename, abspath, isdir
from os import makedirs

import matplotlib.pyplot as plt

from nemere.validation.dissectorMatcher import MessageComparator, FormatMatchScore, DissectorMatcher
from nemere.utils.loader import SpecimenLoader
from nemere.inference.analyzers import *
from nemere.inference.segmentHandler import bcDeltaGaussMessageSegmentation, originalRefinements, \
    symbolsFromSegments
from nemere.utils import reportWriter
from nemere.utils.evaluationHelpers import sigmapertrace

debug = False
"""Some modules and methods contain debug output that can be activated by this flag."""


def mapQualities2Messages(m2q: Dict[AbstractMessage, FormatMatchScore]) \
        -> Dict[float, List[AbstractMessage]]:
    """
    Create a mapping from FMS values to messages of this quality aspects.

    :param m2q: A mapping of Messages to their quality aspects.
    :return: A mapping of FMS (rounded to 3 positions) to a list of messages with that quality.
    """
    q2m = dict()
    for q in m2q.values():
        qkey = round(q.score, 3)
        if qkey not in q2m: # for messages with identical scores, have a list
            q2m[qkey] = list()
        q2m[qkey].append(q.message)
    return q2m


# noinspection PyShadowingNames,PyShadowingNames
def writeResults(tikzcode: str, specimens: SpecimenLoader, inferenceTitle: str, folder="reports"):
    """
    Write NEMESYS inference evaluation results to a report

    :param tikzcode: tikz code of inference examples (e. g. worst, average, best result)
    :param specimens: The input data encasulated in a SpecimenLoader object
    :param inferenceTitle: A title for this inference report
    :param folder: The folder to safe the report to
    :return:
    """

    absFolder = abspath(folder)
    if not isdir(absFolder):
        raise NotADirectoryError("The reports folder {} is not a directory. Reports cannot be written there.".format(
            absFolder))

    pcapName = splitext(basename(specimens.pcapFileName))[0]
    reportFolder = join(absFolder, pcapName + "_{}_fms_{}".format(
        inferenceTitle, time.strftime("%Y%m%d-%H%M%S", time.localtime())))
    makedirs(reportFolder)

    print('Write report to ' + reportFolder)

    # write Format Match Score and Metrics to csv
    with open(join(reportFolder, 'example-inference.tikz'), 'w') as tikzfile:
        tikzfile.write(tikzcode)


# noinspection PyShadowingNames
def bcDeltaPlot(bcdg_mmm: List[BitCongruenceDeltaGauss]):
    """
    Plot BCD(G) values for the messages with the best, average, and worst FMS.

    :param bcdg_mmm: Example message analysis results to plot. Expects three elements in the list.
    """
    from nemere.visualization.multiPlotter import MultiMessagePlotter

    fieldEnds = [comparator.fieldEndsPerMessage(bcdg.message) for bcdg in bcdg_mmm]

    # mark the byte right before the max delta
    inflectionXs = [[offset + int(numpy.nanargmax(wd)) - 1 for offset, wd in a.risingDeltas()] for a in bcdg_mmm]
    inflections = [(pwp, [bcdg.values[p] for p in pwp]) for pwp, bcdg in zip(inflectionXs, bcdg_mmm)]

    pinpointedInflections = [a.inflectionPoints() for a in bcdg_mmm]
    preInflectionXs = [[i - 1 for i in xs] for xs,ys in pinpointedInflections]
    preInflectionPoints = [ (pwp, [bcdg.bcdeltas[p] for p in pwp]) for pwp, bcdg in zip(preInflectionXs, bcdg_mmm)]

    mmp = MultiMessagePlotter(specimens, 'bitCongruenceDeltaGauss', 3, 1, args.interactive)
    # noinspection PyProtectedMember
    for ax in mmp._axes.flat:  # type: plt.Axes
        ax.tick_params(labelsize=7)
        ax.set_xlabel('Byte Position', fontdict={'fontsize':7})
    # aspect ratio 2:3
    mmp.setFigureSize(3.136, 3 * (.667 * 3.136 + 0.14)  )   # 10 pt = 0.139 in
    mmp.plotSubfigs([a.values for a in bcdg_mmm],
                    compareValue=[a.bcdeltas for a in bcdg_mmm],
                    fieldEnds=fieldEnds, fieldEndMarks=False)
    mmp.scatterInEachAx(preInflectionPoints, 'v')
    mmp.scatterInEachAx(inflections, 'o')
    mmp.printMessageBytes([a.message for a in bcdg_mmm], {'size': 4})  # set to 4 for DNS, 2.5 for NTP
    mmp.writeOrShowFigure()






if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Generate segments of messages using the NEMESYS method and evaluate against tshark dissectors: '
                    'Write a report containing the FMS for each message and other evaluation data.')
    parser.add_argument('pcapfilename', help='pcapfilename')
    parser.add_argument('-i', '--interactive', help='open ipython prompt after finishing the analysis.',
                        action="store_true")
    parser.add_argument('-s', '--sigma', type=float, help='sigma for noise reduction (gauss filter)')
    parser.add_argument('-l', '--layer', type=int, default=2,
                        help='Protocol layer relative to IP to consider. Default is 2 layers above IP '
                             '(typically the payload of a transport protocol).')
    parser.add_argument('-r', '--relativeToIP', default=False, action='store_true')
    args = parser.parse_args()
    if not isfile(args.pcapfilename):
        print('File not found: ' + args.pcapfilename)
        exit(1)

    if not args.sigma:
        pcapBasename = basename(args.pcapfilename)
        sigma = sigmapertrace[pcapBasename] if pcapBasename in sigmapertrace else 0.6
    else:
        sigma = args.sigma

    print("Load messages...")
    specimens = SpecimenLoader(args.pcapfilename, layer=args.layer,
                               relativeToIP = args.relativeToIP)
    comparator = MessageComparator(specimens, layer=args.layer,
                               relativeToIP=args.relativeToIP,
                               failOnUndissectable=False, debug=debug)

    ########################

    print("Segment messages...")
    inferenceTitle = 'bcDeltaGauss{:.1f}'.format(sigma)  # +hiPlateaus

    startsegmentation = time.time()
    segmentsPerMsg = bcDeltaGaussMessageSegmentation(specimens, sigma)
    runtimeSegmentation = time.time() - startsegmentation
    refinedPerMsg = originalRefinements(segmentsPerMsg)
    # refinedPerMsg = baseRefinements(segmentsPerMsg)
    runtimeRefinement = time.time() - startsegmentation

    print('Segmented and refined in {:.3f}s'.format(time.time() - startsegmentation))

    symbols = symbolsFromSegments(segmentsPerMsg)
    refinedSymbols = symbolsFromSegments(refinedPerMsg)

    ########################

    comparator.pprintInterleaved(refinedSymbols)

    # calc FMS per message
    print("Calculate FMS...")
    message2quality = DissectorMatcher.symbolListFMS(comparator, refinedSymbols)

    # have a mapping from quality to messages
    quality2messages = mapQualities2Messages(message2quality)
    msg2analyzer = {segs[0].message: segs[0].analyzer for segs in refinedPerMsg}
    minmeanmax = reportWriter.getMinMeanMaxFMS([round(q.score, 3) for q in message2quality.values()])

    # here we only use one message as example of a quality! There may be more messages with the same quality.
    bcdg_mmm = [msg2analyzer[quality2messages[q][0]] for q in minmeanmax]  # type: List[BitCongruenceDeltaGauss]
    bcDeltaPlot(bcdg_mmm)

    ########################

    if args.interactive:
        print('Loaded PCAP in: specimens, comparator')
        print('Inferred messages in: symbols, refinedSymbols')
        print('FMS of messages in: message2quality, quality2messages, minmeanmax')
        IPython.embed()
    else:
        reportWriter.writeReport(message2quality, runtimeRefinement,
                                 comparator, inferenceTitle)
        reportWriter.writeReport(DissectorMatcher.symbolListFMS(comparator, symbols), runtimeSegmentation,
                                 comparator, inferenceTitle + '_withoutRefinement')
