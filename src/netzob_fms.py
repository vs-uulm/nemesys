"""
Infer PCAP with Netzob and compare the result to the tshark dissector of the protocol:

-- compare --
Dissector-label output:
        list of formats: list of (type, length)-tuples
-- to --
Netzob-inference output:
        list of symbols: list of contained fields

In the end, accumulate the comparison into one number to get a quality metric: The Format Match Score for each message.

Usenix WOOT 2018.
"""
import argparse
from os.path import isfile, join
from typing import Dict, Tuple, List

import matplotlib.colors
import matplotlib.pyplot as plt
import IPython

from netzob import all as netzob
from netzob.Model.Vocabulary.Messages.AbstractMessage import AbstractMessage

from nemere.utils.loader import SpecimenLoader
from nemere.validation.messageParser import ParsedMessage
from nemere.validation.dissectorMatcher import MessageComparator, DissectorMatcher
import nemere.validation.netzobFormatMatchScore as FMS


debug = True
"""Some modules and methods contain debug output that can be activated by this flag."""

# iterate similarity threshold from minThresh to maxThresh
minThresh = 75
maxThresh = 75


# optimal similarity threshold for some evaluation traces (from -100s):
optThresh = {
    "dhcp_SMIA2011101X-filtered_maxdiff-"   : 76,
    "dns_ictf2010_maxdiff-"                 : 51,
    "dns_ictf2010-new_maxdiff-"             : 50,
    "nbns_SMIA20111010-one_maxdiff-"        : 53,
    "ntp_SMIA-20111010_maxdiff-"            : 66,
    "smb_SMIA20111010-one-rigid1_maxdiff-"  : 53,
    "awdl-filtered"                         : 57,
    "au-wifi-filtered"                      : 51,
    "ari_syslog_corpus_maxdiff-"            : 47,  # nemesys-reports/NEMEPCA/phase-postgv-3_comparison/netzob-format-206-fms-c4d940b/ari_syslog_corpus_maxdiff-99_clByAlign_20230207-021732
}



def getNetzobInference(l5msgs: List[AbstractMessage], minEquivalence=45):
    """
    Imports the application layer messages from a PCAP and applies Format.clusterByAlignment() to them.

    :param l5msgs: A list of messages to infer
    :param minEquivalence: the similarity threshold for the clustering
    :type minEquivalence: int
    :return: list of symbols inferred by clusterByAlignment from pcap trace
    """
    import time #, gc

    print("Start netzob inference...")
    # profiler = utils.ProfileFunction(netzob.Format.clusterByAlignment, cprofile=True)
    # gc.disable()
    starttime = time.time()
    symbollist = netzob.Format.clusterByAlignment(l5msgs, minEquivalence=minEquivalence, internalSlick=True)
    runtime = time.time() - starttime
    print('Inferred in {:.3f}s'.format(runtime))
    # gc.enable()
    return symbollist, runtime
    # return profiler.run(l5msgs, minEquivalence=minEquivalence, internalSlick=True)


# noinspection PyShadowingNames
def iterSimilarities(minSimilarity=40, maxSimilarity=60) \
        -> Dict[int, Tuple[Dict[netzob.Symbol, List[List[Tuple[str, int]]]], float]]:
    """
    Iterate input parameter similarity threshold for clustering (minEquivalence = 0...100).

    :returns a dict in the of structure of:
        dict (
            similaritythreshold : dict (
                symbol : list ( tformats )
                )
            )
    """
    symFmt = dict()
    for similaritythreshold in range(minSimilarity, maxSimilarity+1):
        print("Similarity {:d}: ".format(similaritythreshold))
        symbols, runtime = getNetzobInference(
            list(specimens.messagePool.keys()), similaritythreshold)  # l5msgs (should be in original order)
        symFmt[similaritythreshold] = (dict(), runtime)
        for symbol in symbols:
            # l2msgs = [specimens.messagePool[msg] for msg in symbol.messages]
            tformats = MessageComparator.uniqueFormats(
                list(comparator.dissections.values()) )
            symFmt[similaritythreshold][0][symbol] = tformats
    return symFmt



#################################################


# noinspection PyShadowingNames,PyShadowingNames,PyShadowingNames
def findFormatExamples(theshSymbols):
    """
    Distinct datatypes per Symbol or Message-List

    for each similarityThreshold's each Symbol determine contained tformats and a sample message of it
    dict ( thresh : dict ( symb : list( tuple(tshark_format), l5msg ) ) )

    **depreciated** in favor of new approach of iterSimilarities()

    :param theshSymbols: dict ( thresh : dict ( symb : ... )
    :return:
    """
    formatsInSymbols = dict()
    for thresh, symbqual in theshSymbols:
        formatsInSymbols[thresh] = dict()
        for symb, rest in symbqual.items():
            formatsInSymbols[thresh][symb] = list()
            for l5msg in symb.messages:  # the L5Messages
                # only retain unique formats (a list of tuples of primitives can simply be compared)
                if tformats[specimens.messagePool[l5msg]] not in (fmt for fmt, msg in formatsInSymbols[thresh][symb]):
                    formatsInSymbols[thresh][symb].append((tformats[specimens.messagePool[l5msg]], l5msg))  # the format for message msg


def reduceBitsToBytes(formatdescbit):
    """
    Reduce length precicion from bits to bytes in a field-length tuple list derived from Scapy.

    **depreciated** since not needed without scapy

    :param formatdescbit: list of tuples (fieldtype, fieldlengthInBits)
    :return: list of tuples (fieldtype, fieldlengthInBytes)
    """
    formatdescbyte = list()
    bittoalign = 0
    for ftype, length in formatdescbit:
        bytesnum = length / 8
        overbit = length % 8
        bittoalign = (overbit + bittoalign) % 8
        bytesnum += (overbit + bittoalign) / 8
        if bytesnum == 0:
            continue  # jump to next field, since current is too small (< 8 bit) to retain
        formatdescbyte.append((ftype, bytesnum))
    return formatdescbyte

#################################################



if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Compare netzob inference and a protocol dissector for a set of messages (PCAP).')
    parser.add_argument('pcapfilename', help='Filename of the PCAP to load.')
    parser.add_argument('--smin', type=int, help='minimum similarity threshold to iterate.')
    parser.add_argument('--smax', type=int, help='maximum similarity threshold to iterate. Omit to only infer at the threshold of smin')
    parser.add_argument('-p', '--profile', help='profile the netzob run.',
                        action="store_true")
    parser.add_argument('-i', '--interactive', help='Show interactive plot instead of writing output to file and '
                                                    'open ipython prompt after finishing the analysis.',
                        action="store_true")
    parser.add_argument('-l', '--layer', type=int, default=2,
                        help='Protocol layer relative to IP to consider. Default is 2 layers above IP '
                             '(typically the payload of a transport protocol).')
    parser.add_argument('-r', '--relativeToIP', default=False, action='store_true')
    args = parser.parse_args()
    if not isfile(args.pcapfilename):
        print('File not found: ' + args.pcapfilename)
        exit(1)

    import time
    swstart = time.time()
    print('\nLoading ...')

    specimens = SpecimenLoader(args.pcapfilename, layer=args.layer, relativeToIP=args.relativeToIP)
    comparator = MessageComparator(specimens, layer=args.layer, relativeToIP=args.relativeToIP,
                               failOnUndissectable=False, debug=debug)
    print('Loaded and dissected in {:.3f}s'.format(time.time() - swstart))

    print(f'\nNetzob Inference of {specimens.pcapFileName}...')
    # dict ( similaritythreshold : dict ( symbol : (quality, fieldcount, exactf, nearf, uospecific) ) )
    if args.smin:
        minThresh = args.smin
        maxThresh = args.smax if args.smax else args.smin
    else:
        # use optimum for trace if a value is known
        for pcap, simthr in optThresh.items():
            if pcap in specimens.pcapFileName:
                minThresh = maxThresh = simthr
                break
    threshSymbTfmtTime = iterSimilarities(minThresh, maxThresh)
    threshSymbTfmt = {t: s for t, (s, r) in threshSymbTfmtTime.items()}
    threshTime = {t: r for t, (s, r) in threshSymbTfmtTime.items()}

    print('\nCalculate Format Match Score...')
    swstart = time.time()

    if args.profile:
        import cProfile #, pstats
        prof = cProfile.Profile()
        prof.enable()
    # (thresh, msg) : fms
    formatmatchmetrics = DissectorMatcher.thresymbolListsFMS(comparator, threshSymbTfmt)
    if args.profile:
        # noinspection PyUnboundLocalVariable
        prof.disable()
    print('Calculated in {:.3f}s'.format(time.time() - swstart))

    # if args.profile:
    #     import io
    #     s = io.StringIO()
    #     sortby = 'cumulative'
    #     ps = pstats.Stats(prof, stream=s).sort_stats(sortby)
    #     ps.print_stats()
    #     print(s.getvalue())
    # # TODO do the profiling
    # IPython.embed()

    print('\nPrepare output data...')
    swstart = time.time()

    # per tformat stats: dict ( message: list((t,s),(t,s)) )
    tformats = comparator.dissections
    distinctFormats = MessageComparator.uniqueFormats(tformats.values())

    # Format Match Score per format and symbol per threshold
    qpfSimilarity = dict()  # lists of x-values of each distinctFormat
    qualityperformat = dict()  # lists of y-values of each distinctFormat
    for df in distinctFormats:
        qualityperformat[df] = list()  # FMS per format
        qpfSimilarity[df] = list()
    for (thresh, msg), metrics in formatmatchmetrics.items():  # per threshold
        # ignore parsing errors
        if metrics.score is not None:
            qualityperformat[metrics.trueFormat].append(metrics.score)
            qpfSimilarity[metrics.trueFormat].append(thresh)

    # TODO biggest/most correct cluster per threshold
    # TODO format correctness, conciseness, (coverage) of each symbol

    # ## Output
    # FMS.printFMS(formatmatchmetrics, False)
    # plot_scatter3d(underOverSpecific, formatMatchScore, similarityThreshold)
    FMS.MessageScoreStatistics.printMinMax(formatmatchmetrics)

    # experimental
    # plt.ion()

    # Format Match Score per format and symbol per threshold
    xkcdc = list(matplotlib.colors.XKCD_COLORS.values())
    for i, df in enumerate(distinctFormats):
        plt.scatter(qpfSimilarity[df], qualityperformat[df], c=xkcdc[i], alpha=0.5, marker=r'.',
                    label="Format {:d} ".format(i))  # + repr(df))
    plt.ticklabel_format(style="plain")
    plt.xlabel("Similarity Threshold")
    plt.ylabel("Format Match Score")
    plt.legend(loc=2)

    print('Prepared in {:.3f}s'.format(time.time() - swstart))
    print('Writing report...')
    swstart = time.time()

    reportFolder = FMS.writeReport(formatmatchmetrics, plt, threshTime,specimens, comparator)
    if args.profile:
        prof.dump_stats(join(reportFolder, 'profiling-netzob-pathparser.profile'))
    print('Written in {:.3f}s'.format(time.time() - swstart))

    ParsedMessage.closetshark()

    # interactive stuff
    if args.interactive:
        # plt.show()
        print("\nAll truths are easy to understand once they are discovered; the point is to discover them. "
              "-- Galileo Galilei\n")
        IPython.embed()



