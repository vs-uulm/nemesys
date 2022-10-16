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
from os.path import isfile, splitext, basename
from typing import Dict, Tuple, List

from netzob import all as netzob
from netzob.Model.Vocabulary.Messages.AbstractMessage import AbstractMessage

from nemere.utils.loader import SpecimenLoader
from nemere.utils.reportWriter import IndividualClusterReport, CombinatorialClustersReport
from nemere.validation.messageParser import ParsedMessage
from nemere.validation.dissectorMatcher import MessageComparator


debug = True
"""Some modules and methods contain debug output that can be activated by this flag."""

# iterate similarity threshold from minThresh to maxThresh
minThresh = 75
maxThresh = 75

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

def writeNetzobPerformanceStatistics(specimens, threshTime):
    import os, csv
    from nemere.utils.evaluationHelpers import reportFolder

    fileNameS = "Netzob-performance-statistics"
    csvpath = os.path.join(reportFolder, fileNameS + '.csv')
    csvWriteHead = False if os.path.exists(csvpath) else True

    print('Write performance statistics to {}...'.format(csvpath))
    with open(csvpath, 'a') as csvfile:
        statisticscsv = csv.writer(csvfile)
        if csvWriteHead:
            statisticscsv.writerow([
                'script', 'pcap', 'threshold', 'runtime'
            ])
        # noinspection PyUnresolvedReferences,PyPackageRequirements
        import __main__ as main
        statisticscsv.writerows([
            [os.path.basename(main.__file__), os.path.basename(specimens.pcapFileName), threshold, runtime]
            for threshold, runtime in threshTime.items() ])





if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Compare netzob inference and a scapy dissector for a set of messages (pcap).')
    parser.add_argument('pcapfilename', help='pcapfilename')
    parser.add_argument('--smin', type=int, help='minimum similarity threshold to iterate.')
    parser.add_argument('--smax', type=int, help='maximum similarity threshold to iterate. Omit to only infer at the threshold of smin')
    parser.add_argument('-p', '--profile', help='profile the netzob run.',
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

    print('\nNetzob Inference ...')
    # dict ( similaritythreshold : dict ( symbol : (quality, fieldcount, exactf, nearf, uospecific) ) )
    if args.smin:
        minThresh = args.smin
        maxThresh = args.smax if args.smax else args.smin
    threshSymbTfmtTime = iterSimilarities(minThresh, maxThresh)
    threshSymbTfmt = {t: s for t, (s, r) in threshSymbTfmtTime.items()}

    threshTime = {t: r for t, (s, r) in threshSymbTfmtTime.items()}
    writeNetzobPerformanceStatistics(specimens, threshTime)

    print('\nCalculate Cluster Statistics...')
    swstart = time.time()

    if args.profile:
        import cProfile #, pstats
        prof = cProfile.Profile()
        prof.enable()
    # here goes the extraction of messages from the symbols
    threshSymbMsgs = {t: {n: [msg for msg in s.messages]
                          for n, s in enumerate(syme.keys())} for t, syme in threshSymbTfmt.items()}

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

    # compare symbols' messages to true message types
    groundtruth = {msg: pm.messagetype for msg, pm in comparator.parsedMessages.items()}
    IndividualClusterReport.messagetypeStatsFile = "messagetype-netzob-statistics"
    CombinatorialClustersReport.messagetypeStatsFile = "messagetype-combined-netzob-statistics"
    for thresh, symbMsgs in threshSymbMsgs.items():
        messageClusters = {symname: [specimens.messagePool[msg] for msg in msglist]
                            for symname, msglist in symbMsgs.items()}

        # TODO replace title by ...inferenceParams
        clusterReport = IndividualClusterReport(groundtruth, splitext(basename(specimens.pcapFileName))[0])
        clusterReport.write(messageClusters, "netzob-thresh={}".format(thresh))
        combinReport = CombinatorialClustersReport(groundtruth, splitext(basename(specimens.pcapFileName))[0])
        combinReport.write(messageClusters, "netzob-thresh={}".format(thresh))

    ParsedMessage.closetshark()

    # # interactive stuff
    # # plt.show()
    # print("\nAll truths are easy to understand once they are discovered; the point is to discover them. -- Galileo Galilei\n")
    # IPython.embed()



