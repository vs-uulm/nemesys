# Script to validate the intended function of the Format Match Score (FMS):
#
# We validate that the FMS is measuring the format match quality of network messages by baseline and dummy segmenters.
#   For dissertation ... with NTP timestamps
import random
from os.path import isfile, basename, splitext
from typing import List

from nemere.inference.analyzers import Value
from nemere.inference.segmentHandler import fixedlengthSegmenter
from nemere.inference.segments import MessageAnalyzer, MessageSegment
from nemere.utils import reportWriter
from nemere.utils.loader import SpecimenLoader
from nemere.validation.dissectorMatcher import MessageComparator, BaseDissectorMatcher

debug = False

# Load small traces (filename: (layer, relativeToIP))
TESTTRACES = {
    # "input/maxdiff-fromOrig/dhcp_SMIA2011101X-filtered_maxdiff-100.pcap": (2, True),
    # "input/maxdiff-fromOrig/dns_ictf2010-new_maxdiff-100.pcap": (2, True),
    # "input/maxdiff-fromOrig/nbns_SMIA20111010-one_maxdiff-100.pcap": (2, True),
    # "input/maxdiff-fromOrig/ntp_SMIA-20111010_maxdiff-100.pcap": (2, True),
    # "input/maxdiff-fromOrig/smb_SMIA20111010-one-rigid1_maxdiff-100.pcap": (2, True),
    "input/maxdiff-fromOrig/binaryprotocols_maxdiff-fromOrig-500.pcap":  (2, True),
}



def segmentOnePerMsg(messagePool, *args):
    """
    :param messagePool:
    :return: one segment for the whole message
    """
    segments = list()
    for l4msg, rmsg in messagePool.items():
        originalAnalyzer = MessageAnalyzer.findExistingAnalysis(Value, MessageAnalyzer.U_BYTE, l4msg, None)
        sequence = [ MessageSegment(originalAnalyzer, 0, len(originalAnalyzer.values)) ]
        segments.append(sequence)
    return segments

def segmentRandom(messagePool, *args):
    segments = list()
    random.seed(42)  # produce "deterministic PRNs"
    for l4msg, rmsg in messagePool.items():
        originalAnalyzer = MessageAnalyzer.findExistingAnalysis(Value, MessageAnalyzer.U_BYTE, l4msg, None)
        sequence = [ ]  # type: List[MessageSegment]
        while len(sequence) == 0 or sequence[-1].nextOffset < len(originalAnalyzer.values):
            nextOffset = sequence[-1].nextOffset if len(sequence) > 0 else 0
            nextLen = random.randint(1, 1 + len(originalAnalyzer.values) - nextOffset)
            segment = MessageSegment(originalAnalyzer, nextOffset, nextLen)
            sequence.append(segment)
        segments.append(sequence)
    return segments

def segmentEachByte(messagePool, *args):
    """
    :param messagePool:
    :return: one segment for each byte of the message
    """
    segments = list()
    for l4msg, rmsg in messagePool.items():
        originalAnalyzer = MessageAnalyzer.findExistingAnalysis(Value, MessageAnalyzer.U_BYTE, l4msg, None)
        sequence = [ MessageSegment(originalAnalyzer, offset, 1) for offset in range(0, len(originalAnalyzer.values)) ]
        segments.append(sequence)
    return segments

def segmentFixedLength(messagePool, specimens, *args):
    """
    :param messagePool:
    :return: 4-byte-fixed
    """
    return fixedlengthSegmenter(4, specimens, Value, None)

def segmentTshark(messagePool, specimens, comparator: MessageComparator):
    """
    :param messagePool:
    :return: one segment for each byte of the message
    """
    segments = list()
    for l4msg, rmsg in messagePool.items():
        originalAnalyzer = MessageAnalyzer.findExistingAnalysis(Value, MessageAnalyzer.U_BYTE, l4msg, None)
        sequence = []
        for length in (flen for t, flen in comparator.dissections[rmsg]):
            nextOffset = sequence[-1].nextOffset if len(sequence) > 0 else 0
            sequence.append(MessageSegment(originalAnalyzer, nextOffset, length))
        segments.append(sequence)
    return segments


SEGMENTERS = {
    "wholeMsg"    : segmentOnePerMsg,   # one segment for the whole message
    "random"      : segmentRandom,      # random segments
    "eachByte"    : segmentEachByte,    # one segment per byte
    "fixedlength" : segmentFixedLength, # 4-byte-fixed
    "tshark"      : segmentTshark       # tshark true fields
}

if __name__ == '__main__':
    # Load small traces
    comparators = []  # type: List[MessageComparator]
    for pcapfilename, pcaparams in TESTTRACES.items():
        if not isfile(pcapfilename):
            print('File not found: ' + pcapfilename)
            exit(1)

        print(f"Load messages from {pcapfilename}...")
        specimens = SpecimenLoader(pcapfilename, layer = pcaparams[0], relativeToIP = pcaparams[1])
        comparator = MessageComparator(specimens, layer = pcaparams[0], relativeToIP = pcaparams[1],
                                       failOnUndissectable=False, debug=debug)
        comparators.append(comparator)

    # repeat for each trace:
    for comparator in comparators:
        for inference, segmenter in SEGMENTERS.items():
            inferenceTitle = f"{inference}|{splitext(basename(comparator.specimens.pcapFileName))[0]}"
            print(f"Segment messages of {comparator.specimens.pcapFileName} with {inferenceTitle}...")
            segments = segmenter(comparator.specimens.messagePool, comparator.specimens, comparator)

            # cprinter = ComparingPrinter(comparator, segments)
            # cprinter.toConsole()

            message2quality = {msg[0].message : BaseDissectorMatcher(comparator, msg).calcFMS() for msg in segments}
            runtime = 0
            # reportsFolder = join("reports", inferenceTitle)
            # mkdir(reportsFolder)
            reportWriter.writeReport(message2quality, runtime, comparator, inferenceTitle, withTitle=True)
    # IPython.embed()

    # TODO Find visualization of format quality to validate the FMS as evaluation measure
    #  without producing a tautology between used and visualized features.



