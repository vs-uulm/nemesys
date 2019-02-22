"""
Module encapsulating evaluation parameters and helper functions to validate aspects of the
NEMESYS and NEMETYL approaches.
"""
from typing import Union, Tuple, List

from inference.analyzers import *
from inference.segmentHandler import segmentsFromLabels
from inference.segments import MessageAnalyzer, TypedSegment




# available analysis methods
analyses = {
    'bcpnm': BitCongruenceNgramMean,
    # 'bcpnv': BitCongruenceNgramStd,  in branch inference-experiments
    'bc': BitCongruence,
    'bcd': BitCongruenceDelta,
    'bcdg': BitCongruenceDeltaGauss,
    'mbhbv': HorizonBitcongruence,

    'variance': ValueVariance,  # Note: VARIANCE is the inverse of PROGDIFF
    'progdiff': ValueProgressionDelta,
    'progcumudelta': CumulatedProgressionDelta,
    'value': Value,
    'ntropy': EntropyWithinNgrams,
    'stropy': Entropy,  # TODO check applicability of (cosine) distance calculation to this feature
}

epspertrace = {
    "dhcp_SMIA2011101X_deduped-100.pcap" : 1.8,
    "nbns_SMIA20111010-one_deduped-100.pcap" : 1.8, # or anything between 1.8 - 2.6
    "smb_SMIA20111010-one_deduped-100.pcap" : 1.6,
#    "dns_ictf2010_deduped-100.pcap" : 1.0,  # None/Anything, "nothing" to cluster
    "ntp_SMIA-20111010_deduped-100.pcap" : 1.5,
    "dhcp_SMIA2011101X_deduped-1000.pcap": 2.4,
    "nbns_SMIA20111010-one_deduped-1000.pcap": 2.4, # or anything between 1.0 - 2.8
    "smb_SMIA20111010-one_deduped-1000.pcap": 2.2,
    "dns_ictf2010_deduped-982-1000.pcap" : 2.4, # or anything between 1.6 - 2.8
    "ntp_SMIA-20111010_deduped-1000.pcap": 2.8
}

epsdefault = 2.4




def annotateFieldTypes(analyzerType: type, analysisArgs: Union[Tuple, None], comparator,
                       unit=MessageAnalyzer.U_BYTE) -> List[Tuple[TypedSegment]]:
    """
    :return: list of lists of segments that are annotated with their field type.
    """
    segmentedMessages = [segmentsFromLabels(
        MessageAnalyzer.findExistingAnalysis(analyzerType, unit,
                                             l4msg, analysisArgs), comparator.dissections[rmsg])
        for l4msg, rmsg in comparator.messages.items()]
    return segmentedMessages