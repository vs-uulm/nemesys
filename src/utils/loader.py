from typing import List
from os.path import isfile
from collections import OrderedDict

from netzob.Import.PCAPImporter.PCAPImporter import PCAPImporter
from netzob.Model.Vocabulary.Messages.AbstractMessage import AbstractMessage
from netzob.Model.Vocabulary.Messages.RawMessage import RawMessage
from netzob.Model.Vocabulary.Messages.L4NetworkMessage import L4NetworkMessage

from validation.messageParser import ParsingConstants

class SpecimenLoader(object):
    """
    Wrapper for loading messages from a PCAP as specimens.
    Prepares and represents messages to support further analysis.
    """

    def __init__(self, pcap: str, layer:int=-1, relativeToIP:bool=False):
        """
        Load the messages from the PCAP file of the given name.

        :param pcap: PCAP input file name.
        """
        if not isfile(pcap):
            raise FileNotFoundError('File not found:', pcap)
        self.pcapFileName = pcap
        self.messagePool = OrderedDict()  # type: OrderedDict[AbstractMessage, RawMessage]
        """maps message representations for Netzob and tshark
        dict of { application layer of messages L5Messages : RawMessage }"""

        if layer <= 0:
            # read messages at layer 5 for the Netzob inference
            l5msgs = PCAPImporter.readFile(pcap, importLayer=5).values()  # type: List[L4NetworkMessage]
        else:
            # read messages at given layer for the Netzob inference
            absLayer = 3 + layer if relativeToIP else layer
            l5msgs = PCAPImporter.readFile(pcap, importLayer=absLayer).values()  # type: List[AbstractMessage]
        # read messages as raw for tshark input
        l1msgs = PCAPImporter.readFile(pcap, importLayer=1).values()  # type: List[RawMessage]
        for k, m in zip(l5msgs, l1msgs):
            self.messagePool[k] = m
            # TODO replace the above quickfix not to read the file a second time (should we?)
            # probably we could use msgs = ParsedMessage.parseMultiple(l1msgs); for m in msgs:
            # ... append(RawMessage(m.protocolbytes))


    def getBaseLayerOfPCAP(self):
        """
        see ParsingConstants.LINKTYPES

        :return: Determine lowest encapulating layer of PCAP.
        """
        try:
            # looking at just one message should reveal lowest encapulating layer of the whole PCAP
            al5msg = next(iter(self.messagePool.keys()))
        except StopIteration:
            raise ValueError('No message could be imported. See previous errors for more details.')
        if al5msg.l2Protocol == 'Ethernet':
            return ParsingConstants.LINKTYPES['ETHERNET']
        elif al5msg.l2Protocol == 'None':  # no ethernet
            if al5msg.l3Protocol == 'IP':
                return ParsingConstants.LINKTYPES['RAW_IP']  # IP
            else:
                raise NotImplementedError("Linktype on layer 3 unknown. Protocol is {}".format(al5msg.l3Protocol))
        else:
            raise NotImplementedError("Linktype on layer 2 unknown. Protocol is {}".format(al5msg.l2Protocol))


    @property
    def maximumMessageLength(self):
        """
        :return: The maximum message length in bytes of the relevant network layer without its encapsulation for
            the messages in this specimen's pool.
        """
        return max(len(line.data) for line in self.messagePool.keys())