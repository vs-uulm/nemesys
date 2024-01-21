from typing import List, Iterable, Union
from os.path import isfile
from collections import OrderedDict
import logging

from scapy.layers.dot11 import RadioTap, Dot11, Dot11FCS
from scapy.packet import Packet, Raw
from scapy.utils import rdpcap
import pcapy

from netzob.Common.NetzobException import NetzobImportException
from netzob.Common.Utils.SortedTypedList import SortedTypedList
from netzob.Import.PCAPImporter.PCAPImporter import PCAPImporter
from netzob.Model.Vocabulary.Messages.AbstractMessage import AbstractMessage
from netzob.Model.Vocabulary.Messages.RawMessage import RawMessage
from netzob.Model.Vocabulary.Messages.L2NetworkMessage import L2NetworkMessage
from netzob.Model.Vocabulary.Messages.L4NetworkMessage import L4NetworkMessage

from nemere.validation.messageParser import ParsingConstants

class BaseLoader(object):
    """
    Wrapper for loading messages from memory as specimens.
    Prepares and represents messages to support further analysis.
    Especially useful for on-the-fly creation of test cases.
    """

    def __init__(self, l5msgs: Iterable[Union[AbstractMessage, RawMessage]], l1msgs=None, baselayer=None):
        """
        Load messages from memory. Base class for other loaders, e.g. loading from PCAP file.

        :param l5msgs:
        :param l1msgs:
        :param baselayer: Optionally, set the base layer explicitly.
        """
        if not hasattr(self, 'pcapFileName'):
            self.pcapFileName = 'from-memory'
        self.messagePool = OrderedDict()  # type: OrderedDict[AbstractMessage, RawMessage]
        """maps the message representations for Netzob and tshark
        dict of { application layer of messages L5Messages : RawMessage }"""
        if not l1msgs:
            l1msgs = l5msgs
        for k, m in zip(l5msgs, l1msgs):
            self.messagePool[k] = m
            # TODO replace the above quickfix not to read the file a second time (should we?)
            # probably we could use msgs = ParsedMessage.parseMultiple(l1msgs); for m in msgs:
            # ... append(RawMessage(m.protocolbytes))
        self._baselayer = baselayer

    def getBaseLayerOfPCAP(self):
        """
        see ParsingConstants.LINKTYPES

        :return: Determine lowest encapsulating layer of PCAP.
        """
        if self._baselayer is not None:
            return self._baselayer

        try:
            # looking at just one message should reveal lowest encapsulating layer of the whole PCAP
            al5msg = next(iter(self.messagePool.keys()))
        except StopIteration:
            raise ValueError('No message could be imported. See previous errors for more details.')
        if isinstance(al5msg, L2NetworkMessage):
            if al5msg.l2Protocol == 'Ethernet':
                return ParsingConstants.LINKTYPES['ETHERNET']
            # for some reason, its not possible to access the class variable "name", so instantiate a dummy object
            elif al5msg.l2Protocol == Dot11().name or al5msg.l2Protocol == Dot11FCS().name:
                return ParsingConstants.LINKTYPES['IEEE802_11']  # 802.11
            elif al5msg.l2Protocol == 'None':  # no ethernet
                if al5msg.l3Protocol == 'IP':
                    return ParsingConstants.LINKTYPES['RAW_IP']  # IP
                else:
                    raise NotImplementedError("Linktype on layer 3 unknown. Protocol is {}".format(al5msg.l3Protocol))
            else:
                raise NotImplementedError("Linktype on layer 2 unknown. Protocol is {}".format(al5msg.l2Protocol))
        else:
            return ParsingConstants.LINKTYPES['undecoded']  # non-decoded raw trace without link type information

    def __repr__(self):
        return type(self).__name__ + ": " + self.pcapFileName + f" on layer {self.getBaseLayerOfPCAP()}"

    @property
    def maximumMessageLength(self):
        """
        :return: The maximum message length in bytes of the relevant network layer without its encapsulation for
            the messages in this specimen's pool.
        """
        return max(len(line.data) for line in self.messagePool.keys())

    @property
    def cumulatedMessageLength(self):
        """
        :return: The sum of all message lengths in bytes of the relevant network layer without its encapsulation for
            the messages in this specimen's pool. I. e., the cumulated size of all payload in the trace.
        """
        return sum(len(line.data) for line in self.messagePool.keys())

class SpecimenLoader(BaseLoader):
    """
    Wrapper for loading messages from a PCAP as specimens.
    Prepares and represents messages to support further analysis.
    """

    def __init__(self, pcap: str, layer:int=-1, relativeToIP:bool=False):
        """
        Load the messages from the PCAP file of the given name.

        >>> from nemere.utils.loader import SpecimenLoader
        >>> sl = SpecimenLoader('../input/hide/random-100-continuous.pcap', layer=0, relativeToIP=True)
        >>> firstmessage = list(sl.messagePool.items())[0]
        >>> print(firstmessage[0].data.hex())  # the whole message
        450000780001000040007c837f0000017f000001da362a9b658bcc70d1a8323979e8fc0affda6f9fca9de16a3b051303601280f2c249abdd05f85ce98aebaf67626c07b4698a3c87abb95af87abf108735bac6ffd8823e80ac498622e7347852e0f2e7a8759911782b00a74b5744109b9fe59f5d311252a1
        >>> print(firstmessage[1].data.hex())  # only the payload of the target layer
        ffffffffffff0000000000000800450000780001000040007c837f0000017f000001da362a9b658bcc70d1a8323979e8fc0affda6f9fca9de16a3b051303601280f2c249abdd05f85ce98aebaf67626c07b4698a3c87abb95af87abf108735bac6ffd8823e80ac498622e7347852e0f2e7a8759911782b00a74b5744109b9fe59f5d311252a1

        :param pcap: PCAP input file name.
        :param layer: The protocol layer to extract. If not set or negative, use the top layer.
        :param relativeToIP: If True, extract the given layer relative to the IP layer.

        """
        if not isfile(pcap):
            raise FileNotFoundError('File not found:', pcap)
        self.pcapFileName = pcap
        self.layer = layer
        self.relativeToIP = relativeToIP
        absLayer = 2 + layer if relativeToIP else layer

        # prevent Netzob from producing debug output in certain cases.
        logging.getLogger().setLevel(30)

        try:
            if layer < 0:
                # read messages at layer 5 for the Netzob inference
                l5msgs = PCAPImporter.readFile(pcap, importLayer=5).values()  # type: List[L4NetworkMessage]
            else:
                # read messages at given layer for the Netzob inference
                l5msgs = PCAPImporter.readFile(pcap, importLayer=absLayer).values()  # type: List[AbstractMessage]
            # read messages as raw for tshark input
            l1msgs = PCAPImporter.readFile(pcap, importLayer=1).values()  # type: List[RawMessage]
        except (NetzobImportException, pcapy.PcapError):
            importer = ScaPyCAPimporter(self.pcapFileName, absLayer)
            l5msgs = importer.messages
            l1msgs = importer.rawMessages
        super().__init__(l5msgs, l1msgs)

    # The int value of some pcapy datalink denotations is different from the tcpdump ones: https://www.tcpdump.org/linktypes.html
    # http://vpnb.leipzig.freifunk.net:8004/srv2/lede/lede-20171116/build_dir/target-mips_24kc_musl/python-pcapy-0.11.1/pcapy.html#idp8777598240
    pcapyDatalinkTranslation = {
        pcapy.DLT_RAW: ParsingConstants.LINKTYPES['RAW_IP']
    }
    """Translates pcapy linktype values to tcpdump ones."""

    def getBaseLayerOfPCAP(self):
        pcap = pcapy.open_offline(self.pcapFileName)
        dl = pcap.datalink()
        # Translates pcapy linktype values to tcpdump ones if in dict, otherwise the value is used unchanged
        return dl if dl not in SpecimenLoader.pcapyDatalinkTranslation else SpecimenLoader.pcapyDatalinkTranslation[dl]


class ScaPyCAPimporter(object):
    def __init__(self, pcapfilename, importLayer=5):
        # l5msgs = PCAPImporter.readFile(pcap, importLayer=absLayer).values()  # type: List[AbstractMessage]
        self.importLayer = importLayer
        self.packets = rdpcap(pcapfilename)
        self._messages = SortedTypedList(AbstractMessage)
        self._rawmessages = SortedTypedList(AbstractMessage)

        for pkt in self.packets:  # type: Packet
            self.packetHandler(pkt)

    @property
    def messages(self):
        return self._messages.values()

    @property
    def rawMessages(self):
        return self._rawmessages.values()

    def packetHandler(self, packet: Packet):
        epoch = packet.time
        l1Payload = bytes(packet)
        if len(l1Payload) == 0:
            return
        # Build the RawMessage
        rawMessage = RawMessage(l1Payload, epoch, source=None, destination=None)

        if isinstance(packet, RadioTap):
            # lift layer to Dot11 if there is a RadioTap dummy frame
            packet = packet.payload
        if self.importLayer == 2:
            (l2Proto, l2SrcAddr, l2DstAddr, l2Payload) = self.__decodeLayer2(packet)
            if len(l2Payload) == 0:
                return
            # Build the L2NetworkMessage
            l2Message = L2NetworkMessage(l2Payload, epoch, l2Proto, l2SrcAddr, l2DstAddr)
            self._messages.add(l2Message)
            self._rawmessages.add(rawMessage)
        else:
            # Use Netzob's PCAPImporter if layer 2 is not WLAN
            raise NetzobImportException("PCAP", "Unsupported import layer. Currently only handles layer 2.",
                                        PCAPImporter.INVALID_LAYER2)

    def __decodeLayer2(self, packet: Packet):
        """Internal method that parses the specified header and extracts
        layer2 related proprieties."""
        l2Proto = packet.name
        if isinstance(packet, Raw):
            print("Ignoring undecoded packet with values:", bytes(packet).hex())
            return l2Proto, None, None, ""
        if isinstance(packet, Dot11):
            l2DstAddr = packet.fields['addr1']  # receiver address, alt: packet.fields['addr3'] destination address
            l2SrcAddr = packet.fields['addr2']  # transmitter address, alt: packet.fields['addr4'] source address
        else:
            raise NetzobImportException("NEMERE_PCAP", "Unsupported layer 2 protocol " + l2Proto,
                                        PCAPImporter.INVALID_LAYER2)
        l2Payload = bytes(packet.payload)
        return l2Proto, l2SrcAddr, l2DstAddr, l2Payload
