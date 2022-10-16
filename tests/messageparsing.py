import unittest
import logging
import sys
from contextlib import contextmanager
from io import StringIO

from netzob.all import PCAPImporter

from nemere.validation.messageParser import ParsedMessage

TESTDHCP = "../input/deduped-orig/dhcp_SMIA2011101X_deduped-100.pcap"
TESTDNS = "../input/deduped-orig/dns_ictf2010_deduped-100.pcap"

HUNDRED_COOKIES = "['63825363']\n"*99 + "['63825363']"
HUNDRED_REQUEST_LIST_ITEMS = """['010f03062c2e2f1f21f92b']
['010f03062c2e2f1f2179f92b']
['010f03062c2e2f1f21f92bfc']
['010f03062c2e2f1f21f92bfc']
['011c02030f06770c2c2f1a792a']
['011c02030f06770c2c2f1a792a']
['011c02030f06770c2c2f1a792a']
['011c02030f06770c2c2f1a792a']
['010f03062c2e2f1f21f92b']
['010f03062c2e2f1f21f92b']
['011c02030f06770c2c2f1a792a']
['011c02030f06770c2c2f1a792a']
['010f03062c2e2f1f21f92bfc']
['010f03062c2e2f1f21f92b']
['010f03062c2e2f1f21f92b']
['010f03062c2e2f1f21f92bfc']
['010f03062c2e2f1f21f92bfc']
['010f03062c2e2f1f21f92bfc']
['010f032c2e2f06']
['010f032c2e2f06']
['010f03062c2e2f1f2179f92bfc']
['010f03062c2e2f1f21f92bfc']
['010f03062c2e2f1f21f92b']
['010f03062c2e2f1f2179f92bfc']
['010f03062c2e2f1f2179f92b']
['010f03062c2e2f1f21f92bfc']
['011c02030f06770c2c2f1a792a']
['011c02030f06770c2c2f1a792a']
['011c02030f06770c2c2f1a792a']
['011c02030f06770c2c2f1a792a']
['011c02030f06770c2c2f1a792a']
['010f03062c2e2f1f21f92bfc']
['010f03062c2e2f1f21f92b']
['010f03062c2e2f1f21f92bfc']
['011c02030f06770c2c2f1a792a']
['011c02030f06770c2c2f1a792a']
['010f03062c2e2f1f2179f92b']
['010f03062c2e2f1f21f92b']
['010f03062c2e2f1f21f92bfc']
['010f03062c2e2f1f212b4d']
['010f03062c2e2f1f21f92b']
['010f03062c2e2f1f21f92bfc']
['010f03062c2e2f1f21f92b']
['010f03062c2e2f1f21f92b']
['010f03062c2e2f1f21f92bfc']
['010f03062c2e2f1f2179f92b']
['010f03062c2e2f1f21f92bfc']
['010f03062c2e2f1f21f92b']
['010f03062c2e2f1f21f92b']
['010f03062c2e2f1f21f92b']
['010f03062c2e2f1f2179f92bfc']
['010f032c2e2f06']
['010f03062c2e2f1f21f92bfc']
['010f03062c2e2f1f212b']
['011c02030f06770c2c2f1a792a']
['011c02030f06770c2c2f1a792a']"""
HUNDRED_MESSAGE_TYPES = """Request
ACK
Request
ACK
Inform
ACK
Inform
ACK
Discover
Discover
Discover
Discover
Request
ACK
Request
ACK
Discover
Discover
Inform
ACK
Request
ACK
Request
ACK
Inform
ACK
Inform
ACK
Inform
ACK
Request
Request
ACK
Inform
ACK
Inform
ACK
Request
ACK
Inform
ACK
Request
ACK
Inform
ACK
Discover
Discover
Discover
Discover
Discover
Inform
ACK
Request
ACK
Inform
ACK
Discover
Request
Request
ACK
Request
ACK
Inform
ACK
Request
Request
ACK
Inform
ACK
Request
ACK
Request
ACK
Inform
ACK
Request
ACK
Inform
ACK
Request
ACK
Request
ACK
Request
ACK
Inform
ACK
Inform
Request
ACK
Inform
ACK
Request
ACK
Discover
Request
Inform
Inform
Inform
Inform"""



@contextmanager
def captured_output():
    """
    Temporarily capture output to be evaluated in assert afterwards.

    shamelssly copied and adapted from
    https://stackoverflow.com/questions/4219717/how-to-assert-output-with-nosetest-unittest-in-python
    """
    new_out, new_err = StringIO(), StringIO()
    old_out, old_err = sys.stdout, sys.stderr
    try:
        sys.stdout, sys.stderr = new_out, new_err
        yield sys.stdout, sys.stderr
    finally:
        sys.stdout, sys.stderr = old_out, old_err


class TestParsedMessage(unittest.TestCase):
    """
    Test nemere.validation.messageParser.ParsedMessage as unittest since the corresponding doctests fail miraculously.
    """

    def setUp(self) -> None:
        """Import DHCP and DNS traces as samples using netzob's PCAPImporter."""
        # prevent Netzob from producing debug output.
        logging.getLogger().setLevel(30)
        self.dhcp = PCAPImporter.readFile(TESTDHCP, importLayer = 1).values()
        self.dns  = PCAPImporter.readFile(TESTDNS,  importLayer = 1).values()
        self.dhcpPms = ParsedMessage.parseMultiple(self.dhcp)
        self.dnsPms = []

    def test_printUnknownTypes(self):
        """Test parsing of DHCP and validate by printUnknownTypes which should return nothing if everything is fine."""
        with captured_output() as (out, err):
            for parsed in self.dhcpPms.values(): parsed.printUnknownTypes()
        output = out.getvalue().strip()
        self.assertEqual(output, '')

    def test_parseMultiple(self):
        """Test parsing of DNS which should return basically nothing."""
        with captured_output() as (out, err):
            self.dnsPms = ParsedMessage.parseMultiple(self.dns)
        output = out.getvalue().strip()
        self.assertEqual(output, 'Wait for tshark output (max 20s)...')

    def test_getValuesByName(self):
        """Test retrieving field values by name."""
        with captured_output() as (out, err):
            for pms in self.dhcpPms.values():
                elements = pms.getValuesByName("dhcp.option.request_list")  # cookie
                if not isinstance(elements,bool) and elements:
                     print(elements)
        output = out.getvalue().strip()
        self.assertEqual(output, HUNDRED_REQUEST_LIST_ITEMS)

    def test_messagetype(self):
        """Test MessageTypeIdentifiers.typeOfMessage."""
        with captured_output() as (out, err):
            for pms in sorted(self.dhcpPms.values(), key=lambda m: m.message.date):
                print(pms.messagetype)
        output = out.getvalue().strip()
        self.assertEqual(output, HUNDRED_MESSAGE_TYPES)

if __name__ == '__main__':
    unittest.main()

# cd src/
# from netzob.all import PCAPImporter
# from nemere.validation.messageParser import ParsedMessage
# TESTDHCP = "../input/deduped-orig/dhcp_SMIA2011101X_deduped-100.pcap"
# dhcp = PCAPImporter.readFile(TESTDHCP, importLayer = 1).values()
# dhcpPms = ParsedMessage.parseMultiple(dhcp)
# from collections import Counter, defaultdict
# fieldCnt = defaultdict(Counter)
# for pms in dhcpPms.values():
#     fc = Counter(pms.getFieldNames())
#     for f,c in fc.items():
#         fieldCnt[f].update([c])




