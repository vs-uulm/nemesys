import unittest
import logging
import sys
from contextlib import contextmanager
from io import StringIO

from netzob.all import PCAPImporter

from nemere.validation.messageParser import ParsedMessage


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
    TESTDHCP = "../input/deduped-orig/dhcp_SMIA2011101X_deduped-100.pcap"
    TESTDNS  = "../input/deduped-orig/dns_ictf2010_deduped-100.pcap"

    def setUp(self) -> None:
        """Import DHCP and DNS traces as samples using netzob's PCAPImporter."""
        # prevent Netzob from producing debug output.
        logging.getLogger().setLevel(30)
        self.dhcp = PCAPImporter.readFile(TestParsedMessage.TESTDHCP, importLayer = 1).values()
        self.dns  = PCAPImporter.readFile(TestParsedMessage.TESTDNS,  importLayer = 1).values()

    def test_printUnknownTypes(self):
        """Test parsing of DHCP and validate by printUnknownTypes which should return nothing if everything is fine."""
        with captured_output() as (out, err):
            pms = ParsedMessage.parseMultiple(self.dhcp)
            for parsed in pms.values(): parsed.printUnknownTypes()
        output = out.getvalue().strip()
        self.assertEqual(output, 'Wait for tshark output (max 20s)...')

    def test_parseMultiple(self):
        """Test parsing of DNS which should return basically nothing."""
        with captured_output() as (out, err):
            pms = ParsedMessage.parseMultiple(self.dns)
        output = out.getvalue().strip()
        self.assertEqual(output, 'Wait for tshark output (max 20s)...')

if __name__ == '__main__':
    unittest.main()
