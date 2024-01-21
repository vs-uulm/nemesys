import unittest, sys, re

from contextlib import contextmanager
from io import StringIO
from itertools import islice
from typing import List

from netzob.Model.Vocabulary.Messages.RawMessage import RawMessage
from netzob.Model.Vocabulary.Symbol import Symbol

from nemere.utils.loader import SpecimenLoader, BaseLoader
from nemere.inference.segmentHandler import bcDeltaGaussMessageSegmentation, refinements, symbolsFromSegments, \
    zerocharPCAmocoSFrefinements
import nemere.visualization.simplePrint as sP


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


class BaseNemesysTest(unittest.TestCase):
    specimens = None  # type: BaseLoader
    output_filename = None
    nemesys_output = None  # type: str
    sigma = 0.6  # type: float

    def run_nemesys(self, refinementFunction):
        with captured_output() as (out, err):
            print("Segment messages...")
            inferenceTitle = 'bcDeltaGauss{:.1f}'.format(self.sigma)  # +hiPlateaus

            segmentsPerMsg = bcDeltaGaussMessageSegmentation(self.specimens, self.sigma)
            refinedPerMsg = refinementFunction(segmentsPerMsg)

            symbols = symbolsFromSegments(segmentsPerMsg)
            refinedSymbols = symbolsFromSegments(refinedPerMsg)  # type: List[Symbol]

            # output visualization of at most 100 messages on terminal
            segprint = sP.SegmentPrinter(refinedPerMsg)
            segprint.toConsole(islice(self.specimens.messagePool.keys(), 100))
            # omit messages longer than 200 bytes (and not more than 100 messages)
            segprint.toTikz(islice((msg for msg in self.specimens.messagePool.keys() if len(msg.data) < 200), 100))
        output = out.getvalue().strip()

        # remove this variable line from the test output
        pattern = re.compile(r"Calculated distances for \d+ segment pairs in 0.\d+ seconds.")
        output_static = pattern.sub("", output)
        # obtain the desired output
        # with open(self.output_filename, "w") as outfile:
        #     outfile.write(output)

        self.assertEqual(output_static, self.nemesys_output)

class NemesysPcapTest(BaseNemesysTest):
    pcapfilename = "../input/maxdiff-fromOrig/dns_ictf2010-new_maxdiff-100.pcap"
    layer = 2
    relativeToIP = True

    def setUp(self):
        self.maxDiff = None

        #print("Load messages...")
        self.specimens = SpecimenLoader(self.pcapfilename, layer=self.layer, relativeToIP=self.relativeToIP)

    def d_test_nemesys_zerocharPCAmocoSF(self):
        self.output_filename = "../tests/resources/nemesys_refined_pcap_test_output.txt"
        with open(self.output_filename, "r") as outfile:
            self.nemesys_output = outfile.read()
        self.run_nemesys(zerocharPCAmocoSFrefinements)

    def test_nemesys_default(self):
        self.output_filename = "../tests/resources/nemesys_default_pcap_test_output.txt"
        with open(self.output_filename, "r") as outfile:
            self.nemesys_output = outfile.read()
        self.run_nemesys(refinements)  # originalRefinements


class NemesysShortMessageTest(BaseNemesysTest):
    messages = [ b'\x00\x00\nA',
                 b'\x00\x00)4',
                 b'\x00\x00Y\x84',
                 b'\x00\x009\xcb',
                 b'\x00\x00!\xca',
                 b'\x00\x00Hj',
                 b"\x00\x00'y",
                 b'\x00\x00St',
                 b'\x00\x00E5',
                 b'\x00\x00Q\xc1' ]

    def setUp(self):
        self.maxDiff = None

        self.specimens = BaseLoader([RawMessage(payload) for payload in self.messages])

    def d_test_nemesys_zerocharPCAmocoSF(self):
        self.output_filename = "../tests/resources/nemesys_refined_shortmessage_test_output.txt"
        with open(self.output_filename, "r") as outfile:
            self.nemesys_output = outfile.read()
        self.run_nemesys(zerocharPCAmocoSFrefinements)

    def test_nemesys_default(self):
        self.output_filename = "../tests/resources/nemesys_default_shortmessage_test_output.txt"
        with open(self.output_filename, "r") as outfile:
            self.nemesys_output = outfile.read()
        self.run_nemesys(refinements)  # originalRefinements


if __name__ == '__main__':
    unittest.main()
