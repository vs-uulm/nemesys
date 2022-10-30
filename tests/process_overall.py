import unittest, sys, re

from contextlib import contextmanager
from io import StringIO
from itertools import islice
from typing import List

from netzob.Model.Vocabulary.Symbol import Symbol

from nemere.utils.loader import SpecimenLoader
from nemere.inference.segmentHandler import bcDeltaGaussMessageSegmentation, refinements, symbolsFromSegments
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


class MyTestCase(unittest.TestCase):
    pcapfilename = "../input/maxdiff-fromOrig/dns_ictf2010-new_maxdiff-100.pcap"
    layer = 2
    relativeToIP = True
    sigma = 0.6

    nemesys_output = ""

    def setUp(self):
        with open("../tests/resources/nemesys_test_output.txt", "r") as outfile:
            self.nemesys_output = outfile.read()
        self.maxDiff = None

    def test_nemesys(self):
        with captured_output() as (out, err):
            print("Load messages...")
            specimens = SpecimenLoader(self.pcapfilename, layer=self.layer, relativeToIP=self.relativeToIP)

            print("Segment messages...")
            inferenceTitle = 'bcDeltaGauss{:.1f}'.format(self.sigma)  # +hiPlateaus

            segmentsPerMsg = bcDeltaGaussMessageSegmentation(specimens, self.sigma)
            refinedPerMsg = refinements(segmentsPerMsg)

            symbols = symbolsFromSegments(segmentsPerMsg)
            refinedSymbols = symbolsFromSegments(refinedPerMsg)  # type: List[Symbol]

            # output visualization of at most 100 messages on terminal
            segprint = sP.SegmentPrinter(refinedPerMsg)
            segprint.toConsole(islice(specimens.messagePool.keys(), 100))
            # omit messages longer than 200 bytes (and not more than 100 messages)
            segprint.toTikz(islice((msg for msg in specimens.messagePool.keys() if len(msg.data) < 200), 100))
        output = out.getvalue().strip()

        # remove this variable line from the test output
        pattern = re.compile(r"Calculated distances for 17581 segment pairs in 0.\d+ seconds.")
        output_static = pattern.sub("", output)
        # obtain the desired output
        # with open("nemesys_test_output.txt", "w") as outfile:
        #     outfile.write(output)

        self.assertEqual(output_static, self.nemesys_output)


if __name__ == '__main__':
    unittest.main()
