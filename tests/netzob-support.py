"""
Test suite to ensure crucial Netzob functionality to work as expected.

NEMESYS requires that Netzob is at least as recent to have the following commits:
    * fix hash generation for Symbol + add one for AbstractField (44d899c/df7094a)
    * fix building of layered messages (57ee01e/9cb7507)
"""
import logging
import unittest, itertools

from netzob.Import.PCAPImporter.PCAPImporter import PCAPImporter
from netzob.Model.Vocabulary.Field import Field
from netzob.Model.Vocabulary.Symbol import Symbol


UPGRADE_NOTE = "Upgrade to current version from https://github.com/netzob!"

class TestAbstractField(unittest.TestCase):
    """
    Test requirement about hashability of AbstractField
    """
    def setUp(self) -> None:
        """
        Construct AbstractFields with the same name.
        Use a subclass implementing AbstractField to instantiate.
        """
        self.afs = [Field(name="NEMESYS")]*5

    def test_hash(self):
        """
        Check if different AbstractField objects return different hashes.
        """
        for afA, afB in itertools.combinations(self.afs, 2):
            if afA == afB:
                continue
            # print(id(afA), id(afB))
            assert hash(afA) != hash(afB), "Netzob version without fixed AbstractField hash generation. " + \
                                            UPGRADE_NOTE

class TestSymbol(unittest.TestCase):
    """
    Test requirement about hashability of Symbol
    """
    def setUp(self) -> None:
        """
        Construct Symbols with the same name.
        """
        self.symbols = [Symbol(fields=[], messages=[], name="NEMESYS")]*5

    def test_hash(self):
        """
        Check if different Symbol objects return different hashes.
        """
        for symA, symB in itertools.combinations(self.symbols, 2):
            if symA == symB:
                continue
            assert hash(symA) != hash(symB), "Netzob version without fixed Symbol hash generation. " + \
                                              UPGRADE_NOTE

class TestPCAPImporter(unittest.TestCase):
    """
    Test import from PCAP file on different layers. Ensure the correct bytes are considered payload.
    """
    # TESTPCAP = "resources/pcaps/test_import_udp.pcap"
    # """relative to the netzob repository folder 'test'"""
    TESTPCAP = "../tests/resources/test_import_udp_courtesy2NetzobTeam.pcap"

    UPGRADE_NOTE = "Netzob version with faulty PCAP import. "

    def setUp(self) -> None:
        # modpath = path.dirname(netzob.__file__)
        # testpath = path.join(modpath, "../../test")
        # self.pcappath = path.join(testpath, TestPCAPImporter.TESTPCAP)
        self.pcappath = TestPCAPImporter.TESTPCAP
        logging.getLogger().setLevel(logging.WARNING)

    def test_layer1(self):
        """
        Raw frame (whole frame is "payload")
        """
        messages = PCAPImporter.readFile(self.pcappath, importLayer=1).values()
        assert messages[0].data == b'\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x08\x00' \
                                   b'E\x00\x003\xdc\x11@\x00@\x11`\xa6\x7f\x00\x00\x01\x7f\x00\x00\x01' \
                                   b'\xe1\xe7\x10\x92\x00\x1f\xfe2' \
                                   b'CMDidentify#\x07\x00\x00\x00Roberto', \
                TestPCAPImporter.UPGRADE_NOTE + UPGRADE_NOTE

    def test_layer2(self):
        """
        Layer 2, e. g. parse Ethernet frame (IP packet is payload)
        """
        messages = PCAPImporter.readFile(self.pcappath, importLayer=2).values()
        assert messages[0].data == b'E\x00\x003\xdc\x11@\x00@\x11`\xa6\x7f\x00\x00\x01\x7f\x00\x00\x01' \
                                   b'\xe1\xe7\x10\x92\x00\x1f\xfe2' \
                                   b'CMDidentify#\x07\x00\x00\x00Roberto', \
            TestPCAPImporter.UPGRADE_NOTE + UPGRADE_NOTE

    def test_layer3(self):
        """
        Layer 3, e. g. parse IP packet (UDP datagram is payload)
        """
        messages = PCAPImporter.readFile(self.pcappath, importLayer=3).values()
        assert messages[0].data == b'\xe1\xe7\x10\x92\x00\x1f\xfe2' \
                                   b'CMDidentify#\x07\x00\x00\x00Roberto', \
            TestPCAPImporter.UPGRADE_NOTE + UPGRADE_NOTE

    def test_layer4plus(self):
        """
        Layer 4, e. g. parse UDP packet (application protocol is payload)
        Layer > 4, does decode like layer=4
        """
        messages4 = PCAPImporter.readFile(self.pcappath, importLayer=4).values()
        messages5 = PCAPImporter.readFile(self.pcappath, importLayer=5).values()
        assert messages4[0].data == messages5[0].data == b'CMDidentify#\x07\x00\x00\x00Roberto', \
            TestPCAPImporter.UPGRADE_NOTE + UPGRADE_NOTE




if "__main__" == __name__:
    unittest.main()
