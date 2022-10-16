"""
Parse a PCAP file and print its dissection.
This script primarily is intended to check whether the dissection of a specific PCAP works and all fields can be
interpreted correctly to create a baseline to compare inferences to.
"""

import time
from argparse import ArgumentParser
from os.path import isfile
from sys import exit
import IPython

from nemere.validation.messageParser import ParsedMessage
from nemere.utils.loader import SpecimenLoader

# logging.basicConfig(level=logging.DEBUG)
# logging.getLogger().setLevel(logging.DEBUG)

if __name__ == '__main__':
    parser = ArgumentParser(
        description='Dissect PCAP with tshark and parse to python.')
    parser.add_argument('pcapfilename', help='pcapfilename')
    parser.add_argument('-l', '--targetlayer', type=int)
    parser.add_argument('-r', '--relativeToIP', default=False, action='store_true')
    args = parser.parse_args()
    if not isfile(args.pcapfilename):
        print('File not found: ' + args.pcapfilename)
        exit(1)
    if args.targetlayer:
        specimens = SpecimenLoader(args.pcapfilename, args.targetlayer, args.relativeToIP)
    else:
        specimens = SpecimenLoader(args.pcapfilename)
    print('Loaded PCAP file:', specimens.pcapFileName)
    pkt = list(specimens.messagePool.values())

    st = time.time()

    ###########################
    # Performance tests, comparing the dissection of one message at a time by the static parseMultiple method,
    # the ParsedMessage constructor and finally the dissection of a batch of all messages.
    ###########################
    # # Single messages with ParsedMessage.parseMultiple test:   Dissection ran in 63.48 seconds.
    # pms = dict()
    # for p in pkt:
    #     pms.update(ParsedMessage.parseMultiple([p]))
    # pms = list(pms.values())

    # Single messages with ParsedMessage constructor test:   Dissection ran in 58.39 seconds.
    # pms = list()
    # for p in pkt:
    #     pms.append(ParsedMessage(p))
    ###########################

    # # Multiple messages with ParsedMessage.parseMultiple test:   Dissection ran in 1.55 seconds.
    # if args.targetlayer:
    #     pms = ParsedMessage.parseMultiple(pkt, args.targetlayer, args.relativeToIP, linktype=specimens.getBaseLayerOfPCAP())
    #     pms = ParsedMessage.parseOneshot(specimens)
    # else:
    #     pms = ParsedMessage.parseMultiple(pkt, linktype=specimens.getBaseLayerOfPCAP())
    pms = ParsedMessage.parseOneshot(specimens)
    pms = list(pms.values())

    print("Dissection ran in {:3.2f} seconds.".format(time.time()-st))
    for pm in pms:  # type: ParsedMessage
        pm.printUnknownTypes()

        ###########################
        # Output of dissected messages as usage example
        ###########################
        # print(pm.getFieldNames())
        # # Simple field value print
        # for fv in pm.getFieldValues():
        #     try:
        #         print(bytes.fromhex(fv).decode("utf8").replace('\r\n', '¶'), end=' | ')
        #     except:
        #         print("(h)", fv, end=' | ')
        #         # IPython.embed()
        # print()
        ###########################

    ParsedMessage.closetshark()

    print('Loaded PCAP in: specimens')
    print('Parsed messages in: pms')

    IPython.embed()

