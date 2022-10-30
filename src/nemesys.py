"""
Reference implementation for calling NEMESYS: NEtwork MEssage Syntax analysYS with an unknown protocol.
Usenix WOOT 2018.

Infer messages from PCAPs by the NEMESYS approach (BCDG-segmentation)
and write inference result to the terminal. Finally drop to an IPython console
and expose API to interact with the result.
"""

import argparse, time
from os.path import isfile
from itertools import islice
from typing import List

import IPython
from netzob.Model.Vocabulary.Symbol import Symbol

from nemere.utils.loader import SpecimenLoader
from nemere.inference.segmentHandler import bcDeltaGaussMessageSegmentation, refinements, symbolsFromSegments
import nemere.visualization.simplePrint as sP

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Generate segments of messages using the NEMESYS method and evaluate against tshark dissectors: '
                    'Write a report containing the FMS for each message and other evaluation data.')
    parser.add_argument('pcapfilename', help='pcapfilename')
    parser.add_argument('-i', '--interactive', help='open ipython prompt after finishing the analysis.',
                        action="store_true")
    parser.add_argument('-s', '--sigma', type=float, help='sigma for noise reduction (gauss filter)')
    parser.add_argument('-l', '--layer', type=int, default=2,
                        help='Protocol layer to consider. Default is layer 2. Use --relativeToIP '
                             'to use a layer relative to IP layer.')
    parser.add_argument('-r', '--relativeToIP', default=False, action='store_true',
                        help='Consider a layer relative to the IP layer (see also --layer flag)')
    args = parser.parse_args()
    if not isfile(args.pcapfilename):
        print('File not found: ' + args.pcapfilename)
        exit(1)

    sigma = 0.6 if not args.sigma else args.sigma

    print("Load messages...")
    specimens = SpecimenLoader(args.pcapfilename, layer=args.layer,
                               relativeToIP=args.relativeToIP)

    print("Segment messages...")
    inferenceTitle = 'bcDeltaGauss{:.1f}'.format(sigma)  # +hiPlateaus

    startsegmentation = time.time()
    segmentsPerMsg = bcDeltaGaussMessageSegmentation(specimens, sigma)
    runtimeSegmentation = time.time() - startsegmentation
    refinedPerMsg = refinements(segmentsPerMsg)
    runtimeRefinement = time.time() - startsegmentation

    print('Segmented and refined in {:.3f}s'.format(time.time() - startsegmentation))

    symbols = symbolsFromSegments(segmentsPerMsg)
    refinedSymbols = symbolsFromSegments(refinedPerMsg)  # type: List[Symbol]

    # output visualization of at most 100 messages on terminal and all into file
    segprint = sP.SegmentPrinter(refinedPerMsg)
    segprint.toConsole(islice(specimens.messagePool.keys(),100))
    # segprint.toTikzFile()
    # omit messages longer than 200 bytes (and not more than 100 messages)
    segprint.toTikzFile(islice((msg for msg in specimens.messagePool.keys() if len(msg.data) < 200), 100))
    # available for output:
    # * nemere.utils.reportWriter.writeSegmentedMessages2CSV
    # * from netzob.Export.WiresharkDissector.WiresharkDissector import WiresharkDissector
    #   WiresharkDissector.dissectSymbols(refinedSymbols, 'ari.lua')

    if args.interactive:

        print("\nAccess inferred symbols via variables: symbols, refinedSymbols")
        print("Access inferred message segments via variables: segmentsPerMsg, refinedPerMsg\n")
        IPython.embed()
