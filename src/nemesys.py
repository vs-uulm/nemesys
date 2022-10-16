"""
Infer messages from PCAPs by the NEMESYS approach (BCDG-segmentation)
and write inference result to the terminal. Finally drop to an IPython console
and expose API to interact with the result.

Usenix WOOT 2018.
"""

import argparse, time
from os.path import isfile
import IPython

from utils.loader import SpecimenLoader
from inference.segmentHandler import bcDeltaGaussMessageSegmentation, refinements, symbolsFromSegments




if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Generate segments of messages using the NEMESYS method and evaluate against tshark dissectors: '
                    'Write a report containing the FMS for each message and other evaluation data.')
    parser.add_argument('pcapfilename', help='pcapfilename')
    parser.add_argument('-i', '--interactive', help='open ipython prompt after finishing the analysis.',
                        action="store_true")
    parser.add_argument('-s', '--sigma', type=float, help='sigma for noise reduction (gauss filter)')
    parser.add_argument('-l', '--layer', type=int, default=2,
                        help='Protocol layer relative to IP to consider. Default is 2 layers above IP '
                             '(typically the payload of a transport protocol).')
    parser.add_argument('-r', '--relativeToIP', default=False, action='store_true')
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
    refinedPerMsg = refinements(segmentsPerMsg, None)
    runtimeRefinement = time.time() - startsegmentation

    print('Segmented and refined in {:.3f}s'.format(time.time() - startsegmentation))

    symbols = symbolsFromSegments(segmentsPerMsg)
    refinedSymbols = symbolsFromSegments(refinedPerMsg)

    # TODO output (colored?) visualization on terminal

    print("Access inferred symbols via variables: symbols, refinedSymbols")
    print("Access inferred message segments via variables: segmentsPerMsg, refinedPerMsg")

    IPython.embed()