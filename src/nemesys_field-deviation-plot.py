"""
Infer messages from PCAPs by the NEMESYS approach (BCDG-segmentation)
and plot the amount deviating boundaries to a histogram.

Usenix WOOT 2018.
"""

import argparse, time
from os.path import isfile

import matplotlib.pyplot as plt
import IPython

from nemere.validation.dissectorMatcher import MessageComparator
from nemere.utils.loader import SpecimenLoader
from nemere.visualization.singlePlotter import SingleMessagePlotter
from nemere.inference.segmentHandler import bcDeltaGaussMessageSegmentation, refinements, symbolsFromSegments


debug = False
"""Some modules and methods contain debug output that can be activated by this flag."""


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Generate segments of messages using the NEMESYS method and evaluate against tshark dissectors: '
                    'Plot the amount deviating boundaries to a histogram.')
    parser.add_argument('pcapfilename', help='pcapfilename')
    parser.add_argument('-i', '--interactive', help='open ipython prompt after finishing the analysis.',
                        action="store_true")
    parser.add_argument('-s', '--sigma', type=float, help='sigma for noise reduction (gauss filter)')
    parser.add_argument('-l', '--layer', type=int, default=2,
                        help='Protocol layer relative to IP to consider. Default is 2 layers above IP '
                             '(typically the payload of a transport protocol).')
    parser.add_argument('-r', '--relativeToIP', default=True, action='store_true')
    parser.add_argument('-c', '--columns', type=int, default=2,
                        help='Adjust width/aspect ratio for use in one USENIX column wide plot (1) or '
                             'for one USENIX column sideways leaving space for the caption (2)')
    args = parser.parse_args()
    if not isfile(args.pcapfilename):
        print('File not found: ' + args.pcapfilename)
        exit(1)

    sigma = 0.6 if not args.sigma else args.sigma

    print("Load messages...")
    specimens = SpecimenLoader(args.pcapfilename, layer=args.layer,
                               relativeToIP=args.relativeToIP)
    comparator = MessageComparator(specimens, layer=args.layer,
                               relativeToIP=args.relativeToIP,
                               failOnUndissectable=False, debug=debug)

    ########################

    print("Segment messages...")
    startsegmentation = time.time()

    inferenceTitle = 'bcDeltaGauss{:.1f}'.format(sigma)  # +hiPlateaus
    segmentsPerMsg = bcDeltaGaussMessageSegmentation(specimens, sigma)
    refinedPerMsg = refinements(segmentsPerMsg, unused=None)

    print('Segmented and refined in {:.3f}s'.format(time.time() - startsegmentation))

    symbols = symbolsFromSegments(segmentsPerMsg)
    refinedSymbols = symbolsFromSegments(refinedPerMsg)

    ########################

    print()

    # Adjust aspect ratio
    if args.columns == 1:
        plt.figure(figsize=(3.136, 3.136 * .667 + 0.14))  # for one USENIX column wide plot
    else:
        plt.figure(figsize=(7, 3.136+0.29))  # for one USENIX column sideways leaving space for the caption... (DHCP)

    smp = SingleMessagePlotter(specimens, 'distances-distribution_' + inferenceTitle, args.interactive)
    plt.tick_params(labelsize=7)
    plt.xlabel('Aligned Byte Position', fontdict={'fontsize': 7})
    smp.heatMapFieldComparison(comparator, refinedSymbols)
    smp.writeOrShowFigure()

    ########################

    if args.interactive:
        IPython.embed()

