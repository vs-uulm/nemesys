"""
Write a topology plot to visualize where centers of true field types are and an type-separation histogram per true field type.
Helps in determining which field types may be distinct enough to be later recognized from a template generated from this ground truth.
Use groundtruth about field segmentation by dissectors and determine the medoid of all segments of one data type.

Takes a PCAP trace of a known protocol, dissects each message into their fields, and yields segments from each of them.
These segments get analyzed by the "value" analysis method which is used as feature to determine their similarity.
Real field types are separated using ground truth and the quality of this separation is visualized.
"""

import argparse
from itertools import chain
from os.path import isfile

from nemere.inference.analyzers import *
from nemere.inference.segmentHandler import segments2types
from nemere.inference.templates import Template, TemplateGenerator, MemmapDC
from nemere.utils.evaluationHelpers import annotateFieldTypes
from nemere.utils.loader import SpecimenLoader
from nemere.validation.dissectorMatcher import MessageComparator
from nemere.visualization.distancesPlotter import DistancesPlotter
from nemere.visualization.multiPlotter import MultiMessagePlotter

debug = False

# fix the analysis method to VALUE
analysisTitle = 'value'
analyzerType = Value
analysisArgs = None
# fix the distance method to canberra
distance_method = 'canberra'





if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Analyze fields as segments of messages and plot field type identification quality.')
    parser.add_argument('pcapfilename', help='Filename of the PCAP to load.')
    parser.add_argument('-i', '--interactive', help='Show interactive plot instead of writing output to file and '
                                                    'open ipython prompt after finishing the analysis.',
                        action="store_true")
    parser.add_argument('-l', '--layer', type=int, default=2,
                        help='Protocol layer relative to IP to consider. Default is 2 layers above IP '
                             '(typically the payload of a transport protocol).')
    parser.add_argument('-r', '--relativeToIP', default=False, action='store_true')
    args = parser.parse_args()
    if not isfile(args.pcapfilename):
        print('File not found: ' + args.pcapfilename)
        exit(1)

    # dissect and label messages
    print("Load messages...")
    specimens = SpecimenLoader(args.pcapfilename, layer=args.layer, relativeToIP=args.relativeToIP)
    comparator = MessageComparator(specimens, layer=args.layer, relativeToIP=args.relativeToIP, debug=debug)

    # segment messages according to true fields from the labels
    print("Segmenting messages...")
    segmentedMessages = annotateFieldTypes(analyzerType, analysisArgs, comparator)
    # # all segments
    filteredSegments = list(chain.from_iterable(segmentedMessages))

    print("Calculating dissimilarities...")
    dc = MemmapDC(filteredSegments)

    print("Generate type groups and templates...")
    typegroups = segments2types(filteredSegments)
    typelabels = list(typegroups.keys())
    templates = TemplateGenerator.generateTemplatesForClusters(dc, [typegroups[ft] for ft in typelabels])

    # labels of templates
    labels = ['Noise'] * len(dc.segments)  # TODO check: list of segment indices (from raw segment list) per message
    #                                           ^ here the question is, whether we like to print resolved segements or representatives
    for l, t in zip(typelabels, templates):
        labels[dc.segments2index([t.medoid])[0]] = l

    print("Plot dissimilarities...")
    sdp = DistancesPlotter(specimens, 'distances-templatecenters', args.interactive)
    sdp.plotSegmentDistances(dc, numpy.array(labels))
    sdp.writeOrShowFigure()

    print("Plot histograms...")
    # import matplotlib.pyplot as plt
    mmp = MultiMessagePlotter(specimens, 'histo-templatecenters', len(templates))
    for figIdx, (typlabl, typlate) in enumerate(zip(typelabels, templates)):
        # h_match histogram of distances to medoid for segments of typlate's type
        match = [di for di, of in typlate.distancesToMixedLength(dc)]
        # abuse template to get distances to non-matching field types
        filtermismatch = [typegroups[tl] for tl in typelabels if tl != typlabl]
        mismatchtemplate = Template(typlate.medoid, list(chain.from_iterable(filtermismatch)))
        # h_mimatch histogram of distances to medoid for segments that are not of typlate's type
        mismatch = [di for di, of in mismatchtemplate.distancesToMixedLength(dc)]
        # plot both histograms h overlapping (i.e. for each "bin" have two bars).
        # the bins denote ranges of distances from the medoid
        mmp.histoToSubfig(figIdx, [match, mismatch], bins=numpy.linspace(0, 1, 20), label=[typlabl, 'not ' + typlabl])
    # plot in subfigures on one page
    mmp.writeOrShowFigure()


    if args.interactive:
        IPython.embed()
