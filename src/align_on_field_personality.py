"""
Use groundtruth about field segmentation by dissectors and align segments
on the similarity of their feature "personality".

Takes a PCAP trace of a known protocol, dissects each message into their fields, and yields segments from each of them.
These segments get analyzed by the given analysis method which is used as feature to determine their similarity.
Similar fields are then aligned.
"""

import argparse, IPython
from os.path import isfile

from inference.templates import TemplateGenerator
from inference.analyzers import *
from inference.segmentHandler import annotateFieldTypes, groupByLength, segments2types
from validation.dissectorMatcher import MessageComparator
from utils.loader import SpecimenLoader
from characterize_fieldtypes import analyses, segments2clusteredTypes, filterSegments, labelForSegment

debug = False





analysis_method = 'progcumudelta'
distance_method = 'canberra'





if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Analyze fields as segments of messages and align on the similarity of their feature "personality".')
    parser.add_argument('pcapfilename', help='Filename of the PCAP to load.')
    parser.add_argument('-i', '--interactive', help='Open ipython prompt after finishing the analysis.',
                        action="store_true")
    # parser.add_argument('analysis', type=str,
    #                     help='The kind of analysis to apply on the messages. Available methods are: '
    #                     + ', '.join(analyses.keys()) + '.')
    # parser.add_argument('--parameters', '-p', help='Parameters for the analysis.')
    args = parser.parse_args()

    if not isfile(args.pcapfilename):
        print('File not found: ' + args.pcapfilename)
        exit(1)
    # if args.analysis not in analyses:
    #     print('Analysis {} unknown. Available methods are:\n' + ', '.join(analyses.keys()) + '.')
    #     exit(2)
    # analyzerType = analyses[args.analysis]
    # analysisArgs = args.parameters
    # analysisTitle = "{}{}".format(args.analysis, "" if not analysisArgs else " ({})".format(analysisArgs))
    analyzerType = analyses[analysis_method]
    analysisArgs = None
    analysisTitle = analysis_method


    # dissect and label messages
    print("Load messages...")
    specimens = SpecimenLoader(args.pcapfilename, 2, True)
    comparator = MessageComparator(specimens, 2, True, debug=debug)

    # segment messages according to true fields from the labels
    print("Segmenting messages...", end=' ')
    segmentedMessages = annotateFieldTypes(analyzerType, analysisArgs, comparator)
    segsByLen = groupByLength(segmentedMessages)
    print("done.")


    for length, segments in segsByLen.items():  # type: int, List[MessageSegment]
        filteredSegments = filterSegments(segments)

        # if length < 3:
        #     continue
        # if len(filteredSegments) < 16:
        #     print("Too few relevant segments for length {} after Filtering. {} segments remaining:".format(
        #         length, len(filteredSegments)
        #     ))
        #     for each in filteredSegments:
        #         print("   ", each)
        #     print()
        #     continue

        typeDict = segments2types(filteredSegments)

        print("Calculate distances...")
        tg = TemplateGenerator(filteredSegments, distance_method)

        print("Aligning...")


        segmentGroups = segments2clusteredTypes(tg, analysisTitle, min_cluster_size=5)
        # re-extract cluster labels for segments
        labels = numpy.array([
            labelForSegment(segmentGroups, seg) for seg in tg.segments
        ])

        # print("Prepare output...")

    if args.interactive:
        IPython.embed()




