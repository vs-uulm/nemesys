import argparse
from os.path import isfile

from validation.dissectorMatcher import MessageComparator
from inference.segments import MessageSegment
from utils.loader import SpecimenLoader



debug = False





if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Generate segments of messages and evaluate against tshark dissectors.')
    parser.add_argument('pcapfilename', help='pcapfilename')
    parser.add_argument('-i', '--interactive', help='open ipython prompt after finishing the analysis.',
                        action="store_true")
    parser.add_argument('-s', '--sigma', type=float, help='sigma for noise reduction (gauss filter)')
    args = parser.parse_args()
    if not isfile(args.pcapfilename):
        print('File not found: ' + args.pcapfilename)
        exit(1)

    sigma = 0.6 if not args.sigma else args.sigma

    print("Load messages...")
    specimens = SpecimenLoader(args.pcapfilename, 2, True)
    comparator = MessageComparator(specimens, 2, True, debug=debug)

    segments = list()
    for rmsg, fmt in comparator.dissections.items():
        fends = MessageComparator.fieldEndsFromLength([f[1] for f in fmt])
        for (ftype, flen), fend in zip(fmt, fends):
            newSeg = MessageSegment()



