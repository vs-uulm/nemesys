import argparse
from os import makedirs
from os.path import isfile, basename, join, splitext, exists

import IPython

from inference.segmentHandler import originalRefinements, baseRefinements, pcaPcaRefinements, pcaMocoRefinements
from utils.evaluationHelpers import analyses, cacheAndLoadDC, annotateFieldTypes, reportFolder

debug = False

# fix the analysis method to VALUE
analysisTitle = 'value'
# fix the distance method to canberra
distance_method = 'canberra'
# use NEMESYS segments
tokenizer = 'nemesys'
# tokenizer = '4bytesfixed'

refinementMethods = [
    "raw",      # unrefined NEMESYS
    "original", # WOOT2018 paper
    "base",     # moco+splitfirstseg
    "PCA",      # 2-pass PCA
    "PCA1",     # 1-pass PCA
    "PCAmoco"   # 2-pass PCA+moco
    ]



if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Cluster NEMESYS segments of messages according to similarity.')
    parser.add_argument('pcapfilename', help='Filename of the PCAP to load.')
    parser.add_argument('-i', '--interactive', help='Show interactive plot instead of writing output to file and '
                                                    'open ipython prompt after finishing the analysis.',
                        action="store_true")
    parser.add_argument('-s', '--sigma', type=float, help='Sigma for noise reduction (gauss filter) in NEMESYS,'
                                                          'default: 0.9')
    parser.add_argument('-p', '--with-plots',
                        help='Generate plots of true field types and their distances.',
                        action="store_true")
    parser.add_argument('-r', '--refinement', help='Select segment refinement method.', choices=refinementMethods)
    args = parser.parse_args()

    if not isfile(args.pcapfilename):
        print('File not found: ' + args.pcapfilename)
        exit(1)
    pcapbasename = basename(args.pcapfilename)
    withPlots = args.with_plots

    analyzerType = analyses[analysisTitle]
    analysisArgs = None

    # # # # # # # # # # # # # # # # # # # # # # # #
    # cache/load the DistanceCalculator to/from the filesystem
    #

    doCache = True
    if args.refinement == "original":
        refinementCallback = originalRefinements
    elif args.refinement == "base":
        refinementCallback = baseRefinements
    elif args.refinement == "PCA":
        refinementCallback = pcaPcaRefinements
    elif args.refinement == "PCAmoco":
        refinementCallback = pcaMocoRefinements
    elif args.refinement == "raw":
        refinementCallback = None
    else:
        print("Unknown refinement", args.refinement, "\nAborting")
        exit(2)
    specimens, comparator, inferredSegmentedMessages, dc, segmentationTime, dist_calc_segmentsTime = cacheAndLoadDC(
        args.pcapfilename, analysisTitle, tokenizer, debug, analyzerType, analysisArgs, args.sigma, True,
        refinementCallback=refinementCallback, disableCache=not doCache
        )

    trueSegmentedMessages = {msgseg[0].message: msgseg
                             for msgseg in annotateFieldTypes(analyzerType, analysisArgs, comparator)}
    # # # # # # # # # # # # # # # # # # # # # # # #

    reportFolder = join(reportFolder, splitext(pcapbasename)[0])
    if not exists(reportFolder):
        makedirs(reportFolder)



    # Experiment: How to slice segments from between zero values.

    # mSlice = (0,40) # dhcp start
    # mSlice = (220, 1000)  # dhcp middle
    mSlice = (0, 1000)
    for msgsegs in inferredSegmentedMessages:
        startZ = 10  # dhcp: 0, 5, 60, 220, 230
        mdata = msgsegs[0].message.data  # type: bytes
        betweenZeros = [None, None]
        while startZ < len(mdata):
            aZero = mdata.find(b"\x00", startZ)
            if aZero < 0:
                break  # no zeros found
            bZero = mdata.find(b"\x00", aZero + 1)
            if bZero > aZero + 1:  # some non-zeros inbetween
                betweenZeros = aZero + 1, bZero
                break
            startZ += 1
        if any(bz is None for bz in betweenZeros):
            betweenZeros = None
        # limit to message slice
        if betweenZeros is not None and any(mSlice[0] > bz or bz > mSlice[1] for bz in betweenZeros):
            betweenZeros = None
        # compare zeros-slicing to nemesys segments:
        comparator.pprint2Interleaved(msgsegs[0].message, [infs.nextOffset for infs in msgsegs],
                                      mark=betweenZeros, messageSlice=mSlice)

    # if its a single zero: add to previous slice if (n bytes) before are chars (extended definition),
    #   otherwise add to subsequent segment.

    # single non-zeros?

    # do nemesys on the resulting non-zero slices to create segments.

    # then refine by PCA, ...


    if args.interactive:
        from tabulate import tabulate
        # globals().update(locals())
        IPython.embed()




