from typing import Tuple, Iterable, Sequence, Union
from tabulate import tabulate

from netzob.Common.Utils.MatrixList import MatrixList
from netzob.Model.Vocabulary.Messages.AbstractMessage import AbstractMessage

from inference.segments import MessageSegment
from inference.templates import DistanceCalculator, Template
from visualization import bcolors as bcolors


def printMatrix(lines: Iterable[Iterable], headers: Iterable=None):
    ml = MatrixList()
    if headers:
        ml.headers = headers

    strlines = [ [ "{:0.3f}".format(cell) if isinstance(cell, float) else str(cell) for cell in row] for row in lines ]
    ml.extend(strlines)
    print(ml)


def alignDescreteValues(listA: list, listB: list) -> Tuple[list, list]:
    """
    Insert None-elements in both lists to place each value in the interval of the first list's values
    at index i like (i-1, i].

    In other words: align B to A with b <= a for all b in B, a in A.

    As a consequence exchangin A and B in the parameters will yield a different result.

    :param listA: the dominant list
    :param listB: the recessive list
    :return: two lists aligned by inserted Nones.
        The gapped dominant list is the first in the tuple.
        Each of its values will be larger or equal to all values of the recessive gapped list up to the same index.
    """
    rest = listB.copy()
    newA = list()
    newB = list()
    for valA in listA:
        consume = 0  # stays 0 until something is to consume in rest
        while len(rest) > consume and rest[consume] <= valA:
            consume += 1  # items at beginning of rest <= current valA

        if consume == 0:
            newA.append(valA)
            newB.append(None)
        if consume > 0:
            newA.extend([None]*(consume-1) + [valA])
            newB.extend(rest[:consume])
        rest = rest[consume:]
    if len(rest) > 0:
        newA.extend([None]*len(rest))
        newB.extend(rest)

    return newA, newB


def tabuSeqOfSeg(sequence: Sequence[Sequence[MessageSegment]]):
    print(tabulate(((sg.bytes.hex() if sg is not None else '' for sg in msg) for msg in sequence),
                   headers=range(len(sequence[0])), showindex="always", disable_numparse=True))



def resolveIdx2Seg(dc: DistanceCalculator, segseq: Sequence[Sequence[int]]):
    """
    Prints tabulated hex representations of (aligned) sequences of indices.

    :param dc: DistanceCalculator to use for resolving indices to MessageSegment objects.
    :param segseq: list of segment indices (from raw segment list) per message.
    """
    print(tabulate([[dc.segments[s].bytes.hex() if s != -1 else None for s in m]
                        for m in segseq], disable_numparse=True, headers=range(len(segseq[0]))))


def printMarkedBytesInMessage(message: AbstractMessage, markStart, markEnd, subStart=0, subEnd=None):
    if subEnd is None:
        subEnd = len(message.data)
    assert markStart >= subStart
    assert markEnd <= subEnd
    sub = message.data[subStart:subEnd]
    relMarkStart = markStart-subStart
    relMarkEnd = markEnd-subStart
    colored = \
        sub[:relMarkStart].hex() + \
        bcolors.colorizeStr(
            sub[relMarkStart:relMarkEnd].hex(),
            10
        ) + \
        sub[relMarkEnd:].hex()
    print(colored)



def markSegmentInMessage(segment: Union[MessageSegment, Template]):
    if isinstance(segment, MessageSegment):
        printMarkedBytesInMessage(segment.message, segment.offset, segment.nextOffset)
    else:
        for bs in segment.baseSegments:
            markSegmentInMessage(bs)



