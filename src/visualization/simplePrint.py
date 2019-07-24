from typing import Tuple, Iterable, Sequence, Dict, List, Union
from tabulate import tabulate

from netzob.Common.Utils.MatrixList import MatrixList
from netzob.Model.Vocabulary.Messages.AbstractMessage import AbstractMessage

from inference.segments import MessageSegment, TypedSegment, AbstractSegment
from inference.fieldTypes import BaseTypeMemento, RecognizedField, RecognizedVariableLengthField
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


def segmentFieldTypes(sequence: Sequence[TypedSegment],
                      recognizedFields: Dict[Union[BaseTypeMemento, str], List[RecognizedVariableLengthField]],
                      fieldNumStart=0):
    """
    Visualization for recognized field type templates in message.
    Abbreviates long zero sequences.

    :param sequence:
    :param recognizedFields:
    :param fieldNumStart: Start numbering of fields with this offset.
    :return:
    """
    sortedRecognizedFieldKeys = sorted(recognizedFields.keys(),
                                       key=lambda x: x.fieldtype if isinstance(x, BaseTypeMemento) else x)
    ftmlines = list()
    for ftm in sortedRecognizedFieldKeys:
        poscon = recognizedFields[ftm]
    # for ftm, poscon in recognizedFields.items():
        ftmline = [ftm.fieldtype if isinstance(ftm, BaseTypeMemento) else ftm]
        conline = [" ^ conf."]
        posconMap = {o: recognized
                     for recognized in poscon
                     for o in range(recognized.position, recognized.end)}  # type: Dict[int, RecognizedField]
        for seg in sequence:
            ftm4seg = ""
            con4seg = ""
            for offset in range(seg.offset, seg.nextOffset):
                if offset in posconMap:
                    if offset == posconMap[offset].position and posconMap[offset].end == posconMap[offset].position + 1:
                        ftm4seg += "()"
                        con4seg += " {:.1f}".format(posconMap[offset].confidence)
                    elif offset == posconMap[offset].position:
                        ftm4seg += "(-"
                        con4seg += " {:.1f}".format(posconMap[offset].confidence)
                    elif offset == posconMap[offset].end - 1:
                        ftm4seg += "-)"
                    else:
                        ftm4seg += "--"
                else:
                    ftm4seg += "  "

            ftmline.append(ftm4seg if set(ftm4seg) != {" "} else "")
            conline.append(con4seg if len(con4seg) > 0 else "")
        # only add lines if they have any content
        if any(ftmline[1:]):  # [cell != "" for cell in ftmline]
            ftmlines.append(ftmline)
            ftmlines.append(conline)

    import tabulate as tabmod
    tabmod.PRESERVE_WHITESPACE = True
    print(tabulate(
        (
            ["tft"] + [sg.fieldtype if sg is not None else '' for sg in sequence],
            ["bytes"] + ['' if sg is None else
                         "00..00" if set(sg.bytes) == {b"\x00"} else sg.bytes.hex() for sg in sequence],
            *ftmlines
        ),
        headers=[""] + list(range(fieldNumStart, fieldNumStart+len(sequence))), disable_numparse=True)
    )
    tabmod.PRESERVE_WHITESPACE = False


def printFieldContext(trueSegmentedMessages: Dict[AbstractMessage, Sequence[TypedSegment]],
                      recognizedField: RecognizedVariableLengthField):
    """
    Get and print true segment context around a selected recognition

    :param trueSegmentedMessages: true segments to use as reference for determining the context of recognizedField.
    :param recognizedField: The field for which to print the context.
    """
    posSegMatch = None  # first segment that starts at or after the recognized field
    for sid, seg in enumerate(trueSegmentedMessages[recognizedField.message]):
        if seg.offset > recognizedField.position:
            posSegMatch = sid
            break
    posSegEnd = None  # last segment that ends after the recognized field
    for sid, seg in enumerate(trueSegmentedMessages[recognizedField.message]):
        if seg.nextOffset > recognizedField.end:
            posSegEnd = sid
            break
    if posSegMatch is not None:
        contextStart = max(posSegMatch - 2, 0)
        if posSegEnd is None:
            posSegEnd = posSegMatch
        contextEnd = min(posSegEnd + 1, len(trueSegmentedMessages))
        segmentFieldTypes(trueSegmentedMessages[recognizedField.message][contextStart:contextEnd],
                          {recognizedField.template: [recognizedField]}, contextStart)


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



