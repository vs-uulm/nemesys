from itertools import chain
from time import strftime
from typing import Tuple, Iterable, Sequence, Dict, List, Union

from tabulate import tabulate

from netzob.Common.Utils.MatrixList import MatrixList
from netzob.Model.Vocabulary.Messages.AbstractMessage import AbstractMessage

from nemere.inference.segments import MessageSegment
from nemere.inference.templates import DistanceCalculator, Template
from nemere.validation.dissectorMatcher import MessageComparator
from nemere.visualization import bcolors as bcolors


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


def markSegNearMatch(segment: Union[Iterable[MessageSegment], MessageSegment, Template],
                     segmentedMessages: List[Sequence[MessageSegment]],
                     comparator: MessageComparator,
                     withContext: Union[bool,int]=False):
    """
    Print messages with the given segment in each message marked (underlined).
    Supports Templates by resolving them to their base segments.

    :param comparator: Comparator representing the true message dissections.
    :param withContext: if a integer value, print this number of bytes as context before and after the segment.
    :param segmentedMessages: the inferred segments (list of tuples of messages) to overlay as colors.
    :param segment: list of segments that should be printed, i. e.,
        marked within the print of the message it is originated from.
    """
    if isinstance(segment, Template):
        segs = segment.baseSegments
    elif isinstance(segment, Iterable):
        segs = segment
    else:
        segs = [segment]

    # print()  # one blank line for visual structure
    for seg in segs:
        inf4seg = inferred4segment(seg, segmentedMessages)
        if isinstance(withContext, int):
            context = (seg.offset - withContext, seg.nextOffset + withContext)
        else:
            context = None
        cprinter = ComparingPrinter(comparator, [inf4seg])
        cprinter.toConsole([seg.message], (seg.offset, seg.nextOffset), context)

    # # a simpler approach - without true fields marked as spaces
    # markSegmentInMessage(segment)

    # # get field number of next true field
    # tsm = trueSegmentedMessages[segment.message]  # type: List[MessageSegment]
    # fsnum, offset = 0, 0
    # while offset < segment.offset:
    #     offset += tsm[fsnum].offset
    #     fsnum += 1
    # markSegmentInMessage(trueSegmentedMessages[segment.message][fsnum])

    # # limit to immediate segment context
    # posSegMatch = None  # first segment that starts at or after the recognized field
    # for sid, seg in enumerate(trueSegmentedMessages[segment.message]):
    #     if seg.offset > segment.offset:
    #         posSegMatch = sid
    #         break
    # posSegEnd = None  # last segment that ends after the recognized field
    # for sid, seg in enumerate(trueSegmentedMessages[segment.message]):
    #     if seg.nextOffset > segment.nextOffset:
    #         posSegEnd = sid
    #         break
    # if posSegMatch is not None:
    #     contextStart = max(posSegMatch - 2, 0)
    #     if posSegEnd is None:
    #         posSegEnd = posSegMatch
    #     contextEnd = min(posSegEnd + 1, len(trueSegmentedMessages))


def inferred4segment(segment: MessageSegment, segmentedMessages: List[Sequence[MessageSegment]]) \
        -> Sequence[MessageSegment]:
    """
    Determine all the segments from segmentedMessages in the message of the one given segment.

    :param segment: The input segment.
    :param segmentedMessages: List of segmented messages to search in.
    :return: All inferred segments for the message which the input segment is from.
    """
    return next(msegs for msegs in segmentedMessages if msegs[0].message == segment.message)


class SegmentPrinter(object):
    """
    Printing of inferred segments within messages without ground truth,
    """

    def __init__(self, segmentsPerMsg: Iterable[Iterable[MessageSegment]]):
        """
        :param segmentsPerMsg: The segments that should be visualized by color changes or other optical features.
        """
        self._segmentedMessages = {msg[0].message:msg
                                   for msg in segmentsPerMsg}  # type: Dict[AbstractMessage, List[MessageSegment]]

    @staticmethod
    def _sliceMessageData(message: AbstractMessage, messageSlice: Tuple[Union[int,None],Union[int,None]]=None):
        """
        Slices a message into a selected substring and returns the absolute offsets of the cuts.

        :param message: Message to slice something out.
        :param messageSlice: Tuple used as parameters of the slice of a message.
            Use None to create an open slice (up to the beginning or end of the message).
        :return:
        """
        msglen = len(message.data)
        absSlice = (
            messageSlice[0] if messageSlice is not None and messageSlice[0] is not None else 0,
            messageSlice[1] if messageSlice is not None and messageSlice[1] is not None else msglen
        )
        dataSnip = message.data if messageSlice is None else message.data[slice(*messageSlice)]
        return dataSnip, absSlice

    @staticmethod
    def _prepareMark(mark: Union[Tuple[int,int], MessageSegment], absSlice: Tuple[int,int]):
        if mark is not None:
            if isinstance(mark, MessageSegment):
                mark = mark.offset, mark.nextOffset
            assert mark[0] >= absSlice[0], repr(mark) + "not valid with message slice" + repr(absSlice)
            assert mark[1] <= absSlice[1], repr(mark) + "not valid with message slice" + repr(absSlice)
        return mark

    # noinspection PyMethodMayBeStatic,PyUnusedLocal
    def _trueFieldEnds(self, message: AbstractMessage):
        """
        Just a dummy, for subcalsses to overwrite if there is ground truth to be printed there.

        :param message: is ignored
        :return: always a empty tuple
        """
        return ()

    def _inferredFieldStarts(self, message: AbstractMessage):
        ifs = [seg.offset for seg in self._segmentedMessages[message]]
        return ifs

    def _inferredFieldEnds(self, message: AbstractMessage):
        ife = [seg.nextOffset for seg in self._segmentedMessages[message]]
        return ife

    def toConsole(self, selectMessages: Iterable[AbstractMessage]=None, mark: Union[Tuple[int,int], MessageSegment]=None,
                  messageSlice: Tuple[Union[int,None],Union[int,None]]=None):
        """
        :param selectMessages: The messages from which to print the byte hex values. Also used to look up the
            true field boundaries, if any, to mark by spaces between in the printed byte hex values.
        :param mark: Start and end indices of a range to mark by underlining.
        :param messageSlice: Tuple used as parameters of the slice builtin to select a subset of all messages to print.
            Use None to create an open slice (up to the beginning or end of the message).
        """
        import nemere.visualization.bcolors as bc

        for msg in selectMessages if selectMessages is not None else self._segmentedMessages.keys():
            dataSnip, absSlice = SegmentPrinter._sliceMessageData(msg, messageSlice)
            mark = SegmentPrinter._prepareMark(mark, absSlice)

            tfe = self._trueFieldEnds(msg)
            # inferred segments starts and ends (not necessarily covering the whole message!)
            ifs = self._inferredFieldStarts(msg)
            ife = self._inferredFieldEnds(msg)

            hexdata = list()  # type: List[str]
            lastcolor = None
            for po, by in enumerate(dataSnip, absSlice[0]):
                # end mark
                if mark is not None and po == mark[1]:
                    hexdata.append(bc.ENDC)
                    # restart color after mark end
                    if lastcolor is not None and lastcolor < po and po not in ifs and po not in ife:
                        hexdata.append(bc.eightBitColor(lastcolor % 231 + 1))

                # have a space in place of each true field end in the hex data.
                if po in tfe:
                    hexdata.append(' ')

                # clear color at segment end
                if po in ife:
                    lastcolor = None
                    hexdata.append(bc.ENDC)
                    # restart mark after color change
                    if mark is not None and mark[0] < po < mark[1]:
                        hexdata.append(bc.UNDERLINE)

                # have a different color per each inferred field
                if po in ifs:
                    assert lastcolor is None, "Some segment overlap prevented unambiguous coloring."
                    if po < absSlice[1]:
                        lastcolor = po
                        hexdata.append(bc.eightBitColor(po % 231 + 1))

                # start mark
                if mark is not None and po == mark[0]:
                    hexdata.append(bc.UNDERLINE)

                # add the actual value
                hexdata.append('{:02x}'.format(by))
            hexdata.append(bc.ENDC)

            print(''.join(hexdata).strip())

    _basestyles = ["every node/.style={font=\\ttfamily, text height=.7em, outer sep=0, inner sep=0}",
                   "tfe/.style={draw, minimum height=1.2em, thick}",
                   "tfelabel/.style={rotate=-20, anchor=north west}",
                   "nonelabel/.style={}"]
    _texhead = "\n\\begin{tikzpicture}[node distance=0pt, yscale=2,\n"
    _texfoot = """
    \end{tikzpicture}

    \\centering
    \\bigskip\ninferred fields: framed box

    """
    _tfemarker = '1ex '

    # noinspection PyUnusedLocal,PyMethodMayBeStatic
    def _msgoffs2label(self, msg, po):
        return ""

    def toTikz(self, selectMessages: Iterable[AbstractMessage] = None, styles = None):
        if styles is None:
            styles = type(self)._basestyles.copy()
        else:
            styles = type(self)._basestyles.copy() + styles

        # start filling the texcode variable
        texcode = type(self)._texhead
        texcode += ",\n".join(styles) + "]"
        for msgid, msg in enumerate(selectMessages if selectMessages is not None else self._segmentedMessages.keys()):
            # true fields infos
            tfe = self._trueFieldEnds(msg)
            # inferred segments list (not necessarily covering the whole message!)
            isegs = self._segmentedMessages[msg] if msg in self._segmentedMessages else []

            hexdata = list()  # type: List[str]
            hexdata.append('\n\n\\coordinate(m{}f0) at (0,{});'.format(msgid, -msgid))
            for po, by in enumerate(msg.data, start=1):
                # add the actual value
                hexdata.append('\\node[right={}of m{}f{}, {}] (m{}f{}) {{{:02x}}};'.format(
                    # have a 1ex space in place of each true field end in the hex data.
                    type(self)._tfemarker if po - 1 in tfe else '', msgid, po - 1,
                    # style for some label at this offset
                    self._msgoffs2label(msg, po),
                    msgid, po, by)
                )
            texcode += '\n'.join(hexdata)

            # have a frame around each inferred field
            fitnodes = list()
            for seg in isegs:
                fitnodes.append(
                    f'\\node[fit=(m{msgid}f{seg.offset + 1})(m{msgid}f{seg.nextOffset}), tfe] {{}};'
                )
            texcode += '\n' + '\n'.join(fitnodes)

        texcode += type(self)._texfoot
        return texcode + "\n"

    def toTikzFile(self, selectMessages: Iterable[AbstractMessage] = None, styles = None, folder = None):
        from os.path import join, isdir, exists
        if folder is None:
            from ..utils.evaluationHelpers import reportFolder
            folder = reportFolder
        if not isdir(folder):
            raise NotADirectoryError(
                "The reports folder {} is not a directory. Reports cannot be written there.".format(
                    folder))
        print('Write tikz to ' + folder)
        filename = join(folder, 'inferredMessages.tikz')
        if exists(filename):
            print("File {} already exists, adding date and proceed...".format(filename))
            filename = filename + strftime("_%Y-%m-%d_%H%M%S")
            # file could still exist
            if exists(filename):
                raise FileExistsError("File already exists. Abort write of tikz file.")
        with open(filename, 'w') as tikzfile:
            tikzfile.write(self.toTikz(selectMessages, styles))


class ComparingPrinter(SegmentPrinter):
    """
    Routines to generate beautiful human-readable representations of true and inferred message syntax for comparison.
    """
    def __init__(self, comparator: MessageComparator, segmentsPerMsg: Sequence[Sequence[MessageSegment]]):
        super().__init__(segmentsPerMsg)
        self._comparator = comparator
        """map of messages to their inferred segments, filled by _mapMessages2Segments."""

    def __colorlabels(self, selectMessages: List[AbstractMessage]):
        """Needs to return a set of all possible color labels to be returned by _offset2colorlabel."""
        return {t[0] for msg in selectMessages for t in
                self._comparator.parsedMessages[self._comparator.messages[msg]].getTypeSequence()}

    def _offset2colorlabel(self, message):
        """offset2type: style label for the field type"""
        pm = self._comparator.parsedMessages[self._comparator.messages[message]]
        typeSequence = pm.getTypeSequence()
        return list(chain.from_iterable([lab] * lgt for lab, lgt in typeSequence))

    def _offset2textlabel(self, message):
        """offset2name: true field name as labels"""
        pm = self._comparator.parsedMessages[self._comparator.messages[message]]
        trueFieldNameMap = dict()
        offset = 0
        for name, lgt in pm.getFieldSequence():
            trueFieldNameMap[offset] = name.replace("_", "\\_")
            offset += lgt
        return trueFieldNameMap

    def _trueFieldEnds(self, message: AbstractMessage):
        pm = self._comparator.parsedMessages[self._comparator.messages[message]]
        typeSequence = pm.getTypeSequence()
        # typeSequence = self.dissections[self._comparator.messages[message]]  # dissections uses RawMessage as keys
        return MessageComparator.fieldEndsFromLength([l for t, l in typeSequence])

    _texfoot = """
    \end{tikzpicture}

    \\centering
    \\bigskip\ntrue fields: SPACE | inferred fields: framed box

    """


