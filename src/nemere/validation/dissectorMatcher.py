"""
DissectorMatcher, FormatMatchScore, and MessageComparator

Methods to comparison of a list of messages' inferences and their dissections
and match a message's inference with its dissector in different ways.
"""
from abc import ABC, abstractmethod
from itertools import chain
from typing import List, Tuple, Dict, Iterable, Generator, Union, Sequence
from collections import OrderedDict
import copy

import math, numpy
from netzob.Model.Vocabulary.Messages.L2NetworkMessage import L2NetworkMessage
from numpy import argmin

from netzob import all as netzob
from netzob.Model.Vocabulary.Messages.AbstractMessage import AbstractMessage

import nemere.visualization.bcolors as bcolors
from nemere.validation.messageParser import ParsedMessage, ParsingConstants
from nemere.inference.segments import MessageSegment, TypedSegment


# TODO find a suitable value
# messageparsetimeout = 600
messageparsetimeout = 60*120

def stop_process_pool(executor):
    # noinspection PyProtectedMember
    for pid, process in executor._processes.items():
        process.terminate()
    executor.shutdown()

class WatchdogTimeout(Exception):
    pass

class FormatMatchScore(object):
    """
    Object to hold all relevant data of an FMS.
    """
    def __init__(self, message = None, symbol = None):
        self.message = message
        self.symbol = symbol
        self.trueFormat = None
        self.score = None
        self.specificyPenalty = None
        self.matchGain = None
        self.specificy = None
        self.nearWeights = None
        self.meanDistance = None  # mean of "near" distances
        self.trueCount = None
        self.inferredCount = None
        self.exactCount = None
        self.nearCount = None
        self.exactMatches = None
        self.nearMatches = None


class BaseComparator(object):
    """Dummy for using nemere.utils.evaluationHelpers.CachedDistances with a unknown protocol."""
    import nemere.utils.loader as sl

    def __init__(self, specimens: sl.SpecimenLoader, layer: int = -1, relativeToIP: bool = False, debug = False):
        self.specimens = specimens
        self.messages = specimens.messagePool  # type: OrderedDict[AbstractMessage, netzob.RawMessage]
        """:type messages: OrderedDict[AbstractMessage, RawMessage]"""
        self.baselayer = specimens.getBaseLayerOfPCAP()
        self.debug = debug
        self._targetlayer = layer
        self._relativeToIP = relativeToIP

class MessageComparator(BaseComparator):
    """
    Formal and visual comparison of a list of messages' inferences and their dissections.

    Functions that are closely coupled to the dissection: Interfaces with tshark to configure the call to it by
    the parameters layer, relativeToIP, and failOnUndissectable and processes the output to directly know the
    dissection result.
    """
    import nemere.utils.loader as sl

    __messageCellCache = dict()  # type: Dict[(netzob.Symbol, AbstractMessage), List]

    def __init__(self, specimens: sl.SpecimenLoader, layer: int = -1, relativeToIP: bool = False,
                 failOnUndissectable=True, debug = False, dissectOneshot=True):
        super().__init__(specimens, layer, relativeToIP, debug)

        self._failOnUndissectable = failOnUndissectable

        # Cache messages that already have been parsed and labeled
        self._messageCache = dict()  # type: Dict[netzob.RawMessage, ]
        if dissectOneshot:
            self._dissections = ParsedMessage.parseOneshot(specimens, failOnUndissectable)
        else:
            self._dissections = self._dissectAndLabel(self.messages.values())


    def _dissectAndLabel(self, messages: Iterable[netzob.RawMessage]) \
            -> Dict[netzob.RawMessage, ParsedMessage]:
        """
        :param messages: List of messages to be dissected - needs to be hashable
        :return: dict of {message: format}, where format is a list of
            2-tuples describing the fields of this L2-message in their byte order
            each 2-tuple contains (field_type, field_length in byte)
        """
        assert isinstance(messages, Iterable)
        if isinstance(messages, Generator):
            messages = tuple(messages)
        if not self.baselayer in ParsingConstants.LINKTYPES.values():
            raise NotImplementedError('PCAP Linktype with number {} is unknown'.format(self.baselayer))

        labeledMessages = dict()
        toparse = [msg for msg in messages if msg not in self._messageCache]
        # for msg in toparse: print("MessageCache miss for {}".format(msg.data))
        mparsed = ParsedMessage.parseMultiple(toparse, self._targetlayer, self._relativeToIP,
                                              failOnUndissectable=self._failOnUndissectable, linktype=self.baselayer)
        for m, p in mparsed.items():
            try:
                self._messageCache[m] = p
            except NotImplementedError as e:
                if self._failOnUndissectable:
                    raise e

        for m in messages:
            if not isinstance(m, netzob.RawMessage):
                raise TypeError("Message needs to be a RawMessage to be parseable by tshark. Message type was ",
                                m.__class__.__name__)

            try:
                labeledMessages[m] = self._messageCache[m]
            except KeyError:  # something went wrong in the last parsing attempt of m
                if self._failOnUndissectable:
                    reparsed = ParsedMessage(m, self._targetlayer, self._relativeToIP,
                                             failOnUndissectable=self._failOnUndissectable)
                    self._messageCache[m] = reparsed
                    labeledMessages[m] = self._messageCache[m]

        ParsedMessage.closetshark()

        return labeledMessages


    @property
    def dissections(self) -> Dict[netzob.RawMessage, List[Tuple[str, int]]]:
        return {message: dissected.getTypeSequence() for message, dissected in self._dissections.items()}

    @property
    def parsedMessages(self) -> Dict[netzob.RawMessage, ParsedMessage]:
        return self._dissections

    @staticmethod
    def fieldEndsPerSymbol(nsymbol: netzob.Symbol, message: AbstractMessage):
        """
        we need to use concrete message values for each field to determine length
        otherwise gaps of alignment get wrongly cumulated

        :param nsymbol:
        :param message:
        :return: The field ends for the specific message and the given symbol
        """
        if not message in nsymbol.messages:
            raise ValueError('Message in input symbol unknown by this comparator.')

        from concurrent.futures.process import ProcessPoolExecutor
        from concurrent.futures import TimeoutError as FutureTOError

        # since netzob.Symbol.getMessageCells is EXTREMELY inefficient,
        # we try to have it run as seldom as possible by caching its output
        if (nsymbol, message) not in MessageComparator.__messageCellCache:
            # DHCP fails due to a very deep recursion here:
            #   Wrap Symbols.getMessageCell in watchdog: run it in process and abort it after a timeout.
            #   TODO Look for my previous solution in siemens repos
            #   Wait only messageparsetimeout seconds for Netzob's MessageParser to return the result
            with ProcessPoolExecutor(max_workers=1) as executor:
                try:
                    future = executor.submit(nsymbol.getMessageCells, encoded=False)
                    mcells = future.result(messageparsetimeout)  # dict of cells keyed by message
                    msgIdMap = {msg.id: msg for msg in nsymbol.messages}
                except FutureTOError:
                    stop_process_pool(executor)
                    raise WatchdogTimeout(f"Parsing of Netzob symbol {nsymbol.name} timed out after "
                                          f"{messageparsetimeout} seconds.")
            # Non-process call:
            # mcells = nsymbol.getMessageCells(encoded=False)  # dict of cells keyed by message
            # for msg, fields in mcells.items():
            #     MessageComparator.__messageCellCache[(nsymbol, msg)] = fields
            #
            # IPC breaks the identity check of messages, since the object instance needs to be copied.
            #   Look up and use the correct message instances.
            # TODO Keep in mind that this might break something since the fields
            #  still do contain references to the copied messages in:
            #       .messages and .parent.messages
            for msg, fields in mcells.items():
                MessageComparator.__messageCellCache[(nsymbol, msgIdMap[msg.id])] = fields

        mcontent = MessageComparator.__messageCellCache[(nsymbol, message)]
        nfieldlengths = [len(field) for field in mcontent]
        # nfieldlengths = [nf.domain.dataType.size[1]/8 for nf in nsymbol.fields]

        #####
        # determine the indices of the field ends in the message byte sequence
        nfieldends = MessageComparator.fieldEndsFromLength(nfieldlengths)
        return nfieldends


    def fieldEndsPerMessage(self, message: AbstractMessage):
        if type(message) == netzob.RawMessage:
            rawMessage = message
        else:
            rawMessage = self.specimens.messagePool[message]
        return MessageComparator.fieldEndsFromLength(
            [flen for t, flen in self.dissections[rawMessage]] )


    @staticmethod
    def fieldEndsFromLength(fieldlengths: List[int]) -> List[int]:
        """
        Converts a sequence of field lengths into the resulting byte positions.

        :param fieldlengths: list of field lengths
        :return: list of field boundary positions
        """
        fieldends = []
        fe = 0
        for fl in fieldlengths:
            fe += fl
            fieldends.append(fe)
        return fieldends


    @staticmethod
    def uniqueFormats(onlyformats: Iterable) -> List[List[Tuple[str, int]]]:
        """
        Filter input format list for only unique formats.

        :param onlyformats: a list of formats (tuples of (fieldtype, fieldlengthInBytes))
        :type onlyformats: Iterable[tuple[str, int]]
        :return: Unique formats
        """
        distinctFormats = list()
        for tf in onlyformats:
            if tf not in distinctFormats:
                distinctFormats.append(tf)
        return distinctFormats


    def pprint2Interleaved(self, message: AbstractMessage, segmentsPerMsg: Sequence[Sequence[MessageSegment]]=tuple(),
                           mark: Union[Tuple[int,int], MessageSegment]=None,
                           messageSlice: Tuple[Union[int,None],Union[int,None]]=None):
        """
        TODO deprecated: use ComparingPrinter directly!

        :param message: The message from which to print the byte hex values. Also used to look up the
            true field boundaries to mark by spaces between in the printed byte hex values.
        :param segmentsPerMsg: The segments that should be visualized by color changes.
        :param mark: Start and end indices of a range to mark by underlining.
        :param messageSlice: Tuple used as parameters of the slice builtin to select a subset of all messages to print.
            Use None to create an open slice (up to the beginning or end of the message).
        """
        from ..visualization.simplePrint import ComparingPrinter
        rawmsg = self.messages[message] if isinstance(message, L2NetworkMessage) else message
        cprinter = ComparingPrinter(self, segmentsPerMsg)
        cprinter.toConsole([rawmsg], mark, messageSlice)


    def __prepareMessagesForPPrint(self, symbols: Iterable[netzob.Symbol]) \
            -> List[List[Tuple[AbstractMessage, List[int], Union[List[int], WatchdogTimeout]]]]:
        """
        Iterate symbols and their messages, determine boundary lists of true and inferred formats.

        :param symbols:
        :return: list (symbols) of list (messages) of tuple with message, true, and inferred field ends
        """
        symfes = list()
        for sym in symbols:
            msgfes = list()
            for msg in sym.messages:  # type: netzob.L4NetworkMessage
                if not msg in self.messages:
                    raise ValueError('Message in input symbol unknown by this comparator.')
                l2msg = self.messages[msg]
                tformat = self.dissections[l2msg]
                msglen = len(msg.data)
                tfe = MessageComparator.fieldEndsFromLength([l for t, l in tformat])
                try:
                    # catch WatchdogTimeout in callers of fieldEndsPerSymbol
                    ife = [0] + MessageComparator.fieldEndsPerSymbol(sym, msg)
                    ife += [msglen] if ife[-1] < msglen else []
                except WatchdogTimeout as e:
                    ife = e
                msgfes.append((msg, tfe, ife))
            symfes.append(msgfes)
        return symfes


    def pprintInterleaved(self, symbols: List[netzob.Symbol]):
        """
        Print terminal output visualizing the interleaved true and inferred format upon the byte values
        of each message in the symbols.

        Also see self.lprintInterleaved doing the same for LaTeX code (and returning it instead of printing).

        :param symbols: Inferred symbols
        """
        import nemere.visualization.bcolors as bc

        for symfes in self.__prepareMessagesForPPrint(symbols):
            for msg, tfe, ife in symfes:
                hexdata = list()  # type: List[str]
                for po, by in enumerate(msg.data):
                    # have a space in place of each true field end in the hex data.
                    if po in tfe:
                        hexdata.append(' ')

                    # have a different color per each inferred field
                    if po in ife:
                        if po > 0:
                            hexdata.append(bc.ENDC)
                        if po < len(msg.data):
                            hexdata.append(bc.eightBitColor(po % 231 + 1))

                    # add the actual value
                    hexdata.append('{:02x}'.format(by))
                if isinstance(ife, WatchdogTimeout):
                    # handle Netzob WatchdogTimeout in fieldEndsPerSymbol
                    print(str(ife), end="   ")
                else:
                    hexdata.append(bc.ENDC)

                print(''.join(hexdata))
        print('true fields: SPACE | inferred fields: color change')


    def lprintInterleaved(self, symbols: List[netzob.Symbol]):
        """
        Generate LaTeX source code visualizing the interleaved true and inferred format upon the byte values
        of each message in the symbols.

        Also see self.pprintInterleaved doing the same for terminal output.

        :param symbols: Inferred symbols
        :return LaTeX code
        """
        tfemarker = '\\enspace '
        ifeendmarker = '}'
        ifestartmarker = '\\fbox{'

        texcode = ""
        for symfes in self.__prepareMessagesForPPrint(symbols):
            for msg, tfe, ife in symfes:
                hexdata = list()  # type: List[str]
                for po, by in enumerate(msg.data):
                    # have a space in place of each true field end in the hex data.
                    if po in tfe:
                        hexdata.append(tfemarker)

                    # have a different color per each inferred field
                    if not isinstance(ife, WatchdogTimeout) and po in ife:
                        if po > 0:
                            hexdata.append(ifeendmarker)
                        if po < len(msg.data):
                            hexdata.append(ifestartmarker)

                    # add the actual value
                    hexdata.append('{:02x}'.format(by))
                if isinstance(ife, WatchdogTimeout):
                    # handle Netzob WatchdogTimeout in fieldEndsPerSymbol
                    note = str(ife) + "\\hspace{3em}"
                else:
                    hexdata.append(ifeendmarker)
                    note = ""
                texcode += '\\noindent\n' + note + '\\texttt{' + ''.join(hexdata) + '}\n\n'
        texcode += '\\bigskip\ntrue fields: SPACE | inferred fields: framed box'
        return texcode


    def tprintInterleaved(self, symbols: Sequence[netzob.Symbol]):
        """
        Generate tikz source code visualizing the interleaved true and inferred format upon the byte values
        of each message in the symbols.

        requires \\usetikzlibrary{positioning, fit}

        Also see self.pprintInterleaved doing the same for terminal output.

        TODO cleanup: adapt and use nemere.visualization.simplePrint.ComparingPrinter (which is functionally equivalent)

        :param symbols: Inferred symbols
        :return LaTeX/tikz code
        """
        tfemarker = '1ex '
        texcode = ""

        ftlabels = set()
        for sym in symbols:
            for msg in sym.messages:
                pm = self.parsedMessages[self.messages[msg]]
                ftlabels.update(t[0] for t in pm.getTypeSequence())
        ftstyles = {lab: "fts" + lab.replace("_", "").replace(" ", "") for lab in ftlabels}  # field-type label to style name
        ftcolornames = {tag: "col" + tag[3:] for lab, tag in ftstyles.items() }  # style name to color name
        ftcolors = list()  # color definition
        for tag in ftcolornames.values():
            lightness = 0  # choose only light colors
            while lightness < .5:
                rgb = numpy.random.rand(3, )
                lightness = 0.5 * min(rgb) + 0.5 * max(rgb)
            # noinspection PyUnboundLocalVariable
            ftcolors.append( f"\definecolor{{{tag}}}{{rgb}}{{{rgb[0]},{rgb[1]},{rgb[2]}}}" )
        texcode += "\n        ".join(ftcolors) + "\n"

        styles = ["every node/.style={font=\\ttfamily, text height=.7em, outer sep=0, inner sep=0}",
                  "tfe/.style={draw, minimum height=1.2em, thick}", "tfelabel/.style={rotate=-20, anchor=north west}"]
        styles += [f"{sty}/.style={{fill={ftcolornames[sty]}}}" for sty in ftstyles.values() ]

        texcode += "\n\\begin{tikzpicture}[node distance=0pt, yscale=2,\n"
        texcode += ",\n".join(styles) + "]"
        for symid, symfes in enumerate(self.__prepareMessagesForPPrint(symbols)):
            for msgid, (msg, tfe, ife) in enumerate(symfes):
                pm = self.parsedMessages[self.messages[msg]]
                offset2type = list(chain.from_iterable( [lab]*lgt for lab, lgt in pm.getTypeSequence() ))
                offset2name = dict()
                offset = 0
                for name, lgt in pm.getFieldSequence():
                    offset2name[offset] = name.replace("_", "\\_")
                    offset += lgt

                smid = symid + msgid
                hexdata = list()  # type: List[str]
                hexdata.append('\n\n\\coordinate(m{}f0) at (0,{});'.format(smid, -smid))
                for po, by in enumerate(msg.data, start=1):
                    # add the actual value
                    hexdata.append('\\node[right={}of m{}f{}, {}{}] (m{}f{}) {{{:02x}}};'.format(
                        # have a 1ex space in place of each true field end in the hex data.
                        tfemarker if po-1 in tfe else '', smid, po-1,
                        # style for the field type
                        ftstyles[offset2type[po-1]],
                        f", label={{[tfelabel]below:\\sffamily\\tiny {offset2name[po-1]}}}" if po-1 in offset2name else "",
                        smid, po, by)
                    )
                texcode += '\n'.join(hexdata)

                # have a frame per each inferred field
                fitnodes = list()
                if isinstance(ife, WatchdogTimeout):
                    # handle Netzob WatchdogTimeout in fieldEndsPerSymbol
                    fitnodes.append(
                        '\\node[] at (m{}f0) {{{}}};'.format(smid, str(ife))
                    )
                else:
                    for pol, por in zip(ife[:-1], ife[1:]):
                        fitnodes.append(
                            '\\node[fit=(m{}f{})(m{}f{}), tfe] {{}};'.format(smid, pol+1, smid, por)
                        )
                        # TODO add the inferred field's "most true" type as label
                texcode += '\n' + '\n'.join(fitnodes)

        texcode += """
\end{tikzpicture}

\\centering
\\bigskip\ntrue fields: SPACE | inferred fields: framed box

True field type colors:\\\\
"""
        for lab, tag in ftstyles.items():
            texlab = lab.replace("_", "\\_")
            texcode += f"\\colorbox{{{ftcolornames[tag]}}}{{{texlab}}}\\\\\n"

        return texcode + "\n"


    def segment2typed(self, segment: MessageSegment) -> Tuple[float, Union[TypedSegment, MessageSegment]]:
        overlapRatio, overlapIndex, overlapStart, overlapEnd = self.fieldOverlap(segment)
        messagetype, fieldname, fieldtype = self.lookupField(segment)

        # return a typed version of the segment and the ratio of overlapping bytes to the segment length
        return overlapRatio, TypedSegment(segment.analyzer, segment.offset, segment.length, fieldtype)


    def fieldOverlap(self, segment: MessageSegment):
        """
        Overlap info between segment and its closest true field.

        :return: 4-tuple of the
            ratio of overlap (with the segment length as base),
            the index of the field in the field sequence (see parsedMessage#getFieldSequence()),
            start and end offsets of the overlap.
        """
        parsedMessage = self.parsedMessages[self.messages[segment.message]]
        fieldSequence = list()
        off = 0
        for _, l in parsedMessage.getFieldSequence():
            off += l
            fieldSequence.append(off)
        # fieldSequence = self.fieldEndsPerMessage(segment.message)
        # print(".", end="")

        # the largest true offset smaller or equal to the given segment start
        beforeOff = max([0] + [trueOff for trueOff in fieldSequence if trueOff <= segment.offset])
        # the smallest true offset larger or equal to the given segment end
        afterOff = min([trueOff for trueOff in fieldSequence if trueOff >= segment.nextOffset] +
                       [len(segment.message.data)])
        # true offsets fully enclosed in the inferred segment
        enclosed = sorted(trueOff for trueOff in fieldSequence if segment.offset < trueOff < segment.nextOffset)

        # longest overlap
        overlapStart, overlapEnd = 0, 0
        for start, end in zip([segment.offset] + enclosed, enclosed + [segment.nextOffset]):
            if overlapEnd - overlapStart < end - start:
                overlapStart, overlapEnd = start, end

        # if more than half of the longest overlap is outside of the inferred segment, there is no type match
        if overlapStart == segment.offset and (overlapEnd - beforeOff)/(overlapEnd - overlapStart) <= .5 \
                or overlapEnd == segment.nextOffset and (afterOff - overlapStart)/(overlapEnd - overlapStart) <= .5:
            # no true field with sufficient overlap exists
            return 0.0, segment

        overlapRatio = (overlapEnd - overlapStart) / segment.length

        assert overlapEnd > 0, "No possible overlap could be found. Investigate!"
        trueEnd = afterOff if overlapEnd == segment.nextOffset else overlapEnd

        if not trueEnd in fieldSequence:
            print("Field sequence is not matching any possible overlap. Investigate!")
            import IPython; IPython.embed()
        assert trueEnd in fieldSequence, "Field sequence is not matching any possible overlap. Investigate!"
        overlapIndex = fieldSequence.index(trueEnd)

        return overlapRatio, overlapIndex, overlapStart, overlapEnd


    def lookupField(self, segment: MessageSegment):
        """
        Look up the field name for a segment.
        For determining this fields overlap (ratio, field index, overlap start and end) use #fieldOverlap().

        Caveat: Returns "close matches" without any notification that it is not an exact match,
        i. e., the first offset of the segment's message fields that is not less than the segment's offset,
        regardless of its length. It still is the field with the largest overlap as determined by #fieldOverlap(),
        But this may be just, e.g., 2 out of 10 bytes if all other fields in scope of this segment are one byte long.

        :param segment: The segment to look up
        :return: Message type (from MessageTypeIdentifiers in module ..messageParser),
            field name (from tshark dissector's nomenclature),
            field type (from ParsingConstants in module ..messageParser)
        """
        parsedMessage = self.parsedMessages[self.messages[segment.message]]
        overlapRatio, overlapIndex, overlapStart, overlapEnd = self.fieldOverlap(segment)

        return parsedMessage.messagetype, \
               parsedMessage.getFieldSequence()[overlapIndex][0], parsedMessage.getTypeSequence()[overlapIndex][0]


    def segmentInfo(self, segment: MessageSegment):
        """
        Print the infos about the given segment

        :param segment: The segment to look up
        """
        pmLookup = self.lookupField(segment)
        print("Message type:", pmLookup[0])
        print("Field name:  ", pmLookup[1])
        print("Field type:  ", pmLookup[2])
        print("Byte values: ", segment.bytes)
        print("Hex values:  ", segment.bytes.hex())


    def lookupValues4FieldName(self, fieldName: str):
        """
        Lookup the values for a given field name in all messages.
        # TODO comparator.lookupValues4FieldName with list of messages (i.e., cluster elements)

        >>> from nemere.utils.loader import SpecimenLoader
        >>> from nemere.validation.dissectorMatcher import MessageComparator
        >>> from collections import Counter
        >>> specimens = SpecimenLoader("../input/deduped-orig/ntp_SMIA-20111010_deduped-100.pcap", 2, True)
        >>> comparator = MessageComparator(specimens, 2, True, debug=False)
        >>> lv = comparator.lookupValues4FieldName("ntp.ppoll")
        >>> Counter(lv).most_common()
        [('0a', 43), ('06', 41), ('09', 6), ('0e', 4), ('08', 2), ('0f', 2), ('0d', 2)]

        :param fieldName: name of field (according to tshark nomenclature)
        :return: List of values of all fields carrying the given field name
        """
        values = list()
        for pm in self.parsedMessages.values():
            values.extend(pm.getValuesByName(fieldName))
        return values


class AbstractDissectorMatcher(ABC):
    """
    Incorporates methods to match a message's inference with its dissector in different ways.

    Dissections been are done by MessageComparator so this class does not need direct interaction
    with tshark nor any knowledge of layer, relativeToIP, and failOnUndissectable.
    """
    @abstractmethod
    def __init__(self, mc: MessageComparator, message: AbstractMessage=None):
        """
        Prepares matching of one message's (or message type's) inference with its dissector.

        :param mc: the object holding the low level message and dissection information
        :param message: Message in segments to match.
        """
        self.debug = False
        self._message = message
        self._comparator = mc
        """set of specimens message is contained in"""

        tformat = self._comparator.parsedMessages[self._comparator.messages[self._message]].getTypeSequence()
        tfieldlengths = [fieldlength for sfieldtype, fieldlength in tformat]
        self._dissectionFields = MessageComparator.fieldEndsFromLength(tfieldlengths)
        """Lists of field ends including message end"""
        self._inferredFields = None  # must be filled by subclass!

    @property
    def inferredFields(self) -> List[int]:
        """
        :return: List of inferred field ends.
        """
        return self._inferredFields

    @property
    def dissectionFields(self):
        """
        :return: List of true field ends according to the dissection.
        """
        return self._dissectionFields

    def exactMatches(self) -> List[int]:
        """
        :return: exact matches of field ends in dissection and inference (excluding beginning and end of message)
        """
        return [dife for dife in self._dissectionFields[:-1] if dife in self._inferredFields[:-1]]

    def nearMatches(self) -> Dict[int, int]:
        """
        :return: near matches of field ends in dissection and inference (excluding beginning and end of message).
            Depends on the scopes returned by self.dissectorFieldEndScopes()
        """
        difescopes = self.dissectorFieldEndScopes()
        nearmatches = dict()  # dife : nearest infe if in scope
        for dife, piv in difescopes.items():
            ininscope = [infe for infe in self._inferredFields if piv[0] <= infe <= piv[1]]
            if len(ininscope) == 0:
                continue
            closest = argmin([abs(dife - infe) for infe in ininscope]).astype(int)
            # noinspection PyTypeChecker
            nearmatches[dife] = ininscope[closest]
        return nearmatches

    def dissectorFieldEndScopes(self) -> Dict[int, Tuple[int, int]]:
        """
        :return: Byte position ranges (scopes) of field ends that are no exact matches and are longer than zero.
        Implementation is independent from allDissectorFieldEndScopes!
        """
        exactMatches = self.exactMatches()

        difescopes = dict()
        for idxl in range(len(self._dissectionFields) - 1):
            center = self._dissectionFields[idxl]
            if center in exactMatches:
                continue # if there is an exact match on this field,
                # do not consider any other inferred field ends in its scope.

            left = self._dissectionFields[idxl - 1]
            right = self._dissectionFields[idxl + 1]

            # single byte fields never can have near matches, therefore the if ... else
            pivl = left + (center - left) // 2 if center - left > 1 else center
            pivr = center - 1 + (right - center) // 2 if right - center > 1 else center # shift the right pivot
            # one byte towards the center to have unique assignments of pivots to field ends.

            if pivl == pivr:
                continue # omit the field end in the search for near matches if its surrounded by 1-byte fields.

            difescopes[center] = (pivl, pivr)
        return difescopes

    def allDissectorFieldEndScopes(self) -> Dict[int, Tuple[int, int]]:
        """
        :return: All byte position ranges (scopes) of field ends regardless whether they are exact matches.
        """
        difescopes = dict()
        for idxl in range(len(self._dissectionFields) - 1):
            center = self._dissectionFields[idxl]
            left = self._dissectionFields[idxl - 1]
            right = self._dissectionFields[idxl + 1]

            # single byte fields never can have near matches, therefore the if ... else
            pivl = left + (center - left) // 2 if center - left > 1 else center
            pivr = center - 1 + (right - center) // 2 if right - center > 1 else center # shift the right pivot
            # one byte towards the center to have unique assignments of pivots to field ends.

            difescopes[center] = (pivl, pivr)

        return difescopes

    def inferredInDissectorScopes(self) -> Dict[int, List[int]]:
        """
        :return: any matches of field ends in dissection and inference (excluding beginning and end of message).
            Depends on the scopes returned by self.allDissectorFieldEndScopes()
        """
        difescopes = self.allDissectorFieldEndScopes()
        nearmatches = dict()  # dife : nearest infe if in scope
        for dife, piv in difescopes.items():
            ininscope = [infe for infe in self._inferredFields if piv[0] <= infe <= piv[1]]
            if len(ininscope) == 0:
                continue
            nearmatches[dife] = ininscope
        return nearmatches

    def distancesFromDissectorFieldEnds(self) -> Dict[int, List[int]]:
        """
        get distances for all inferred fields per true field.

        :return: dict(true field ends: List[signed inferred distances])
            negative distances are inferred fields ends left to the true field end
        """
        inferredForTrue = self.inferredInDissectorScopes()
        return {tfe: [ife - tfe for ife in ifes] for tfe, ifes in inferredForTrue.items()}

    def calcFMS(self):
        exactmatches = self.exactMatches()
        nearmatches = self.nearMatches()  # TODO check the associated inferred field index (sometimes wrong?)
        nearestdistances = {tfe: min(dists) for tfe, dists
                            in self.distancesFromDissectorFieldEnds().items()
                            if tfe in nearmatches}
        # fieldendscopes = dm.dissectorFieldEndScopes()

        exactcount = len(exactmatches)
        nearcount = len(nearmatches)

        msglen = len(self._message.data)

        fieldcount = len(self.dissectionFields)
        if 0 in self.dissectionFields:
            fieldcount -= 1
        if msglen in self.dissectionFields:
            fieldcount -= 1

        inferredcount = len(self.inferredFields)
        if 0 in self.inferredFields:
            inferredcount -= 1
        if msglen in self.inferredFields:
            inferredcount -= 1

        # nearmatches weighted by distance, /2 to increase spread -> less intense penalty for deviation from 0
        nearweights = {tfe: math.exp(- ((dist / 2) ** 2))
                       for tfe, dist in nearestdistances.items()}

        # penalty for over-/under-specificity (normalized to true field count)
        try:
            specificyPenalty = math.exp(- ((fieldcount - inferredcount) / fieldcount) ** 2)
        except ZeroDivisionError:
            raise ZeroDivisionError(f"Likely cause: Unknown protocol or missing dissector."
                                    f"fieldcount: {fieldcount} - inferredcount: {inferredcount}"
                                    f"\nOffending message:\n{self._message.data.hex()}")
        matchGain = (exactcount + sum([nearweights[nm] for nm in nearmatches.keys()])) / fieldcount
        score = specificyPenalty * (
            # exact matches + weighted near matches
            matchGain)

        fms = FormatMatchScore(self._message)
        fms.trueFormat = self._comparator.parsedMessages[self._comparator.messages[self._message]].getTypeSequence()
        fms.score = score
        fms.symbol = self.inferredFields   # sets the list of field ends instead of a Netzob Symbol
        fms.specificyPenalty = specificyPenalty
        fms.matchGain = matchGain
        fms.nearWeights = nearweights
        fms.meanDistance = numpy.mean(list(nearestdistances.values())) if len(nearestdistances) > 0 else numpy.nan
        fms.trueCount = fieldcount
        fms.inferredCount = inferredcount
        fms.exactCount = exactcount
        fms.nearCount = nearcount
        fms.specificy = fieldcount - inferredcount
        fms.exactMatches = exactmatches
        fms.nearMatches = nearmatches

        return fms


class BaseDissectorMatcher(AbstractDissectorMatcher):
    """
    Incorporates methods to match a message's inference with its dissector in different ways.

    Dissections been are done by MessageComparator so this class does not need direct interaction
    with tshark nor any knowledge of layer, relativeToIP, and failOnUndissectable.
    """
    def __init__(self, mc: MessageComparator, messageSegments: List[MessageSegment]):
        """
        Prepares matching of one message's (or message type's) inference with its dissector.

        :param mc: the object holding the low level message and dissection information
        :param messageSegments: Message in segments from inference to match in offset order,
            The corresponding dissection is implicit by the message.
        """
        # check messageSegments is consistent for exactly one message (and is not empty)
        assert all(messageSegments[0].message == seg.message for seg in messageSegments)
        super().__init__(mc, messageSegments[0].message)

        self.__messageSegments = messageSegments
        """Message in segments from inference"""
        self._inferredFields = [0] + [seg.nextOffset for seg in messageSegments]
        if self._inferredFields[-1] < len(self._message.data):
            self._inferredFields += [len(self._message.data)]

    def calcFMS(self):
        fms = super().calcFMS()
        fms.symbol = self.__messageSegments   # sets the list of segments instead of a Netzob Symbol
        return fms


class DissectorMatcher(AbstractDissectorMatcher):
    """
    Incorporates methods to match a message's inference with its dissector in different ways.

    TODO: check where this can be replaced by BaseDissectorMatcher, which performs better,
      due to the omitted parsing of Netzob Symbols.

    Dissections been are done by MessageComparator so this class does not need direct interaction
    with tshark nor any knowledge of layer, relativeToIP, and failOnUndissectable.
    """

    def __init__(self, mc: MessageComparator, inferredSymbol: netzob.Symbol, message: AbstractMessage=None):
        """
        Prepares matching of one message's (or message type's) inference with its dissector.

        :param mc: the object holding the low level message and dissection information
        :param inferredSymbol: Symbol from inference to match, The corresponding dissection is implicit by the message.
        :param message: If not given, uses the first message in the inferredSymbol, otherwise this message is used
            to determine a dissection and a instantiation of the inference for comparison.

        :raises WatchdogTimeout: If Netzob symbol parsing times out
        """
        """L4 Message"""
        if message:
            assert message in inferredSymbol.messages
        else:
            assert len(inferredSymbol.messages) > 0
            message = inferredSymbol.messages[0]
        super().__init__(mc, message)

        self._inferredSymbol = inferredSymbol
        """Symbol from inference"""
        try:
            self._inferredFields, self._dissectionFields = self._inferredandtrueFieldEnds(inferredSymbol)
            """Lists of field ends including message end"""
        except RuntimeError as e:
            print("Runtime error (probably due to a Netzob message parsing error) gracefully handled.")
            raise e

    def _inferredandtrueFieldEnds(self, nsymbol: netzob.Symbol, tformat: List[Tuple]=None) \
            -> Tuple[List[int],List[int]]:
        """
        Determines the field ends of an inferred Symbol
        and the true field ends for the first message in the Symbol
        (probably filtering the messages in the Symbol according to a specific given true format).

        :param nsymbol: a netzob symbol
        :param tformat: the format tuple (see LabelMessages.dissectAndLabel()).
            If not given, determines and uses the true format of the first message in the Symbol.

        :return: inferred field ends; true field ends

        :raises WatchdogTimeout: If Netzob symbol parsing times out
        """
        # Fallback, which uses only the first message in the symbol to determine a dissection
        if tformat is None:
            if self.debug:
                print("Determine true formats for symbol via tshark...")
            # get the raw message for the first layer 5 message in nsymbol and dissect
            tformat = list(self._comparator.dissections[self._comparator.messages[nsymbol.messages[0]]])
            samplemessage = nsymbol.messages[0]
        else:
            l2msgs = {self._comparator.messages[msg]: msg for msg in nsymbol.messages}
            tformats = {k: self._comparator.dissections[k] for k in l2msgs.keys()}
            samplemessage = None  # initialize to later check on it
            # get a sample for the tformat
            for m, tf in tformats.items():
                if tf == tformat:
                    samplemessage = l2msgs[m]  # Take the first message of this tformat.
                    break
            if not samplemessage:
                raise ValueError("No sample message could be determined.")

        #####
        # prepare dissection for message symbol.messages[0]
        # true format, only one message contained here
        #
        # basic idea:
        #   format (list of byte lengths)
        #   ---.
        #       `> Packet
        #             ---. message
        #                 `> (list of field values)
        # print ", ".join([str(fieldlength) for sfieldtype, fieldlength in sformats[0]])
        tfields = []
        offset = 0
        counter = 0
        tfieldlengths = [ fieldlength for sfieldtype, fieldlength in tformat ]
        for fieldlength in tfieldlengths:
            # By reduceBitsToBytes we don't consider fields smaller than one byte
            tfields.append(netzob.Field(netzob.Raw(nbBytes=fieldlength), name='Field{:02d}'.format(counter)))
            counter += 1
            offset += fieldlength

        if self.debug:
            ssymbol = netzob.Symbol(name="Tshark Dissection", fields=tfields, messages=[samplemessage])
            ssymbol.addEncodingFunction(netzob.TypeEncodingFunction(netzob.HexaString))
            print(bcolors.HEADER + "Tshark Dissection:" + bcolors.ENDC)
            print(ssymbol)

        #####
        # prepare inferred format
        # fist, make a copy of nsymbol to only retain the first message
        if self.debug:
            ncsymbol = netzob.Symbol(copy.deepcopy(nsymbol.fields), nsymbol.messages, name=nsymbol.name)
            ncsymbol.clearEncodingFunctions()
            ncsymbol.addEncodingFunction(netzob.TypeEncodingFunction(netzob.HexaString))
            ncsymbol.clearMessages()
            ncsymbol.messages = [samplemessage]
            print("\n{}Inference:{}".format(bcolors.HEADER, bcolors.ENDC))
            print(ncsymbol)

        #####
        # Lists of inferred and dissector field end indices in byte within message:
        # determine the indices of the field ends in the message byte sequence
        # ... Here the WatchdogTimeout is raised
        nfieldends = MessageComparator.fieldEndsPerSymbol(nsymbol, samplemessage)
        tfieldends = MessageComparator.fieldEndsFromLength(tfieldlengths)

        if self.debug:
            print("\n" + bcolors.HEADER + "Compare inference and dissector..." + bcolors.ENDC)
            print("Tshark   Field Ends " + str(tfieldends))
            print("Inferred Field Ends " + str(nfieldends))

        return nfieldends, tfieldends

    def calcFMS(self):
        fmslist = list()
        for msg in self._inferredSymbol.messages:
            # TODO calculate independent FMSs for each symbol member message, since currently
            #  this does result in an FMS that is identical for all messages within the symbol!
            fms = super().calcFMS()
            fms.symbol = self._inferredSymbol
            fmslist.append(fms)
        return fmslist

    @staticmethod
    def symbolListFMS(mc: MessageComparator, symbols: List[netzob.Symbol]) -> Dict[AbstractMessage, FormatMatchScore]:
        """
        Calculate Format Matching Score for a list of symbols and name the symbols by adding a sequence number.

        :param mc: The message comparator by which to obtain dissections for the messages in the symbols.
        :param symbols: list of inferred symbols
        :return: OrderedDict of messages mapping to their FormatMatchScore
        """
        matchprecisions = OrderedDict()
        for counter, symbol in enumerate(symbols):
            symbol.name = "{:s}{:2d}".format(symbol.name, counter)
            try:
                try:
                    dm = DissectorMatcher(mc, symbol)
                except WatchdogTimeout as e:
                    print(e, "Continuing with next symbol...")
                    for msg in symbol.messages:
                        matchprecisions[msg] = FormatMatchScore(msg, symbol)  # add empty dummy FMS
                    continue
                fmslist = dm.calcFMS()
                for fms in fmslist:
                    matchprecisions[fms.message] = fms
            except RuntimeError as e:
                print("\n\n# # # Messages # # #\n")
                for msg in symbol.messages:
                    # # add dummy entries without values to denote (most probably) failed message parsing by Netzob
                    # matchprecisions[msg] = FormatMatchScore(msg, symbol)
                    print(msg.data.hex())
                print()
                raise e

        return matchprecisions

    @staticmethod
    def thresymbolListsFMS(mc: MessageComparator,
                           threshSymbTfmt: Dict[int, Dict[netzob.Symbol, List[List[Tuple[str, int]]]]]) \
            ->  Dict[Tuple[int, AbstractMessage], FormatMatchScore]:
        """
        Annotates a hierarchical dict of (parameters : Symbols : Formats) with each entry's Format Match Sore.

        :param mc: The message comparator by which to obtain dissections for the messages in the symbols.
        :param threshSymbTfmt: a two layer hierarchical dict with FMS
        :return: a one layer dict of (threshold and message) mapping to a FMS
        """
        formatmatchmetrics = dict()
        for thresh, symbTfmt in threshSymbTfmt.items():
            msgfms = DissectorMatcher.symbolListFMS(mc, list(symbTfmt.keys()))
            for msg, fms in msgfms.items():
                formatmatchmetrics[(thresh, msg)] = fms
        return formatmatchmetrics
