"""
DissectorMatcher, FormatMatchScore, and MessageComparator

Methods to comparison of a list of messages' inferences and their dissections
and match a message's inference with its dissector in different ways.
"""

from typing import List, Tuple, Dict, Iterable, Generator
from collections import OrderedDict
import copy

from numpy import argmin

from netzob import all as netzob
from netzob.Model.Vocabulary.Messages.AbstractMessage import AbstractMessage

import visualization.bcolors as bcolors
from validation.messageParser import ParsedMessage, ParsingConstants
from inference.segments import TypedSegment


class FormatMatchScore(object):
    """
    Object to hold all relevant data of an FMS.
    """
    message = None
    symbol = None
    trueFormat = None
    score = None
    specificyPenalty = None
    matchGain = None
    specificy = None
    nearWeights = None
    meanDistance = None  # mean of "near" distances
    trueCount = None
    inferredCount = None
    exactCount = None
    nearCount = None
    exactMatches = None
    nearMatches = None



class MessageComparator(object):
    """
    Formal and visual comparison of a list of messages' inferences and their dissections.

    Functions that are closely coupled to the dissection: Interfaces with tshark to configure the call to it by
    the parameters layer, relativeToIP, and failOnUndissectable and processes the output to directly know the
    dissection result.
    """
    import utils.loader as sl

    __messageCellCache = dict()  # type: Dict[(netzob.Symbol, AbstractMessage), List]

    def __init__(self, specimens: sl.SpecimenLoader,
                 layer: int = -1, relativeToIP: bool = False, failOnUndissectable=True,
                 debug = False):
        self.specimens = specimens
        self.messages = specimens.messagePool  # type: OrderedDict[AbstractMessage, netzob.RawMessage]
        """:type messages: OrderedDict[AbstractMessage, RawMessage]"""
        self.baselayer = specimens.getBaseLayerOfPCAP()
        self.debug = debug

        # Cache messages that already have been parsed and labeled
        self._messageCache = dict()  # type: Dict[netzob.RawMessage, ]
        self._targetlayer = layer
        self._relativeToIP = relativeToIP
        self._failOnUndissectable = failOnUndissectable

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

        # since netzob.Symbol.getMessageCells is EXTREMELY inefficient,
        # we try to have it run as seldom as possible by caching its output
        if (nsymbol, message) not in MessageComparator.__messageCellCache:
            # TODO wrap Symbols.getMessageCell in watchdog: run it in process and abort it after a timeout.
            # Look for my previous solution in siemens repos
            # DHCP fails due to a very deep recursion here:
            mcells = nsymbol.getMessageCells(encoded=False)  # dict of cells keyed by message
            for msg, fields in mcells.items():
                MessageComparator.__messageCellCache[(nsymbol, msg)] = fields

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


    def __prepareMessagesForPPrint(self, symbols: Iterable[netzob.Symbol]) \
            -> List[List[Tuple[AbstractMessage, List[int], List[int]]]]:
        """

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
                ife = [0] + MessageComparator.fieldEndsPerSymbol(sym, msg)
                ife += [msglen] if ife[-1] < msglen else []
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
        import visualization.bcolors as bc

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
                    if po in ife:
                        if po > 0:
                            hexdata.append(ifeendmarker)
                        if po < len(msg.data):
                            hexdata.append(ifestartmarker)

                    # add the actual value
                    hexdata.append('{:02x}'.format(by))
                hexdata.append(ifeendmarker)
                texcode += '\\noindent\n\\texttt{' + ''.join(hexdata) + '}\n\n'
        texcode += '\\bigskip\ntrue fields: SPACE | inferred fields: framed box'
        return texcode


    def tprintInterleaved(self, symbols: Iterable[netzob.Symbol]):
        """
        Generate tikz source code visualizing the interleaved true and inferred format upon the byte values
        of each message in the symbols.

        requires \\usetikzlibrary{positioning, fit}

        Also see self.pprintInterleaved doing the same for terminal output.

        :param symbols: Inferred symbols
        :return LaTeX code
        """
        tfemarker = '1ex '
        texcode = """
\\begin{tikzpicture}[node distance=0pt, yscale=.5,
        every node/.style={font=\\ttfamily, text height=.7em, outer sep=0, inner sep=0},
        tfe/.style={draw, minimum height=1.2em, thick}]
"""
        for symid, symfes in enumerate(self.__prepareMessagesForPPrint(symbols)):
            for msgid, (msg, tfe, ife) in enumerate(symfes):
                smid = symid + msgid
                hexdata = list()  # type: List[str]
                hexdata.append('\n\n\\coordinate(m{}f0) at (0,{});'.format(smid, -smid))
                for po, by in enumerate(msg.data, start=1):
                    # add the actual value
                    hexdata.append('\\node[right={}of m{}f{}] (m{}f{}) {{{:02x}}};'.format(
                        # have a 1ex space in place of each true field end in the hex data.
                        tfemarker if po-1 in tfe else '', smid, po-1, smid, po, by))
                texcode += '\n'.join(hexdata)

                # have a frame per each inferred field
                fitnodes = list()
                for pol, por in zip(ife[:-1], ife[1:]):
                    fitnodes.append(
                        '\\node[fit=(m{}f{})(m{}f{}), tfe] {{}};'.format(smid, pol+1, smid, por)
                    )
                texcode += '\n' + '\n'.join(fitnodes)

        texcode += """
\end{tikzpicture}

\\bigskip\ntrue fields: SPACE | inferred fields: framed box
"""
        return texcode






class DissectorMatcher(object):
    """
    Incorporates methods to match a message's inference with its dissector in different ways.

    Dissections been are done by MessageComparator so this class does not need direct interaction
    with tshark nor any knowledge of layer, relativeToIP, and failOnUndissectable.
    """

    def __init__(self, mc: MessageComparator, inferredSymbol: netzob.Symbol, message:AbstractMessage=None):
        """
        Prepares matching of one message's (or message type's) inference with its dissector.

        :param mc: the object holding the low level message and dissection information
        :param inferredSymbol: Symbol from inference to match, The corresponding dissection is implicit by the message.
        :param message: If not given, uses the first message in the inferredSymbol, otherwise this message is used
            to determine a dissection and a instantiation of the inference for comparison.
        """
        self.debug = False

        self.__message = None
        """L4 Message"""
        if message:
            assert message in inferredSymbol.messages
            self.__message = message
        else:
            assert len(inferredSymbol.messages) > 0
            self.__message = inferredSymbol.messages[0]

        self.__comparator = mc
        """set of specimens message is contained in"""
        self.__inferredSymbol = inferredSymbol
        """Symbol from inference"""
        self.__inferredFields, self.__dissectionFields = self._inferredandtrueFieldEnds(inferredSymbol)
        """Lists of field ends including message end"""


    @property
    def inferredFields(self):
        """
        :return: List of inferred field ends.
        """
        return self.__inferredFields


    @property
    def dissectionFields(self):
        """
        :return: List of true field ends according to the dissection.
        """
        return self.__dissectionFields


    def exactMatches(self) -> List[int]:
        """
        :return: exact matches of field ends in dissection and inference (excluding beginning and end of message)
        """
        return [dife for dife in self.__dissectionFields[:-1] if dife in self.__inferredFields[:-1]]


    def nearMatches(self) -> Dict[int, int]:
        """
        :return: near matches of field ends in dissection and inference (excluding beginning and end of message).
            Depends on the scopes returned by self.dissectorFieldEndScopes()
        """
        difescopes = self.dissectorFieldEndScopes()
        nearmatches = dict()  # dife : nearest infe if in scope
        for dife, piv in difescopes.items():
            ininscope = [infe for infe in self.__inferredFields if piv[0] <= infe <= piv[1]]
            if len(ininscope) == 0:
                continue
            closest = argmin([abs(dife - infe) for infe in ininscope]).astype(int)
            nearmatches[dife] = ininscope[closest]
        return nearmatches


    def inferredInDissectorScopes(self) -> Dict[int, List[int]]:
        """
        :return: any matches of field ends in dissection and inference (excluding beginning and end of message).
            Depends on the scopes returned by self.allDissectorFieldEndScopes()
        """
        difescopes = self.allDissectorFieldEndScopes()
        nearmatches = dict()  # dife : nearest infe if in scope
        for dife, piv in difescopes.items():
            ininscope = [infe for infe in self.__inferredFields if piv[0] <= infe <= piv[1]]
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



    def dissectorFieldEndScopes(self) -> Dict[int, Tuple[int, int]]:
        """
        :return: Byte position ranges (scopes) of field ends that are no exact matches and are longer than zero.
        Implementation is independent from allDissectorFieldEndScopes!
        """
        exactMatches = self.exactMatches()

        difescopes = dict()
        for idxl in range(len(self.__dissectionFields)-1):
            center = self.__dissectionFields[idxl]
            if center in exactMatches:
                continue # if there is an exact match on this field,
                # do not consider any other inferred field ends in its scope.

            left = self.__dissectionFields[idxl-1]
            right = self.__dissectionFields[idxl+1]

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
        for idxl in range(len(self.__dissectionFields)-1):
            center = self.__dissectionFields[idxl]
            left = self.__dissectionFields[idxl-1]
            right = self.__dissectionFields[idxl+1]

            # single byte fields never can have near matches, therefore the if ... else
            pivl = left + (center - left) // 2 if center - left > 1 else center
            pivr = center - 1 + (right - center) // 2 if right - center > 1 else center # shift the right pivot
            # one byte towards the center to have unique assignments of pivots to field ends.

            difescopes[center] = (pivl, pivr)

        return difescopes


    def _inferredandtrueFieldEnds(self, nsymbol: netzob.Symbol, tformat: List[Tuple]=None)\
            -> Tuple[List[int],List[int]]:
        """
        Determines the field ends of an inferred Symbol
        and the true field ends for the first message in the Symbol
        (probably filtering the messages in the Symbol according to a specific given true format).

        :param nsymbol: a netzob symbol
        :param tformat: the format tuple (see LabelMessages.dissectAndLabel()).
            If not given, determines and uses the true format of the first message in the Symbol.

        :return: inferred field ends; true field ends
        """
        # Fallback, which uses only the first message in the symbol to determine a dissection
        if tformat is None:
            if self.debug:
                print("Determine true formats for symbol via tshark...")
            # get the raw message for the first layer 5 message in nsymbol and dissect
            tformat = list(self.__comparator.dissections[self.__comparator.messages[nsymbol.messages[0]]])
            samplemessage = nsymbol.messages[0]
        else:
            l2msgs = { self.__comparator.messages[msg]: msg for msg in nsymbol.messages }
            tformats = {k: self.__comparator.dissections[k] for k in l2msgs.keys()}
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
        nfieldends = MessageComparator.fieldEndsPerSymbol(nsymbol, samplemessage)
        tfieldends = MessageComparator.fieldEndsFromLength(tfieldlengths)

        if self.debug:
            print("\n" + bcolors.HEADER + "Compare inference and dissector..." + bcolors.ENDC)
            print("Tshark   Field Ends " + str(tfieldends))
            print("Inferred Field Ends " + str(nfieldends))

        return nfieldends, tfieldends


    def calcFMS(self):
        import math, numpy
        from netzob.Model.Vocabulary.Messages.RawMessage import RawMessage

        fmslist = list()
        l2rmsgs = {self.__comparator.messages[msg]: msg for msg in self.__inferredSymbol.messages}
        tformats = {k: self.__comparator.dissections[k] for k in l2rmsgs.keys()}  # l2msg: tuple
        for l2msg, l4msg in l2rmsgs.items():  # type: (RawMessage, AbstractMessage)
            exactmatches = self.exactMatches()
            nearmatches = self.nearMatches()  # TODO check the associated inferred field index (sometimes wrong?)
            nearestdistances = {tfe: min(dists) for tfe, dists
                                in self.distancesFromDissectorFieldEnds().items()
                                if tfe in nearmatches}
            # fieldendscopes = dm.dissectorFieldEndScopes()

            exactcount = len(exactmatches)
            nearcount = len(nearmatches)
            fieldcount = len(self.dissectionFields) - 1
            inferredcount = len(self.inferredFields) - 1

            # nearmatches weighted by distance, /2 to increase spread -> less intense penalty for deviation from 0
            nearweights = {tfe: math.exp(- ((dist / 2) ** 2))
                           for tfe, dist in nearestdistances.items()}

            # penalty for over-/under-specificity (normalized to true field count)
            try:
                specificyPenalty = math.exp(- ((fieldcount - inferredcount) / fieldcount) ** 2)
            except ZeroDivisionError:
                raise ZeroDivisionError("Offending message:\n{}".format(l4msg.data.hex()))
            matchGain = (exactcount + sum([nearweights[nm] for nm in nearmatches.keys()])) / fieldcount
            score = specificyPenalty * (
                # exact matches + weighted near matches
                matchGain)

            fms = FormatMatchScore()
            fms.message = l4msg
            fms.symbol = self.__inferredSymbol
            fms.trueFormat = tformats[l2msg]
            fms.score = score
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

            dm = DissectorMatcher(mc, symbol)
            fmslist = dm.calcFMS()
            for fms in fmslist:
                matchprecisions[fms.message] = fms

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
