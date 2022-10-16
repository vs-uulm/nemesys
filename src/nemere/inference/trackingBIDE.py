from typing import Hashable, Iterable, Sequence, Dict, List, Tuple
import numpy, uuid, logging
from tabulate import tabulate

from netzob.Model.Vocabulary.Messages.AbstractMessage import AbstractMessage


class HashableByteSequence(Hashable):
    """
    Make a byte sequence hashable and recognizable if ohash is self defined.

    **Note** that the __hash__() and __eq__ functions are somewhat **abused** and may not behave pythonic!
    """

    def __init__(self, sequence:bytes, ohash:int=None):
        self._sequence = sequence
        if ohash is None:
            self._hash = uuid.uuid4().int
        else:
            self._hash = ohash

    def __hash__(self) -> int:
        return self._hash

    def __eq__(self, other):
        """We define HashableByteSequence objects having the same hash value as equal."""
        return hash(other) == hash(self)

    @property
    def sequence(self):
        return self._sequence


class BIDEracker(object):
    """
    Occurrence tracking BIDE-based closed sequence mining.
    """

    MIN_SUPPORT = 0.6

    def __init__(self, sequences: Sequence[HashableByteSequence]):
        """
        import inference.trackingBIDE as bide
        from tabulate import tabulate
        subsfreq = bide.BIDEracker(bide.HashableByteSequence(msg.data[:6], hash(msg)) for msg in specimens.messagePool.keys())
        print(tabulate(sorted([(cnt, bv.hex()) for bv, (cnt, occ) in subsfreq.mostFrequent(1).items()])))
        """
        self._occurranceDrop = 60
        # length: subsequence: (count, message: offset)  (last dict is "support" of length - 1)
        self._subsequenceLookup = dict()
        self._sequenceCount = len(sequences)
        self.fillL1(sequences)
        self.printExtensions()
        self.iterateExtensions()

    @property
    def minSupport(self):
        """
        :return: Min support threshold: fixed ratio of the number of sequences
        """
        return self._sequenceCount * type(self).MIN_SUPPORT

    def countAllOccurences(self):
        allcounts = {byteValue: count for freqSeq in self._subsequenceLookup.values()
                     for byteValue, (count, occurrences) in freqSeq.items()}
        return sorted(allcounts.items(), key=lambda x: x[1])

    def fillL1(self, sequences: Iterable[HashableByteSequence]):
        # fill L1 (all one-byte sequences and their positions)
        self._subsequenceLookup[1] = dict()
        for sequence in sequences:
            for offset, intValue in enumerate(sequence.sequence):
                byteValue = bytes([intValue])
                if not byteValue in self._subsequenceLookup[1]:
                    self._subsequenceLookup[1][byteValue] = [0, dict()]
                if not sequence in self._subsequenceLookup[1][byteValue][1]:
                    self._subsequenceLookup[1][byteValue][1][sequence] = list()
                self._subsequenceLookup[1][byteValue][0] += 1  # count
                self._subsequenceLookup[1][byteValue][1][sequence].append(offset)  # location

        # # initial pruning
        # allocc = countAllOccurences(subsequenceLookup)
        # # print([count for byteValue, count in allocc])
        # knee = numpy.percentile([count for byteValue, count in allocc], 66)
        # print("k", 1, "knee", knee)
        # for byteValue in list(subsequenceLookup[1]):  # since we change the dict, iter over copy of keys
        #     count, occurrences = subsequenceLookup[1][byteValue]
        #     if knee > count:  # threshold for being frequent
        #         # remove subsequence just added in k if it is not frequent
        #         del subsequenceLookup[1][byteValue]

    def iterateExtensions(self):
        logger = logging.getLogger(__name__)
        k = 2
        # iterate ks as long as there are common sequences with length k
        while len(self._subsequenceLookup[k - 1]) > 0:
            self._subsequenceLookup[k] = dict()
            minSupport = self.minSupport
            # debug
            if isinstance(self, DynamicBIDEracker):
                logger.debug(f"k {k:4d} | knee {minSupport:.2f}")

            # if 8 < k < 12:
            #     print("next iteration")
            #     print(tabulate([(bv.hex(), co, lok) for lok, bco in self._subsequenceLookup.items()
            #                     for bv, (co, oc) in bco.items() if bv[:2] == b"\x81\x82"]))
            #
            everythingfrequent = {(sequence, o): locK for locK, freqSeq in self._subsequenceLookup.items() for
                                  count, occurrences in freqSeq.values()
                                  for sequence, offsets in occurrences.items() for o in offsets if
                                  minSupport <= count}  # threshold for being frequent
            # sortedoccurences = sorted(subsequenceLookup[k - 1].items(), key=lambda x: x[1][0],
            #                           reverse=True)  # key: bytevalue's count

            # extend frequent prefixes known from k-1
            # for byteValue, (count, occurrences) in sortedoccurences:  # type: bytes, (int, Dict[AbstractMessage, List[int]])
            for sequence, offset in everythingfrequent.keys():  # search for all the frequent strings in k's supporters
                # for message, offsets in occurrences.items():  # search for all the frequent strings in k's supporters
                #     for o in sorted(offset):
                if len(sequence.sequence) < offset + k \
                        or (sequence, offset) not in everythingfrequent \
                        or not any((sequence, k_1o) in everythingfrequent and everythingfrequent[(sequence, k_1o)] >= k_1l
                               for k_1o, k_1l in zip(range(offset+1, offset + k + 1), range(k-1, 0, -1))):
                    # message does not contain an extension of prefix at position o for sequence length k
                    #  ... or no frequent extension
                    continue

                byteValue = sequence.sequence[offset:offset + k]
                if not byteValue in self._subsequenceLookup[k]:
                    self._subsequenceLookup[k][byteValue] = [0, dict()]
                if not sequence in self._subsequenceLookup[k][byteValue][1]:
                    self._subsequenceLookup[k][byteValue][1][sequence] = list()
                # add all frequent occurrences in k's supporters
                self._subsequenceLookup[k][byteValue][0] += 1  # count
                self._subsequenceLookup[k][byteValue][1][sequence].append(offset)  # location

            # print(tabulate(sorted([(cn, bv.hex()) for bv, (cn, sp) in subsequenceLookup[k].items()], key=lambda x: x[0])))

            # pruning
            # all of the new sequences that are frequent will cause the occurrences in their supporters to be removed
            if len(self._subsequenceLookup[k]) > 0:
                for byteValue, (count, occurrences) in self._subsequenceLookup[k].items():
                    # print(count, byteValue)
                    for sequence, offsets in occurrences.items():
                        for ofs in offsets:
                            assert sequence.sequence[ofs:ofs + k] == byteValue

                # newlyfrequent = {(sequence, o): k for count, occurrences in self._subsequenceLookup[k].values()
                #                  for sequence, offsets in occurrences.items() for o in offsets if
                #                  minSupport <= count}  # threshold for being frequent

                # iterate all frequent sequences newly found in k
                for byteValue in list(self._subsequenceLookup[k].keys()):  # since we change the dict, iter over copy of keys
                    count, occurrences = self._subsequenceLookup[k][byteValue]
                    # threshold for being absolutely frequent...
                    if minSupport > count:
                        # remove subsequence just added in k if it is not frequent
                        del self._subsequenceLookup[k][byteValue]
                        continue  # k's sequence to be infrequent causes its support in k-1 to remain valid (and potentially frequent)
                    # if byteValue[:2] == b"\x63\x82\x53\x63":
                    #     print(count, byteValue.hex())

                    # ... and for being locally frequent (prefix + extension)
                    keepPrefix = False
                    for ext in range(1, k):
                        # print(byteValue[:-ext], ext)
                        if byteValue[:-ext] in self._subsequenceLookup[k-ext]:
                            if count < self._subsequenceLookup[k-ext][byteValue[:-ext]][0] * self._occurranceDrop / 100:
                                # remove subsequence just added in k if it is not frequent
                                if byteValue in self._subsequenceLookup[k]:
                                    del self._subsequenceLookup[k][byteValue]
                                    keepPrefix = True
                                # if byteValue[:2] == b"\x81\x82":  bytes.fromhex("d23d15")
                                #    print("remove", byteValue)
                                continue
                            break
                        else:
                            pass
                            # print(byteValue[:-1].hex(), "not in", k, "-1")
                    if keepPrefix:
                        continue  # k's sequence to be locally infrequent causes its support in k-1 to remain valid (and potentially frequent)

                    # prefix = byteValue[:-1]
                    # find all occurrences in all supporters in k-1 for k's frequent string and remove from k-1
                    for sequence, offsets in occurrences.items():
                        for o in sorted(offsets):
                            # print("newly:", sequence.sequence[o:o+k])
                            # remove only if frequent from supporter is completely contained in new frequent
                            #   meaning: remove all occurrences in any supporters for any k_1o >= o and k_1l < k
                            for k_1o, k_1l in zip(range(o, o + k + 1), range(k, 0, -1)):
                                if (sequence, k_1o) in everythingfrequent and \
                                        everythingfrequent[(sequence, k_1o)] <= k_1l:
                                    locK = everythingfrequent[(sequence, k_1o)]
                                    byVa = sequence.sequence[k_1o:k_1o + locK]
                                    # print("rem sup occ:", k_1o, " - ", message.data[k_1o:k_1o+locK])
                                    # print(subsequenceLookup[locK][byVa][1][message])
                                    if k_1o in self._subsequenceLookup[locK][byVa][1][sequence]:
                                        self._subsequenceLookup[locK][byVa][1][sequence].remove(k_1o)
                                    self._subsequenceLookup[locK][byVa][0] -= 1
                #
                # if 8 < k < 12:
                #     print("after pruning")
                #     print(tabulate([(bv.hex(), co, lok) for lok, bco in self._subsequenceLookup.items()
                #                     for bv, (co, oc) in bco.items() if bv[:2] == b"\x81\x82"]))
                #
                # prune k-1's supporters whose occurrence count have fallen to zero as they have been accounted for in k (in the previous loop)
                for locK, bco in self._subsequenceLookup.items():
                    if locK == k:
                        continue
                    for byteValue in list(bco):  # since we change the dict, iter over copy of keys
                        count, occurrences = self._subsequenceLookup[locK][byteValue]
                        if count <= 0:  # may be negative is multiple longer frequent strings have had this one as supporting occurence
                            del self._subsequenceLookup[locK][byteValue]
                        else:
                            for sequence in list(occurrences):
                                if len(occurrences[sequence]) == 0:
                                    del self._subsequenceLookup[locK][byteValue][1][sequence]
            k += 1

    def printExtensions(self, start=1):
        for kLoc in range(start, max(self._subsequenceLookup.keys())):
            print("### k:", kLoc)
            print(tabulate(
                sorted([(cn, bv.hex()) for bv, (cn, sp) in self._subsequenceLookup[kLoc].items()], key=lambda x: x[0])))


    def mostFrequent(self, start=1) -> Dict[bytes, Tuple[int, Dict[HashableByteSequence, List[int]]]]:
        """
        :param start: minimum substring length (k) to include
        :return: dict of {byte value: (occurrence count, occurrences in terms of {byte sequence: list of offsets})}
        """
        # bytevalue: (count, occurrences)
        retVal = dict()
        for kLoc in range(start, max(self._subsequenceLookup.keys())):
            for bv in self._subsequenceLookup[kLoc].keys():
                if bv in retVal:
                    print("collision!")
            retVal.update(self._subsequenceLookup[kLoc].items())
        return retVal


class DynamicBIDEracker(BIDEracker):
    """
    Dynamic occurrance tracking BIDE-based closed sequence mining.
    """
    def __init__(self, sequences: Sequence[HashableByteSequence], percentile=80):
        """
        import inference.trackingBIDE as bide
        from tabulate import tabulate
        subsfreq = bide.DynamicBIDEracker(bide.HashableByteSequence(msg.data[:6], hash(msg)) for msg in specimens.messagePool.keys())
        print(tabulate(sorted([(cnt, bv.hex()) for bv, (cnt, occ) in subsfreq.mostFrequent(1).items()])))

        :param percentile: The percentile of occurrences per prefix-extension iteration as threshold for being frequent.
        """
        self._percentile = percentile
        super().__init__(sequences)

    @property
    def minSupport(self):
        """
        Dynamically determine min support.
        :return: Min support dynamically determined from the kee of the occurrences distribution.
        """
        # determine k's min support by counting all occurrences
        allocc = self.countAllOccurences()
        knee = numpy.percentile([count for byteValue, count in allocc], self._percentile)
        return knee


class DynamicMessageBIDE(DynamicBIDEracker):
    """
    Apply BIDEracker to netzob messages.

    import inference.trackingBIDE as bide
    from tabulate import tabulate
    subsfreq = bide.DynamicMessageBIDE(specimens.messagePool.keys())
    print(tabulate(sorted([(cnt, bv.hex()) for bv, (cnt, occ) in subsfreq.mostFrequent(1).items()])))
    """
    def __init__(self, messages: Iterable[AbstractMessage], percentile=80):
        sequences = [HashableByteSequence(msg.data, hash(msg)) for msg in messages]
        super().__init__(sequences, percentile)


class MessageBIDE(BIDEracker):
    """
    Apply BIDEracker to netzob messages.

    import inference.trackingBIDE as bide
    from tabulate import tabulate
    subsfreq = bide.MessageBIDE(specimens.messagePool.keys())
    print(tabulate(sorted([(cnt, bv.hex()) for bv, (cnt, occ) in subsfreq.mostFrequent(1).items()])))
    """
    def __init__(self, messages: Iterable[AbstractMessage]):
        sequences = [HashableByteSequence(msg.data, hash(msg)) for msg in messages]
        super().__init__(sequences)


