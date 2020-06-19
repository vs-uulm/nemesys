from typing import List, Tuple

from netzob.Model.Vocabulary.Messages.AbstractMessage import AbstractMessage

import inference.templates as TG


class SegmentLabel(object):
    """
    Assignes a template to a position (offset) in a message
    """
    label = None  # type: TG.Template
    offset = None  # type: int
    message = None  # type: AbstractMessage

    def __init__(self, label, offset, message):
        self.label = label
        self.offset = offset
        self.message = message


    @property
    def segment(self):
        """
        :return: The segment this label is based on
        """
        for bsegment in self.label.baseSegments:
            if bsegment.message == self.message:
                return bsegment
        return None


class LabeledMessage(object):
    """
    Sequence of segment labels that constitute one message.
    """
    # sorted by offset (natural sequence of message)
    _segLabels = None  # type: List[SegmentLabel]

    def __init__(self, segmentLabels: List[SegmentLabel]):
        """
        :param segmentLabels: List of segment labels that constitute one message.
        """
        for sl in segmentLabels[1:]:
            if segmentLabels[0].message != sl.message:
                raise ValueError("Segments from different messages cannot be in one LabeledMessage object.")
        self._segLabels = sorted(segmentLabels, key=lambda l: l.offset)  # insert sorted by offset
        # self._baseSegments = sorted([sl.segment for sl in self._segLabels], key=lambda s: s.offset)


    @property
    def message(self) -> AbstractMessage:
        return self._segLabels[0].message

    @property
    def segmentLabels(self):
        return self._segLabels

    @property
    def analysisValues(self):
        return self._segLabels[0].segment.analyzer.values


    def getLabelForByte(self, pos: int) -> List[SegmentLabel]:
        candidateLabels = list()
        for sl in self._segLabels:
            if sl.offset > pos:
                break  # return an empty list if no label is assigned to the given byte
            if sl.offset < pos < sl.offset + len(sl.label.values):
                candidateLabels.append(sl)  # add possibly multiple labels if segment labels overlap.
        return candidateLabels


    def getLabelOffsets(self):
        return [labels.offset for labels in self.segmentLabels]


    def getLabelLengths(self):
        return [len(labels.label.values) for labels in self.segmentLabels]


    @property
    def segmentBytes(self):
        # .hex()
        return [sl.message.data[sl.offset:sl.offset + len(sl.label.values)] for sl in self._segLabels]


    def getLabelSequence(self) -> Tuple[TG.Template, ...]:
        """
        some kind of handy representation of the whole labeled message.

        :return:
        """
        return tuple([lmsl.label for lmsl in self.segmentLabels])

    def getSymbol(self):
        """

        :return: Netzob Symbol of the message as labeled in this object.
        """
        from netzob.Model.Vocabulary.Symbol import Symbol, Field
        return Symbol([Field(segment) for segment in self.segmentBytes], messages=[self.message])

    def __repr__(self):
        return 'LabeledMessage {:02x}'.format(hash(tuple(self._segLabels)) % 0xffff )  # TODO caveat collisions
