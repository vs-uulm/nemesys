from ..messageParser import ParsingConstants, MessageTypeIdentifiers

class MTID_ICMP(MessageTypeIdentifiers):
    FOR_PROTCOL = {
        'icmp': 'icmp.ident'
    }
    NAMED_TYPES = {}

class ICMP(ParsingConstants):
    COMPATIBLE_TO = b'3.2.5'
    MESSAGE_TYPE_IDS = MTID_ICMP

    # This means the full name including the '_raw' suffix, if desired
    IGNORE_FIELDS = [ 'icmp.ident_le_raw', 'icmp.seq_le_raw' ]

    EXCLUDE_SUB_FIELDS = []
    INCLUDE_SUBFIELDS = []
    RECORD_STRUCTURE = []

    # mapping of field names to general value types.
    TYPELOOKUP = dict()
    """:type: Dict[str, str]"""

    prehooks = dict()
    posthooks = dict()