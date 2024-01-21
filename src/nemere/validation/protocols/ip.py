from ..messageParser import ParsingConstants, MessageTypeIdentifiers

class MTID_IP(MessageTypeIdentifiers):
    FOR_PROTCOL = {
        'ip': 'ip.version'
    }
    NAMED_TYPES = {}

class IP(ParsingConstants):
    COMPATIBLE_TO = b'3.2.5'
    MESSAGE_TYPE_IDS = MTID_IP

    # This means the full name including the '_raw' suffix, if desired
    IGNORE_FIELDS = [ 'ip.hdr_len_raw', 'ip.addr_raw', 'ip.src_host_raw', 'ip.host_raw', 'ip.dst_host_raw' ]
    # ip.hdr_len is the second half-byte of ip.version - tshark contains the full byte in both field raw values

    EXCLUDE_SUB_FIELDS = []
    INCLUDE_SUBFIELDS = []
    RECORD_STRUCTURE = []

    # mapping of field names to general value types.
    TYPELOOKUP = dict()
    """:type: Dict[str, str]"""

    # ip.version
    # ip.dsfield
    # ip.len
    # ip.id
    # ip.flags
    # ip.frag_offset
    # ip.ttl
    # ip.proto
    # ip.checksum
    # ip.src
    # ip.dst

    prehooks = dict()
    posthooks = dict()