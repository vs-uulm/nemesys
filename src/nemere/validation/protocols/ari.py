from ..messageParser import ParsingConstants, MessageTypeIdentifiers

class MTID_ARI(MessageTypeIdentifiers):
    FOR_PROTCOL = {
        'ari': [ {
            'field': 'ari.gid',
            'filter': lambda v: True,
            'select': lambda w: int.from_bytes(bytes.fromhex(w), "big") & 0xf801  # mask out bits 5-15
        } ]
    }
    NAMED_TYPES = {}

class ARI(ParsingConstants):
    COMPATIBLE_TO = b'3.2.5'
    MESSAGE_TYPE_IDS = MTID_ARI

    IGNORE_FIELDS = ['_ws.lua.text_raw', 'ari.message_id_raw', 'ari.gmid_raw', 'ari.seq_num_raw', 'ari.ack_opt_raw',
                     'ari.unknown_4_raw', 'ari.unknown_8_raw', 'ari.unknown_10_raw',
                     'ari.tlv.mandatory_raw', 'ari.tlv.codec.name_raw', 'ari.tlv.type_desc_raw',
                     'ari.tlv.unknown_0_raw', 'ari.tlv.unknown_2_raw', 'ari.tlv.data_uint_raw',
                     'ari.ibiuint8.value_raw', 'ari.ibiuint16.value_raw', 'ari.ibiuint32.value_raw',
                     'ari.utauint8.value_raw', 'ari.utauint16.value_raw', 'ari.utauint32.value_raw',
                     'ari.tlv.version_raw', 'ari.tlv.data_asstring_uint_value_raw',
                     'gsm_sms_raw', 'ari.ibibool.value_raw', 'ari.utabool.value_raw',
                     'ari.ibiltecellinfot.index_raw', 'ari.ibiltecellinfot.mcc_raw', 'ari.ibiltecellinfot.mnc_raw',
                     'ari.ibiltecellinfot.band_info_raw', 'ari.ibiltecellinfot.area_code_raw',
                     'ari.ibiltecellinfot.cell_id_raw', 'ari.ibiltecellinfot.earfcn_raw', 'ari.ibiltecellinfot.pid_raw',
                     'ari.ibiltecellinfot.latitude_raw', 'ari.ibiltecellinfot.longitude_raw',
                     'ari.ibiltecellinfot.bandwidth_raw', 'ari.ibiltecellinfot.deployment_type_raw',
                     ]
    EXCLUDE_SUB_FIELDS = ['ari.gmid', 'ari.seq_num', 'gsm_sms']
    INCLUDE_SUBFIELDS = ['_ws.lua.text']
    RECORD_STRUCTURE = []

    # mapping of field names to general value types.
    TYPELOOKUP = dict()
    """:type: Dict[str, str]"""

     # {'int', 'chars', 'bytes', 'checksum', 'crypto', 'flags', '???', 'macaddr', 'addr', 'ipv6', 'int_le',
     #        'timestamp', 'enum', 'unknown', 'pad', 'id', 'ipv4', 'timestamp_le'}
    TYPELOOKUP['ari.proto_flag'] = 'flags'  # has value: dec07eab
    TYPELOOKUP['ari.gid'] = 'id'  # has value: 98c3
    TYPELOOKUP['ari.length'] = 'int_le'  # has value: 0d04
    TYPELOOKUP['ari.message_name'] = 'id'  # has value: 07e7
    TYPELOOKUP['ari.transaction'] = 'id'  # has value: 0000
    TYPELOOKUP['ari.tlv.id'] = 'id'  # has value: 0200
    TYPELOOKUP['ari.tlv.length'] = 'int_le'  # has value: 1000
    TYPELOOKUP['ari.tlv.data'] = 'bytes'  # has value: 01000000

    prehooks = dict()
    posthooks = dict()