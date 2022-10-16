from ..messageParser import ParsingConstants, MessageTypeIdentifiers

class MTID_AWDL_PD(MessageTypeIdentifiers):
    FOR_PROTCOL = dict()
    FOR_PROTCOL['wlan.mgt'] = [ 'wlan.fixed.category_code', 'awdl_pd.tag.id',   # use sequence of tag ids as type
                                'wlan.fixed.ftm.param.delim1' ]

    NAMED_TYPES = {
        # 'wlan.fixed.category_code': {'09': 'Protected Dual of Public Action'},
        # 'awdl_pd.tag.id': {
        #     '01': 'Additional Request Parameters (req)',
        #     '02': 'Measurement Information (meas)',
        #     '03': 'Security Parameters (sec)',
        #     '04': 'Toast Parameters (seq)',
        #     '05': 'Multi Frame Buffer (mf_buf)',
        # }
        'wlan.fixed.category_code': {'09': 'PDPA'},
        'awdl_pd.tag.id': {
            '01': 'req',
            '02': 'meas',
            '03': 'sec',
            '04': 'seq',
            '05': 'mf_buf',
        },
        'wlan.fixed.ftm.param.delim1': {
            '00b2': 'si0',
            '01b2': 'si1'
        }
    }

class AWDL_PD(ParsingConstants):
    COMPATIBLE_TO = b'3.2.5'
    MESSAGE_TYPE_IDS = MTID_AWDL_PD

    IGNORE_FIELDS = ['wlan.tagged.all_raw', 'wlan.tag_raw', 'awdl_pd_raw',
                     'wlan.fixed.ftm.param.delim1_tree', 'wlan.fixed.ftm.param.delim2_tree',
                     'wlan.fixed.ftm.param.delim3_tree', 'awdl_pd.samples_raw', 'awdl_pd.mfbuf.fragments_raw' ]
    # 'awdl_pd.mfbuf.reassembled.data_raw'
    EXCLUDE_SUB_FIELDS = ['awdl_pd.samples_tree', 'awdl_pd.mfbuf.fragments', 'wlan.fixed.ftm.param.delim1_tree',
                          'wlan.fixed.ftm.param.delim2_tree', 'wlan.fixed.ftm.param.delim3_tree']
    INCLUDE_SUBFIELDS = ['wlan.tagged.all', 'awdl_pd',  # 'wlan.tag' : handled by WLAN module
                         'Additional Request Parameters (req)', 'Toast Parameters (seq)',
                         'Security Parameters (sec)', 'Measurement Information (meas)',
                         'Multi Frame Buffer (mf_buf)',
                         ]
    RECORD_STRUCTURE = []

    # mapping of field names to general value types.
    TYPELOOKUP = dict()
    """:type: Dict[str, str]"""

    TYPELOOKUP['wlan.fixed.publicact'] = 'enum'  # has value: 21
    TYPELOOKUP['wlan.fixed.followup_dialog_token'] = 'int'  # has value: 08
    TYPELOOKUP['wlan.fixed.ftm_tod'] = 'int_le'  # has value: 000000000000
    TYPELOOKUP['wlan.fixed.ftm_toa'] = 'int_le'  # has value: b00400000000
    TYPELOOKUP['wlan.fixed.ftm_tod_err'] = 'enum'  # has value: 0000
    TYPELOOKUP['wlan.fixed.ftm_toa_err'] = 'enum'  # has value: 0000
    TYPELOOKUP['wlan.fixed.trigger'] = 'enum'  # has value: 01
    TYPELOOKUP['wlan.fixed.ftm.param.delim1'] = 'flags'  # has value: 01b2
    TYPELOOKUP['wlan.fixed.ftm.param.delim2'] = 'flags'  # has value: 3239011e
    TYPELOOKUP['wlan.fixed.ftm.param.delim3'] = 'flags'  # has value: 240100

    TYPELOOKUP['awdl_pd.version'] = 'int'  # has value: 01
    TYPELOOKUP['awdl_pd.tag.id'] = 'int'  # has value: 05
    TYPELOOKUP['awdl_pd.tag.length'] = 'int'  # has value: 26
    TYPELOOKUP['awdl_pd.mfbuf.reserved'] = 'flags'  # has value: 0100
    TYPELOOKUP['awdl_pd.mfbuf.chunk_offset'] = 'int_le'  # has value: e001
    TYPELOOKUP['awdl_pd.mfbuf.total_len'] = 'int_le'  # has value: 0002
    TYPELOOKUP['awdl_pd.mfbuf.chunk_data'] = 'bytes'  # has value: 8c0000003efe...
    TYPELOOKUP['awdl_pd.mfbuf.sample_component'] = 'int_le'  # single I/I signal measurement value
    TYPELOOKUP['awdl_pd.meas.reserved1'] = 'int_le'  # has value: 0300
    TYPELOOKUP['awdl_pd.meas.phy_error'] = 'enum'  # has value: 00000000
    TYPELOOKUP['awdl_pd.meas.reserved2'] = 'pad'  # has value: 0000
    TYPELOOKUP['awdl_pd.sec.reserved1'] = 'int_le'  # has value: 0100
    TYPELOOKUP['awdl_pd.sec.reserved2'] = 'pad'  # has value: 00
    TYPELOOKUP['awdl_pd.sec.phy_ri_rr_len'] = 'int'  # has value: 08
    TYPELOOKUP['awdl_pd.sec.phy_ri'] = 'bytes'  # has value: 7e62931b35647313
    TYPELOOKUP['awdl_pd.sec.phy_rr'] = 'bytes'  # has value: 49128de496b25b55
    TYPELOOKUP['awdl_pd.unknown'] = 'pad'  # has value: 0000000000000000000000000000000000
    TYPELOOKUP['awdl_pd.req.length'] = 'int_le'  # has value: 1b00
    TYPELOOKUP['awdl_pd.req.unk1'] = 'int'  # has value: 00
    TYPELOOKUP['awdl_pd.req.unk2'] = 'int'  # has value: 06
    TYPELOOKUP['awdl_pd.req.unk3'] = 'int_le'  # has value: 06100000
    TYPELOOKUP['awdl_pd.seq.length'] = 'int_le'  # has value: 0100
    TYPELOOKUP['awdl_pd.seq.data1'] = 'bytes'  # has value: 01150a
    TYPELOOKUP['awdl_pd.seq.data2'] = 'pad'  # has value: 00

    prehooks = dict()
    posthooks = dict()

