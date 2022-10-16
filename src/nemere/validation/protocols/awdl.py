from typing import List, Tuple, Union

from ..messageParser import ParsingConstants, MessageTypeIdentifiers


class MessageTypeIdentifiers_AWDL(MessageTypeIdentifiers):
    FOR_PROTCOL = dict()
    # AWDL
    FOR_PROTCOL['wlan.mgt'] = [ 'wlan.fixed.category_code', 'awdl.type', 'awdl.subtype',
                                'awdl.datastate.extflags']  # see nemesys-reports/protocol-awdl/discriminator-candidates.ods


    NAMED_TYPES = {
        'wlan.fixed.category_code': {'7f': 'Vendor Specific'},
        'awdl.type': {'08': 'AWDL'},
        'awdl.subtype': {
            '00': 'Periodic Synchronization Frame (PSF) (0)',
            '03': 'Master Indication Frame (MIF) (3)' }
    }

class AWDL(ParsingConstants):
    COMPATIBLE_TO = b'3.2.5'
    MESSAGE_TYPE_IDS = MessageTypeIdentifiers_AWDL

    IGNORE_FIELDS = [ 'awdl.fixed.all_raw', 'awdl.tagged.all_raw', 'awdl.tag_raw' ]
    EXCLUDE_SUB_FIELDS = [
        'awdl.version_tree', 'awdl.dns.name', 'awdl.dns.target', 'awdl.dns.ptr', 'awdl.arpa',
        'awdl.datastate.flags_tree', 'awdl.datastate.social_channel_map_tree', 'awdl.datastate.extflags_tree',
        'awdl.serviceparams.valuess', 'awdl.ht.capabilities_tree', 'awdl.ht.ampduparam_tree', 'awdl.ht.mcsset',

        # TODO actually, these should be walked
        'awdl.channelseq.channel_list'
    ]
    INCLUDE_SUBFIELDS = [ 'awdl.fixed.all',  'awdl.tagged.all', 'awdl.tag' ]
    # names of field nodes in the json that have a record structure (list[list[tuples], not list[tuples[str, tuple]]).
    RECORD_STRUCTURE = [  ]

    # mapping of field names to general value types.
    TYPELOOKUP = dict()
    """:type: Dict[str, str]"""

    # awdl
    TYPELOOKUP['awdl.type'] = 'enum'  # has value: 08
    TYPELOOKUP['awdl.version'] = 'int'  # has value: 10
    TYPELOOKUP['awdl.subtype'] = 'enum'  # has value: 00
    TYPELOOKUP['awdl.reserved'] = 'unknown'  # has value: 00
    TYPELOOKUP['awdl.phytime'] = 'timestamp_le'  # has value: 7392fa8b
    TYPELOOKUP['awdl.targettime'] = 'timestamp_le'  # has value: f891fa8b
    TYPELOOKUP['awdl.unknown'] = 'unknown'  # has value: 00
    TYPELOOKUP['awdl.tag.number'] = 'int'  # has value: 02
    TYPELOOKUP['awdl.tag.length'] = 'int_le'  # has value: 2000
    TYPELOOKUP['awdl.tag.padding'] = 'pad'  # has value: 0000

    TYPELOOKUP['awdl.syncparams.txchannel'] = 'int'  # has value: 95
    TYPELOOKUP['awdl.syncparams.txcounter'] = 'int_le'  # has value: 3000
    TYPELOOKUP['awdl.syncparams.masterchan'] = 'int'  # has value: 95
    TYPELOOKUP['awdl.syncparams.guardtime'] = 'int'  # has value: 00
    TYPELOOKUP['awdl.syncparams.awperiod'] = 'int_le'  # has value: 1000
    TYPELOOKUP['awdl.syncparams.afperiod'] = 'int_le'  # has value: 6e00
    TYPELOOKUP['awdl.syncparams.awdlflags'] = 'flags'  # has value: 0018
    TYPELOOKUP['awdl.syncparams.aw.ext_len'] = 'int_le'  # has value: 1000
    TYPELOOKUP['awdl.syncparams.aw.common_len'] = 'int_le'  # has value: 1000
    TYPELOOKUP['awdl.syncparams.aw.remaining'] = 'int_le'  # has value: 0000
    TYPELOOKUP['awdl.syncparams.ext.min'] = 'int'  # has value: 03
    TYPELOOKUP['awdl.syncparams.ext.max_multicast'] = 'int'  # has value: 03
    TYPELOOKUP['awdl.syncparams.ext.max_unicast'] = 'int'  # has value: 03
    TYPELOOKUP['awdl.syncparams.ext.max_af'] = 'int'  # has value: 03
    TYPELOOKUP['awdl.syncparams.master'] = 'macaddr'  # has value: eea1c937585c
    TYPELOOKUP['awdl.syncparams.presencemode'] = 'enum'  # has value: 04
    TYPELOOKUP['awdl.syncparams.awseqcounter'] = 'int_le'  # has value: a153
    TYPELOOKUP['awdl.syncparams.apbeaconalignment'] = 'int_le'  # has value: 0000

    TYPELOOKUP['awdl.electionparams.flags'] = 'flags'  # has value: 00
    TYPELOOKUP['awdl.electionparams.id'] = 'int'  # has value: 0000
    TYPELOOKUP['awdl.electionparams.distance'] = 'int'  # has value: 00
    TYPELOOKUP['awdl.electionparams.unknown'] = 'unknown'  # has value: 00
    TYPELOOKUP['awdl.electionparams.master'] = 'macaddr'  # has value: 126adc00a260
    TYPELOOKUP['awdl.electionparams.mastermetric'] = 'int_le'  # has value: 09020000
    TYPELOOKUP['awdl.electionparams.selfmetric'] = 'int_le'  # has value: 09020000

    TYPELOOKUP['awdl.electionparams2.master'] = 'macaddr'  # has value: 126adc00a260
    TYPELOOKUP['awdl.electionparams2.other'] = 'macaddr'  # has value: 126adc00a260
    TYPELOOKUP['awdl.electionparams2.mastercounter'] = 'int_le'  # has value: f9030000
    TYPELOOKUP['awdl.electionparams2.disstance'] = 'int_le'  # has value: 00000000
    TYPELOOKUP['awdl.electionparams2.mastermetric'] = 'int_le'  # has value: 09020000
    TYPELOOKUP['awdl.electionparams2.selfmetric'] = 'int_le'  # has value: 09020000
    TYPELOOKUP['awdl.electionparams2.unknown'] = 'unknown'  # has value: 00000000
    TYPELOOKUP['awdl.electionparams2.reserved'] = 'unknown'  # has value: 00000000
    TYPELOOKUP['awdl.electionparams2.selfcounter'] = 'int_le'  # has value: f9030000

    TYPELOOKUP['awdl.channelseq.channels'] = 'int'  # has value: 0f
    TYPELOOKUP['awdl.channelseq.encoding'] = 'enum'  # has value: 01
    TYPELOOKUP['awdl.channelseq.duplicate'] = 'flags'  # has value: 00
    TYPELOOKUP['awdl.channelseq.step_count'] = 'int'  # has value: 03
    TYPELOOKUP['awdl.channelseq.fill_channel'] = 'enum'  # has value: ffff
    TYPELOOKUP[
        'awdl.channelseq.channel_list'] = 'int_le'  # has value: 1d9700000000000000000000000000002b061d971d9700000000000000000000
    # TODO actually a list of int_le

    TYPELOOKUP['awdl.datastate.flags'] = 'flags'  # has value: 239f
    TYPELOOKUP['awdl.datastate.countrycode'] = 'chars'  # has value: 555300
    TYPELOOKUP['awdl.datastate.social_channel_map'] = 'flags'  # has value: 0700
    TYPELOOKUP['awdl.datastate.social_channel'] = 'int'  # has value: 0000
    TYPELOOKUP['awdl.datastate.infra_bssid'] = 'macaddr'  # has value: 703a0e888052
    TYPELOOKUP['awdl.datastate.infra_channel'] = 'int_le'  # has value: 6800
    TYPELOOKUP['awdl.datastate.infra_addr'] = 'macaddr'  # has value: 126adc00a260
    TYPELOOKUP['awdl.datastate.own_awdladdr'] = 'macaddr'  # has value: 42915dbee89b
    TYPELOOKUP['awdl.datastate.unicast_options_length'] = 'int_le'  # has value: 0400
    TYPELOOKUP['awdl.datastate.unicast_options'] = 'flags'  # has value: 00000000
    TYPELOOKUP['awdl.datastate.extflags'] = 'flags'  # has value: 2d00
    TYPELOOKUP['awdl.datastate.logtrigger'] = 'int'  # has value: 0000f903
    TYPELOOKUP['awdl.datastate.undecoded'] = 'unknown'  # has value: 000014400300c0320000e0040000

    TYPELOOKUP['awdl.serviceparams.sui'] = 'int_le'  # has value: c800
    TYPELOOKUP['awdl.serviceparams.valuess'] = 'flags'  # has value: 101088804001408008

    TYPELOOKUP['awdl.ht.unknown'] = 'unknown'  # has value: 0000
    TYPELOOKUP['awdl.ht.capabilities'] = 'flags'  # has value: 6f01
    TYPELOOKUP['awdl.ht.ampduparam'] = 'flags'  # has value: 17
    TYPELOOKUP['awdl.ht.mcsset'] = 'flags'  # has value: ffff

    TYPELOOKUP['awdl.synctree.addr'] = 'macaddr'  # has value: 126adc00a260
    TYPELOOKUP['awdl.arpa.flags'] = 'flags'  # has value: 03
    TYPELOOKUP['awdl.arpa'] = 'chars'  # has value: 0c4e6f6168732d4970686f6e65c00c
    TYPELOOKUP['awdl.version.device_class'] = 'int'  # has value: 02

    TYPELOOKUP['awdl.dns.name.len'] = 'int_le'  # has value: 1000
    TYPELOOKUP['awdl.dns.name'] = 'chars'  # has value: 0c393666303861646338313632c007
    TYPELOOKUP['awdl.dns.type'] = 'enum'  # has value: 10
    TYPELOOKUP['awdl.dns.data_len'] = 'int_le'  # has value: 0a00
    TYPELOOKUP['awdl.dns.unknown'] = 'unknown'  # has value: 0000
    TYPELOOKUP['awdl.dns.txt'] = 'chars'  # has value: 09666c6167733d353033
    TYPELOOKUP['awdl.dns.ptr'] = 'chars'  # has value: 0d313939707036696472747a3473c000
    TYPELOOKUP['awdl.dns.priority'] = 'int_le'  # has value: 0000
    TYPELOOKUP['awdl.dns.weight'] = 'int_le'  # has value: 0000
    TYPELOOKUP['awdl.dns.port'] = 'int'  # has value: 2242 (not little endian here!)
    TYPELOOKUP['awdl.dns.target'] = 'chars'  # has value: 0c4e6f6168732d4970686f6e65c00c


    # noinspection PyUnusedLocal
    @staticmethod
    def _hookAWDLtag(value: list, siblings: List[Tuple[str, str]]) -> Union[List[Tuple[str, str]], None]:
        """
        Hook to parse the Service Response (2) of an awdl.tag.

        :param value: hex value of the field we are working on
        :param siblings: subfields that we know of by now
        :return: tuple of field name and value to add as new field
        """
        from ..messageParser import ParsedMessage

        # retrieve the tag type ("number"), we are interested only in "Service Response (2)"
        tagnumbers = [tag[1] for tag in value if tag[0] == 'awdl.tag.number']
        # print(tagnumbers[0])
        if len(tagnumbers) != 1 or tagnumbers[0] != '2':
            return None
        if not value[-1][1][0][0].startswith('awdl.dns'):
            # unexpected format
            raise RuntimeWarning("Unexpected format of 'Service Response' AWDL tag. Ignoring.")
        fields = ParsedMessage.walkSubTree(value[-1][1])
        return fields


    prehooks = dict()
    posthooks = dict()
    # noinspection PyUnresolvedReferences
    posthooks['awdl.tag'] = _hookAWDLtag.__func__
