import re
from typing import List, Tuple, Union

from ..messageParser import ParsingConstants, MessageTypeIdentifiers


class MessageTypeIdentifiers_WLAN(MessageTypeIdentifiers):
    # wlan discriminators
    FOR_PROTCOL = dict()
    FOR_PROTCOL['wlan.mgt'] = [
        'wlan.fixed.beacon'
        # 'wlan.fc.type_subtype' # would be correct, however we do not see the outer wlan frame
    ]

    # names for message types based on discriminator values
    NAMED_TYPES = {
        'wlan.fixed.beacon' : { # all values of this fieldtype should be used as workaround for the missing outer frame
            '6400': 'Beacon frame',
            '6600': 'Beacon frame',
            'c800': 'Beacon frame',
        }
        # see above # 'wlan.fc.type_subtype' : { '0008' : 'Beacon frame' }
    }

class WLAN(ParsingConstants):
    COMPATIBLE_TO = b'3.2.5'
    MESSAGE_TYPE_IDS = MessageTypeIdentifiers_WLAN

    # NOTE: in most cases, you would want to append **_raw**
    IGNORE_FIELDS = [
        'wlan.tag_raw', 'wlan.tag',
        'wlan.tagged.all_raw', 'wlan.fixed.all_raw',
        'wlan.tag.vendor.oui.type_raw',
        # 'wlan.wfa.ie.type_raw',
        'wps.vendor_id_raw', 'wlan.qbss.version_raw', 'wlan.tim.aid_raw',
        'wlan.mobility_domain.ft_capab.ft_over_ds_raw', 'wlan.mobility_domain.ft_capab.resource_req_raw',

        'wps.config_methods.usba_raw', 'wps.config_methods.ethernet_raw', 'wps.config_methods.label_raw',
        'wps.config_methods.display_raw', 'wps.config_methods.virt_display_raw', 'wps.config_methods.phy_display_raw',
        'wps.config_methods.nfcext_raw', 'wps.config_methods.nfcint_raw', 'wps.config_methods.nfcinf_raw',
        'wps.config_methods.pushbutton_raw', 'wps.config_methods.virt_pushbutton_raw',
        'wps.config_methods.phy_pushbutton_raw', 'wps.config_methods.keypad_raw',
        'wps.primary_device_type.category_raw', 'wps.primary_device_type.subcategory_network_infrastructure_raw',
        'wps.primary_device_type.subcategory_printers_scanners_faxes_copiers_raw',
        'wps.primary_device_type.subcategory_computer_raw', 'wps.primary_device_type.subcategory_displays_raw',

        'wlan.rsn.pcs_raw', 'wlan.rsn.pcs.list_raw', 'wlan.rsn.akms_raw', 'wlan.rsn.akms.list_raw'
    ]
    EXCLUDE_SUB_FIELDS = [
        'wlan.tag_raw', 'wlan.tag', 'wlan.ext_tag',

        'wlan.fixed.baparams_tree', 'wlan.fixed.ssc_tree', 'wlan.txbf_tree', 'wlan.asel_tree', 'wlan.extcap_tree',
        'wlan.ht.capabilities_tree', 'wlan.ht.ampduparam_tree', 'wlan.ht.mcsset', 'wlan.ht.info.delim1_tree',
        'wlan.ht.info.delim2_tree', 'wlan.ht.info.delim3_tree', 'wlan.vht.mcsset', 'wlan.rmcap_tree',
        'wlan.fixed.capabilities_tree', 'wlan.country_info.fnm', 'wlan.tim.bmapctl_tree',
        'wlan.erp_info_tree', 'wlan.htex.capabilities_tree',
        'wlan.rsn.gcs_tree', 'wlan.rsn.capabilities_tree',
        'wlan.rsn.gmcs_tree', 'wlan.20_40_bc_tree',
        'wlan.wfa.ie.wpa.mcs_tree', 'wlan.wfa.ie.wpa.ucs.list', 'wlan.wfa.ie.wpa.akms.list',
        'wlan.wfa.ie.wme.qos_info_tree', 'wlan.wfa.ie.wme.acp', 'wlan.atheros.ie.advcap.cap_tree',
        'wlan.hta.capabilities_tree', 'wlan.vs.ht.mcsset', 'wlan.vs.ht.capabilities_tree', 'wlan.vs.ht.ampduparam_tree',
        'wlan.vs.ht.mcsset', 'wlan.vs.htex.capabilities_tree', 'wlan.vs.txbf_tree', 'wlan.vs.asel_tree',
        'wlan.vht.op', 'wlan.vht.op.basicmcsmap_tree', 'wlan.vs.routerboard.subitem_tree',
    ]
    INCLUDE_SUBFIELDS = [
        'wlan.fixed.all',  'wlan.tagged.all', 'Fixed parameters',
        'Version: 0x10', 'UUID E', 'WFD Device Information',
        'Vendor Extension', 'Request Type: Enrollee, Info only (0x00)', 'Primary Device Type',
        'Association State: Not associated (0x0000)', 'Response Type: AP (0x03)',
        'Configuration Error: No Error (0x0000)', # 'wlan.ext_tag', 'wlan.ext_tag.he_mac_caps_tree'
        'wlan.rsn.akms.list', 'wlan.rsn.akms_tree', 'wlan.rsn.pcs.list', 'wlan.rsn.pcs_tree'
    ]
    # 'RF Bands: 2.4 GHz (0x01)', 'RF Bands: 2.4 and 5 GHz (0x03)',
    # 'Config Methods: 0x3108', 'Config Methods: 0x4388'
    # 'Manufacturer:  ', 'Model Name:  ', 'Model Number:  ', 'Device Name:  ',
    # 'Manufacturer: Celeno Communication, Inc.', 'Model Name: Celeno Wireless AP 2.4G', 'Model Number: CL1800',
    # 'Device Name: CelenoAP2.4G', 'Serial Number: 12345678', 'Device Password ID: PIN (default) (0x0000)',
    # 'P2P Capability: Device 0x4  Group 0x1', 'P2P Device ID: b2:5a:da:23:c9:fd', 'Ap Setup Locked: 0x01',
    # 'Selected Registrar: 0x00',
    # 'Wifi Protected Setup State: Not configured (0x01)', 'Wifi Protected Setup State: Configured (0x02)',
    INCLUDE_SUBFIELDS_RE = [ re.compile(pattern) for pattern in [
        'Manufacturer: .*', 'Model Name: .*', 'Model Number: .*', 'Device Name: .*',
        'RF Bands: .*', 'Config Methods: .*', 'Serial Number: .*', 'Device Password ID: .*',
        'P2P Capability: .*', 'P2P Device ID: .*', 'Ap Setup Locked: .*', 'Selected Registrar: .*',
        'Wifi Protected Setup State: .*'
    ]]
    # names of field nodes in the json that have a record structure (list[list[tuples], not list[tuples[str, tuple]]).
    RECORD_STRUCTURE = [  ]

    # mapping of field names to general value types.
    TYPELOOKUP = dict()
    """:type: Dict[str, str]"""

    TYPELOOKUP['wlan.fixed.timestamp'] = 'timestamp_le'  # has value: 34f23e7b3c000000
    TYPELOOKUP['wlan.fixed.beacon'] = 'int_le'  # has value: 6400
    TYPELOOKUP['wlan.fixed.capabilities'] = 'flags'  # has value: 1104
    TYPELOOKUP['wlan.fixed.action_code'] = 'enum'  # has value: 00
    TYPELOOKUP['wlan.fixed.dialog_token'] = 'int'  # has value: 5c
    TYPELOOKUP['wlan.fixed.baparams'] = 'flags'  # has value: 0310
    TYPELOOKUP['wlan.fixed.batimeout'] = 'int_le'  # has value: 0000
    TYPELOOKUP['wlan.fixed.ssc'] = 'int_le'  # has value: b0d1

    TYPELOOKUP['wlan.tag.number'] = 'enum'  # has value: dd
    TYPELOOKUP['wlan.tag.length'] = 'int_le'  # has value: ff
    TYPELOOKUP['wlan.tag.vendor.oui.type'] = 'enum'  # has value: 0b

    TYPELOOKUP['wlan.ssid'] = 'chars'  # has value: 465249545a21426f7820574c414e2033313730
    TYPELOOKUP['wlan.supported_rates'] = 'int'  # has value: 82
    TYPELOOKUP['wlan.extended_supported_rates'] = 'int'  # has value: 30

    TYPELOOKUP['wlan.ht.capabilities'] = 'flags'  # has value: ad01
    TYPELOOKUP['wlan.ht.ampduparam'] = 'flags'  # has value: 17
    TYPELOOKUP['wlan.ht.mcsset'] = 'flags'  # has value: ffffff00000000000000000000000000
    TYPELOOKUP['wlan.ht.info.primarychannel'] = 'int'  # has value: 01
    TYPELOOKUP['wlan.ht.info.delim1'] = 'flags'  # has value: 08
    TYPELOOKUP['wlan.ht.info.delim2'] = 'flags'  # has value: 1500
    TYPELOOKUP['wlan.ht.info.delim3'] = 'flags'  # has value: 0000
    TYPELOOKUP['wlan.vht.op'] = 'flags'  # has value: 000100
    TYPELOOKUP['wlan.vht.op.basicmcsmap'] = 'enum'  # has value: fcff
    TYPELOOKUP['wlan.vht.capabilities'] = 'flags'  # has value: 32008003
    # TODO actually this is a tree with tag.number = 17 (VHT Capabilities) containing type, length, and a series of flags
    TYPELOOKUP['wlan.vht.mcsset'] = 'flags'  # has value: faff0000faff0000
    TYPELOOKUP['wlan.hta.control_channel'] = 'enum'  # has value: 01
    TYPELOOKUP['wlan.hta.capabilities'] = 'flags'  # has value: 00
    TYPELOOKUP['wlan.htex.capabilities'] = 'flags'  # has value: 0000
    TYPELOOKUP['wlan.txbf'] = 'flags'  # has value: 00000000
    TYPELOOKUP['wlan.asel'] = 'flags'  # has value: 00

    TYPELOOKUP['wlan.ds.current_channel'] = 'enum'  # has value: 01
    TYPELOOKUP['wlan.tim.dtim_count'] = 'int'  # has value: 02
    TYPELOOKUP['wlan.tim.dtim_period'] = 'int'  # has value: 03
    TYPELOOKUP['wlan.tim.bmapctl'] = 'flags'  # has value: 00
    TYPELOOKUP['wlan.tim.partial_virtual_bitmap'] = 'flags'  # has value: 000000000000000000
    TYPELOOKUP['wlan.erp_info'] = 'flags'  # has value: 04
    TYPELOOKUP['wlan.rsn.version'] = 'int_le'  # has value: 0100
    TYPELOOKUP['wlan.rsn.gcs'] = 'addr'  # has value: 000fac02  # actually is is addr + enum
    TYPELOOKUP['wlan.rsn.pcs.count'] = 'int_le'  # has value: 0100
    TYPELOOKUP['wlan.rsn.pcs.list'] = 'unknown'  # has value: 000fac04
    TYPELOOKUP['wlan.rsn.akms.count'] = 'int_le'  # has value: 0100
    TYPELOOKUP['wlan.rsn.akms.list'] = 'unknown'  # has value: 000fac02
    TYPELOOKUP['wlan.rsn.pcs.oui'] = 'addr'  # has value: 000fac
    TYPELOOKUP['wlan.rsn.pcs.type'] = 'enum'  # has value: 04
    TYPELOOKUP['wlan.rsn.akms.oui'] = 'addr'  # has value: 000fac
    TYPELOOKUP['wlan.rsn.akms.type'] = 'enum'  # has value: 02
    TYPELOOKUP['wlan.rsn.capabilities'] = 'flags'  # has value: 0000
    TYPELOOKUP['wlan.rsn.pmkid.count'] = 'int_le'  # has value: 0000
    TYPELOOKUP['wlan.rsn.gmcs'] = 'addr'  # has value: 000fac06  # actually is is addr + enum
    TYPELOOKUP['wlan.20_40_bc'] = 'flags'  # has value: 00

    TYPELOOKUP['wlan.wfa.ie.wpa.version'] = 'int_le'  # has value: 0100
    TYPELOOKUP['wlan.wfa.ie.wpa.mcs'] = 'addr'  # has value: 0050f202
    TYPELOOKUP['wlan.wfa.ie.wpa.ucs.count'] = 'int_le'  # has value: 0100
    TYPELOOKUP['wlan.wfa.ie.wpa.ucs.list'] = 'addr'  # has value: 0050f202
    TYPELOOKUP['wlan.wfa.ie.wpa.akms.count'] = 'int_le'  # has value: 0100
    TYPELOOKUP['wlan.wfa.ie.wpa.akms.list'] = 'addr'  # has value: 0050f202
    TYPELOOKUP['wlan.wfa.ie.type'] = 'enum'  # has value: 02
    TYPELOOKUP['wlan.wfa.ie.wme.subtype'] = 'enum'  # has value: 01
    TYPELOOKUP['wlan.wfa.ie.wme.version'] = 'enum'  # has value: 01
    TYPELOOKUP['wlan.wfa.ie.wme.qos_info'] = 'flags'  # has value: 80
    TYPELOOKUP['wlan.wfa.ie.wme.reserved'] = 'enum'  # has value: 00
    TYPELOOKUP['wlan.wfa.ie.wme.acp'] = 'flags'  # has value: 03a40000

    TYPELOOKUP['wlan.qbss.version'] = 'enum'
    TYPELOOKUP['wlan.qbss.scount'] = 'int_le'  # has value: 0000
    TYPELOOKUP['wlan.qbss.cu'] = 'int'  # has value: 3b
    TYPELOOKUP['wlan.qbss.adc'] = 'int_le'  # has value: 0000
    TYPELOOKUP['wlan.obss.spd'] = 'int_le'  # has value: 1400
    TYPELOOKUP['wlan.obss.sad'] = 'int_le'  # has value: 0a00
    TYPELOOKUP['wlan.obss.cwtsi'] = 'int_le'  # has value: 2c01
    TYPELOOKUP['wlan.obss.sptpc'] = 'int_le'  # has value: c800
    TYPELOOKUP['wlan.obss.satpc'] = 'int_le'  # has value: 1400
    TYPELOOKUP['wlan.obss.wctdf'] = 'int_le'  # has value: 0500
    TYPELOOKUP['wlan.obss.sat'] = 'int_le'  # has value: 1900
    TYPELOOKUP['wlan.extcap'] = 'flags'  # has value: 05
    TYPELOOKUP['wlan.rmcap'] = 'flags'  # has value: 72
    TYPELOOKUP['wlan.country_info.code'] = 'chars'  # has value: 4e4c
    TYPELOOKUP['wlan.country_info.environment'] = 'enum'  # has value: 20
    TYPELOOKUP['wlan.country_info.fnm'] = 'enum'  # has value: 010d14
    TYPELOOKUP['wlan.ap_channel_report.channel_list'] = 'int'  # has value: 01
    TYPELOOKUP['wlan.ap_channel_report.operating_class'] = 'enum'  # has value: 21
    TYPELOOKUP['wlan.supopeclass.current'] = 'enum'  # has value: 51

    TYPELOOKUP['wps.type'] = 'enum'  # has value: 1044
    TYPELOOKUP['wps.length'] = 'int_le'  # has value: 0001
    TYPELOOKUP['wps.version'] = 'enum'  # has value: 10
    TYPELOOKUP['wps.wifi_protected_setup_state'] = 'enum'  # has value: 02
    TYPELOOKUP['wps.uuid_e'] = 'addr'  # has value: e7a17c8b6184cf40054d6178ec45fb8e
    TYPELOOKUP['wps.rf_bands'] = 'enum'  # has value: 03
    TYPELOOKUP['wps.response_type'] = 'enum'  # has value: 03
    TYPELOOKUP['wps.manufacturer'] = 'chars'  # has value: 42726f6164636f6d
    TYPELOOKUP['wps.model_name'] = 'chars'  # has value: 42726f6164636f6d
    TYPELOOKUP['wps.model_number'] = 'chars'  # has value: 313233343536
    TYPELOOKUP['wps.serial_number'] = 'chars'  # has value: 31323334
    TYPELOOKUP['wps.primary_device_type'] = 'enum'  # has value: 00060050f2040001
    TYPELOOKUP['wps.device_name'] = 'chars'  # has value: 42726f6164636f6d4150
    TYPELOOKUP['wps.config_methods'] = 'flags'  # has value: 0000
    TYPELOOKUP['wps.request_type'] = 'enum'  # has value: 00
    TYPELOOKUP['wps.association_state'] = 'enum'  # has value: 0000
    TYPELOOKUP['wps.configuration_error'] = 'enum'  # has value: 0000
    TYPELOOKUP['wps.device_password_id'] = 'enum'  # has value: 0000
    TYPELOOKUP['wps.ap_setup_locked'] = 'enum'  # has value: 01
    TYPELOOKUP['wps.selected_registrar'] = 'enum'  # has value: 00

    TYPELOOKUP['wlan.atheros.ie.type'] = 'enum'  # has value: 01
    TYPELOOKUP['wlan.atheros.ie.subtype'] = 'enum'  # has value: 01
    TYPELOOKUP['wlan.atheros.ie.version'] = 'enum'  # has value: 00
    TYPELOOKUP['wlan.atheros.ie.advcap.cap'] = 'flags'  # has value: 00
    TYPELOOKUP['wlan.atheros.ie.advcap.defkey'] = 'enum'  # has value: ff7f
    TYPELOOKUP['wlan.atheros.data'] = 'bytes'  # has value: 08000a00

    TYPELOOKUP['wlan.powercon.local'] = 'int'  # has value: 00
    TYPELOOKUP['wlan.tcprep.trsmt_pow'] = 'int'  # has value: 14
    TYPELOOKUP['wlan.tcprep.link_mrg'] = 'int'  # has value: 00
    TYPELOOKUP['wlan.ext_bss.mu_mimo_capable_sta_count'] = 'int_le'  # has value: 0000
    TYPELOOKUP['wlan.ext_bss.ss_underutilization'] = 'int'  # has value: f3
    TYPELOOKUP['wlan.ext_bss.observable_sec_20mhz_utilization'] = 'int'  # has value: 8b
    TYPELOOKUP['wlan.ext_bss.observable_sec_40mhz_utilization'] = 'int'  # has value: 00
    TYPELOOKUP['wlan.ext_bss.observable_sec_80mhz_utilization'] = 'int'  # has value: 00

    TYPELOOKUP['wlan.tag.symbol_proprietary.oui'] = 'addr'  # has value: 00a0f8
    TYPELOOKUP['wlan.tag.symbol_proprietary.extreme.assoc_clients'] = 'int_le'  # has value: 0000
    TYPELOOKUP['wlan.tag.symbol_proprietary.extreme.load_kbps'] = 'int_le'  # has value: 0a00
    TYPELOOKUP['wlan.tag.symbol_proprietary.extreme.load_pps'] = 'int_le'  # has value: 3000
    TYPELOOKUP['wlan.tag.symbol_proprietary.extreme.client_txpower'] = 'int_le'  # has value: 0000
    TYPELOOKUP['wlan.tag.symbol_proprietary.extreme.timestamp'] = 'timestamp_le'  # has value: 2ef82f60

    TYPELOOKUP['wlan.vs.ht.mcsset'] = 'flags'  # has value: 00000000000000000000000000000000
    TYPELOOKUP['wlan.vs.ht.capabilities'] = 'flags'  # has value: ee19
    TYPELOOKUP['wlan.vs.ht.ampduparam'] = 'flags'  # has value: 1f
    TYPELOOKUP['wlan.vs.htex.capabilities'] = 'flags'  # has value: 0000
    TYPELOOKUP['wlan.vs.txbf'] = 'flags'  # has value: 00000000
    TYPELOOKUP['wlan.vs.asel'] = 'flags'  # has value: 00
    TYPELOOKUP['wlan.vs.pren.type'] = 'enum'  # has value: 04
    TYPELOOKUP['wlan.vs.pren.unknown_data'] = 'unknown'  # has value: 08bf0cb279ab03aaff0000aaff0000c005000100fcff
    TYPELOOKUP['wlan.vs.extreme.subtype'] = 'enum'  # has value: 03
    TYPELOOKUP['wlan.vs.extreme.subdata'] = 'bytes'  # has value: 00010000000000004871adb700140001949b2c0d883f00010c01
    TYPELOOKUP['wlan.vs.routerboard.unknown'] = 'unknown'  # has value: 0000
    TYPELOOKUP['wlan.vs.routerboard.subitem'] = 'bytes'
                                        # has value: 011e001000000066190600004534384438433932384337440000000000000000
    TYPELOOKUP['wlan.vs.routerboard.subitem'] = 'bytes'  # has value: 05026c09
    TYPELOOKUP['wlan.vs.aruba.subtype'] = 'enum'  # has value: 04
    TYPELOOKUP['wlan.vs.aruba.data'] = 'bytes'  # has value: 0809

    TYPELOOKUP['wlan.mobility_domain.mdid'] = 'addr'  # has value: 38de
    TYPELOOKUP['wlan.mobility_domain.ft_capab'] = 'flags'  # has value: 01
    TYPELOOKUP['wlan.cisco.ccx1.unknown'] = 'unknown'  # has value: 05008f000f00ff035900
    TYPELOOKUP['wlan.cisco.ccx1.name'] = 'chars'  # has value: 41502d46494c2d303832360000000000
    TYPELOOKUP['wlan.cisco.ccx1.clients'] = 'int'  # has value: 00
    TYPELOOKUP['wlan.cisco.ccx1.unknown2'] = 'unknown'  # has value: 00002d
    TYPELOOKUP['wlan.aironet.type'] = 'enum'  # has value: 00
    TYPELOOKUP['wlan.aironet.dtpc'] = 'int'  # has value: 11
    TYPELOOKUP['wlan.aironet.dtpc_unknown'] = 'unknown'  # has value: 00
    TYPELOOKUP['wlan.aironet.data'] = 'bytes'  # has value: 0104
    TYPELOOKUP['wlan.aironet.version'] = 'int'  # has value: 05
    TYPELOOKUP['wlan.aironet.clientmfp'] = 'enum'  # has value: 01

    # TODO ext_tag actually is a complex substructure seldom seen
    TYPELOOKUP['wlan.ext_tag'] = 'unknown'  # has value: 23010808180080203002000d009f08000000fdfffdff391cc7711c07
    TYPELOOKUP['wlan.tag_raw'] = 'unknown'  # has value: dd080050f20800120010
    TYPELOOKUP['wps.vendor_extension'] = 'unknown'  # has value: 00372a000120
    TYPELOOKUP['wlan.tag.vendor.data'] = 'unknown'  # has value: 0200101c0000
    TYPELOOKUP['padding'] = 'pad'  # has value: 00

    # noinspection PyUnusedLocal
    @staticmethod
    def _hookUnknownOUItag(value: list, siblings: List[Tuple[str, str]]) -> Union[List[Tuple[str, str]], None]:
        """
        Hook to parse wlan.tag excluding some dumb vendor extensions that are not contained completely in the subfields.
        These are with 'Tag Number: Vendor Specific (221)' ('wlan.tag.number'):
            * 'Type: Unknown (0x08)': the Unknown OUI type 0x08 of Microsoft in an wlan.wfa.ie.type.
            * 'Vendor Specific OUI Type: 22' of the Wi-Fi Alliance
            * 'Advertisement Protocol element: ANQP' (tag number 108 == 0x6c)
            * wlan.interworking: tag number '107' == '6b'
            * wlan.supopeclass: tag number '59' == '3b'
            * 'WFD Device Information': tag number '10' == '0a'

        :param value: hex value of the field we are working on
        :param siblings: subfields that we know of by now
        :return: tuple of field name and value to add as new field
        """
        from ..messageParser import ParsedMessage

        # retrieve the tag type ("number"), we are interested only in 'Tag Number: Vendor Specific (221)'
        tagnumbers = [tag[1] for tag in value if tag[0] == 'wlan.tag.number']

        if len(tagnumbers) == 1 and tagnumbers[0] in ['108', '107', '59']:
            return None

        if len(tagnumbers) == 1 and tagnumbers[0] in ['221']:
            ietypenumbers = [ietype[1] for ietype in value if ietype[0] == 'wlan.tag.vendor.oui.type']
            if len(ietypenumbers) == 1 and ietypenumbers[0] in ['8', '9', '10', '22']:
                return None

        return ParsedMessage.walkSubTree(value)

    # noinspection PyUnusedLocal
    @staticmethod
    def _hookUnknownOUIraw(value: str, siblings: List[Tuple[str, str]]) -> Union[List[Tuple[str, str]], None]:
        """
        Hook to parse the Unknown OUI type 0x08 of Microsoft in an wlan.wfa.ie.type.
        Vendor Specific OUI Type: 22

        :param value: hex value of the field we are working on
        :param siblings: subfields of our common parent that we know of by now
        :return: tuple of field name and value to add as new field
        """
        #          🢇 .tag.number = 221     🢇 .tag.vendor.oui.type                    🢇 wlan.tag.number
        if value[0:2] == 'dd' and value[10:12] in ['08', '09', '0a', '16'] or value[0:2] in ['6c', '6b', '3b']:
            return [('wlan.tag_raw', value)]
        return None

    # noinspection PyUnusedLocal
    @staticmethod
    def _hookSupopeclass(value: str, siblings: List[Tuple[str, str]]) -> Union[List[Tuple[str, str]], None]:
        return [('padding', '00')]

    # noinspection PyUnusedLocal
    @staticmethod
    def _hookExtTag(value: str, siblings: List[Tuple[str, str]]) -> Union[List[Tuple[str, str]], None]:
        # retrieve the tag type ("number"), we are interested only in 'Tag Number: Element ID Extension (255)'
        tagnumbers = [tag[1][0] for tag in value if tag[0] == 'wlan.tag.number_raw']
        # and its length
        taglengths = [tag[1][0] for tag in value if tag[0] == 'wlan.ext_tag.length_raw']
        if len(tagnumbers) == 1 and tagnumbers[0][0] in ['ff'] and len(taglengths) == 1:
            return [('wlan.tag.number', tagnumbers[0]), ('wlan.ext_tag.length', taglengths[0])]
        return None

    prehooks = dict()
    # noinspection PyUnresolvedReferences
    prehooks['wlan.ext_tag'] = _hookExtTag.__func__

    posthooks = dict()
    # noinspection PyUnresolvedReferences
    posthooks['wlan.tag'] = _hookUnknownOUItag.__func__
    # noinspection PyUnresolvedReferences
    posthooks['wlan.tag_raw'] = _hookUnknownOUIraw.__func__
    # noinspection PyUnresolvedReferences
    posthooks['wlan.supopeclass.current_raw'] = _hookSupopeclass.__func__

