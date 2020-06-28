"""
ANSI color definitions from blender
"""
HEADER = '\033[95m'
OKBLUE = '\033[94m'
OKGREEN = '\033[92m'
WARNING = '\033[93m'
FAIL = '\033[91m'
ENDC = '\033[0m'
BOLD = '\033[1m'
UNDERLINE = '\033[4m'

def eightBitColor(colorCode: int):
    """
    see https://en.wikipedia.org/wiki/ANSI_escape_code#8-bit

    :param colorCode: 8-bit color code
    :return: escape sequence for this color.
    """
    if colorCode > 255:
        raise IndexError('only 8 bit color codes (max value 255) are allowed.')
    return '\033[38:5:{:d}m'.format(colorCode)


def colorizeStr(string: str, color: int):
    """
    enclose a string in ANSI escape sequences for a color

    :param string: string to colorize
    :param color: 8-bit color code
    :return: string pre- and appended with the ANSI escapes
    """
    return eightBitColor(color) + string + ENDC