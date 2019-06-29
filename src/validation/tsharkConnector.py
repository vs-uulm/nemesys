import subprocess, io, struct, time
from queue import Queue
from tempfile import NamedTemporaryFile
from typing import Dict

class TsharkConnector(object):
    """
    Class to manage a tshark process and encapsulate the communication with the process' input and output.

    ## Parsing PCAPs with tshark
    For validating inferences with FMS, we parse PCAPs with tshark yielding JSON with raw data via `-T json -x`.
    We use the same process of tshark without restarting to improve performance.
    tshark however detects repeated parsing of the same packet as a retransmission or other "fatal" errors in TCP and
    the payload of the encapsulated protocol does not get dissected any more.
    To prevent this, we use the following tshark parameter to turn off sequence number related analysis for TCP:
    `-o tcp.analyze_sequence_numbers: FALSE`
    """

    # __tsharkline = ["tshark", "-l", "-r", "-", "-T", "json", "-x"]
    # __tsharkline = ["tshark", "-Q", "-a", "duration:20", "-l", "-n", "-i", "-", "-T", "json", "-x"]

    # tshark params:
    # -Q : keep quiet, output only real errors on stderr not some infos
    # -a duration:600 : stop the process after five minutes
    # -l : flush output buffering after each packet
    # -n : Disable network object name resolution (such as hostname, TCP and UDP port names)
    # -i - : capture on stdin
    # -T json : set JSON output format
    # -x : print hex of raw data (and ASCII interpretation)
    # -o tcp.analyze_sequence_numbers:FALSE :
    #       prevent error messages associated with the circumstance that it is no true trace tshark gets to dissect
    #       here. Spares the necessity of restarting the tshark process after every packet.
    __tsharkline = ["/usr/bin/tshark", "-Q", "-a", "duration:600", "-l", "-n", "-i", "-", "-T", "json", "-x",
                  "-o", "tcp.analyze_sequence_numbers:FALSE"]

    def __init__(self, linktype : int):
        self.__linktype = linktype
        self.__tshark = None  # type: subprocess.Popen
        self.__tsharkqueue = Queue()
        self.__tempfile = None  # type: io.BufferedRandom
        self.__tempreader = None  # type: io.BufferedReader
        self.__version = None


    @property
    def linktype(self):
        return self.__linktype


    @property
    def version(self):
        return self.__version


    def writePacket(self, paketdata: bytes):
        """
        Write a byte sequence as input message to the tshark process for dissection.

        :param paketdata: The raw bytes of the packet.
        :return: The tshark process that handles the dissection.
        """
        packet = struct.pack("IIII", int(time.time()), 0, len(paketdata), len(paketdata))

        # run tshark to generate json of dissection
        cmd = self._retrieveProcess()

        cmd.stdin.write(packet)  # stdin type: io.BufferedWriter
        cmd.stdin.write(paketdata)
        cmd.stdin.flush()


    @staticmethod
    def __readlines(pipe: io.BufferedReader, queue: Queue):
        """
        Read the output of tshark from pipe into queue.
        Since stdout looses some bytes at the beginning of the read of subsequent messages after a pause,
        we needed the workaround via a temp file.

        :param pipe: The reader to get the tshark output from.
        :param queue: The queue to write each line of pipe into.
            Originally, this was intended for multiprocessing. May come in handy in the future.
        :return:
        """
        # st = time.time()

        # Wait for file content
        for x in range(500):
            try:
                peeked = pipe.peek(10)  # type: bytes
                if peeked:
                    break
                time.sleep(.01)
            except ValueError as e:
                raise e
        emptywaitcycles = 200
        while emptywaitcycles > 0:
            line = pipe.readline()
            if line and line != "\n":
                # print(line.decode("utf-8"), end='')
                queue.put(line)
            else:
                emptywaitcycles -= 1
                # st = time.time()
                for x in range(5):
                    # sometimes the last "]\n" comes only after a delay
                    if pipe.peek(10):
                        line = pipe.readline()
                        queue.put(line)
                        # print("Peeked '{}' after finish at {}s: {}".format(line, time.time()-st, pipe.peek(40)[:40]))
                        break
                    time.sleep(.01)
                break


    def readPacket(self):
        """
        Read a dissected packet definition from the queue.

        :raises:
            A TimeoutError if no data is waiting in the queue.

            A ValueError if the JSON was incomplete.

        :return: A JSON string, trimmed and superficially validated.
        """
        assert self.__tempreader is not None, "Call writePacket() first"
        TsharkConnector.__readlines(self.__tempreader, self.__tsharkqueue)

        if self.__tsharkqueue.empty():
            raise TimeoutError("tshark timed out with no result.")

        tjson = ""
        while not self.__tsharkqueue.empty():
            tjson += self.__tsharkqueue.get_nowait().decode("utf-8")

        if tjson == ']\n':
            return None
        tjsonS = tjson.strip(", \n")
        tjsonS = tjsonS.strip()
        tjsonBegin = tjsonS[:40].replace(" ","").replace("\n","")
        if tjsonBegin.startswith(',{"_index":'):
            tjsonS = tjsonS[1:]
        if tjsonBegin.startswith('{"_index":'):
            tjsonS = "[\n" + tjsonS
        if not tjsonS.startswith("["):
            # Rather fail than amend by >>> tjsonS = "[\n" + tjsonS
            raise ValueError("Result from tshark was incomplete. It started with: {}".format(tjsonS[:40]))
        if not tjsonS.endswith("]"):
            tjsonS += "]\n"

        return tjsonS


    def _retrieveProcess(self) -> subprocess.Popen:
        """
        Retrieve the running tshark process or start a new one if none is open.

        :return: A running tshark process, to await packets written to it via :func:`_tsharkWritePacket()`.
        """

        if self.__tshark is None or self.__tshark.poll() is not None:
            self.__version = TsharkConnector.checkTsharkCompatibility()[0]

            header = struct.pack("IHHIIII", 0xa1b2c3d4, 2, 4, 0, 0, 0x7fff, self.__linktype)

            # create tempfile
            self.__tempfile = NamedTemporaryFile()
            self.__tempreader = open(self.__tempfile.name, "rb")
            self.__tshark = subprocess.Popen(TsharkConnector.__tsharkline,
                                        stdout=self.__tempfile, stdin=subprocess.PIPE)
            self.__tshark.stdin.write(header)
            time.sleep(.3)
        return self.__tshark


    def terminate(self, wait=None):
        """
        Closes the running tshark process if any.
        Should be called after all messages are parsed and always before the program is closed.

        :param wait: Wait for the process with timeout (see Popen.wait)
        """
        if self.__tshark is None or self.__tshark.poll() is not None:
            self.__tshark.terminate()
            if wait:
                self.__tshark.wait(wait)
        if self.__tempreader:
            self.__tempreader.close()
        if self.__tempfile:
            self.__tempfile.close()


    def isRunning(self):
        """
        :return: whether a tshark process is running.
        """
        return self.__tshark.poll() is None if self.__tshark else False


    @staticmethod
    def checkTsharkCompatibility():
        versionstring = subprocess.check_output(("tshark", "-v"))
        versionlist = versionstring.split(maxsplit=4)
        if versionlist[2] < b'2.1.1':
            raise Exception('ERROR: The installed tshark does not support JSON output, which is required for '
                            'dissection parsing. Found tshark version {}. '
                            'Upgrade!\”'.format(versionlist[2].decode()))
        if versionlist[2] not in (b'2.2.6', b'2.6.3', b'2.6.5'):
            print("WARNING: Unchecked version {} of tshark in use! Dissections may be misfunctioning of faulty. "
                  "Check compatibility of JSON output!\n".format(versionlist[2].decode()))
            return versionlist[2], False
        return versionlist[2], True


    def __getstate__(self):
        """
        Handling of runtime specific object attributes for pickling. This basically omits all instances of
            io.BufferedReader, io.BufferedRandom, and subprocess.Popen
            that need to be freshly instanciated after pickle.load() anyway.

        :return: The dict of this object for use in pickle.dump()
        """
        return {
            '_TsharkConnector__linktype': self.__linktype,
            '_TsharkConnector__version': self.__version,
        }


    def __setstate__(self, state: Dict):
        """
        Handling of runtime specific object attributes for pickling.

        :param state: The dict of this object got from pickle.load()
        :return:
        """
        self.__linktype = state['_TsharkConnector__linktype']
        self.__version = state['_TsharkConnector__version']
        self.__tsharkqueue = Queue()
        self.__tempfile = None
        self.__tempreader = None
        self.__tshark = None


