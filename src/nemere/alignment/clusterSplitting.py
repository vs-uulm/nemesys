"""
Module to split clusters based on fields with high frequency values.
"""

from os.path import exists, join
from typing import List, Union, Tuple, Dict, Hashable
import numpy, csv
from collections import Counter
from tabulate import tabulate

from nemere.inference.segments import MessageSegment
from nemere.alignment.alignMessages import SegmentedMessages
from nemere.utils.evaluationHelpers import reportFolder
from nemere.utils.reportWriter import IndividualClusterReport

debug = True





class ClusterSplitter(object):
    """
    Class to split clusters based on fields without rare values.
    It uses static constraints about what is determined to be a rare value (#__getPivotFieldIds()).
    """
    exoticValueStats = join(reportFolder, "exotic-values-statistics.csv")

    def __init__(self, fieldLenThresh: int,
                 alignedClusters: Dict[Hashable, List[Tuple[MessageSegment]]],
                 messageClusters: Dict[Hashable, List[Tuple[MessageSegment]]],
                 segmentedMessages: SegmentedMessages):
        """
        Split clusters based on fields with high frequency values.

        :param fieldLenThresh:
        :param alignedClusters:
        :param messageClusters:
        :param segmentedMessages:
        """
        self._fieldLenThresh = fieldLenThresh
        self._clusterReplaceMap = dict()
        self._alignedClusters = alignedClusters
        self._messageClusters = messageClusters
        self._segmentedMessages = segmentedMessages

        # output
        self._labels = None

        # for evaluation/debug
        self.__runtitle = None
        self.__trace = None
        self.__clusterPrecisions = None


    def activateCVSout(self, runtitle: Union[str, Dict], trace: str, clusterPrecisions: Dict[Hashable, float]):
        """
        Activate writing of exotic field statistics to CSV for evaluation.

        Example:
            >>> recs = RelaxedExoticClusterSplitter()  # doctest: +SKIP
            >>> recs.activateCVSout("{}-{}-eps={:.2f}-min_samples={}".format(  # doctest: +SKIP
            ...         tokenizer, type(clusterer).__name__, clusterer.eps, clusterer.min_samples),
            ...         comparator.specimens.pcapFileName, {cs[0] : cs[2] for cs in clusterStats})

        :param clusterPrecisions:
        :param runtitle:
        :param trace:
        :return:
        """
        self.__runtitle = runtitle
        self.__trace = trace
        self.__clusterPrecisions = clusterPrecisions


    def _writeCSVline(self, aNum: Hashable, clusterSize, exotic, valCounts4fields: Dict[int, Counter]):
        """
        Writing of the set of lines for one cluster of exotic field statistics to CSV for evaluation.

        :param aNum: Cluster label
        :param clusterSize: Number of messages in the cluster
        :param exotic: The field IDs that have rare values.
        :param valCounts4fields: The value count for the fields of this cluster
        """
        if self.__runtitle is not None and self.__trace is not None and self.__clusterPrecisions is not None:
            # aNum and cPrec may only be None if runtitle and trace are not set
            assert aNum is not None and aNum in self.__clusterPrecisions
            cPrec = self.__clusterPrecisions[aNum]
            print("Cluster should", "" if cPrec < 1 else "not", "be split. Precision is", cPrec)

            headers = [
                          'trace', 'cluster_label', 'precision', 'cluster_size', 'field',
                          'num_vals', 'maxdiff_n', 'maxdiff_v', 'sum<n', 'sum>=n', 'mean<n', 'mean>=n',
                          'stdev<n', 'stdev>=n', 'median<n', 'median>=n'
                      ]
            if not isinstance(self.__runtitle, str):
                infCols = IndividualClusterReport.inferenceColumns(self.__runtitle)
                headers = list(infCols.keys()) + headers
                infParams = list(infCols.values())
            else:
                headers = ['run_title'] + headers
                infParams = [self.__runtitle]

            for fidx in exotic:
                scnt = sorted(valCounts4fields[fidx].values())
                diffmax = (numpy.diff(scnt).argmax() + 1) if len(scnt) > 1 else "-"
                csvWriteHead = False if exists(ClusterSplitter.exoticValueStats) else True
                with open(ClusterSplitter.exoticValueStats, 'a') as csvfile:
                    exoticcsv = csv.writer(csvfile)  # type: csv.writer
                    if csvWriteHead:
                        exoticcsv.writerow(headers)
                    fieldParameters = [*infParams, self.__trace,
                        aNum, cPrec, clusterSize, fidx, len(scnt)]
                    if len(scnt) > 1:
                        exoticcsv.writerow([
                            *fieldParameters, diffmax, scnt[diffmax],
                            sum(scnt[:diffmax]), sum(scnt[diffmax:]),
                            numpy.mean(scnt[:diffmax]), numpy.mean(scnt[diffmax:]),
                            numpy.std(scnt[:diffmax]), numpy.std(scnt[diffmax:]),
                            numpy.median(scnt[:diffmax]), numpy.median(scnt[diffmax:])
                        ])
                    else:
                        exoticcsv.writerow(fieldParameters + [""] * 10)


    def _getPivotFieldIds(self, fields: List[List[MessageSegment]], freqThresh: int, aNum = None):
        """
        Determine fields that should be used as pivots to split the cluster. Fields that contain only frequent
        values are returned.

        :param fields: All fields of the cluster's messages.
        :param freqThresh:
        :param aNum: Cluster label
        :return: Pivots to split the cluster.
        """
        valCounts4fields = {fidx: Counter(tuple(seg.values) for seg in segs if seg is not None)
                            for fidx, segs in enumerate(fields)}  # type: Dict[int, Counter]
        distinctVals4fields = [{tuple(val.values) for val in fld if val is not None} for fld in fields]
        # amount of distinct values per field
        valAmount4fields = [len(valSet) for valSet in distinctVals4fields]

        pivotFieldIds = [
            fidx for fidx, vCnt in enumerate(valAmount4fields)
                if 1 < vCnt <= freqThresh
                    # omit fields that have many gaps
                    and len([True for val in fields[fidx] if val is None]) <= freqThresh
                    # omit fields longer than fieldLenThresh
                    and not any(val.length > self._fieldLenThresh for val in fields[fidx] if val is not None)
                    # omit fields that have zeros
                    and not any(set(val.values) == {0} for val in fields[fidx] if val is not None)
                    # remove fields with exotic values
                    and not any(cnt <= freqThresh for cnt in valCounts4fields[fidx].values())
            ]

        self._writeCSVline(aNum, len(fields[0]), pivotFieldIds, valCounts4fields)

        return pivotFieldIds


    @staticmethod
    def _addSplit4Cluster(alignedCluster: List[Tuple[MessageSegment]], exotic: List[int]):
        """
        Create a dict of spilts for the given cluster.

        :param alignedCluster: The aligned cluster to get splits for.
        :param exotic: Field IDs for this cluster to use as pivots.
        :return: A dict of pivot field IDs as keys and lists of aligned message segments as values.
        """
        clusterSplits = dict()  # type: Dict[Union[Tuple, None], List[Tuple[MessageSegment]]]
        for msgsegs in alignedCluster:
            # concatenate multiple distinct field combinations
            pivotVals = tuple([(pId, *msgsegs[pId].values) if msgsegs[pId] is not None else None
                               for pId in exotic])
            if pivotVals not in clusterSplits:
                clusterSplits[pivotVals] = list()
            clusterSplits[pivotVals].append(msgsegs)
        return clusterSplits



    def split(self):
        """
        In-place (!) cluster re-assignment: Modifies the contained (reference from constructor!) clusters.
        """
        for aNum, aClu in self._alignedClusters.items():
            if aNum == -1:
                continue
            freqThresh = numpy.floor(numpy.log(len(aClu)))  # numpy.round(numpy.log(len(aClu)))
            fields = [fld for fld in zip(*aClu)]  # type: List[Tuple[MessageSegment]]
            assert len(fields[0]) == len(aClu)

            print("\nCluster {} of size {} - threshold {:.2f}".format(aNum, len(aClu), numpy.log(len(aClu))))
            pivotFieldIds = self._getPivotFieldIds(fields, freqThresh, aNum)

            if len(pivotFieldIds) == 0:
                if debug:  # print debug info only if we do not split
                    print("no pivot fields left")
                continue  # conditions not met for splitting: next cluster
            elif len(pivotFieldIds) > 2:
                if debug:
                    print("too many pivot fields:", len(pivotFieldIds))
                continue  # conditions not met for splitting: next cluster

            self._clusterReplaceMap[aNum] = ClusterSplitter._addSplit4Cluster(
                self._alignedClusters[aNum], pivotFieldIds)

            print("replace cluster", aNum, "by")
            print(tabulate((self._clusterReplaceMap[aNum].keys())))


        # replace clusters by their splits
        for aNum, clusterSplits in self._clusterReplaceMap.items():
            for nci, cluSpl in enumerate(clusterSplits.values()):  # type: int, List[Tuple[MessageSegment]]
                newCluLabel = "{}s{}".format(aNum, nci)

                msgs = [next(seg for seg in msgsegs if seg is not None).message for msgsegs in cluSpl]
                self._messageClusters[newCluLabel] = [msgsegs for msgsegs in self._messageClusters[aNum]
                                                if msgsegs[0].message in msgs]

                clusteralignment, alignedsegments = self._segmentedMessages.alignMessageType(
                    self._messageClusters[newCluLabel])
                self._alignedClusters[newCluLabel] = alignedsegments

            del self._alignedClusters[aNum]
            del self._messageClusters[aNum]

        # labels for distance plot
        msgLabelMap = {tuple(msgsegs): clunu for clunu, msgs in self._messageClusters.items() for msgsegs in msgs}
        self._labels = numpy.array([msgLabelMap[tuple(seglist)] for seglist in self._segmentedMessages.messages])


    @property
    def alignedClusters(self):
        """
        :return: Resulting aligned clusters as lists of message segments with gaps in a dict with the label as key.
        """
        if self._labels is None:
            raise RuntimeWarning("split() method has not been run yet.")
        return self._alignedClusters


    @property
    def messageClusters(self):
        """
        :return: Resulting clusters as lists of messages in a dict with the label as key.
        """
        if self._labels is None:
            raise RuntimeWarning("split() method has not been run yet.")
        return self._messageClusters


    @property
    def labels(self):
        """
        :return: Resulting cluster labels in the order of the message list.
        """
        if self._labels is None:
            raise RuntimeError("split() method has not been run yet.")
        return self._labels



class RelaxedExoticClusterSplitter(ClusterSplitter):
    """
    Class to split clusters based on fields without rare values.
    It uses dynamic constraints about what is determined to be a rare value (#__getPivotFieldIds()).
    Thus it performs a relaxed detection to allow for fields polluted with noise.
    """


    def _getPivotFieldIds(self, fields: List[List[MessageSegment]], freqThresh: int, aNum = None):
        """
        Determine fields that should be used as pivots to split the cluster. Fields that contain mostly frequent
        values and only a minority of exotic values that account for noise are returned.

        :param fields: All fields of the cluster's messages.
        :param freqThresh:
            * Number of different values a field may contain at max to be considered,
            * and min amount of one value in a field to be "frequent",
            * and max frequency
        :param aNum: Cluster label
        :return: Pivots to split the cluster.
        """
        clusterSize = len(fields[0])
        valCounts4fields = {fidx: Counter(tuple(seg.values) for seg in segs if seg is not None)
                            for fidx, segs in enumerate(fields)}
        distinctVals4fields = [{tuple(val.values) for val in fld if val is not None} for fld in fields]
        # amount of distinct values per field
        valAmount4fields = [len(valSet) for valSet in distinctVals4fields]

        preExotic = [
            fidx for fidx, vCnt in enumerate(valAmount4fields)
                if 1 < vCnt <= freqThresh
                    # omit fields that have many (more than freqThresh) gaps
                    and len([True for val in fields[fidx] if val is None]) <= freqThresh
                    # omit fields longer than fieldLenThresh
                    and not any([val.length > self._fieldLenThresh for val in fields[fidx] if val is not None])
                    # omit fields that have zeros
                    and not any([set(val.values) == {0} for val in fields[fidx] if val is not None])
            ]

        self._writeCSVline(aNum, clusterSize, preExotic, valCounts4fields)

        newExotic = list()
        for fidx in preExotic:
            scnt = sorted(valCounts4fields[fidx].values())
            # That is the same:
            assert valAmount4fields[fidx] == len(scnt)
            if len(scnt) > 1:
                if scnt[0] > freqThresh >= len(scnt):
                    newExotic.append(fidx)
                    continue

                # the pivot index and value to split the sorted list of type amounts
                # iVal, pVal = next((i, cnt) for i, cnt in enumerate(scnt) if cnt > freqThresh)
                iVal = numpy.diff(scnt).argmax() + 1
                # the special case of any(cnt <= freqThresh for cnt in scnt) is relaxedly included here
                numValues_u = len(scnt) - iVal
                # if there are no or only one frequent value, do not split
                if numValues_u > 1:
                    # pVal = scnt[iVal]
                    mean_u = numpy.mean(scnt[iVal:])
                    halfsRatio = sum(scnt[:iVal]) / sum(scnt[iVal:])
                    if halfsRatio < 0.1 and mean_u > 2 * clusterSize / numpy.log(clusterSize):
                        newExotic.append(fidx)

        return newExotic
