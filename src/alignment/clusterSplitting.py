from os.path import exists
from typing import List, Union, Tuple, Dict, Any
import numpy
from collections import Counter
from tabulate import tabulate

from inference.segments import MessageSegment


class ClusterSplitter(object):

    exoticValueStats = "reports/exotic-values-statistics.csv"


    def __init__(self, fieldLenThresh: int, alignedClusters: Dict[Any, Tuple[MessageSegment]]):
        self._fieldLenThresh = fieldLenThresh
        self._clusterReplaceMap = dict()
        self._alignedClusters = alignedClusters


    def _getPivotFieldIds(self, fields: List[List[MessageSegment]], valAmount4fields: List[int], freqThresh: int):
        valCounts4fields = {fidx: Counter(tuple(seg.values) for seg in segs if seg is not None)
                            for fidx, segs in enumerate(fields)}

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
        return pivotFieldIds

    def _addSplit4Cluster(self, aNum: int):
        clusterSplits = dict()  # type: Dict[Union[Tuple, None], List[Tuple[MessageSegment]]]
        for msgsegs in aClu:
            globals().update(locals())
            # concatenate multiple distinct field combinations
            pivotVals = tuple([(pId, *msgsegs[pId].values) if msgsegs[pId] is not None else None
                               for pId in newExotic])
            if pivotVals not in clusterSplits:
                clusterSplits[pivotVals] = list()
            clusterSplits[pivotVals].append(msgsegs)
        clusterReplaceMap[aNum] = clusterSplits
        print("replace cluster", aNum, "by")
        print(tabulate((clusterSplits.keys())))


    def split(self) -> Dict[int, Tuple[MessageSegment]]:
        for aNum, aClu in self._alignedClusters.items():
            if aNum == -1:
                continue
            freqThresh = numpy.floor(numpy.log(len(aClu)))  # numpy.round(numpy.log(len(aClu)))
            fields = [fld for fld in zip(*aClu)]  # type: List[List[MessageSegment]]
            distinctVals4fields = [{tuple(val.values) for val in fld if val is not None} for fld in fields]
            # amount of distinct values per field
            valAmount4fields = [len(valSet) for valSet in distinctVals4fields]

            pivotFieldIds = self._getPivotFieldIds(fields, valAmount4fields, freqThresh)

            self._addSplit4Cluster()

        # replace clusters by their splits
        for aNum, clusterSplits in clusterReplaceMap.items():
            for nci, cluSpl in enumerate(clusterSplits.values()):  # type: int, List[Tuple[MessageSegment]]
                newCluLabel = (aNum+1) * 100 + nci

                msgs = [next(seg for seg in msgsegs if seg is not None).message for msgsegs in cluSpl]
                globals().update(locals())
                messageClusters[newCluLabel] = [msgsegs for msgsegs in messageClusters[aNum]
                                                if msgsegs[0].message in msgs]

                clusteralignment, alignedsegments = sm.alignMessageType(messageClusters[newCluLabel])
                alignedClusters[newCluLabel] = alignedsegments

            del alignedClusters[aNum]
            del messageClusters[aNum]

        # labels for distance plot
        msgLabelMap = {tuple(msgsegs): clunu for clunu, msgs in messageClusters.items() for msgsegs in msgs}
        labels = numpy.array([msgLabelMap[tuple(seglist)] for seglist in segmentedMessages])


    def align(self):
        pass



class RelaxedExoticClusterSplitter(ClusterSplitter):


    def _getPivotFieldIds(self, fields: List[List[MessageSegment]], valAmount4fields: List[int], freqThresh: int):
        valCounts4fields = {fidx: Counter(tuple(seg.values) for seg in segs if seg is not None)
                            for fidx, segs in enumerate(fields)}

        preExotic = [
            fidx for fidx, vCnt in enumerate(valAmount4fields)
                if 1 < vCnt <= freqThresh
                    # omit fields that have many gaps
                    and len([True for val in fields[fidx] if val is None]) <= freqThresh
                    # omit fields longer than fieldLenThresh
                    and not any([val.length > self._fieldLenThresh for val in fields[fidx] if val is not None])
                    # omit fields that have zeros
                    and not any([set(val.values) == {0} for val in fields[fidx] if val is not None])
            ]

        for fidx in preExotic:
            scnt = sorted(valCounts4fields[fidx].values())
            diffmax = (numpy.diff(scnt).argmax() + 1) if len(scnt) > 1 else "-"
            csvWriteHead = False if exists(exoticValueStats) else True
            with open(exoticValueStats, 'a') as csvfile:
                exoticcsv = csv.writer(csvfile)  # type: csv.writer
                if csvWriteHead:
                    exoticcsv.writerow([
                        'run_title', 'trace', 'cluster_label', 'precision', 'cluster_size', 'field',
                        'num_vals',
                        'maxdiff_n', 'maxdiff_v', 'sum<n', 'sum>=n', 'mean<n', 'mean>=n',
                        'stdev<n', 'stdev>=n', 'median<n', 'median>=n'
                    ])
                fieldParameters = ["{}-{}-eps={:.2f}-min_samples={}".format(
                    tokenizer, type(clusterer).__name__, clusterer.eps, clusterer.min_samples),
                    comparator.specimens.pcapFileName,
                    aNum, cPrec, len(aClu), fidx, len(scnt)]
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

        newExotic = list()
        for fidx in preExotic:
            scnt = sorted(valCounts4fields[fidx].values())
            if len(scnt) > 1:
                if scnt[0] > freqThresh and len(scnt) <= freqThresh:
                    newExotic.append(fidx)
                    continue

                # the pivot index and value to split the sorted list of type amounts
                # iVal, pVal = next((i, cnt) for i, cnt in enumerate(scnt) if cnt > freqThresh)
                iVal = numpy.diff(scnt).argmax() + 1
                # the special case of any(cnt <= freqThresh for cnt in scnt) is relaxedly included here
                numValues_u = len(scnt) - iVal
                # if there are no or only one frequent value, do not split
                if numValues_u > 1:
                    pVal = scnt[iVal]
                    mean_u = numpy.mean(scnt[iVal:])
                    halfsRatio = sum(scnt[:iVal]) / sum(scnt[iVal:])
                    if halfsRatio < 0.1 and mean_u > 2 * len(aClu) / numpy.log(len(aClu)):
                        newExotic.append(fidx)