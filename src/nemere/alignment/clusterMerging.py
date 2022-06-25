from itertools import chain, combinations, compress
from typing import Iterable

from sklearn.cluster import DBSCAN
from networkx import Graph
from networkx.algorithms.components.connected import connected_components

from nemere.inference.templates import DistanceCalculator, Template
from nemere.alignment.hirschbergAlignSegments import HirschbergOnSegmentSimilarity
from nemere.inference.analyzers import *



debug = False


class ClusterAligner(object):
    FC_DYN = b"DYNAMIC"
    FC_GAP = "GAP"

    def __init__(self, alignedClusters: Dict[int, List], dc: DistanceCalculator):
        self.alignedClusters = alignedClusters
        self.dc = dc

    def generateHirsch(self, mmg=(0, -1, 5)):
        """TODO fix the non-symmetric DYN-DYN similarity."""
        alignedFields = {clunu: [field for field in zip(*cluelms)] for clunu, cluelms in self.alignedClusters.items() if
                         clunu != -1}
        statDynFields = dict()
        for clunu, alfi in alignedFields.items():
            statDynFields[clunu] = list()
            for fvals in alfi:
                fvalsGapless = [val for val in fvals if val is not None]
                # if len(fvalsGapless) < len(fvals): there are GAPs in there
                if all([val.bytes == fvalsGapless[0].bytes for val in fvalsGapless]):
                    # This leaves only an example in the field list:
                    # The association to the original message is lost for the other segments in this field!
                    statDynFields[clunu].append(fvalsGapless[0])
                else:
                    dynTemp = Template(tuple(ClusterMerger.FC_DYN), fvalsGapless)
                    dynTemp.medoid = self.dc.findMedoid(dynTemp.baseSegments)
                    statDynFields[clunu].append(dynTemp)
        # generate a similarity matrix for field-classes (static-dynamic)
        statDynValues = list(set(chain.from_iterable(statDynFields.values())))


        fcSimMatrix = numpy.array([[
            # 1.0 if fcL.bytes == fcK.bytes else
            1.0 - 0.4 * fcL.distToNearest(fcK.baseSegments, self.dc)  # DYN-DYN similarity
            if isinstance(fcL, Template) and isinstance(fcK, Template)
            else 1.0 - fcL.distToNearest(fcK, self.dc)  # DYN-STA similarity, modified 00 field
                 * (0.6 if set(fcK.bytes) != {0} else 0.1)
            if isinstance(fcL, Template) and isinstance(fcK, MessageSegment)
            else 1.0 - fcK.distToNearest(fcL, self.dc)  # STA-DYN similarity, modified 00 field  min(0.2,
                 * (0.6 if set(fcK.bytes) != {0} else 0.1)
            if isinstance(fcK, Template) and isinstance(fcL, MessageSegment)
            else 1.0 - self.dc.pairDistance(fcK, fcL)  # STA-STA similarity, modified 00 field
                 * (0.4 if set(fcK.bytes) != {0} or set(fcL.bytes) != {0} else 0.1)
            if isinstance(fcK, MessageSegment) and isinstance(fcL, MessageSegment)
            else 0.0
            for fcL in statDynValues] for fcK in statDynValues])

        # fcdc = DistanceCalculator(statDynValues)
        # fcSimMatrix = fcdc.similarityMatrix()

        fclassHirsch = HirschbergOnSegmentSimilarity(fcSimMatrix, *mmg)
        return fclassHirsch, statDynFields, statDynValues


    @staticmethod
    def printShouldMerge(connectedClusters: Iterable[Iterable[int]], clusterStats):   # clusters: Iterable[Tuple[int, int]]
        print("connected:", connectedClusters)

        print("should merge:")
        # identify over-specified clusters and list which ones ideally would be merged:
        clusterMostFreqStats = {stats[1]: [] for stats in clusterStats if stats is not None}
        # 'cluster_label', 'most_freq_type', 'precision', 'recall', 'cluster_size'
        for stats in clusterStats:
            if stats is None:
                continue
            cluster_label, most_freq_type, precision, recall, cluster_size = stats
            clusterMostFreqStats[most_freq_type].append((cluster_label, precision))
        overspecificClusters = {mft: stat for mft, stat in clusterMostFreqStats.items() if len(stat) > 1}
        superClusters = list()
        for mft, stat in overspecificClusters.items():
            superCluster = [str(label) if precision > 0.5 else "[{}]".format(label) for label, precision in stat]
            superClusters.append(superCluster)
            print("    \"{}\"  ({})".format(mft, ", ".join(superCluster)))

        print("missed clusters for merge:")
        missedmerges = [[ocel[0] for ocel in oc] for oc in overspecificClusters.values()]
        mergeCandidates = [el for cchain in connectedClusters for el in cchain]
        for sc in missedmerges:
            for sidx, selement in reversed(list(enumerate(sc))):
                if selement in mergeCandidates:
                    del sc[sidx]
        print(missedmerges)
        print()

        return missedmerges

        #     # alignedFieldClasses to look up aligned field candidates from cluster pairs
        #     # in dhcp-1000, cluster pair 0,7
        #     from inference.templates import TemplateGenerator
        #     t0019 = alignedFieldClasses[(0, 7)][0][19]
        #     t7119 = alignedFieldClasses[(0, 7)][1][19]
        #     tg0019 = TemplateGenerator.generateTemplatesForClusters(dc, [t0019.baseSegments])[0]
        #     dc.findMedoid(t0019.baseSegments)
        #     dc.pairDistance(tg0019.medoid, t7119)
        #
        #     # 1:19 should be aligned to 0:21
        #     tg021 = TemplateGenerator.generateTemplatesForClusters(dc, [alignedFieldClasses[(0, 7)][0][21].baseSegments])[0]
        #
        #     dc.pairDistance(tg021.medoid, t7119)
        #     # 0.37706280402265446



class ClusterMerger(ClusterAligner):

    def __init__(self, alignedClusters: Dict[int, List], dc: DistanceCalculator,
                 messageTuplesClusters: Dict[int, List[Tuple[MessageSegment]]]):
        self.mismatchGrace = .2
        super().__init__(alignedClusters, dc)
        self.messageTuplesClusters = messageTuplesClusters

    # # # # # # # # # # # # # # # # # # #
    # An experimentation protocol for design decisions exists!
    # # # # # # # # # # # # # # # # # # #
    def _alignFieldClasses(self, mmg=(0, -1, 5)):
        fclassHirsch, statDynFields, statDynValues = self.generateHirsch(mmg)

        statDynValuesMap = {sdv: idx for idx, sdv in enumerate(statDynValues)}
        statDynIndices = {clunu: [statDynValuesMap[fc] for fc in sdf] for clunu, sdf in statDynFields.items()}

        # fclassHirsch = NWonSegmentSimilarity(fcSimMatrix, *mmg)
        clusterpairs = combinations(statDynFields.keys(), 2)
        alignedFCIndices = {(clunuA, clunuB): fclassHirsch.align(statDynIndices[clunuA], statDynIndices[clunuB])
                            for clunuA, clunuB in clusterpairs}
        alignedFieldClasses = {clupa: ([statDynValues[a] if a > -1 else ClusterMerger.FC_GAP for a in afciA],
                                       [statDynValues[b] if b > -1 else ClusterMerger.FC_GAP for b in afciB])
                               for clupa, (afciA, afciB) in alignedFCIndices.items()}
        # from alignment.hirschbergAlignSegments import NWonSegmentSimilarity
        # IPython.embed()
        return alignedFieldClasses

    def _gapMerging4nemesys(self, alignedFieldClasses):
        from tabulate import tabulate
        import IPython

        # if two fields are adjacent gap/static and static/static in any order (see nbns clusters 0,2)
        # merge the static/static fields (to be aligned with the static one without the gap)
        # # # DOES NOT ACTUALLY CHANGE THE SEGMENTATION # # #
        # merging is not recursive
        alignedFieldClassesRefined = dict()
        for cluPair, alignedFC in alignedFieldClasses.items():
            changes = []  # tuple: fid to remove, (+?-1, 0?1) to replace

            for fid, (afcA, afcB) in enumerate(zip(*alignedFC)):
                if afcA == ClusterMerger.FC_GAP and isinstance(afcB, MessageSegment) \
                        or afcB == ClusterMerger.FC_GAP and isinstance(afcA, MessageSegment):
                    # if static is left and right of gap, choose the static/static side with the lower SSdist
                    mergeLeft = None
                    mergeRight = None

                    # check if all are statics to the left or to the right
                    if fid > 0 and isinstance(alignedFC[0][fid - 1], MessageSegment) \
                            and isinstance(alignedFC[1][fid - 1], MessageSegment):
                        mergeLeft = self.dc.pairDistance(alignedFC[0][fid - 1], alignedFC[1][fid - 1])
                    if fid < len(alignedFC[0]) - 1 and isinstance(alignedFC[0][fid + 1], MessageSegment) \
                            and isinstance(alignedFC[1][fid + 1], MessageSegment):
                        mergeRight = self.dc.pairDistance(alignedFC[0][fid + 1], alignedFC[1][fid + 1])
                    if mergeLeft is None and mergeRight is None:
                        continue

                    segAtGAP = afcA if afcB == ClusterMerger.FC_GAP else afcB
                    if mergeRight is None \
                            or mergeLeft is not None and mergeLeft < mergeRight:
                        segAtSTA = alignedFC[0][fid - 1] if afcB == ClusterMerger.FC_GAP else alignedFC[1][fid - 1]
                    elif mergeLeft is None \
                            or mergeRight is not None and mergeLeft >= mergeRight:
                        segAtSTA = alignedFC[0][fid + 1] if afcB == ClusterMerger.FC_GAP else alignedFC[1][fid + 1]
                    else:
                        print("tertium non datur.")
                        segAtSTA = None  # type: Union[MessageSegment, None]
                        IPython.embed()
                    changes.append((fid, (
                        fid-1 if segAtGAP.offset > segAtSTA.offset else fid+1,
                        0 if afcB == ClusterMerger.FC_GAP else 1
                    )))

            # actually perform found changes
            if changes:
                newFC = ([*alignedFC[0]], [*alignedFC[1]])

                cid = 0
                while cid < len(changes):
                    rid, (mid, sab) = changes[cid]
                    mergedValues = tuple(newFC[sab][mid].values) + tuple(newFC[sab][rid].values) if rid > mid \
                        else tuple(newFC[sab][rid].values) + tuple(newFC[sab][mid].values)
                    mergedSTA = None
                    for seg in self.dc.segments:
                        if tuple(seg.values) == mergedValues:
                            # use any segment, regardless of origin
                            mergedSTA = seg.baseSegments[0] if isinstance(seg, Template) else seg
                            break
                    if mergedSTA is None:
                        notMergedValues = (bytes(newFC[sab][mid].values).hex(), bytes(newFC[sab][rid].values).hex()) \
                            if rid > mid else (bytes(newFC[sab][rid].values).hex(), bytes(newFC[sab][mid].values).hex())
                        if debug:
                            print("We will not merge {} | {}.".format(*notMergedValues))
                        # This is not possible without recalculating dc...
                        # ... which we probably will not want anyways:
                        # If we assume that the segmenter is right in most cases but not all
                        # ... its likely that the valid segment boundaries are the unmerged ones
                        # ... and we have not found a valid boundary modification here in the first place
                        del changes[cid]
                    else:
                        newFC[sab][mid] = mergedSTA
                        cid += 1
                        if debug:
                            print("Will merge to", mergedSTA)
                for rid, (mid, sab) in reversed(changes):
                    del newFC[0][rid]
                    del newFC[1][rid]
                alignedFieldClassesRefined[cluPair] = newFC

                if changes and debug:
                    print(tabulate(
                        [(clunu, *[fv.bytes.hex() if isinstance(fv, MessageSegment) else
                                   fv.bytes.decode() if isinstance(fv, Template) else fv for fv in fvals])
                         for clunu, fvals in zip(cluPair, alignedFieldClasses[cluPair])]
                    ))
                    print(tabulate(
                        [(clunu, *[fv.bytes.hex() if isinstance(fv, MessageSegment) else
                                   fv.bytes.decode() if isinstance(fv, Template) else fv for fv in fvals])
                         for clunu, fvals in zip(cluPair, newFC)]
                    ))
                    print()
            else:
                alignedFieldClassesRefined[cluPair] = alignedFieldClasses[cluPair]
        return alignedFieldClassesRefined

    def _generateMatchingConditions(self, alignedFieldClasses):
        return self._generateMatchingConditionsAlt1(alignedFieldClasses)

    def _generateMatchingConditionsAlt1(self, alignedFieldClasses):
        """
        fixed threshold for DYN-STA mix: 0.7 > afcA.distToNearest(afcB)

        :param alignedFieldClasses:
        :return:
        """
        # noinspection PyTypeChecker
        #                               0      1       2       3       4         5       6       7          8
        return {(clunuA, clunuB): [("Agap", "Bgap", "equal", "Azero", "Bzero", "BinA", "AinB", "DSdist", "SSdist")] + [
            (afcA == ClusterMerger.FC_GAP, afcB == ClusterMerger.FC_GAP,
             isinstance(afcA, (MessageSegment, Template)) and isinstance(afcB, (
             MessageSegment, Template)) and afcA.bytes == afcB.bytes,
             isinstance(afcA, MessageSegment) and set(afcA.bytes) == {0},
             isinstance(afcB, MessageSegment) and set(afcB.bytes) == {0},
             isinstance(afcA, Template) and isinstance(afcB, MessageSegment) and afcB.bytes in [bs.bytes for bs in
                                                                                                afcA.baseSegments],
             isinstance(afcB, Template) and isinstance(afcA, MessageSegment) and afcA.bytes in [bs.bytes for bs in
                                                                                                afcB.baseSegments],
             # TODO alt1
             0.7 > afcA.distToNearest(afcB, self.dc)  # 0.2 dhcp-1000: +2merges // 9,30
                 if isinstance(afcA, Template) and isinstance(afcB, MessageSegment)
                 else 0.7 > afcB.distToNearest(afcA, self.dc)
                 if isinstance(afcB, Template) and isinstance(afcA, MessageSegment)
                 else False,

             self.mismatchGrace > self.dc.pairDistance(afcA, afcB)
                 if isinstance(afcA, MessageSegment) and isinstance(afcB, MessageSegment)
                 else False
             )
            for afcA, afcB in zip(*alignedFieldClasses[(clunuA, clunuB)])
        ] for clunuA, clunuB in alignedFieldClasses.keys()}

    def _generateMatchingConditionsAlt2(self, alignedFieldClasses):
        """
        alternative dynamic threshold for DYN-STA mix: dist(STA, DYN.medoid) <= DYN.maxDistToMedoid()

        :param alignedFieldClasses:
        :return:
        """
        # TODO *mismatchGrace:
        #  DHCP has rather large distances for some cluster pairs here - other protocols are good with .2

        # noinspection PyTypeChecker
        #                               0      1       2       3       4         5       6       7          8
        return {(clunuA, clunuB): [("Agap", "Bgap", "equal", "Azero", "Bzero", "BinA", "AinB", "MSdist", "SSdist")] + [
            (afcA == ClusterMerger.FC_GAP, afcB == ClusterMerger.FC_GAP,
             isinstance(afcA, (MessageSegment, Template)) and isinstance(afcB, (
             MessageSegment, Template)) and afcA.bytes == afcB.bytes,
             isinstance(afcA, MessageSegment) and set(afcA.bytes) == {0},
             isinstance(afcB, MessageSegment) and set(afcB.bytes) == {0},
             isinstance(afcA, Template) and isinstance(afcB, MessageSegment) and afcB.bytes in [bs.bytes for bs in
                                                                                                afcA.baseSegments],
             isinstance(afcB, Template) and isinstance(afcA, MessageSegment) and afcA.bytes in [bs.bytes for bs in
                                                                                                afcB.baseSegments],
             # TODO alt2
             (afcA.maxDistToMedoid(self.dc) + (1 - afcA.maxDistToMedoid(self.dc)) * self.mismatchGrace >
                   self.dc.pairDistance(afcA.medoid, afcB))
                 if isinstance(afcA, Template) and isinstance(afcB, MessageSegment) # TODO *mismatchGrace
                 else (afcB.maxDistToMedoid(self.dc) + (1 - afcB.maxDistToMedoid(self.dc)) * self.mismatchGrace >
                       self.dc.pairDistance(afcB.medoid, afcA))                     # TODO *mismatchGrace
                 if isinstance(afcB, Template) and isinstance(afcA, MessageSegment)
                 else False,

             self.mismatchGrace > self.dc.pairDistance(afcA, afcB)
                 if isinstance(afcA, MessageSegment) and isinstance(afcB, MessageSegment)
                 else False
             )
            for afcA, afcB in zip(*alignedFieldClasses[(clunuA, clunuB)])
        ] for clunuA, clunuB in alignedFieldClasses.keys()}

    @staticmethod
    def _selectMatchingClusters(alignedFieldClasses, matchingConditions):

        def lenAndTrue(boolist, length=2, truths=0):
            return len(boolist) <= length and len([a for a in boolist if a]) > truths

        def basicMatch(clunuA, clunuB):
            """
            any of "Agap","Bgap","equal","Azero","Bzero","BinA","AinB" are true

            :param clunuA:
            :param clunuB:
            :return:
            """
            condA = all([any(condResult[:7]) for condResult in matchingConditions[(clunuA, clunuB)][1:]])
            # dynStaPairs may not exceed 10% of fields (ceiling) to match
            # condB = len([True for c in matchingConditions[(clunuA, clunuB)][1:] if c[5] or c[6]]) \
            #         <= ceil(.1 * len(matchingConditions[(clunuA, clunuB)][1:]))
            condB = True

            if condA:
                print(
                    (clunuA, clunuB), "basic match", "accepted" if condB else "rejected: too many dynStaPairs"
                )

            return condA and condB


        def limitedSSdistOnlyMatch(clunuA, clunuB):
            """
            if merging is based solely on SSdist for any field, allow only one other "if not equal"

            :param clunuA:
            :param clunuB:
            :return:
            """
            condA = all([any(condResult[:8]) for condResult in matchingConditions[(clunuA, clunuB)][1:]])
            condB = lenAndTrue([not any(condResult[:7]) and condResult[8]  # True if solely SSdist for field
                            for condResult in matchingConditions[(clunuA, clunuB)][1:] if not condResult[2]])

            if condA:
                print(
                    (clunuA, clunuB), "SSdist only match", "accepted" if condB else "rejected: too many other non-equals"
                )

            return condA and condB
            # and not any([True for condResult in matchingConditions[(clunuA, clunuB)][1:]
            #           if not condResult[2] and any(condResult[5:8])])  # prevents ntp merging of (1, 6) solely on ntp.stratum STA-STA

        def onlyMSdistMatch(clunuA, clunuB):
            """
            where there is not any condition 0 to 6 true, use MSdist for this field match

            :param clunuA:
            :param clunuB:
            :return:
            """
            condA = all([condResult[7]
                 for condResult in matchingConditions[(clunuA, clunuB)][1:] if not any(condResult[:7])])

            if condA:
                print(
                    (clunuA, clunuB), "MSdist only match", "accepted"
                )

            # ! this is the key for DHCP
            # condResult[2:7] lets (ALL) queries be merged for DNS / removes a number of merges in DHCP /
            # replaces one valid merge for NTP by two invalid ones
            return condA

        return [
            (clunuA, clunuB) for clunuA, clunuB in alignedFieldClasses.keys()
            if basicMatch(clunuA, clunuB) # or limitedSSdistOnlyMatch(clunuA, clunuB)
              # or onlyMSdistMatch(clunuA, clunuB)
        ]

    def _mergeClusters(self, messageClusters, clusterStats, alignedFieldClasses,
                       matchingClusters, matchingConditions):
        import IPython
        from tabulate import tabulate
        from nemere.utils.evaluationHelpers import printClusterMergeConditions
        from nemere.inference.templates import Template

        remDue2gaps = [
            clunuAB for clunuAB in matchingClusters
            if not len([True for a in matchingConditions[clunuAB][1:] if a[0] == True or a[1] == True])
                   <= numpy.ceil(.4 * len(matchingConditions[clunuAB][1:]))
        ]
        print("\nremove due to more than 40% gaps:")
        print(tabulate(
            [(clupair,
              len([True for a in matchingConditions[clupair][1:] if a[0] == True or a[1] == True]),
              len(matchingConditions[clupair]) - 1)
             for clupair in remDue2gaps],
            headers=("clpa", "gaps", "fields")
        ))
        print()

        remDue2gapsInARow = list()
        for clunuAB in matchingClusters:
            for flip in (0, 1):
                rowOfGaps = [a[flip] for a in matchingConditions[clunuAB][1:]]
                startOfGroups = [i for i, g in enumerate(rowOfGaps) if g and i > 1 and not rowOfGaps[i - 1]]
                endOfGroups = [i for i, g in enumerate(rowOfGaps) if
                               g and i < len(rowOfGaps) - 1 and not rowOfGaps[i + 1]]
                if len(startOfGroups) > 0 and startOfGroups[-1] == len(rowOfGaps) - 1:
                    endOfGroups.append(startOfGroups[-1])
                if len(endOfGroups) > 0 and endOfGroups[0] == 0:
                    startOfGroups = [0] + startOfGroups
                # field index before and after all gap groups longer than 2
                groupOfLonger = [(sog - 1, eog + 1) for sog, eog in zip(startOfGroups, endOfGroups) if sog < eog - 1]
                for beforeGroup, afterGroup in groupOfLonger:
                    if not ((beforeGroup < 0
                             or isinstance(alignedFieldClasses[clunuAB][flip][beforeGroup], MessageSegment))
                            or (afterGroup >= len(rowOfGaps)
                                or isinstance(alignedFieldClasses[clunuAB][flip][afterGroup], MessageSegment))):
                        remDue2gapsInARow.append(clunuAB)
                        break
                if clunuAB in remDue2gapsInARow:
                    # already removed
                    break
        print("\nremove due to more than 2 gaps in a row not surounded by STAs:")
        print(remDue2gapsInARow)
        print()
        # remove pairs based on more then 25% gaps
        matchingClusters = [
            clunuAB for clunuAB in matchingClusters
            if clunuAB not in remDue2gaps and clunuAB not in remDue2gapsInARow
        ]

        # search in filteredMatches for STATIC - DYNAMIC - STATIC with different static values and remove from matchingClusters
        # : the matches on grounds of the STA value in DYN condition, with the DYN role(s) in a set in the first element of each tuple
        dynStaPairs = list()
        for clunuPair in matchingClusters:
            dynRole = [
                clunuPair[0] if isinstance(alignedFieldClasses[clunuPair][0][fieldNum], Template) else clunuPair[1]
                for fieldNum, fieldCond in enumerate(matchingConditions[clunuPair][1:])
                if not any(fieldCond[:5]) and (fieldCond[5] or fieldCond[6] or fieldCond[7])]
            if dynRole:
                dynStaPairs.append((set(dynRole), clunuPair))
        dynRoles = set(chain.from_iterable([dynRole for dynRole, clunuPair in dynStaPairs]))
        # List of STA roles for each DYN role
        staRoles = {dynRole: [clunuPair[0] if clunuPair[1] in dr else clunuPair[1] for dr, clunuPair in dynStaPairs
                              if dynRole in dr] for dynRole in dynRoles}
        removeFromMatchingClusters = list()
        # for each cluster that holds at least one DYN field class...
        for dynRole, staRoleList in staRoles.items():
            try:
                # alt: use staMismatch and subsequent accesses to remove whole group of connected clusters at once
                # staMismatch = False
                staValues = dict()
                clunuPairs = dict()
                # match the STA values corresponding to the DYN fields...
                for staRole in staRoleList:
                    clunuPair = (dynRole, staRole) if (dynRole, staRole) in matchingConditions else (staRole, dynRole) \
                        if (staRole, dynRole) in matchingConditions else None
                    if clunuPair is None:
                        # print("Skipping ({}, {})".format(staRole, dynRole))
                        continue
                    cluPairCond = matchingConditions[clunuPair]
                    fieldMatches = [fieldNum for fieldNum, fieldCond in enumerate(cluPairCond[1:])
                                    if not any(fieldCond[:5]) and (fieldCond[5] or fieldCond[6] or fieldCond[7])]
                    dynTemplates = [
                        (alignedFieldClasses[clunuPair][0][fieldNum], alignedFieldClasses[clunuPair][1][fieldNum])
                        if dynRole == clunuPair[0] else
                        (alignedFieldClasses[clunuPair][1][fieldNum], alignedFieldClasses[clunuPair][0][fieldNum])
                        for fieldNum in fieldMatches]
                    for dynT, custva in dynTemplates:
                        if not isinstance(dynT, Template) or not isinstance(custva, MessageSegment):
                            continue
                        if dynT not in staValues:
                            # set the current static value to the STA-field values for the DYN template
                            staValues[dynT] = custva
                            clunuPairs[dynT] = clunuPair
                        elif staValues[dynT].values != custva.values and self.dc.pairDistance(staValues[dynT], custva) > 0.1:
                            #
                            # staMismatch = True
                            #
                            # if dc.pairDistance(dynT.medoid, staValues[dynT]) > dc.pairDistance(dynT.medoid, custva):
                            #     removeFromMatchingClusters.append(clunuPairs[dynT])
                            # else:
                            #     removeFromMatchingClusters.append(clunuPair)
                            #
                            # print("prevValue {} and {} currentValues".format(staValues[dynT], custva))
                            if staValues[dynT].values not in [bsv.values for bsv in dynT.baseSegments]:
                                # and dc.pairDistance(dynT.medoid, staValues[dynT]) > 0.15:
                                print("remove", clunuPairs[dynT], "because of", staValues[dynT])
                                removeFromMatchingClusters.append(clunuPairs[dynT])
                            if custva.values not in [bsv.values for bsv in dynT.baseSegments]:
                                # and dc.pairDistance(dynT.medoid, custva) > 0.15:
                                print("remove", clunuPair, "because of", custva)
                                removeFromMatchingClusters.append(clunuPair)
                            #
                            # break
                            #
                    #
                    # if staMismatch:
                    #     break
                #
                # if staMismatch:
                #     # mask to remove the clunuPairs of all combinations with this dynRole
                #     removeFromMatchingClusters.extend([
                #         (dynRole, staRole) if (dynRole, staRole) in matchingConditions else (staRole, dynRole)
                #         for staRole in staRoleList
                #     ])
            except KeyError as e:
                print("KeyError:", e)
                IPython.embed()
                raise e
        print("remove for transitive STA mimatch:", removeFromMatchingClusters)
        print()

        # if a chain of matches would merge more than .66 of all clusters, remove that chain
        dracula = Graph()
        dracula.add_edges_from(set(matchingClusters) - set(removeFromMatchingClusters))
        connectedDracula = list(connected_components(dracula))
        for clusterChain in connectedDracula:
            if len(clusterChain) > .66 * len(self.alignedClusters):  # TODO increase to .66 (nbns tshark)
                for clunu in clusterChain:
                    for remainingPair in set(matchingClusters) - set(removeFromMatchingClusters):
                        if clunu in remainingPair:
                            removeFromMatchingClusters.append(remainingPair)

        remainingClusters = set(matchingClusters) - set(removeFromMatchingClusters)
        if len(remainingClusters) > 0:
            print("Clusters could be merged:")
            for clunuAB in remainingClusters:
                printClusterMergeConditions(clunuAB, alignedFieldClasses, matchingConditions, self.dc)

        print("remove finally:", removeFromMatchingClusters)

        print("remain:", remainingClusters)
        chainedRemains = Graph()
        chainedRemains.add_edges_from(remainingClusters)
        connectedClusters = list(connected_components(chainedRemains))

        # for statistics
        if clusterStats is not None:
            missedmerges = ClusterClusterer.printShouldMerge(connectedClusters, clusterStats)
            missedmergepairs = [k for k in remainingClusters if any(
                [k[0] in mc and k[1] in mc or
                 k[0] in mc and k[1] in chain.from_iterable([cc for cc in connectedClusters if k[0] in cc]) or
                 k[0] in chain.from_iterable([cc for cc in connectedClusters if k[1] in cc]) and k[1] in mc
                 for mc in missedmerges]
            )]

        singleClusters = {ck: ml for ck, ml in messageClusters.items() if not chainedRemains.has_node(ck)}
        mergedClusters = {str(mergelist):
                              list(chain.from_iterable([messageClusters[clunu] for clunu in mergelist]))
                          for mergelist in connectedClusters}
        mergedClusters.update(singleClusters)

        return mergedClusters

    def merge(self, forNemesys=False, clusterStats=None):
        alignedFieldClasses = self._alignFieldClasses((0, -1, 5))  # TODO alt1
        # alignedFieldClasses = clustermerger._alignFieldClasses((0, -5, 5))  # TODO alt2
        if forNemesys:
            alignedFieldClasses = self._gapMerging4nemesys(alignedFieldClasses)
        matchingConditions = self._generateMatchingConditions(alignedFieldClasses)
        matchingClusters = ClusterMerger._selectMatchingClusters(alignedFieldClasses, matchingConditions)
        return self._mergeClusters(self.messageTuplesClusters, clusterStats, alignedFieldClasses,
                                   matchingClusters, matchingConditions)


class ClusterClusterer(ClusterAligner):

    def __init__(self, alignedClusters: Dict[int, List], dc: DistanceCalculator):
        super().__init__(alignedClusters, dc)
        self.clusterOrder = [clunu for clunu in sorted(alignedClusters.keys()) if clunu != -1]
        self.distances = self.calcClusterDistances()

    def calcClusterDistances(self, mmg=(0, -1, 5)):
        from nemere.inference.segmentHandler import matrixFromTpairs

        fclassHirsch, statDynFields, statDynValues = self.generateHirsch(mmg)

        statDynValuesMap = {sdv: idx for idx, sdv in enumerate(statDynValues)}
        statDynIndices = {clunu: [statDynValuesMap[fc] for fc in sdf] for clunu, sdf in statDynFields.items()}

        clusterpairs = combinations(self.clusterOrder, 2)   # self.clusterOrder = sorted(statDynFields.keys())
        nwscores = [(clunuA, clunuB, fclassHirsch.nwScore(statDynIndices[clunuA], statDynIndices[clunuB])[-1])
                            for clunuA, clunuB in clusterpairs]

        # arrange similarity matrix from nwscores
        similarityMatrix = matrixFromTpairs(nwscores, self.clusterOrder)
        # fill diagonal with max similarity per pair
        for ij in range(similarityMatrix.shape[0]):
            # The max similarity for a pair is len(shorter) * self._score_match
            # see Netzob for reference
            similarityMatrix[ij,ij] = len(statDynIndices[ij]) * fclassHirsch.score_match

        # calculate distance matrix
        minScore = min(fclassHirsch.score_gap, fclassHirsch.score_match, fclassHirsch.score_mismatch)
        base = numpy.empty(similarityMatrix.shape)
        maxScore = numpy.empty(similarityMatrix.shape)
        for i in range(similarityMatrix.shape[0]):
            for j in range(similarityMatrix.shape[1]):
                maxScore[i, j] = min(  # The max similarity for a pair is len(shorter) * self._score_match
                    # the diagonals contain the max score match for the pair, calculated in _calcSimilarityMatrix
                    similarityMatrix[i, i], similarityMatrix[j, j]
                )
                minDim = min(len(statDynIndices[i]), len(statDynIndices[j]))
                base[i, j] = minScore * minDim

        distanceMatrix = 100 - 100 * ((similarityMatrix - base) / (maxScore - base))
        assert distanceMatrix.min() >= 0, "prevent negative values for highly mismatching messages"
        return distanceMatrix

    def neighbors(self):
        neighbors = list()
        for idx, dists in enumerate(self.distances):  # type: int, numpy.ndarray
            neighbors.append(sorted([(i, d) for i, d in enumerate(dists) if i != idx], key=lambda x: x[1]))
        return neighbors

    def autoconfigureDBSCAN(self):
        """
        Auto configure the clustering parameters epsilon and minPts regarding the input data

        :return: minpts, epsilon
        """
        from scipy.ndimage.filters import gaussian_filter1d
        from math import log, ceil

        neighbors = self.neighbors()
        sigma = log(len(neighbors))
        knearest = dict()
        smoothknearest = dict()
        seconddiff = dict()
        seconddiffMax = (0, 0, 0)
        # can we omit k = 0 ?
        # No - recall and even more so precision deteriorates for dns and dhcp (1000s)
        for k in range(0, ceil(log(len(neighbors) ** 2))):  # first log(n^2)   alt.: // 10 first 10% of k-neighbors
            knearest[k] = sorted([nfori[k][1] for nfori in neighbors])
            smoothknearest[k] = gaussian_filter1d(knearest[k], sigma)
            # max of second difference (maximum positive curvature) as knee (this not actually the knee!)
            seconddiff[k] = numpy.diff(smoothknearest[k], 2)
            seconddiffargmax = seconddiff[k].argmax()
            # noinspection PyArgumentList
            diffrelmax = seconddiff[k].max() / smoothknearest[k][seconddiffargmax]
            if 2 * sigma < seconddiffargmax < len(neighbors) - 2 * sigma and diffrelmax > seconddiffMax[2]:
                seconddiffMax = (k, seconddiffargmax, diffrelmax)

        k = seconddiffMax[0]
        x = seconddiffMax[1] + 1

        epsilon = smoothknearest[k][x]
        min_samples = round(sigma)
        print("eps {:0.3f} autoconfigured from k {}".format(epsilon, k))
        return epsilon, min_samples

    def clusterMessageTypesDBSCAN(self, eps = 1.5, min_samples = 3) \
            -> Tuple[Dict[int, List[int]], numpy.ndarray, DBSCAN]:
        clusterer = DBSCAN(metric='precomputed', eps=eps,
                         min_samples=min_samples)

        print("Messages: DBSCAN epsilon:", eps, "min samples:", min_samples)
        clusterClusters, labels = self._postprocessClustering(clusterer)
        return clusterClusters, labels, clusterer

    def _postprocessClustering(self, clusterer: Union[DBSCAN]) \
            -> Tuple[Dict[int, List[int]], numpy.ndarray]:
        clusterer.fit(self.distances)

        labels = clusterer.labels_  # type: numpy.ndarray
        assert isinstance(labels, numpy.ndarray)
        ulab = set(labels)
        clusterClusters = dict()
        for l in ulab:  # type: int
            class_member_mask = (labels == l)
            clusterClusters[l] = [seg for seg in compress(self.clusterOrder, class_member_mask)]

        print(len([ul for ul in ulab if ul >= 0]), "Clusters found",
              "(with noise {})".format(len(clusterClusters[-1]) if -1 in clusterClusters else 0))

        return clusterClusters, labels

    @staticmethod
    def mergeClusteredClusters(clusterClusters: Dict[int, List[int]],
                               messageClusters: Dict[int, List[AbstractMessage]]):
        mergedClusters = {str(mergelist):
                              list(chain.from_iterable([messageClusters[clunu] for clunu in mergelist]))
                          for mergelabel, mergelist in clusterClusters.items() if mergelabel != -1}
        singleClusters = {ck: ml for ck, ml in messageClusters.items() if ck not in chain.from_iterable(clusterClusters.values())}
        mergedClusters.update(singleClusters)
        return mergedClusters




