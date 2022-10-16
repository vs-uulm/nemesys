import math
from collections import Counter, MutableSequence
from itertools import combinations
from typing import List, Iterable, Sequence, Tuple, Union

import scipy.stats
import numpy, re
from networkx import Graph
from networkx.algorithms.components.connected import connected_components
from tabulate import tabulate

from nemere.inference.segmentHandler import segments2types, filterChars
from nemere.inference.segments import MessageSegment, TypedSegment, AbstractSegment
from nemere.inference.templates import DelegatingDC, AbstractClusterer, Template, TypedTemplate, FieldTypeTemplate
from nemere.utils.loader import SpecimenLoader
from nemere.utils.baseAlgorithms import tril
from nemere.visualization.distancesPlotter import DistancesPlotter


class ClusterLabel(object):
    def __init__(self, clusterNumber: Union[None, str, int] = None):
        self.clusterNumber = None  # type: Union[None, str]
        if clusterNumber is None:
            self.isNoise = True  # type: bool
        else:
            self.isNoise = False
            if isinstance(clusterNumber, int):
                self.clusterNumber = f"{clusterNumber:02d}"
            else:
                self.clusterNumber = clusterNumber

        self.analysisTitle = None  # type: Union[None, str]
        self.lengthsString = None  # type: Union[None, str]
        self.mostFrequentTypes = None  # type: Union[None, Sequence[Tuple[str, int]]]
        self.clusterSize = None  # type: Union[None, str]
        self.maxDist = None  # type: Union[None, float]

    def __repr__(self):
        """
        Generates a string representation of this label, based on the information present in the instance.
        Options are:
            * if its noise (isNoise): lengthsString and clusterSize
            * non-noise: lengthsString, clusterSize, clusterNumber, and maxDist
                * optionally with analysisTitle being "split", "merged", or "singular"
                * optionally with mostFrequentTypes
            * otherwise just "tf" followed by clusterNumber
        """
        if self.isNoise:
            if self.lengthsString is not None and self.clusterSize is not None:
                return 'Noise: {} Seg.s ({} bytes)'.format(self.clusterSize, self.lengthsString)
            else:
                return "Noise"
        if self.lengthsString is not None and self.clusterSize is not None \
                and self.clusterNumber is not None and self.maxDist is not None:
            if self.analysisTitle in ("split", "merged", "singular"):
                prepend = self.analysisTitle + " "
            else:
                prepend = ""
            # prevent overflowing plots
            clusterNumber = self.clusterNumber
            if len(self.clusterNumber) > 17:
                clusterSplit = self.clusterNumber.split("+")
                if len(clusterSplit) > 4:
                    clusterNumber = "+".join(clusterSplit[:2] + ["..."] + clusterSplit[-2:])
            if self.mostFrequentTypes is not None:
                return prepend + 'Cluster #{} ({:.2f} {}): {} Seg.s ($d_{{max}}$={:.3f}, {} bytes)'.format(
                    clusterNumber,
                    self.mostFrequentRatio,
                    self.mostFrequentTypes[0][0],
                    self.clusterSize,
                    self.maxDist,
                    self.lengthsString)
            return prepend + 'Cluster #{}: {} Seg.s ($d_{{max}}$={:.3f}, {} bytes)'.format(
                clusterNumber,
                self.clusterSize,
                self.maxDist,
                self.lengthsString)
        else:
            return f"tf{self.clusterNumber}"

    def toString(self):
        """More classical string representation than repr"""
        if self.isNoise:
            if self.analysisTitle and self.lengthsString and self.clusterSize:
                return '{} ({} bytes), Noise: {} Seg.s'.format(self.analysisTitle, self.lengthsString, self.clusterSize)
            else:
                return "Noise"
        if self.analysisTitle and self.lengthsString and self.clusterSize and self.clusterNumber \
                and self.clusterSize and self.maxDist:
            if self.mostFrequentTypes:
                return '{} ({} bytes), Cluster #{} ({:.2f} {}): {} Seg.s ($d_{{max}}$={:.3f})'.format(
                    self.analysisTitle,
                    self.lengthsString,
                    self.clusterNumber,
                    self.mostFrequentRatio,
                    self.mostFrequentTypes[0][0],
                    self.clusterSize,
                    self.maxDist)
            return '{} ({} bytes), Cluster #{}: {} Seg.s ($d_{{max}}$={:.3f})'.format(
                self.analysisTitle,
                self.lengthsString,
                self.clusterNumber,
                self.clusterSize,
                self.maxDist)
        else:
            return f"tf{self.clusterNumber}"

    @property
    def mostFrequentRatio(self) -> Union[None, float]:
        if isinstance(self.mostFrequentTypes, Sequence):
            return self.mostFrequentTypes[0][1] / sum(s for t, s in self.mostFrequentTypes)
        return None


class SegmentClusterCauldron(object):
    """
    Container class for results of the clustering of segments
    """
    noise: List[AbstractSegment]
    clusters: List[List[AbstractSegment]]

    def __init__(self, clusterer: AbstractClusterer, analysisTitle: str):
        """
        Cluster segments according to the distance of their feature vectors.
        Keep and label segments classified as noise.

        Start post processing of clusters (splitting, merging, singular/regular clusters, ...) after the desired
        preparation of clusters (e.g., by extractSingularFromNoise, appendCharSegments, ...) by calling
        **clustersOfUniqueSegments()**
        before advanced function are available.

        :param clusterer: Clusterer object that contains all the segments to be clustered
        :type analysisTitle: the string to be used as label for the result
        """
        self._analysisTitle = analysisTitle
        self.clusterer = clusterer

        print("Clustering segments...")
        self.noise, *self.clusters = clusterer.clusterSimilarSegments(False)
        if any(isinstance(seg, Template) for seg in clusterer.segments):
            distinct = "distinct "
        else:
            dc = self.clusterer.distanceCalculator
            self.noise = list({ dc.segments[dc.segments2index([tSegment])[0]] for tSegment in self.noise })
            distinct = ""
        print("{} clusters generated from {} {}segments".format(len(self.clusters), len(clusterer.segments), distinct))
        self.unisegClusters  = None  # type: Union[None, SegmentClusters]
        self.regularClusters = None  # type: Union[None, SegmentClusters]
        self.singularClusters    = None  # type: Union[None, SegmentClusters]
        self.originalRegularClusters = None  # type: Union[None, SegmentClusters]

    def extractSingularFromNoise(self):
        """
        Extract "large" templates from noise that should rather be its own cluster.
        Works in place, i.e. changing the contained cluster containers.
        """
        from nemere.inference.templates import Template

        for idx, seg in reversed(list(enumerate(self.noise.copy()))):  # type: int, MessageSegment
            freqThresh = math.log(len(self.clusterer.segments))
            if isinstance(seg, Template):
                if len(seg.baseSegments) > freqThresh:
                    self.clusters.append([self.noise.pop(idx)])  # .baseSegments

    def appendCharSegments(self, charSegments: List[AbstractSegment]):
        """Append the given char segments to the cluster container. Needs """
        if len(charSegments) > 0:
            self.clusters.append(charSegments)

    @staticmethod
    def truncateList(values: Iterable, maxLen=5):
        """Truncate a list of values if its longer than maxLen by adding ellipsis."""
        output = [str(v) for v in values]
        if len(output) > maxLen:
            return output[:2] + ["..."] + output[-2:]
        return output

    # # # # # # # # # # # # # # # # # # # # # # # # # # #

    def clustersOfUniqueSegments(self):
        """
        Consolidate cluster contents so that the same value in different segments is only represented once per cluster.
        The clusters are also stored in the instance as property self.unisegClusters

        :return: structure of segments2clusteredTypes
        """
        segmentClusters = [   # identical to the second iteration of segments2clusteredTypes()
            ( self._clusterLabel(segs), type(self)._segmentsLabels(segs) )
            for segs in self.clusters
        ]
        dc = self.clusterer.distanceCalculator
        self.unisegClusters = SegmentClusters(dc)
        for cLabel, elements in segmentClusters:
            # consolidates multiple (raw) segments in their respective templates,
            #     while using a set to distinguish multiple identical templates with different labels
            # TODO this is somewhat strange, since it assumes that the same segment
            #  can appear multiple times across clusters. I don't remember, why I thought this is necessary.
            uniqueSegments = {(sLabel, dc.segments[dc.segments2index([tSegment])[0]])
                              for sLabel, tSegment in elements}
            self.unisegClusters.append((cLabel, sorted(uniqueSegments, key=lambda x: x[1].values)))
        self.__mixed()
        self._regularAndSingularClusters()
        return [(self.analysisLabel(), self.unisegClusters.clusters)]

    def __mixed(self):
        """Replace duplicate clusters of different types with one cluster of segments with a [mixed] label."""
        # TODO This becomes obsolete
        #  if the duplicate checks in SegmentClusters would be implemented as stated in the TODOs there.
        mixedSegments = [seg for seg, cnt in Counter(
            tSegment for cLabel, elements in self.unisegClusters for sLabel, tSegment in elements).items()
                         if cnt > 1]
        for tSegment in mixedSegments:
            mixedClusters = [elements for cLabel, elements in self.unisegClusters
                             if tSegment in (sElem for sLabel, sElem in elements)]
            # if len(mixedClusters) >= 2:
            #     print("len(mixedClusters) >= 2  # that would be strange and we needed to find some solution then")
            #     IPython.embed()
            assert len(mixedClusters) < 2  # that would be strange and we needed to find some solution then
            toReplace = [sIdx for sIdx, sTuple in enumerate(mixedClusters[0]) if sTuple[1] == tSegment]
            for rIdx in reversed(sorted(toReplace)):
                del mixedClusters[0][rIdx]
            mixedClusters[0].append(("[mixed]", tSegment))
            # TODO fix function: mixedClusters contains only the elements of the cluster
            #  and thus the change is never written to the actual list

    def _regularAndSingularClusters(self):
        """
        Fill lists with clusters that contain at least three distinct values (regular) and less (singular).
        see also nemere.inference.segmentHandler.extractEnumClusters()
        """
        self.regularClusters = SegmentClusters(self.clusterer.distanceCalculator)
        self.singularClusters = SegmentClusters(self.clusterer.distanceCalculator)
        for uc in self.unisegClusters:
            if len({seg[1].bytes for seg in uc[1]}) > 3:
                self.regularClusters.append(uc)
            else:
                # TODO evaluate nemere.inference.segmentHandler.extractEnumClusters
                self.singularClusters.append(uc)
        # redundantly store regular clusters before any mering and splitting
        self.originalRegularClusters = SegmentClusters(self.clusterer.distanceCalculator)
        self.originalRegularClusters.extend(self.regularClusters)
        self.regularClusters.mergeOnDensity()
        self.regularClusters.splitOnOccurrence()
        self.unisegClusters = SegmentClusters(self.clusterer.distanceCalculator)
        self.unisegClusters.extend(self.regularClusters)
        self.unisegClusters.extend(self.singularClusters)

    def analysisLabel(self):
        """Generate a label for the analysis the clusters in self are the result of."""
        segLengths = set()
        if self.noise:
            segLengths.update({seg.length for seg in self.noise})
        for segs in self.clusters:
            segLengths.update({seg.length for seg in segs})

        return '{} ({} bytes) {}'.format(
            self._analysisTitle,
            next(iter(segLengths)) if len(segLengths) == 1 else 'mixedamount',
            self.clusterer if self.clusterer else 'n/a')

    @staticmethod
    def _segmentsLabels(cluster: List[AbstractSegment]):
        """Generate an empty label for each segment in the given cluster. Used by subclasses.
        The order of segments is NOT retained!"""
        labeledSegments = [(None, seg) for seg in cluster]
        return labeledSegments

    def _clusterLabel(self, cluster: List[AbstractSegment]):
        """
        Generate a label for the given cluster, containing its index number some statistics.
        The method recognizes any known cluster and the noise per identity of the list of Segments.
        """
        segLenStr = " ".join(SegmentClusterCauldron.truncateList({seg.length for seg in cluster}))
        # the label for noise is a bit different than for the others
        if cluster == self.noise:
            cLabel = ClusterLabel()
            cLabel.analysisTitle = self._analysisTitle
            cLabel.lengthsString = segLenStr
            cLabel.clusterSize = len(self.noise)
            return cLabel
        else:
            # raises a ValueError if cluster is not known
            cLabel = ClusterLabel(self.clusters.index(cluster))
            cLabel.analysisTitle = self._analysisTitle
            cLabel.lengthsString = " ".join(SegmentClusterCauldron.truncateList({seg.length for seg in cluster}))
            cLabel.clusterSize = len(cluster)
            cLabel.maxDist = self.clusterer.distanceCalculator.distancesSubset(cluster).max()
            return cLabel

    def exportAsTemplates(self):
        fTypeTemplates = list()
        for i in self.regularClusters.clusterIndices:
            # generate FieldTypeTemplates (padded nans) - Templates as is
            ftype = FieldTypeTemplate(self.unisegClusters.clusterElements(i))
            ftype.fieldtype = self.unisegClusters.clusterLabel(i)
            fTypeTemplates.append(ftype)
        # treat all singular clusters as one
        singularElements = [element for i in self.singularClusters.clusterIndices
                            for element in self.singularClusters.clusterElements(i)]
        if len(singularElements) > 0:
            singularLabel = ClusterLabel(
                "+".join(self.singularClusters[i][0].clusterNumber for i in self.singularClusters.clusterIndices))
            singularLabel.analysisTitle = "singular"
            singularLabel.clusterSize = sum(
                len(e.baseSegments) if isinstance(e, Template) else 1 for e in singularElements)
            # noinspection PyArgumentList
            singularLabel.maxDist = self.clusterer.distanceCalculator.distancesSubset(singularElements).max()
            singularLabel.lengthsString = " ".join(SegmentClusterCauldron.truncateList({
                seg.length for seg in singularElements}))
            ftype = FieldTypeTemplate(singularElements)
            ftype.fieldtype = str(singularLabel)
            fTypeTemplates.append(ftype)
        return fTypeTemplates


class TypedSegmentClusterCauldron(SegmentClusterCauldron):
    """
    Container class for results of the clustering of segments and the evaluation of the clustering result.
    """
    noise: List[TypedSegment]
    clusters: List[List[TypedSegment]]

    def __init__(self, clusterer: AbstractClusterer, analysisTitle: str):
        """
        Cluster segments according to the distance of their feature vectors.
        Keep and label segments classified as noise.

        :param clusterer: Clusterer object that contains all the segments to be clustered
        :type analysisTitle: the string to be used as label for the result
        """
        assert all(isinstance(seg, TypedSegment) for seg in clusterer.segments), \
            "This class is used for evaluating the result quality. Thus, its necessary to use segments that are " \
            "annotated with there true data type. See annotateFieldTypes()"
        super().__init__(clusterer,analysisTitle)

    def segments2clusteredTypes(self):
        """
        TODO replace nemere.inference.segmentHandler.segments2clusteredTypes in callers

        :return: List/Tuple structure of annotated analyses, clusters, and segments.
        List [ of
            Tuples (
                 "analysis label",
                 List [ of cluster
                    Tuples (
                        "cluster label",
                        List [ of segment
                            Tuples (
                                "segment label (e. g. field type)",
                                MessageSegment object
                            )
                        ]
                    )
                ]
            )
        ]
        """
        segmentClusters = list()
        if self.noise:
            segmentClusters.append((
                self._clusterLabel(self.noise),
                type(self)._segmentsLabels(self.noise)
            ))
        for segs in self.clusters:
            segmentClusters.append((
                self._clusterLabel(segs),
                type(self)._segmentsLabels(segs)
            ))
        return [(self.analysisLabel(), segmentClusters)]

    @staticmethod
    def _segmentsLabels(cluster: List[TypedSegment]):
        """Generate a label for each segment in the given cluster. The order of segments is NOT retained!"""
        typeGroups = segments2types(cluster)
        labeledSegments = list()
        for ftype, tsegs in typeGroups.items():  # [label, segments]
            occurence = len(tsegs)
            labeledSegments.extend([(
                "{}: {} Seg.s".format(ftype, occurence),
                tseg
            ) for tseg in tsegs])
        return labeledSegments

    def _clusterLabel(self, cluster: List[TypedSegment]):
        """
        Generate a label for the given cluster, containing its index number some statistics.
        The method recognizes any known cluster and the noise per identity of the list of Segments.
        """
        cLabel = super()._clusterLabel(cluster)
        if cluster != self.noise:
            cLabel.mostFrequentTypes = TypedSegmentClusterCauldron.getMostFrequentTypes(cluster)
        return cLabel

    @staticmethod
    def getMostFrequentTypes(cluster: List[TypedSegment]):
        typeGroups = segments2types(cluster)
        return sorted(((ftype, len(tsegs)) for ftype, tsegs in typeGroups.items()),
                                   key=lambda x: -x[1])

    def label4segment(self, seg: AbstractSegment) -> Union[bool, str]:
        """
        Prepare string label of seg for usage in plot legend. Returns False if no information about
        the segment is present in this instance.
        """
        # simple case: directly in one cluster
        if seg in self.noise:
            return str(self._clusterLabel(self.noise))
        for i in self.unisegClusters.clusterIndices:
            if seg in self.unisegClusters.clusterElements(i):
                return self.unisegClusters.clusterLabel(i)
        # complex case: seg is a template (that was not in a cluster directly)
        #  and we need to check all groups/clusters for the basesegments of the template (seg).
        if isinstance(seg, Template):
            inGroup = None  # type: Union[None, str]
            for bs in seg.baseSegments:
                if bs in self.noise:
                    inGroup = str(self._clusterLabel(self.noise))
                for i in self.unisegClusters.clusterIndices:
                    if bs in self.unisegClusters.clusterElements(i):
                        name = self.unisegClusters.clusterLabel(i)
                        if inGroup is None or inGroup == name:
                            inGroup = name
                        else:
                            return "[mixed]"
            if inGroup is not None:
                return inGroup
            else:
                # not anywhere to be found
                return False
        # not anywhere to be found
        return False

class SegmentClusterContainer(MutableSequence):
    """Container for Clusters of unique segments."""

    _il3q = 99  # parameter: percentile q to remove extreme outliers from distances (for std calc)
    _il3t = .4  # parameter: threshold for the ratio of stds to indicate a linear chain
    _il4t = 3.  # parameter: threshold for the ratio of the increase of the matrix traces of the sorted distance matrix
                #               in direction of the largest extent of the cluster

    def __init__(self, dc: DelegatingDC):
        self._clusters = list()  # type: List[Tuple[ClusterLabel, List[Tuple[str, TypedSegment]]]]
        self._distanceCalculator = dc

    def insert(self, index: int, o: Tuple[ClusterLabel, List[Tuple[str, TypedSegment]]]) -> None:
        # TODO check if o[1] is already in cluster #i and do not do anything (raise an Error?) except updating the element labels to [mixed] if they were not identical in o[1] and cluster[i][1]
        self._clusters.insert(index, o)

    def __getitem__(self, i: int) -> Tuple[ClusterLabel, List[Tuple[str, TypedSegment]]]:
        return self._clusters.__getitem__(i)

    def __setitem__(self, i: int, o: Tuple[ClusterLabel, List[Tuple[str, TypedSegment]]]) -> None:
        # TODO check if o[1] is already in cluster #i and do not do anything (raise an Error?) except updating the element labels to [mixed] if they were not identical in o[1] and cluster[i][1]
        self._clusters.__setitem__(i, o)

    def __delitem__(self, i: int) -> None:
        self._clusters.__delitem__(i)

    def __len__(self) -> int:
        return self._clusters.__len__()

    def __contains__(self, o: Tuple[ClusterLabel, List[Tuple[str, TypedSegment]]]):
        # TODO check if o[1] is already in cluster #i and do not do anything (raise an Error?) except updating the element labels to [mixed] if they were not identical in o[1] and cluster[i][1]
        return self._clusters.__contains__(o)

    def clusterContains(self, i: int, o: Tuple[str, TypedSegment]):
        # TODO check if o[1] is already in cluster #i and do not do anything (raise an Error?) except updating the element labels to [mixed] if they were not identical in o[1] and cluster[i][1]
        for lab, ele in self._clusters[i][1]:
            if lab == o[0] and ele.values == o[1].values:
                return True
        return False

    def clusterIndexOfSegment(self, segment: TypedSegment):
        for ci in self.clusterIndices:
            if segment in self.clusterElements(ci):
                return ci
        return None

    def clusterLabel(self, i: int) -> str:
        return str(self._clusters[i][0])

    def clusterElements(self, i: int) -> List[TypedSegment]:
        return [b for a,b in self._clusters[i][1]]

    @property
    def clusters(self):
        return self._clusters

    def dcSubMatrix(self, i: int):
        return self._distanceCalculator.distancesSubset(self.clusterElements(i))

    def __repr__(self):
        return "\n".join(self.clusterLabel(i) for i in self.clusterIndices)

    # # # # # # # # # # # # # # # # # # # # # # # # # # #

    def _modClusterLabel(self, subc: List[Tuple[str, Union[AbstractSegment, Template]]],
                         hasGT: bool, clusterNumber: str, analysisTitle: str):
        cLabel = ClusterLabel(clusterNumber)
        cLabel.analysisTitle = analysisTitle
        cLabel.lengthsString = " ".join(SegmentClusterCauldron.truncateList({seg.length for l, seg in subc}))
        cLabel.clusterSize = sum(len(e.baseSegments) if isinstance(e, Template) else 1 for e in subc)
        # noinspection PyArgumentList
        cLabel.maxDist = self._distanceCalculator.distancesSubset([seg for l, seg in subc]).max()
        if hasGT:
            # noinspection PyTypeChecker
            typedSegs = [seg for l, seg in subc]  # type: List[TypedSegment]
            cLabel.mostFrequentTypes = TypedSegmentClusterCauldron.getMostFrequentTypes(typedSegs)
        return cLabel

    def splitOnOccurrence(self):
        """
        Split clusters if they have extremely polarized occurrences
        (e.g., many unique values, very few very high occurring values).
        As pivot use ln(occSum). (Determination of a knee has some false positives
            and is way too complex for its benefit).
        """
        splitClusters = list()
        rankThreshold = 95
        for i in self.clusterIndices:
            # if rank > 95 and std > ln(occSum) split at ln(occSum)
            pivot = math.log(self.occurrenceSum(i))
            if self.occurrenceLnPercentRank(i) > rankThreshold \
                    and numpy.std(self.occurrences(i)) > pivot:
                hasGT = False
                if all(isinstance(seg, (TypedSegment, TypedTemplate)) for seg in self.clusterElements(i)):
                    hasGT = True
                # perform the splitting
                subcA = list()  # type: List[Tuple[str, Union[AbstractSegment, Template]]]
                subcB = list()  # type: List[Tuple[str, Union[AbstractSegment, Template]]]
                for l,e in self[i][1]:
                    if isinstance(e, Template) and len(e.baseSegments) > pivot:
                        subcA.append((l,e))
                    else:
                        subcB.append((l,e))

                cLabel = self._modClusterLabel(subcA, hasGT, f"{self[i][0].clusterNumber}s0", "split")
                splitClusters.append( (cLabel, subcA) )

                cLabel = self._modClusterLabel(subcB, hasGT, f"{self[i][0].clusterNumber}s1", "split")
                splitClusters.append( (cLabel, subcB ) )
            else:
                splitClusters.append(self[i])
        self._clusters = splitClusters
        return splitClusters

    def mergeOnDensity(self):
        """Merge nearby (single-linked) clusters with very similar densities."""
        import warnings

        epsilonDensityThreshold = 0.01
        neighborDensityThreshold = 0.002

        # median values for the 1-nearest neighbor ("minimum" distance).
        minmedians = [numpy.median([self._distanceCalculator.neighbors(ce, self.clusterElements(i))[1][1]
                                  for ce in self.clusterElements(i)])
                    for i in self.clusterIndices]
        maxdists = [self.maxDist(i) for i in self.clusterIndices]

        trils = [self.trilFlat(i) for i in self.clusterIndices]
        cpDists = { (i, j): self._distanceCalculator.distancesSubset(self.clusterElements(i), self.clusterElements(j))
            for i, j in combinations(self.clusterIndices, 2) }

        # in case of empty distances, the median may be requested from an empty list. This is no problem, thus ignore.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            vals = list()
            for i, j in combinations(self.clusterIndices, 2):
                # density in $\epsilon$-neighborhood around nearest points between similar clusters
                #   $\epsilon$-neighborhood: link segments (closest in two clusters) s_lc_ic_j
                #   d(s_lc_ic_j, s_k) <= $\epsilon$ for all (s_k in c_i) != s_lc_ic_j
                # the nearest points ("link segments") between the clusters
                coordmin = numpy.unravel_index(cpDists[(i, j)].argmin(), cpDists[(i, j)].shape)
                # index of the smaller cluster
                smallCluster = i if maxdists[i] < maxdists[j] else j
                # extent of the smaller cluster
                smallClusterExtent = maxdists[smallCluster]
                # density as median distances in $\epsilon$-neighborhood with smallClusterExtent as $\epsilon$
                dists2linki = numpy.delete(self.dcSubMatrix(i)[coordmin[0]], coordmin[0])
                dists2linkj = numpy.delete(self.dcSubMatrix(j)[coordmin[1]], coordmin[1])
                densityi = numpy.median(dists2linki[dists2linki <= smallClusterExtent / 2])
                densityj = numpy.median(dists2linkj[dists2linkj <= smallClusterExtent / 2])

                vals.append((
                    (i,j),                                # 0: indices tuple
                    trils[i].mean(), None, minmedians[i], # 1*, 2, 3*: of distances in cluster i
                    None,                                 # 4
                    trils[j].mean(), None, minmedians[j], # 5*, 6, 7*: of distances in cluster j
                    cpDists[(i, j)].min(),                # 8*: min of distances between i and j
                    None,                                 # 9
                    densityi,                             # 10*: density within epsilon around link segment in i
                    densityj                              # 11*: density within epsilon around link segment in j
                ))

        # merge cluster conditions: areVeryCloseBy and linkHasSimilarEpsilonDensity or areSomewhatCloseBy and haveSimilarDensity
        areVeryCloseBy =                [bool(v[8] < v[1] or v[8] < v[5]) for v in vals]
        linkHasSimilarEpsilonDensity =  [bool(abs(v[10] - v[11]) < epsilonDensityThreshold) for v in vals]
        # closer as the mean between both cluster's "densities" normalized to the extent of the cluster
        areSomewhatCloseBy =            [bool(v[8] < numpy.mean([v[3] / v[1], v[7] / v[5]])) for v in vals]
        haveSimilarDensity =            [bool(abs(v[3] - v[7]) < neighborDensityThreshold) for v in vals]

        # filter pairs of clusters to merge that
        #  areVeryCloseBy and linkHasSimilarEpsilonDensity
        #  or areSomewhatCloseBy and haveSimilarDensity
        toMerge = [
            ij[0] for ij, ca1, ca2, cb1, cb2 in
            zip(vals, areVeryCloseBy, linkHasSimilarEpsilonDensity, areSomewhatCloseBy, haveSimilarDensity)
            if ca1 and ca2 or cb1 and cb2
        ]

        # determine chains of merging candidates by graph analysis
        dracula = Graph()
        dracula.add_nodes_from(self.clusterIndices)
        dracula.add_edges_from(toMerge)
        connectedDracula = list(connected_components(dracula))

        # now actually merge
        mergedClusters = list()  # type: List[Tuple[ClusterLabel, List[Tuple[str, AbstractSegment]]]]
        for connected in connectedDracula:
            if len(connected) == 1:
                mergedClusters.append(self[next(iter(connected))])
            else:
                mc = list()  # type: List[Tuple[str, TypedSegment]]
                cnums = list()
                for cid in connected:
                    cnums.append(self[cid][0].clusterNumber)
                    mc.extend(self[cid][1])

                cLabel = self._modClusterLabel(mc,
                                               all(isinstance(seg, (TypedSegment, TypedTemplate)) for seg in mc),
                                               "+".join(cnums), "merged")
                mergedClusters.append( ( cLabel, mc ) )
        self._clusters = mergedClusters
        return mergedClusters

    # # # # # # # # # # # # # # # # # # # # # # # # # # #

    def trilFlat(self, i: int) -> numpy.ndarray:
        """
        :param i: cluster index
        :return: The values of the lower triangle of the distance matrix of the given cluster omitting the diagonal
            as a list.
        """
        return tril(self.dcSubMatrix(i))

    def occurrences(self, i: int) -> List[int]:
        """

        You may put this list in a Counter:
        >>> from collections import Counter
        >>> from nemere.utils.baseAlgorithms import generateTestSegments
        >>> from nemere.inference.templates import DelegatingDC
        >>> from nemere.validation.clusterInspector import SegmentClusterContainer
        >>> segments = generateTestSegments()
        >>> dc = DelegatingDC(segments)
        Calculated distances for 37 segment pairs in ... seconds.
        >>> someclusters = SegmentClusterContainer(dc)
        >>> someclusters.append(("Sample Cluster", list(("Some Type", s) for s in segments)))
        >>> cnt = Counter(someclusters.occurrences(0))
        >>> # noinspection PyUnresolvedReferences
        >>> sum(a*b for a,b in cnt.items()) == someclusters.occurrenceSum(0)
        True

        :param i: Cluster index
        :return: The numbers of occurrences of distinct values (not the values themselves!)
        """
        return [len(e.baseSegments) if isinstance(e, Template) else 1 for e in self.clusterElements(i)]

    def occurrenceSum(self, i: int):
        """
        The sum of occurrences equals the true number of segments including value duplicates.

        :param i: Cluster index
        :return: The sum of occurrences of all values.
        """
        return sum(self.occurrences(i))

    def occurrenceLnPercentRank(self, i: int):
        """
        "%rank of ln(sum(occ))": is a measure of the occurrence and value diversity.
        The percent-rank (80% means that 80% of the scores in a are below the given score) for the occurrences
        with the score ln(#elements).

        :param i:
        :return: percent rank
        """
        lnsumocc = math.log(self.occurrenceSum(i))  # ln of amount of all values (also identical ones)
        return scipy.stats.percentileofscore(self.occurrences(i), lnsumocc)

    def distinctValues(self, i: int):
        """
        :param i: Cluster index
        :return: Number of differing values
        """
        return len(self.clusterElements(i))

    def maxDist(self, i: int):
        """
        :param i: Cluster index
        :return: The maximum distance between any two segments in cluster i.
        """
        if self.distinctValues(i) < 2:
            # If cluster contains only one template, we define the (maximum) distance to itself to be zero.
            #   (Less than 1 template/segment should not happen, but we handle it equally and do not fail,
            #   should it happen.)
            return 0
        dist = self.trilFlat(i)  # type: numpy.ndarray
        # noinspection PyArgumentList
        return dist.max()

    def remotestSegments(self, i: int):
        """
        Determine the segment with the maximum sum of distances to all other segments (A)
        and the segment farthest away from this (C).

        >>> from pprint import pprint
        >>> from tabulate import tabulate
        >>> from nemere.utils.baseAlgorithms import generateTestSegments
        >>> from nemere.inference.templates import DistanceCalculator
        >>> from nemere.validation.clusterInspector import SegmentClusterCauldron
        >>> from nemere.inference.templates import DBSCANsegmentClusterer
        >>>
        >>> segments = generateTestSegments()
        >>> dc = DelegatingDC(segments)
        Calculated distances for 37 segment pairs in ... seconds.
        >>> clusterer = DBSCANsegmentClusterer(dc, segments)
        DBSCANsegmentClusterer: eps 0.476 autoconfigured (Kneedle on ECDF with S 0.8) from k 2
        >>> cauldron = SegmentClusterCauldron(clusterer, "DocTest")
        Clustering segments...
        DBSCAN epsilon: 0.476, minpts: 2
        1 clusters generated from 9 segments
        >>> listOfClusters = cauldron.clustersOfUniqueSegments()
        >>> pprint(listOfClusters)
        [('DocTest (mixedamount bytes) DBSCAN eps 0.476 mpt 2',
          [(Cluster #00: 6 Seg.s ($d_{max}$=0.440, 2 3 4 bytes),
            [(None, MessageSegment 4 bytes at (0, 4): 01020304 | values: (1, 2, 3...),
             (None, MessageSegment 3 bytes at (0, 3): 010304 | values: (1, 3, 4)),
             (None, MessageSegment 2 bytes at (0, 2): 0203 | values: (2, 3)),
             (None, MessageSegment 3 bytes at (0, 3): 020304 | values: (2, 3, 4)),
             (None, MessageSegment 2 bytes at (0, 2): 0204 | values: (2, 4)),
             (None,
              MessageSegment 4 bytes at (0, 4): 03020304 | values: (3, 2, 3...)])])]
        >>> vals = list()
        >>> for i in range(len(cauldron.regularClusters)):
        ...     idxA, idxC, segA, segC = cauldron.regularClusters.remotestSegments(i)
        ...     cxDistances = cauldron.regularClusters.dcSubMatrix(i)
        ...     directAC = cxDistances[idxA, idxC]
        ...     vals.append((
        ...         cauldron.regularClusters.clusterLabel(i),
        ...         cauldron.regularClusters.maxDist(i),
        ...         directAC,
        ...         cauldron.regularClusters.maxDist(i) == directAC  # mostly but not always True
        ...     ))
        >>> print(tabulate(vals))
        ---------------------------------------------------  -------  -------  -
        Cluster #00: 6 Seg.s ($d_{max}$=0.440, 2 3 4 bytes)  0.44043  0.44043  1
        ---------------------------------------------------  -------  -------  -

        :param i: Cluster index
        :return: The two segments as MessageSegment and the index in the cluster elements list self.clusterElements(i).
        """
        distances = self.dcSubMatrix(i)  # type: numpy.ndarray
        idxA = distances.sum(0).argmax()
        idxC = distances[idxA].argmax()
        segA = self.clusterElements(i)[idxA]
        segC = self.clusterElements(i)[idxC]
        return idxA, idxC, segA, segC

    def elementLengths(self, i: int) -> numpy.ndarray:
        segLens = numpy.array(list({e.length for e in self.clusterElements(i)}))
        return segLens

    def charSegmentCount(self, i: int):
        return len(filterChars(self.clusterElements(i)))

    @staticmethod
    def mostFrequentTypes(cluster: List[Tuple[str, TypedSegment]]):
        segLabelExtractor = re.compile(r"(\w*): (\d*) Seg.s")
        allLabels = {l for l,e in cluster}
        typeStats = [next(segLabelExtractor.finditer(l)).groups() for l in allLabels]
        mostFrequentTypes = sorted(((ftype, int(tsegs)) for ftype, tsegs in typeStats),
                                   key=lambda x: -x[1])
        return mostFrequentTypes

    def distancesSortedByLargestExtent(self, i: int):
        smi = self.dcSubMatrix(i)
        dfari = smi[self.remotestSegments(i)[0]]
        cmi = self.clusterElements(i)
        idxi = sorted(range(len(cmi)), key=lambda x: dfari[x])
        sfari = [cmi[e] for e in idxi]
        return self._distanceCalculator.distancesSubset(sfari)

    def traceMeanDiff(self, i: int):
        """
        Difference of the first and the mean of all other diffs of all $k$-traces (sums of the
            $k$th superdiagonals) of the sorted distance matrix for the segments in this cluster.
            It is sorted by the distance from the remotest segment in the cluster.

        :param i:
        :return:
        """
        #
        #
        sortedSMi = self.distancesSortedByLargestExtent(i)
        trMeans = [sortedSMi.trace(k) / (sortedSMi.shape[0] - k) for k in range(sortedSMi.shape[0])]
        return trMeans[1] / numpy.diff(trMeans[1:]).mean()
        # plt.pcolormesh(sortedSMi)
        # plt.title(cauldron.regularClusters.clusterLabel(i))
        # plt.show()
        # print(cauldron.regularClusters.clusterLabel(i))

    # # # # # # # # # # # # # # # # # # # # # # # # # # #



    @property
    def clusterIndices(self):
        return range(len(self))


class SegmentClusters(SegmentClusterContainer):

    def plotDistances(self, i: int, specimens: SpecimenLoader, comparator=None):
        if len(self.clusterElements(i)) < 2:
            print("Too few elements to plot in", self.clusterLabel(i), "Ignoring it.")
            return
        dists = self.dcSubMatrix(i)
        postfix = ""
        fnLabels = None
        if comparator:
            fnLabels = [set([comparator.lookupField(bs)[1] for bs in seg.baseSegments]
                            if isinstance(seg, Template) else [comparator.lookupField(seg)[1]]) for seg in dists]
            fnLabels = [next(iter(fl)) if len(fl) == 1 else repr(fl)[1:-1].replace("'", "") for fl in fnLabels]
            postfix = "-fnLabeled"
        sdp = DistancesPlotter(specimens, f"distances-cluster{i}" + postfix, False)
        if comparator:
            sdp.plotManifoldDistances(self.clusterElements(i), dists, numpy.array(fnLabels))
        else:
            idxA, idxC, segA, segC = self.remotestSegments(i)
            labels = [None]*len(self.clusterElements(i))
            labels[idxA] = "A"
            labels[idxC] = "C"
            sdp.plotManifoldDistances(self.clusterElements(i), dists, numpy.array(labels))
        sdp.axesFlat[1].set_title(self.clusterLabel(i))
        sdp.writeOrShowFigure()
