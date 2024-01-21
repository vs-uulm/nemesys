import math
from collections import Counter, MutableSequence
from itertools import combinations, chain
from typing import List, Iterable, Sequence, Tuple, Union

import scipy.stats
import numpy, pandas, re
from networkx import Graph
from networkx.algorithms.components.connected import connected_components
from tabulate import tabulate

from nemere.inference.segmentHandler import segments2types, filterChars
from nemere.inference.segments import MessageSegment, TypedSegment, AbstractSegment
from nemere.inference.templates import DelegatingDC, AbstractClusterer, Template, TypedTemplate, FieldTypeTemplate
from nemere.utils.loader import SpecimenLoader
from nemere.utils.evaluationHelpers import StartupFilecheck
from nemere.utils.baseAlgorithms import ecdf, tril
from nemere.visualization.distancesPlotter import DistancesPlotter
from nemere.visualization.multiPlotter import MultiMessagePlotter


class ClusterLabel(object):
    """
    Helper to generate cluster labels from cluster properties.
    """
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
        """More classical string representation than `__repr__()`"""
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
        """
        :return: The most frequent field type in the cluster. Requires `self.mostFrequentTypes` to be set.
        """
        if isinstance(self.mostFrequentTypes, Sequence):
            return self.mostFrequentTypes[0][1] / sum(s for t, s in self.mostFrequentTypes)
        return None


class SegmentClusterCauldron(object):
    """
    Container class for results of the clustering of segments.
    """
    noise: List[AbstractSegment]
    clusters: List[List[AbstractSegment]]

    def __init__(self, clusterer: AbstractClusterer, analysisTitle: str):
        """
        Cluster segments according to the dissimilarities of their feature vectors.
        Keep and label segments classified as noise.

        Start post processing of clusters (splitting, merging, singular/regular clusters, ...) after the desired
        preparation of clusters (e.g., by extractSingularFromNoise, appendCharSegments, ...) by calling
        `clustersOfUniqueSegments()`
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
        """Append the given char segments to the cluster container."""
        if len(charSegments) > 0:
            self.clusters.append(charSegments)

    @staticmethod
    def truncateList(values: Iterable, maxLen=5):
        """Truncate a list of values if its longer than maxLen by adding ellipsis dots."""
        output = [str(v) for v in values]
        if len(output) > maxLen:
            return output[:2] + ["..."] + output[-2:]
        return output

    # # # # # # # # # # # # # # # # # # # # # # # # # # #

    def clustersOfUniqueSegments(self):
        """
        Consolidate cluster contents so that the same value in different segments is only represented once per cluster.
        The clusters are also stored in the instance as property `self.unisegClusters`

        :return: structure of `segments2clusteredTypes`
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
        """
        self.regularClusters = SegmentClusters(self.clusterer.distanceCalculator)
        self.singularClusters = SegmentClusters(self.clusterer.distanceCalculator)
        for uc in self.unisegClusters:
            if len({seg[1].bytes for seg in uc[1]}) > 3:
                self.regularClusters.append(uc)
            else:
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
        """
        :return: List of field type templates generated from the clusters in this cauldron.
        """
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
        Cluster segments according to the dissimilarities of their feature vectors.
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
        """
        Determine the most frequent field types from the given list of typed segments.
        :param cluster: List of typed segments
        :return: Sorted list of most frequent types as tuples of a type and its count.
        """
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
    """Container for clusters of unique segments."""

    _il3q = 99
    """parameter: percentile q to remove extreme outliers from dissimilarities (for std calc)"""
    _il3t = .4
    """parameter: threshold for the ratio of stds to indicate a linear chain"""
    _il4t = 3.
    """
    parameter: threshold for the ratio of the increase of the matrix traces of the sorted dissimilarity matrix
    in direction of the largest extent of the cluster"""

    def __init__(self, dc: DelegatingDC):
        """
        Initialize an empty container for segment clusters.
        :param dc: Distance calculator that contains entries for all segments that will be contained in any clusters
            that will be added to this container.
        """
        self._clusters = list()  # type: List[Tuple[ClusterLabel, List[Tuple[str, TypedSegment]]]]
        self._distanceCalculator = dc

    def insert(self, index: int, o: Tuple[ClusterLabel, List[Tuple[str, TypedSegment]]]) -> None:
        """
        Insert the entry o before for the cluster with the given index.

        TODO check if o[1] is already in cluster #i and do not do anything (raise an Error?)
          except updating the element labels to [mixed] if they were not identical in o[1] and cluster[i][1]

        :param index: Cluster index.
        :param o: Cluster tuple structure.
        """
        self._clusters.insert(index, o)

    def __getitem__(self, i: int) -> Tuple[ClusterLabel, List[Tuple[str, TypedSegment]]]:
        """
        Return the entry for the cluster with the given index.

        :param i: Cluster index.
        :return: Cluster tuple structure.
        """
        return self._clusters.__getitem__(i)

    def __setitem__(self, i: int, o: Tuple[ClusterLabel, List[Tuple[str, TypedSegment]]]) -> None:
        """
        Set/replace the entry for the cluster with the given index with o.

        TODO check if o[1] is already in cluster #i and do not do anything (raise an Error?)
          except updating the element labels to [mixed] if they were not identical in o[1] and cluster[i][1]

        :param i: Cluster index.
        :param o: Cluster tuple structure.
        """
        self._clusters.__setitem__(i, o)

    def __delitem__(self, i: int) -> None:
        """
        Remove the cluster with the given index from this container.
        :param i: Cluster index.
        """
        self._clusters.__delitem__(i)

    def __len__(self) -> int:
        """
        :return: Number of clusters in this container.
        """
        return self._clusters.__len__()

    def __contains__(self, o: Tuple[ClusterLabel, List[Tuple[str, TypedSegment]]]):
        """
        Check if o exists in this container.

        TODO check if o[1] is already in cluster #i and do not do anything (raise an Error?)
          except updating the element labels to [mixed] if they were not identical in o[1] and cluster[i][1]

        :param o: Cluster tuple structure.
        :return: True if the given cluster o exists in this container.
        """
        return self._clusters.__contains__(o)

    def clusterContains(self, i: int, o: Tuple[str, TypedSegment]):
        """
        Check if the entry o exists in the cluster given by the index.

        TODO check if o[1] is already in cluster #i and do not do anything (raise an Error?)
          except updating the element labels to [mixed] if they were not identical in o[1] and cluster[i][1]

        :param i: Cluster index.
        :param o: Cluster elements.
        :return: True if o is in the cluster with the given index, False otherwise.
        """
        for lab, ele in self._clusters[i][1]:
            if lab == o[0] and ele.values == o[1].values:
                return True
        return False

    def clusterIndexOfSegment(self, segment: TypedSegment):
        """
        :param segment: Segment to search for.
        :return: Index of the cluster that contains the given segment,
          None if the segment is not in any of the clusters of this container.
        """
        for ci in self.clusterIndices:
            if segment in self.clusterElements(ci):
                return ci
        return None

    def clusterLabel(self, i: int) -> str:
        """
        :param i: Cluster index.
        :return: Label of the cluster with the given index.
        """
        return str(self._clusters[i][0])

    def clusterElements(self, i: int) -> List[TypedSegment]:
        """
        :param i: Cluster index.
        :return: List of segments contained in the cluster with the given index.
        """
        return [b for a,b in self._clusters[i][1]]

    @property
    def clusters(self):
        """
        :return: Raw list of clusters in this container.
        """
        return self._clusters

    def dcSubMatrix(self, i: int):
        """
        :param i: Cluster index.
        :return: Dissimilarity matrix for the cluster with the given index.
        """
        return self._distanceCalculator.distancesSubset(self.clusterElements(i))

    def __repr__(self):
        """
        :return: Textual representation of this cluster container.
        """
        return "\n".join(self.clusterLabel(i) for i in self.clusterIndices)

    # # # # # # # # # # # # # # # # # # # # # # # # # # #

    def _modClusterLabel(self, subc: List[Tuple[str, Union[AbstractSegment, Template]]],
                         hasGT: bool, clusterNumber: str, analysisTitle: str):
        """
        Prepare a new cluster label for the given cluster subc from its properties.

        :param subc: Cluster to create a label for.
        :param hasGT: Flag to indicate if ground truth information should be used. Requires TypedSegment in subc.
        :param clusterNumber: Cluster index.
        :param analysisTitle: Title of the analysis to use in the label.
        :return: Label for the cluster with the given index.
        """
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
        As pivot use `ln(occSum)`. (Determination of a knee has some false positives
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

        # median values for the 1-nearest neighbor ("minimum" dissimilarity).
        minmedians = [numpy.median([self._distanceCalculator.neighbors(ce, self.clusterElements(i))[1][1]
                                  for ce in self.clusterElements(i)])
                    for i in self.clusterIndices]
        maxdists = [self.maxDist(i) for i in self.clusterIndices]

        trils = [self.trilFlat(i) for i in self.clusterIndices]
        cpDists = { (i, j): self._distanceCalculator.distancesSubset(self.clusterElements(i), self.clusterElements(j))
            for i, j in combinations(self.clusterIndices, 2) }

        # In case of empty dissimilarities, the median may be requested from an empty list.
        #   This is no problem, thus ignore.
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
                # density as median dissimilarity in $\epsilon$-neighborhood with smallClusterExtent as $\epsilon$
                dists2linki = numpy.delete(self.dcSubMatrix(i)[coordmin[0]], coordmin[0])
                dists2linkj = numpy.delete(self.dcSubMatrix(j)[coordmin[1]], coordmin[1])
                densityi = numpy.median(dists2linki[dists2linki <= smallClusterExtent / 2])
                densityj = numpy.median(dists2linkj[dists2linkj <= smallClusterExtent / 2])

                vals.append((
                    (i,j),                                # 0: indices tuple
                    trils[i].mean(), None, minmedians[i], # 1*, 2, 3*: of dissimilarities in cluster i
                    None,                                 # 4
                    trils[j].mean(), None, minmedians[j], # 5*, 6, 7*: of dissimilarities in cluster j
                    cpDists[(i, j)].min(),                # 8*: min of dissimilarities between i and j
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
        :return: The values of the lower triangle of the dissimilarity matrix of the given cluster omitting the diagonal
            as a list.
        """
        return tril(self.dcSubMatrix(i))

    def occurrences(self, i: int) -> List[int]:
        """
        Count distinct values in a cluster.

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

    def traverseViaNearest(self, i: int):
        """
        Path through the cluster hopping from nearest neighbor to nearest neighbor that has not already been visited.
        All segments contained in the cluster with index i are visited this way.

        >>> from tabulate import tabulate
        >>> from nemere.utils.baseAlgorithms import generateTestSegments
        >>> from nemere.inference.templates import DelegatingDC
        >>> from nemere.validation.clusterInspector import SegmentClusterContainer
        >>> segments = generateTestSegments()
        >>> dc = DelegatingDC(segments)
        Calculated distances for 37 segment pairs in ... seconds.
        >>> someclusters = SegmentClusterContainer(dc)
        >>> someclusters.append(("Sample Cluster", list(("Some Type", s) for s in segments)))
        >>> path = zip(*someclusters.traverseViaNearest(0))
        >>> print(tabulate(path))
        ----------------------------------------------------------------  ---------
        MessageSegment 4 bytes at (0, 4): 00000000 | values: (0, 0, 0...  1
        MessageSegment 4 bytes at (0, 4): 01020304 | values: (1, 2, 3...  0.125
        MessageSegment 4 bytes at (0, 4): 03020304 | values: (3, 2, 3...  0.214355
        MessageSegment 3 bytes at (0, 3): 020304 | values: (2, 3, 4)      0.111084
        MessageSegment 3 bytes at (0, 3): 010304 | values: (1, 3, 4)      0.367676
        MessageSegment 2 bytes at (0, 2): 0204 | values: (2, 4)           0.0714111
        MessageSegment 2 bytes at (0, 2): 0203 | values: (2, 3)           0.700684
        MessageSegment 3 bytes at (0, 3): 250545 | values: (37, 5, 69)    0.57666
        ----------------------------------------------------------------  ---------

        :param i: Cluster index
        :return: The path as a list of segments and the dissimilarities along this path
        """
        idxA, idxC, segA, segC = self.remotestSegments(i)
        distances = self.dcSubMatrix(i)
        numpy.fill_diagonal(distances, numpy.nan)
        segmentReference = self.clusterElements(i)
        if Counter(segmentReference).most_common(1)[0][1] > 1:
            raise ValueError("Duplicate element in cluster.")
        df = pandas.DataFrame(distances, index=segmentReference, columns=segmentReference)

        segB = segA
        path = [segA]
        distsAlongPath = list()
        while df.shape[0] > 1:
            nearest = df[segB].idxmin(0)
            path.append(nearest)
            distsAlongPath.append(df.at[segB, nearest])
            df.drop(segB, axis=0, inplace=True)
            df.drop(segB, axis=1, inplace=True)
            assert all(numpy.isnan(numpy.diag(df)))  # be sure we removed symmetrical
            segB = nearest
        return path, numpy.array(distsAlongPath)

    def maxDist(self, i: int):
        """
        :param i: Cluster index
        :return: The maximum dissimilarity between any two segments in cluster i.
        """
        if self.distinctValues(i) < 2:
            # If cluster contains only one template, we define the (maximum) dissimilarity to itself to be zero.
            #   (Less than 1 template/segment should not happen, but we handle it equally and do not fail,
            #   should it happen.)
            return 0
        dist = self.trilFlat(i)  # type: numpy.ndarray
        # noinspection PyArgumentList
        return dist.max()

    def remotestSegments(self, i: int):
        """
        Determine the segment with the maximum sum of dissimilarities to all other segments (A)
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
        """
        :param i: Cluster index.
        :return: The lengths in bytes of the elements in cluster i.
        """
        segLens = numpy.array(list({e.length for e in self.clusterElements(i)}))
        return segLens

    def charSegmentCount(self, i: int):
        """
        :param i: Cluster index.
        :return: Number of character segments in cluster i.
        """
        return len(filterChars(self.clusterElements(i)))

    @staticmethod
    def mostFrequentTypes(cluster: List[Tuple[str, TypedSegment]]):
        """
        :param cluster: Cluster as a list of labels and typed segments.
        :return: Sorted list of the most frequent types extraced from the element labels.
        """
        segLabelExtractor = re.compile(r"(\w*): (\d*) Seg.s")
        allLabels = {l for l,e in cluster}
        typeStats = [next(segLabelExtractor.finditer(l)).groups() for l in allLabels]
        mostFrequentTypes = sorted(((ftype, int(tsegs)) for ftype, tsegs in typeStats),
                                   key=lambda x: -x[1])
        return mostFrequentTypes

    def distancesSortedByLargestExtent(self, i: int):
        """
        :param i: Cluster index.
        :return: Dissimilarities of all elements in the cluster with index i
            sorted by the dissimilarity from the remotest segment in the cluster.
        """
        smi = self.dcSubMatrix(i)
        dfari = smi[self.remotestSegments(i)[0]]
        cmi = self.clusterElements(i)
        idxi = sorted(range(len(cmi)), key=lambda x: dfari[x])
        sfari = [cmi[e] for e in idxi]
        return self._distanceCalculator.distancesSubset(sfari)

    def traceMeanDiff(self, i: int):
        """
        Difference of the first and the mean of all other diffs of all $k$-traces (sums of the
            $k$th superdiagonals) of the sorted dissimilarity matrix for the segments in this cluster.
            It is sorted by the dissimilarity from the remotest segment in the cluster.

        :param i: Cluster index.
        :return: Dissimilarity difference value.
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

    def shapeIndicators(self, i: int):
        """
        Heuristic indicators of whether the cluster with index i is linear-chain or chaotic/globular-blob shaped.

        The heuristics are:
        * indiLinear1: indicates a probable linear chain if the dissimilarities along one path through the cluster
            are less than the maximum of all dissimilarities multiplied with the path length through the cluster.
        * indiLinear3: indicates a probable linear chain if the ratio between
            the standard deviation of all dissimilarities and the standard deviation of dissimilarities along a path
            through the cluster is below the threshold `_il3t`.
        indiChaotic: indicates a probable blob/chaotic cluster if the dissimilarities along one path through the cluster
            are greater than the maximum of all dissimilarities multiplied with the path length through the cluster.

        :param i: Cluster index.
        :return: Tuple of three Boolean indicators. The two first for linear chained, the third for chaotic.
        """
        path, distsAlongPath = self.traverseViaNearest(i)
        maxD = self.maxDist(i)
        distList = self.trilFlat(i)

        # deal with extreme outliers for indiLinear3
        distMask = distList < numpy.percentile(distList, type(self)._il3q)
        diapMask = distsAlongPath < numpy.percentile(distsAlongPath, type(self)._il3q)
        # standard deviation of all dissimilarities without extreme outliers (outside 99-percentile)
        distStdwoO = numpy.std(distList[distMask], dtype=numpy.float64)
        # standard deviation of dissimilarities along path without extreme outliers (outside 99-percentile)
        diapStdwoO = numpy.std(distsAlongPath[diapMask], dtype=numpy.float64)

        indiLinear1 = bool(sum(distsAlongPath) < maxD * math.log(len(path)))  # indicates probable linear chain
        indiLinear3 = bool(diapStdwoO / distStdwoO < type(self)._il3t)  # indicates probable linear chain
        indiChaotic = bool(sum(distsAlongPath) > maxD * 1.2 * math.log(len(path)))  # indicates probable blob/chaotic cluster

        return indiLinear1, indiLinear3, indiChaotic

    def indiLinear1(self, i: int):
        """
        One of multiple heuristic indicators of whether the cluster with index i is linear-chain shaped.
        see `self.shapeIndicators()`

        :param i: Cluster index.
        :return: True if the heuristic indicates a linear-chain shaped cluster, False otherwise.
        """
        return self.shapeIndicators(i)[0]

    def indiLinear3(self, i: int):
        """
        One of multiple heuristic indicators of whether the cluster with index i is linear-chain shaped.
        see `self.shapeIndicators()`

        :param i: Cluster index.
        :return: True if the heuristic indicates a linear-chain shaped cluster, False otherwise.
        """
        return self.shapeIndicators(i)[1]

    def indiLinear4(self, i: int):
        """
        One of multiple heuristic indicators of whether the cluster with index i is linear-chain shaped.
        Checks whether the difference of the first and the mean of all other diffs of all $k$-traces
        are below threshold `self._il4t`.

        :param i: Cluster index.
        :return: True if the heuristic indicates a linear-chain shaped cluster, False otherwise.
        """
        return self.traceMeanDiff(i) < self._il4t

    def indiChaotic(self, i: int):
        """
        Heuristic indicator of whether the cluster with index i is chaotic/globular-blob shaped.
        see `self.shapeIndicators()`

        :param i: Cluster index.
        :return: True if the heuristic indicates a chaotic/globular-blob shaped cluster, False otherwise.
        """
        return self.shapeIndicators(i)[2]

    def isAddr(self, i: int):
        """
        Heuristic indicator of whether the cluster with index i contains address fields.

        :param i: Cluster index.
        :return: True if the heuristic indicates the named field types to be prevalent in the cluster i,
            False otherwise.
        """
        return self.occurrenceLnPercentRank(i) < 35
            # don't care about shape
            #  and self.shapeIndicators(i)[1:] == (True, False)  # don't care about indicator 1

    def isSequence(self, i: int):
        """
        Heuristic indicator of whether the cluster with index i contains sequential numbering fields.

        :param i: Cluster index.
        :return: True if the heuristic indicates the named field types to be prevalent in the cluster i,
            False otherwise.
        """
        return 70 <= self.occurrenceLnPercentRank(i) <= 92.5 \
               and self.indiLinear4(i) == True and self.indiChaotic(i) == False

    def isIdFlagsInt(self, i: int):
        """
        Heuristic indicator of whether the cluster with index i contains identifier, flags, or arbitrary integer fields.

        :param i: Cluster index.
        :return: True if the heuristic indicates the named field types to be prevalent in the cluster i,
            False otherwise.
        """
        return self.occurrenceLnPercentRank(i) > 85 \
               and all(2 <= self.elementLengths(i)) and all(self.elementLengths(i) <= 4)

    def isTimestamp(self, i: int):
        """
        Heuristic indicator of whether the cluster with index i contains timestamp fields.

        :param i: Cluster index.
        :return: True if the heuristic indicates the named field types to be prevalent in the cluster i,
            False otherwise.
        """
        return self.occurrenceLnPercentRank(i) > 95 \
               and self.indiLinear4(i) == False and self.indiChaotic(i) == True \
               and all(3 <= self.elementLengths(i)) and all(self.elementLengths(i) <= 8)

    def isPayload(self, i: int):
        """
        Heuristic indicator of whether the cluster with index i contains payload fields.
        Currently only considers character sequences as payload.

        :param i: Cluster index.
        :return: True if the heuristic indicates the named field types to be prevalent in the cluster i,
            False otherwise.
        """
        # CHARS        [ or (BYTES > 8) ]
        return self.charSegmentCount(i) / len(self.clusterElements(i)) > .80 # \
               # or all(self.elementLengths(i) > 8)

    def isPad(self, i: int):
        """
        Heuristic indicator of whether the cluster with index i contains padding fields.

        :param i: Cluster index.
        :return: True if the heuristic indicates the named field types to be prevalent in the cluster i,
            False otherwise.
        """
        valCnt = Counter(chain.from_iterable(e.values for e in self.clusterElements(i)))
        mcVal, mcCnt = valCnt.most_common(1)[0]
        # most common is null byte and nulls are almost the exclusive content
        return mcVal == 0 and mcCnt / sum(valCnt.values()) > .95

    @property
    def clusterIndices(self):
        """
        :return: Range of all valid cluster indices in this container.
        """
        return range(len(self))

    def semanticTypeHypotheses(self):
        """
        :return: String value of the most probably field type according to the heuristic about the semantics
            of the fields in the clusters of this container.
            A dict of indices mapped to the semantic hypthesis for the respective cluster
        """
        hypo = dict()
        for i in self.clusterIndices:
            if self.isPad(i):
                hypo[i] = "pad"
            elif self.distinctValues(i) <= 3:
                hypo[i] = None          # Flags/Addresses? see singular cluster rules for flags/id/addr
            elif self.isPayload(i):     # prio 0
                hypo[i] = "payload"
            elif self.isTimestamp(i):   # prio 2 (swapped to 1)
                hypo[i] = "timestamp"
            elif self.isIdFlagsInt(i):  # prio 1 (swapped to 2)
                hypo[i] = "id/flags/int"
            elif self.isSequence(i):    # prio 3
                hypo[i] = "addr/seq"  # "sequence"  TODO try to use distribution of values (equally distributed dissimilarities?)
            elif self.isAddr(i):        # prio 4
                hypo[i] = "addr/seq"  # "addr"
            else:
                hypo[i] = None
        return hypo


class SegmentClusters(SegmentClusterContainer):
    """
    Extension of the container for clusters of unique segments with further kinds of cluster properties to be evaluated.
    This may support future work to identify semnatics of field types and thus also contains a number of helper
    functions to plot and print cluster properties, representations, and hypothesis statistics.
    """

    def plotDistances(self, i: int, specimens: SpecimenLoader, comparator=None):
        """
        Plot the dissimilarities between all segments of the given cluster in this container as Topology Plots.

        :param i: Cluster index.
        :param specimens: SpecimenLoader object to determine the source of the data
            for retaining the correct link to the evaluated trace.
        :param comparator: Compare to ground truth, if available.
        """
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

    def plotDistributions(self, specimens: SpecimenLoader):
        """
        Plot the dissimilarity distribution density per cluster in this container.

        :param specimens: SpecimenLoader object to determine the source of the data
            for retaining the correct link to the evaluated trace.
        """
        plotTitle = "Distances Distribution Density per Cluster"
        mmp = MultiMessagePlotter(specimens, plotTitle, math.ceil(len(self)))
        subTitles = list()
        # spn: sub plot number, clu: cluster
        for spn in range(len(self)):
            cl = self.clusterLabel(spn)
            subTitles.append(cl)

            dist = self.trilFlat(spn)
            cnts, bins, _ = mmp.axes[spn].hist(dist, bins=30)
            # mmp.axes[spn].legend()

            mu, sigma = scipy.stats.norm.fit(dist)
            bestfit = scipy.stats.norm.pdf(bins, mu, sigma)
            yscale = sum(cnts) / sum(bestfit)
            mmp.axes[spn].plot(bins, bestfit*yscale)
        mmp.nameEachAx(subTitles)
        mmp.writeOrShowFigure()

    def plotPincipalComponents(self, specimens: SpecimenLoader, sampleSize=None):
        """
        Print up to the first ten eigenvalues of the covariance matrix of the dissimilarities per cluster.

        :param specimens: For instantiating the plotter class and thus to determine the source of the data
            for retaining the correct link to the evaluated trace.
        :param sampleSize: If not None, enforce comparable results by random subsampling of clusters to
            sampleSize and ignoring smaller clusters.
        """
        plotTitle = "Eigenvalues per Cluster"
        mmp = MultiMessagePlotter(specimens, plotTitle, math.ceil(len(self)))
        subTitles = list()
        vals = list()
        for i in range(len(self)):
            cl = self.clusterLabel(i)
            subTitles.append(cl)
            if sampleSize is not None:
                if len(self.clusterElements(i)) < sampleSize:
                    print("Ignoring cluster", cl, "with element count", len(self.clusterElements(i)))
                    continue
                dists = self._distanceCalculator.distancesSubset(
                    numpy.random.choice(self.clusterElements(i), sampleSize, False))
            else:
                dists = self.dcSubMatrix(i)
            cov = numpy.cov(dists)
            eigenValuesAndVectors = numpy.linalg.eigh(cov)
            vals.append((cl, *eigenValuesAndVectors[0][10::-1]))
            mmp.plotToSubfig(i, eigenValuesAndVectors[0][10::-1])
        print(tabulate(vals))
        mmp.nameEachAx(subTitles)
        mmp.writeOrShowFigure()

    def shapiro(self, i: int):
        """
        Perform the Shapiro-Wilk test for normality on cluster i.

        Result: Probably no test for normal distribution will work. Clusters with few elements tend to become false
        positives while in any case the densities in the histograms will be truncated at 0 and the cluster "boundary".

        :param i: Cluster index.
        :return: True if the Shapiro-Wilk test passes, False otherwise.
        """
        dists = self.trilFlat(i)
        subsDists = numpy.random.choice(dists, 5000, False) if len(dists) > 5000 else dists
        # noinspection PyUnresolvedReferences
        return scipy.stats.shapiro(subsDists)[1] > 0.05

    def skewedness(self, i: int):
        """
        Calculate a measure for the skewedness of the given cluster cluster.
        The used skewedness measure is `(mode - mean)/stdev`.

        :param i: Cluster index.
        :return: Tuple of mode, median, mean, standard deviation, and skewedness
            calculated from the dissimilarities in the cluster i.
        """
        dists = self.trilFlat(i)
        # noinspection PyUnresolvedReferences
        mode = scipy.stats.mode(dists).mode[0]
        median = numpy.median(dists)
        mean = numpy.mean(dists)
        stdev = numpy.std(dists)
        skn = (mode - median) / stdev
        return mode, median, mean, stdev, skn

    def occurrencePercentile(self, i: int, q=95):
        """
        Percentile with the given q of the value occurrence frequencies in cluster i.

        :param i: Cluster index.
        :param q: Quantile.
        :return: The percentile value.
        """
        return numpy.percentile(numpy.array(self.occurrences(i)), q)

    def eig_eigw(self, i: int, rank=1):
        """
        Deprecated.

        :param i: Cluster index.
        :return: Direct eigenvalues up to rank.
        """
        dists = self.dcSubMatrix(i)
        return numpy.absolute(scipy.linalg.eigh(dists)[0])[:rank]

    def eig_hasDirection(self, i: int):
        """
        Deprecated.

        :param i: Cluster index.
        :return: True if given cluster with index i has a "direction" in terms of a "significantly large"
            first eigenvalue relative to the maximal dissimilarity.
        """
        dirFactThresh = 5
        ew = self.eig_eigw(i,2)
        # ew0diff1 = ew[0] > 2 * ev[1]  # this distinguishes nothing helpful
        ew0snf = ew[0] > dirFactThresh * self.maxDist(i)
        return ew0snf # and ew0diff1

    def pc_percentile(self, i: int, q=85):
        """
        Percentile with the given q of the principle component scores (eigenvalues)
        of the covariance matrix of cluster with index i.

        :param i: Cluster index.
        :param q: Quantile.
        :return: The percentile value.
        """
        dists = self.dcSubMatrix(i)
        cov = numpy.cov(dists)
        eigenValuesAndVectors = numpy.linalg.eigh(cov)
        return numpy.percentile(eigenValuesAndVectors[0], q)

    def pc_stats(self, i: int, q=95):
        """
        Set of principle component statistics for cluster i including the percentile of princile component scores
        for the given quantile q.
        Useful for checking the three-sigma rule of thumb with quantiles: 68–**95**–99.7.

        :param i: Cluster index.
        :param q: Quantile.
        :return: Tuple of
            * The cluster label,
            * ratio of the sum of the first two and the last two scores (eigenvalues),
            * the percentile of the scores above of the given quantile q,
            * the sum of scores above of the given quantile q,
            * the variance of dissimilarities concentrated on first PCs,
            * and the number of differing values in the cluster i.
        """
        eigenValuesAndVectors = numpy.linalg.eigh(numpy.cov(self.dcSubMatrix(i)))
        p95 = self.pc_percentile(i, q)
        pcS = sum(eigenValuesAndVectors[0] > p95)  # larger 1: "has extent"
        return (
            self.clusterLabel(i),
            sum(eigenValuesAndVectors[0][:-2]) / sum(eigenValuesAndVectors[0][-2:]),
            p95,
            pcS,
            pcS / self.distinctValues(i),  # value > 0.1: ratio of significant PCs among all components,
            # works as indicator for shape in nbns, not for smb
            # interpretation: variance of dissimilarities concentrated on first PCs.
            self.distinctValues(i)
        )

    def entropy(self, i: int):
        """Not implemented."""
        raise NotImplementedError()

    def ecdf(self, i: int):
        """
        :param i: Cluster index.
        :return: The empirical cummulative distribution function of the cluster i.
        """
        dist = self.trilFlat(i)  # type: numpy.ndarray
        return ecdf(list(dist))

    def plotECDFs(self, specimens: SpecimenLoader):
        """
        Plot the empirical cummulative distribution function of the cluster i.

        :param specimens: SpecimenLoader object to determine the source of the data
            for retaining the correct link to the evaluated trace.
        """
        plotTitle = "Empirical Cumulated Distribution Function per Cluster"
        mmp = MultiMessagePlotter(specimens, plotTitle, math.ceil(len(self)))
        subTitles = list()
        for spn in range(len(self)):
            cl = self.clusterLabel(spn)
            print(cl)
            subTitles.append(cl)

            ecdValues = self.ecdf(spn)
            mmp.axes[spn].plot(*ecdValues)
        mmp.nameEachAx(subTitles)
        mmp.writeOrShowFigure()

    def triangularDistances(self, i: int):
        """
        Evaluate triangle inequality for the remotest elements in the given cluster.

        :param i: Cluster index.
        :return: Values of how much longer the detour over any B is than the direct path A--C in the cluster i.
        """
        idxA, idxC, segA, segC = self.remotestSegments(i)
        distances = self.dcSubMatrix(i)
        directAC = distances[idxA,idxC]

        viaB = distances[idxA] + distances[idxC]
        assert viaB[idxA] == directAC  # this is path A--A--C
        assert viaB[idxC] == directAC  # this is path A--C--C
        viaB[idxC] = numpy.nan
        viaB[idxA] = numpy.nan

        detourB = viaB - directAC  # how much longer than the direct path A--C are we for each B?
        return detourB
        # TODO how to compare to a single threshold (as a tuned parameter)?

    def shapeStats(self, filechecker: StartupFilecheck, doPrint=False):
        """
        Collects the different shape statistics that are the basis for sematic heuristics
        of the clusters in this container. Writes the cluster shape statistics to a CSV and returns it as a list.

        :param filechecker: Use the filechecker class to determine a suitable report folder to write the CSV to.
        :param doPrint: Flag to request the statistics table to be printed to the console in addition.
        :return: The statistics as multi-dimensional list.
        """
        """filechecker.pcapstrippedname"""
        from tabulate import tabulate
        import csv

        vals = list()
        for i in range(len(self)):
            if self.distinctValues(i) < 2:
                print("Omit cluster with less than two elements", self.clusterLabel(i), "(id/flags?)")
                continue

            idxA, idxC, segA, segC = self.remotestSegments(i)
            path, distsAlongPath = self.traverseViaNearest(i)
            maxD = self.maxDist(i)
            distList = self.trilFlat(i)

            # deal with extreme outliers for indiLinear3
            distMask = distList < numpy.percentile(distList, type(self)._il3q)
            diapMask = distsAlongPath < numpy.percentile(distsAlongPath, type(self)._il3q)
            # standard deviation of all dissimilarities without extreme outliers (outside 99-percentile)
            distStdwoO = numpy.std(distList[distMask], dtype=numpy.float64)
            # standard deviation of dissimilarities along path without extreme outliers (outside 99-percentile)
            diapStdwoO = numpy.std(distsAlongPath[diapMask], dtype=numpy.float64)

            # indicates probable linear chain
            indiLinear1 = bool(sum(distsAlongPath) < maxD * math.log(len(path)))
            # (indicates probable linear chain)
            indiLinear2 = bool(path.index(segC) / len(path) > .9)
            # indicates probable linear chain
            indiLinear3 = bool(diapStdwoO / distStdwoO < type(self)._il3t)
            # indicates probable blob/chaotic cluster
            indiChaotic = bool(sum(distsAlongPath) > maxD * 1.2 * math.log(len(path)))
            # everything else: probable non-linear (multi-linear) substructure
            assert self.shapeIndicators(i) == (indiLinear1, indiLinear3, indiChaotic)

            # shorter float can cause "RuntimeWarning: overflow encountered in reduce" for large clusters
            distStd = numpy.std(distList, dtype=numpy.float64)
            diapStd = numpy.std(distsAlongPath, dtype=numpy.float64)

            # collect statistics
            vals.append((
                self.clusterLabel(i),
                numpy.mean(distList),
                distStd,
                maxD,
                sum(distsAlongPath),
                sum(distsAlongPath) - maxD,
                path.index(segC),
                len(path),
                numpy.mean(distsAlongPath),
                diapStd,
                numpy.percentile(distsAlongPath, 95),
                diapStdwoO,
                distStdwoO,
                self.traceMeanDiff(i),

                indiLinear1,
                indiLinear2,
                indiLinear3,
                self.indiLinear4(i),
                indiChaotic,

                math.log(self.occurrenceSum(i)),
                self.occurrenceLnPercentRank(i),
                self.occurrencePercentile(i),
                numpy.std(self.occurrences(i)),
                numpy.median(self.occurrences(i)),
                self.distinctValues(i),
            ))
        headers = [
            "cluster", "meanDist", "stdDist", "maxDist", "distsAlongPath", "detour", "segCinPath", "pathLength",
            "mean(distsAlongPath)", "std(distsAlongPath)", "percentile(distsAlongPath, 95)", "distStdwoO", "diapStdwoO",
            "traceMeanDiff",
            "indiLinear1", "indiLinear2", "indiLinear3", "indiLinear4", "indiChaotic",
            "lnOccSum", "lnOccPercRank", "95percentile", "occStd", "occMedian", "distinctV" ]
        if doPrint:
            print(tabulate(vals, headers=headers))
        from os.path import join
        reportFile = join(filechecker.reportFullPath, "shapeStats-" + filechecker.pcapstrippedname + ".csv")
        print("Write cluster shape statistics to", reportFile)
        with open(reportFile, 'a') as csvfile:
            statisticscsv = csv.writer(csvfile)
            statisticscsv.writerow(headers)
            statisticscsv.writerows(vals)
        return vals

    def routingPath(self, i: int):
        """Not implemented."""
        raise NotImplementedError("Find a routing algorithm to find the shortest path from the most distant elements.")

    def linkedClustersStats(self):
        """
        Determine nearby (single-linked) clusters with very similar densities to evaluate
        the criteria used to merge clusters.
        The actual merging is performed by function `SegmentClusterContainer.mergeOnDensity()`

        :return: The values used to determine whether to merge clusters.
        """
        # median values for the 1-nearest neighbor ("minimum" dissimilarity).
        minmedians = [numpy.median([self._distanceCalculator.neighbors(ce, self.clusterElements(i))[1][1]
                                  for ce in self.clusterElements(i)])
                    for i in self.clusterIndices]
        # minmeans = [numpy.mean([self._distanceCalculator.neighbors(ce, self.clusterElements(i))[1][1]
        #                         for ce in self.clusterElements(i)])
        #             for i in self.clusterIndices()]
        maxdists = [self.maxDist(i) for i in self.clusterIndices]

        trils = [self.trilFlat(i) for i in self.clusterIndices]
        cpDists = { (i, j): self._distanceCalculator.distancesSubset(self.clusterElements(i), self.clusterElements(j))
            for i, j in combinations(self.clusterIndices, 2) }
        # plt.plot(*ecdf(next(iter(cpDists.values())).flat))

        vals = list()
        for i, j in combinations(self.clusterIndices, 2):
            # density in $\epsilon$-neighborhood around nearest points between similar clusters
            #   $\epsilon$-neighborhood: link segments (closest in two clusters) s_lc_ic_j
            #   d(s_lc_ic_j, s_k) <= $\epsilon$ for all (s_k in c_i) != s_lc_ic_j
            # the nearest points between the clusters
            coordmin = numpy.unravel_index(cpDists[(i, j)].argmin(), cpDists[(i, j)].shape)
            # index of the smaller cluster
            smallCluster = i if maxdists[i] < maxdists[j] else j
            # extent of the smaller cluster
            smallClusterExtent = maxdists[smallCluster]
            # density as median dissimilarities in $\epsilon$-neighborhood with smallClusterExtent as $\epsilon$
            dists2linki = numpy.delete(self.dcSubMatrix(i)[coordmin[0]], coordmin[0])
            dists2linkj = numpy.delete(self.dcSubMatrix(j)[coordmin[1]], coordmin[1])
            densityi = numpy.median(dists2linki[dists2linki <= smallClusterExtent / 2])
            densityj = numpy.median(dists2linkj[dists2linkj <= smallClusterExtent / 2])

            trili = trils[i]
            trilj = trils[j]
            vals.append((
                self.clusterLabel(i),                       # 0
                trili.mean(), trili.std(), minmedians[i],   # 1, 2, 3
                self.clusterLabel(j),                       # 4
                trilj.mean(), trilj.std(), minmedians[j],   # 5, 6, 7
                cpDists[(i, j)].min(),                      # 8
                cpDists[(i, j)].max(),                      # 9
                densityi,                                   # 10
                densityj                                    # 11
            ))

        # merge cluster condition!! working on nbns with no FP!
        areVeryCloseBy = [bool(v[8] < v[1] or v[8] < v[5]) for v in vals]
        linkHasSimilarEpsilonDensity = [bool(abs(v[10] - v[11]) < 0.01) for v in vals]
        # closer as the mean between both cluster's "densities" normalized to the extent of the cluster
        areSomewhatCloseBy = [bool(v[8] < numpy.mean([v[3] / v[1], v[7] / v[5]])) for v in vals]
        haveSimilarDensity = [bool(abs(v[3] - v[7]) < 0.002) for v in vals]

        print(tabulate([
            ( *v, areVeryCloseBy[ij], linkHasSimilarEpsilonDensity[ij],
              areSomewhatCloseBy[ij], haveSimilarDensity[ij] ) for ij, v in enumerate(vals)]))
        # works with test traces:
        #  nbns! smb!
        #  dhcp: no! => + linkHasSimilarEpsilonDensity
        #  dns!
        #  ntp!
        return vals
