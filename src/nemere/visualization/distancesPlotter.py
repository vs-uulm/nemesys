import numpy
import matplotlib.pyplot as plt
from matplotlib import cm, colors

from typing import List, Any, Union, Sequence, Tuple, Hashable
from itertools import compress

from sklearn import manifold
from sklearn.decomposition import PCA

from netzob.Model.Vocabulary.Messages.RawMessage import RawMessage

from nemere.utils.loader import BaseLoader
from nemere.inference.segments import MessageSegment, TypedSegment, AbstractSegment
from nemere.inference.templates import Template, TypedTemplate, DistanceCalculator, FieldTypeTemplate
from nemere.visualization.plotter import MessagePlotter


class DistancesPlotter(MessagePlotter):
    """
    Plot distances between points of high dimensionality using manifold data embedding into a 2-dimensional plot.
    """

    def __init__(self, specimens: BaseLoader, analysisTitle: str,
                 isInteractive: bool=False, plotSegmentValues=False):
        super().__init__(specimens, analysisTitle, isInteractive)
        self._autoLegend = False

        # # plot configuration
        plt.rc('xtick', labelsize=10)  # fontsize of the tick labels
        plt.rc('ytick', labelsize=10)  # fontsize of the tick labels

        self._fig, self._axes = plt.subplots(1,2, figsize=(10,5))  # type: plt.Figure, numpy.ndarray
        if not isinstance(self._axes, numpy.ndarray):
            self._axes = numpy.array(self._axes)
        self._fig.set_size_inches(16, 8)
        # self.cm = cm.Set1  # has 9 colors
        # self.cm = cm.tab20 # 20 colors
        # noinspection PyUnresolvedReferences
        self.cm = cm.jet  # type: colors.LinearSegmentedColormap
        """label color map"""
        # noinspection PyUnresolvedReferences
        self.fcm = cm.cubehelix
        """type color map"""
        self.labsize = 150
        """label markers: size factor"""
        self.typsize = 30
        """type markers: size factor"""
        self.maxSamples = 1000
        """subsample the elements to plot if they are above this threshold"""
        self._plotSegmentValues = plotSegmentValues

    @property
    def axesFlat(self):
        return self._axes.flat

    def subsample(self,
                  segments: List[Union[MessageSegment, TypedSegment, TypedTemplate, Template, RawMessage, Any]],
                  distances: numpy.ndarray, labels: numpy.ndarray):
        """
        subsample the elements to plot if they are above the maxSamples threshold.

        :param segments: The original segments, messages or other elements to be plotted.
        :param distances: The pairwise distances between all of the original segments
        :param labels: The labels for each of the original segments
        :return: if subsampling was necessary, a tuple of
            (originalSegmentCount, and subsampled values for segments, distances, labels),
            else: False
        """
        originalSegmentCount = len(segments)
        if originalSegmentCount > 2 * self.maxSamples:
            import math
            ratiorev = originalSegmentCount / self.maxSamples
            step2keep = math.floor(ratiorev)
            lab2idx = dict()
            for idx, lab in enumerate(labels):
                if lab not in lab2idx:
                    lab2idx[lab] = list()
                lab2idx[lab].append(idx)
            # copy list to remove elements without side-effects
            segments = segments.copy()
            # to save the indices to be removed
            idx2rem = list()
            # determines a subset evenly distributed over all clusters while honoring the ratio to reduce to.
            for lab, ics in lab2idx.items():
                keep = set(ics[::step2keep])
                idx2rem.extend(set(ics) - keep)
            idx2rem = sorted(idx2rem, reverse=True)
            for idx in idx2rem:
                del segments[idx]
            labels = numpy.delete(labels, idx2rem, 0)
            distances = numpy.delete(numpy.delete(distances, idx2rem, 0), idx2rem, 1)
            return originalSegmentCount, segments, distances, labels
        else:
            return False

    @staticmethod
    def manifoldPositions(distances: numpy.ndarray):
        """prepare the 2 dimensionally projected positions for the input"""
        # prepare MDS
        seed = numpy.random.RandomState(seed=3)
        mds = manifold.MDS(n_components=2, max_iter=3000, eps=1e-9, random_state=seed,
                           dissimilarity="precomputed", n_jobs=1)
        pos = mds.fit(distances).embedding_
        # Rotate the data
        clf = PCA(n_components=2)
        return clf.fit_transform(pos)

    @staticmethod
    def uniqueLabels(labels: numpy.ndarray,
                     segments: List[Union[MessageSegment, TypedSegment, TypedTemplate, Template, RawMessage, Any]]) \
            -> List:
        """identify unique labels"""
        allabels = set(labels)
        if None in allabels:
            allabels.remove(None)
        if False in allabels:
            allabels.remove(False)
        if all(isinstance(l, numpy.integer) or l.isdigit() for l in allabels if l != "Noise"):
            ulab = sorted(allabels,
                          key=lambda l: -1 if l == "Noise" else int(l))
        else:
            ulab = sorted(allabels)

        # omit noise in cluster labels if types are plotted anyway.
        # the different handling is necessary due to the different noise markers in segments and messages.
        if any(isinstance(seg, (TypedSegment, TypedTemplate)) for seg in segments):
            for l in ulab:
                # find a string label containing "Noise" and remove it
                if isinstance(l, str) and "Noise" in l:
                    ulab.remove(l)
        elif isinstance(segments[0], RawMessage) and segments[0].messageType != "Raw":
            for l in ulab:
                # find a -1 integer label and remove it
                try:
                    if int(l) == -1:
                        ulab.remove(l)
                except ValueError:
                    pass  # not a problem, just keep the cluster, since its not noise.
        return ulab


    def plotManifoldDistances(self,
                              segments: List[Union[MessageSegment, TypedSegment, TypedTemplate, Template, RawMessage, Any]],
                              distances: numpy.ndarray,
                              labels: numpy.ndarray,
                              templates: List=None, plotEdges = False, countMarkers = False):
        # noinspection PyUnresolvedReferences
        """
        Plot distances of segments according to (presumably multidimensional) features.
        This function abstracts from the actual feature by directly taking a precomputed similarity matrix and
        arranging the segments relative to each other according to their distances using Multidimensional Scaling (MDS).
        See module `manifold` from package `sklearn`.

        If segments is a list of `TypedSegment` or `MessageSegment`, this function plots the feature values of each
        given segment overlaying each other besides the distances; they are colored according to the given labels.

        >>> from netzob.Model.Vocabulary.Messages.RawMessage import RawMessage
        >>> from nemere.utils.loader import BaseLoader
        >>> from nemere.inference.analyzers import Value
        >>>
        >>> bytedata = [
        ...     bytes([1, 2, 3, 4]),
        ...     bytes([   2, 3, 4]),
        ...     bytes([   1, 3, 4]),
        ...     bytes([   2, 4   ]),
        ...     bytes([   2, 3   ]),
        ...     bytes([20, 30, 37, 50, 69, 2, 30]),
        ...     bytes([        37,  5, 69       ]),
        ...     bytes([70, 2, 3, 4]),
        ...     bytes([3, 2, 3, 4])
        ...     ]
        >>> messages  = [RawMessage(bd) for bd in bytedata]
        >>> specimens = BaseLoader(messages)
        >>> analyzers = [Value(message) for message in messages]
        >>> segments  = [TypedSegment(analyzer, 0, len(analyzer.message.data)) for analyzer in analyzers]
        >>> for seg in segments[:4]:
        ...     seg.fieldtype = "ft1"
        >>> for seg in segments[4:6]:
        ...     seg.fieldtype = "ft2"
        >>> for seg in segments[6:]:
        ...     seg.fieldtype = "ft3"
        >>> DistanceCalculator.debug = False
        >>> dc = DistanceCalculator(segments, thresholdFunction=DistanceCalculator.neutralThreshold, thresholdArgs=None)
        Calculated distances for 37 segment pairs in ... seconds.
        >>> dp = DistancesPlotter(specimens, "test", False)
        >>> dp.plotManifoldDistances(segments, dc.distanceMatrix, numpy.array([1,2,3,1,1,0,1,0,2]))
        >>> dp.writeOrShowFigure()  # doctest: +SKIP

        :param segments: If `segments` is a list of `TypedSegment`s, field types are marked as small markers
            within the label marker. labels containing "Noise" then are not explicitly marked like the other labeled
            segments
        :param distances: The precomputed similarity matrix:
            symmetric matrix, rows/columns in the order of `segments`
        :param labels: Labels of strings (or ints or any other printable type) identifying the cluster for each segment
        :param templates: Templates of clusters to be printed alongside with the feature values.
            CURRENTLY UNTESTED
        :param plotEdges: Plot of edges between each pair of segment markers.
            Caution: Adds n^2 lines which takes very long compared to the scatterplot and
            quickly becomes a huge load especially when rendering the plot as PDF.
        :param countMarkers: add text labels with information at positions with multiple markers
        """
        assert isinstance(segments, Sequence)
        assert isinstance(distances, numpy.ndarray)
        assert isinstance(labels, numpy.ndarray)
        assert len(segments) == distances.shape[0] == distances.shape[1]

        from nemere.utils.evaluationHelpers import unknown

        axMDS, axSeg = self._axes  # type: plt.Axes, plt.Axes
        axMDS.set_aspect('equal', adjustable='datalim')

        # subsample if segment count is larger than maxSamples
        subret = self.subsample(segments, distances, labels)
        if subret:
            originalSegmentCount, segments, distances, labels = subret
            if self._plotSegmentValues:
                botlef = (0, -5)
            else:
                botlef = (0.1, 0.1)
            # noinspection PyTypeChecker
            axSeg.text(*botlef, 'Subsampled: {} of {} segments'.format(len(segments), originalSegmentCount))
            # without subsampling, existing values need not to be overwritten

        pos = DistancesPlotter.manifoldPositions(distances)

        # identify unique labels
        ulab = DistancesPlotter.uniqueLabels(labels, segments)
        if templates is None:
            templates = ulab
        # prepare color space
        cIdx = [int(round(each)) for each in numpy.linspace(2, self.cm.N - 2, len(ulab))]

        # CLUSTERS (large bobbles): iterate unique labels and scatter plot each of these clusters
        for c, (l, t) in enumerate(zip(ulab, templates)):  # type: int, (Any, Template)
            lColor = self.cm(cIdx[c])
            class_member_mask = (labels == l)
            try:
                x = list(compress(pos[:, 0].tolist(), class_member_mask))
                y = list(compress(pos[:, 1].tolist(), class_member_mask))
                # "If you want to specify the same RGB or RGBA value for all points, use a 2-D array with a single row."
                # see https://matplotlib.org/api/_as_gen/matplotlib.pyplot.scatter.html:
                axMDS.scatter(x, y, c=colors.to_rgba_array(lColor), alpha=.6,
                              s = self.labsize,
                              # s=s-(c*s/len(ulab)),  #
                              lw=0, label=str(l))
            except IndexError as e:
                print(pos)
                print(distances)
                print(segments)
                raise e

            if isinstance(t, Template) and self._plotSegmentValues:
                axSeg.plot(t.values, c=lColor, linewidth=4)

        # GROUND TRUTH (small bobbles): include field type labels for TypedSegments input
        if any(isinstance(seg, (TypedSegment, TypedTemplate, RawMessage)) for seg in segments):
            if any(isinstance(seg, (TypedSegment, TypedTemplate)) for seg in segments):
                ftypes = numpy.array([seg.fieldtype if isinstance(seg, (TypedSegment, TypedTemplate))
                                      else unknown for seg in segments])  # PP
            elif any(isinstance(seg, RawMessage) and seg.messageType != 'Raw' for seg in segments):
                ftypes = numpy.array([msg.messageType if isinstance(msg, RawMessage) and msg.messageType != 'Raw'
                                      else unknown for msg in segments])  # PP
            else:
                ftypes = set()
            # identify unique types
            utyp = sorted(set(ftypes))
            # prepare color space
            cIdx = [int(round(each)) for each in numpy.linspace(30, self.fcm.N - 30, len(utyp))]
            # iterate unique types and scatter plot each of these groups
            for n, ft in enumerate(utyp):  # PP
                fColor = self.fcm(cIdx[n])
                type_member_mask = (ftypes == ft)
                x = list(compress(pos[:, 0].tolist(), type_member_mask))
                y = list(compress(pos[:, 1].tolist(), type_member_mask))
                # "If you want to specify the same RGB or RGBA value for all points, use a 2-D array with a single row."
                # see https://matplotlib.org/api/_as_gen/matplotlib.pyplot.scatter.html:
                axMDS.scatter(x, y, c=colors.to_rgba_array(fColor), alpha=1,
                          s=self.typsize,
                          lw=0, label=str(ft))

                if isinstance(segments[0], (TypedSegment, TypedTemplate)) and self._plotSegmentValues:
                    for seg in compress(segments, type_member_mask):
                        axSeg.plot(seg.values, c=fColor, alpha=0.05)
        elif isinstance(segments[0], MessageSegment) and self._plotSegmentValues:
            for c, l in enumerate(ulab):
                lColor = self.cm(cIdx[c])
                class_member_mask = (labels == l)
                for seg in compress(segments, class_member_mask):
                    axSeg.plot(seg.values, c=lColor, alpha=0.1)
        elif self._plotSegmentValues:
                axSeg.text(.5, .5, 'nothing to plot\n(message alignment)', horizontalalignment='center')

        # place the label/type legend in the (otherwise empty) axSeg subfigure
        if isinstance(segments[0], RawMessage) or not self._plotSegmentValues:
            legendHandles, legendLabels = axMDS.get_legend_handles_labels()
            # axMDS.legend(bbox_to_anchor=(1.04,1), scatterpoints=1, shadow=False)
            axSeg.legend(handles=legendHandles, labels=legendLabels, loc='best', scatterpoints=1, shadow=False)
            axSeg.patch.set_alpha(0.0)
            axSeg.axis('off')
        else:
            # place the label/type legend at the best position
            axMDS.legend(scatterpoints=1, loc='best', shadow=False)


        if plotEdges:
            # plotting of edges takes a long time compared to the scatterplot (and especially when rendering the PDF)
            from matplotlib.collections import LineCollection
            # Plot the edges
            lines = [[pos[i, :], pos[j, :]]
                        for i in range(len(pos)) for j in range(len(pos))]
            values = numpy.abs(distances)
            # noinspection PyUnresolvedReferences
            lc = LineCollection(lines,
                                zorder=0, cmap=plt.cm.Blues,
                                norm=plt.Normalize(0, values.max()))
            # lc.set_alpha(.1)
            lc.set_array(distances.flatten())
            lc.set_linewidths(0.5 * numpy.ones(len(segments)))
            axMDS.add_collection(lc)

        if countMarkers:
            # Count markers at identical positions and plot text with information about the markers at this position
            from collections import Counter
            import math
            if isinstance(segments[0], (TypedSegment, TypedTemplate)):
                # TODO for TypedTemplates we rather need to count the number of base segments, so for now this is not accurate
                coordCounter = Counter(
                    [(posX, posY, seg.fieldtype) for seg, lab, posX, posY in zip(
                        segments, labels, pos[:, 0].tolist(), pos[:, 1].tolist())]
                )
            else:
                # TODO for Templates we rather need to count the number of base segments, so for now this is not accurate
                coordCounter = Counter(
                    [(posX, posY, lab) for lab, posX, posY in zip(
                        labels, pos[:, 0].tolist(), pos[:, 1].tolist())]
                )
            for (posX, posY, lab), cnt in coordCounter.items():
                if cnt > 1:
                    theta = hash(str(lab)) % 360
                    r = 1
                    posXr = posX + r * math.cos(theta)
                    posYr = posY + r * math.sin(theta)
                    axMDS.text(posXr, posYr, "{}: {}".format(lab, cnt), withdash=True)

        if self._fig.canvas.toolbar is not None:
            self._fig.canvas.toolbar.update()


    def _plot2dDistances(self, segments: List[MessageSegment], labels: List,
                               templates: List = None):
        axMDS, axSeg = self._axes

        ulab = sorted(set(labels))
        cIdx = [each for each in numpy.linspace(0, self.cm.N - 2, len(ulab))]

        if templates is None:
            templates = ulab

        coords = numpy.array([seg.values for seg in segments])  # type: numpy.ndarray

        # s = 150  # size factor
        for c, (l, t) in enumerate(zip(ulab, templates)):  # type: int, (Any, Template)
            lColor = self.cm(int(round(cIdx[c])))
            class_member_mask = (labels == l)
            try:
                x = list(compress(coords[:, 0].tolist(), class_member_mask))

                if coords.shape[1] > 1:
                    y = list(compress(coords[:, 1].tolist(), class_member_mask))
                    # "If you want to specify the same RGB or RGBA value for all points, use a 2-D array with a single row."
                    # see https://matplotlib.org/api/_as_gen/matplotlib.pyplot.scatter.html:
                    axMDS.scatter(x, y, c=colors.to_rgba_array(lColor), alpha=.6,
                                  # s=s - (c * s / len(ulab)),
                                  lw=0, label=str(l))
                else:
                    axMDS.scatter(x, [0] * len(x), c=colors.to_rgba_array(lColor), alpha=.6,
                                  # s=s - (c * s / len(ulab)),
                                  lw=0, label=str(l))
            except IndexError as e:
                print(segments)
                raise e

            for seg in compress(segments, class_member_mask):
                axSeg.plot(seg.values, c=lColor, alpha=0.05)
            if isinstance(t, Template):
                axSeg.plot(t.values, c=lColor, linewidth=4)

        axMDS.legend(scatterpoints=1, loc='best', shadow=False)

        if self._fig.canvas.toolbar is not None:
            self._fig.canvas.toolbar.update()



    def plotSegmentDistances(self, dc: DistanceCalculator, labels: numpy.ndarray):
        """
        Plot distances between points of high dimensionality using manifold data embedding into a 2-dimensional plot.

        :param dc: A template generator object of segments that all have pairwise similarities assigned
            of which to derive segment groups, similarities, and templates.
        :param labels: list of labels in the order of segments as they are contained in tg.segments.
        """
        assert type(labels) == numpy.ndarray
        assert len(dc.segments) == len(dc.distanceMatrix)  # all have pairwise similarities assigned

        segGroup = dc.segments
        similarities = dc.distanceMatrix
        self.plotManifoldDistances(segGroup, similarities, labels)


    @staticmethod
    def plotSegmentDistanceDistribution(dc: DistanceCalculator):
        """
        Plot distribution of distances to identify density boundaries for clusters.

        TODO test

        :param dc:
        :return:
        """
        from nemere.utils.baseAlgorithms import tril
        statistics = list()
        cltrs = [[dc.segments[idx] for idx, *rest in cluster] for cluster in dc.groupByLength().values()]
        for cluster in cltrs:
            similarities = tril(dc.distancesSubset(cluster))
            statistics.append(tril(similarities))

        plt.rc('xtick', labelsize=6)  # fontsize of the tick labels
        plt.rc('ytick', labelsize=6)  # fontsize of the tick labels
        # rows = math.ceil(5* math.sqrt(len(statistics) / 35))
        # cols = math.ceil(7* math.sqrt(len(statistics) / 35))
        rows = 4
        cols = 5
        fig, axes = plt.subplots(rows, cols, figsize=(8,5))
        for ax, stat in zip(axes.flat, [s for s in statistics if len(s) > 10]):  # type: plt.Axes, numpy.ndarray
            ax.hist(stat, bins=50)
            ax.axvline(x=float(numpy.mean(stat)*3), c='r')
            ax.axvline(x=float(numpy.mean(stat)+numpy.std(stat)), c='g')
            ax.axvline(x=float(numpy.mean(stat)+numpy.std(stat)*2.5), c='tab:orange')
            ax.set_title(str(len(stat)))
        plt.tight_layout()
        plt.show()


class SegmentTopology(object):
    """Create a distance Topology plot for the given Segment cluster data."""

    from nemere.utils.evaluationHelpers import TitleBuilder, StartupFilecheck, unknown
    # leads to import clash:
    # from nemere.utils.reportWriter import SegmentClusterGroundtruthReport

    # show only largest clusters
    clusterCutoff = 15

    def __init__(self, clusterStats: List[Tuple[Hashable, str, float, float, int]],
                 fTypeTemplates: List[FieldTypeTemplate], noise: List[AbstractSegment],
                 dc: DistanceCalculator):
        # look up inferred data types for the segments in the selected subset of clusters and generate labels for them.
        clusterStatsLookup = {stats[0]: (stats[4], stats[2], stats[1])  # label, mostFreqentType, precision, recall, numSegsinCuster
                              for stats in clusterStats if stats is not None}
        sortedClusters = sorted(fTypeTemplates, key=lambda x: -len(x.baseSegments))
        if type(self).clusterCutoff > 0:
            selectedClusters = [ftt for ftt in sortedClusters
                                if clusterStatsLookup[ftt.fieldtype][2] != type(self).unknown][:type(self).clusterCutoff]
        else:
            selectedClusters = sortedClusters
        omittedClusters = [ftt for ftt in sortedClusters if ftt not in selectedClusters]
        clustermask = {segid: "{}: {} seg.s ({:.2f} {})".format(ftt.fieldtype, *clusterStatsLookup[ftt.fieldtype])
            for ftt in selectedClusters for segid in dc.segments2index(ftt.baseSegments)}
        clustermask.update({segid: "Noise" for segid in dc.segments2index(
            noise + [bs for ftt in omittedClusters for bs in ftt.baseSegments]
        )})
        self.labels = numpy.array([clustermask[segid] for segid in range(len(dc.segments))])
        self.dc = dc

    def writeFigure(self, specimens: BaseLoader, inferenceParams: TitleBuilder,
                    elementsReport: "SegmentClusterGroundtruthReport", filechecker: StartupFilecheck):
        print("Plot distances...")
        if type(self).clusterCutoff > 0:
            inferenceParams.postProcess = "largest{}clusters".format(type(self).clusterCutoff)
        atitle = 'segment-distances_' + inferenceParams.plotTitle

        sdp = DistancesPlotter(specimens, atitle, False)
        # hand over selected subset of clusters to plot
        sdp.plotManifoldDistances(
            [elementsReport.typedMatchTemplates[seg][1] if elementsReport.typedMatchTemplates[seg][0] > 0.5
             else seg for seg in self.dc.segments],
            self.dc.distanceMatrix, self.labels)
        # sdp.plotSegmentDistances(dc, labels)
        sdp.writeOrShowFigure(filechecker.reportFullPath)
        del sdp
