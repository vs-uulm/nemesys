


import numpy
import matplotlib.pyplot as plt
from matplotlib import cm

from typing import List, Any
from itertools import compress

from visualization.plotter import MessagePlotter
from utils.loader import SpecimenLoader
from inference.segments import MessageSegment, TypedSegment
from inference.templates import TemplateGenerator, Template



class DistancesPlotter(MessagePlotter):
    """
    Plot distances between points of high dimensionality using manifold data embedding into a 2-dimensional plot.
    """

    def __init__(self, specimens: SpecimenLoader, analysisTitle: str,
                 isInteractive: bool=False):
        super().__init__(specimens, analysisTitle, isInteractive)

        plt.rc('xtick', labelsize=10)  # fontsize of the tick labels
        plt.rc('ytick', labelsize=10)  # fontsize of the tick labels

        self._fig, self._axes = plt.subplots(1,2, figsize=(10,5))  # type: plt.Figure, numpy.ndarray
        if not isinstance(self._axes, numpy.ndarray):
            self._axes = numpy.array(self._axes)
        self._fig.set_size_inches(16, 9)
        # self._cm = cm.Set1  # has 9 colors
        # self._cm = cm.tab20 # 20 colors
        self._cm = cm.jet




    def _plotManifoldDistances(self, segments: List[MessageSegment], similarities: numpy.ndarray, labels: List, templates: List=None):
        from matplotlib.collections import LineCollection
        from sklearn import manifold
        from sklearn.decomposition import PCA

        seed = numpy.random.RandomState(seed=3)
        mds = manifold.MDS(n_components=2, max_iter=3000, eps=1e-9, random_state=seed,
                           dissimilarity="precomputed", n_jobs=1)
        pos = mds.fit(similarities).embedding_
        # print(similarities)

        # Rotate the data
        clf = PCA(n_components=2)

        pos = clf.fit_transform(pos)

        fig = self._fig
        axMDS, axSeg = self._axes


        s = 150
        ulab = sorted(set(labels))
        cIdx = [each for each in numpy.linspace(0, self._cm.N-2, len(ulab))]
        if templates is None:
            templates = ulab
        for c, (l, t) in enumerate(zip(ulab, templates)):  # type: int, (Any, Template)
            lColor = self._cm(int(round(cIdx[c])))
            class_member_mask = (labels == l)
            try:
                x = list(compress(pos[:, 0].tolist(), class_member_mask))
                y = list(compress(pos[:, 1].tolist(), class_member_mask))
                axMDS.scatter(x, y, c=lColor, alpha=.6,
                              s=s-(c*s/len(ulab)), lw=0, label=str(l))
            except IndexError as e:
                print(pos)
                print(similarities)
                print(segments)
                raise e

            for seg in compress(segments, class_member_mask):
                axSeg.plot(seg.values, c=lColor, alpha=0.05)
            if isinstance(t, Template):
                axSeg.plot(t.values, c=lColor, linewidth=4)

        axMDS.legend(scatterpoints=1, loc='best', shadow=False)

        # TODO plotting of edges takes a long time compared to the scatterplot (and especially rendering the PDF is a PITA)
        # # Plot the edges
        # lines = [[pos[i, :], pos[j, :]]
        #             for i in range(len(pos)) for j in range(len(pos))]
        # values = numpy.abs(similarities)
        # lc = LineCollection(lines,
        #                     zorder=0, cmap=plt.cm.Blues,
        #                     norm=plt.Normalize(0, values.max()))
        # # lc.set_alpha(.1)
        # lc.set_array(similarities.flatten())
        # lc.set_linewidths(0.5 * numpy.ones(len(segments)))
        # axMDS.add_collection(lc)

        # for l in ulab:
        #     class_member_mask = (labels == l)
        #     for seg in itertools.compress(segments, class_member_mask):
        #         axSeg.plot(seg.values, c=matplotlib.cm.Set1(l+1))

        # plt.tight_layout()
        fig.canvas.toolbar.update()
        # plt.show()
        # fig.clf()

    def _plot2dDistances(self, segments: List[MessageSegment], labels: List,
                               templates: List = None):
        fig = self._fig
        axMDS, axSeg = self._axes

        ulab = sorted(set(labels))
        cIdx = [each for each in numpy.linspace(0, self._cm.N - 2, len(ulab))]

        if templates is None:
            templates = ulab

        coords = numpy.array([seg.values for seg in segments])  # type: numpy.ndarray

        s = 150
        for c, (l, t) in enumerate(zip(ulab, templates)):  # type: int, (Any, Template)
            lColor = self._cm(int(round(cIdx[c])))
            class_member_mask = (labels == l)
            try:
                x = list(compress(coords[:, 0].tolist(), class_member_mask))
                if coords.shape[1] > 1:
                    y = list(compress(coords[:, 1].tolist(), class_member_mask))
                    axMDS.scatter(x, y, c=lColor, alpha=.6,
                                  s=s - (c * s / len(ulab)), lw=0, label=str(l))
                else:
                    axMDS.scatter(x, [0] * len(x), c=lColor, alpha=.6,
                                  s=s - (c * s / len(ulab)), lw=0, label=str(l))
            except IndexError as e:
                print(segments)
                raise e

            for seg in compress(segments, class_member_mask):
                axSeg.plot(seg.values, c=lColor, alpha=0.05)
            if isinstance(t, Template):
                axSeg.plot(t.values, c=lColor, linewidth=4)

        axMDS.legend(scatterpoints=1, loc='best', shadow=False)

        fig.canvas.toolbar.update()



    def plotDistances(self, tg: TemplateGenerator, labels: numpy.ndarray):
        """
        Plot distances between points of high dimensionality using manifold data embedding into a 2-dimensional plot.

        # :param segments: Segments of equal lengths to calculate pairwise distances from and plot these.
        :param tg: A template generator object of segments of one length to derive similarities.
        :param labels: list of labels in the order of segments as they are contained in tg.segments.
        """
        # segments: List[MessageSegment]
        # tg = TemplateGenerator(segments)  # type: TemplateGenerator
        """A template generator object of which to derive segment groups, similarities, and templates."""

        assert type(labels) == numpy.ndarray

        lenGrps = tg._groupByLength()
        if len(lenGrps) > 1:
            raise ValueError('Only segments of one single length in the TemplateGenerator are accepted as input.')

        # Keep loop for later extension
        for lenGrp in lenGrps.values():
            if len(lenGrp) < 2:
                continue
            # # Keep for extension to generate cluster labels
            # similarities = tg._similaritiesInLengthGroup(lenGrp)
            # labels = tg.getClusterLabels(similarities)

            # Prevent ordering errors (remove if we choose to support multiple plots (lenGrps) at once)
            assert lenGrp == tg.segments

            ulab = set(labels)
            clusters = list()
            for l in ulab:
                class_member_mask = (labels == l)
                clusters.append([ seg for seg in compress(lenGrp, class_member_mask) ])

            if len(tg.segments[0].values) > 2:
                similarities = tg.distanceMatrix
                self._plotManifoldDistances(lenGrp, similarities, labels)
            else:
                self._plot2dDistances(lenGrp, labels)





    @staticmethod
    def plotSegmentDistanceDistribution(tg):
        """
        Plot distribution of distances to identify density boundaries for clusters.

        :param tg:
        :return:
        """
        statistics = list()
        cltrs = tg._groupByLength()
        for cluster in cltrs.values():
            similarities = tg._similaritiesInLengthGroup(cluster)
            mask = numpy.tril(numpy.ones(similarities.shape)) != 0
            statistics.append(
                similarities[mask]
            )

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