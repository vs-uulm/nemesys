from os.path import splitext, basename, join, exists
from typing import List

import matplotlib.pyplot as plt

from nemere.utils.evaluationHelpers import reportFolder, uulmColors


class MessagePlotter(object):
    """
    Define basic functions and properties to plot messages.
    """
    from nemere.utils.loader import SpecimenLoader

    STYLE_MAINLINE     = { 'linewidth': .6, 'alpha': .6, 'c': uulmColors['uulm-in'] }
    STYLE_BLUMAINLINE =  { 'linewidth': .6, 'alpha': .6, 'c': uulmColors['uulm-med']}
    STYLE_ALTMAINLINE  = { 'linewidth': .6, 'alpha': 1,  'c': uulmColors['uulm-in'] }
    STYLE_COMPARELINE  = { 'linewidth': .2, 'alpha': .6, 'c': 'black'}
    STYLE_FIELDENDLINE = { 'linewidth': .5, 'linestyle': '--', 'alpha': .6, 'c': uulmColors['uulm'] }
    STYLE_CORRELATION  = dict(linewidth=.4, alpha=.6, c=uulmColors['uulm-mawi'])

    def __init__(self, specimens: SpecimenLoader, analysisTitle: str, isInteractive: bool=False):
        """
        Define basic properties to plot messages.

        :param specimens: The pool of messages, the data to be plotted originated from.
        :param analysisTitle: A freely chosen title to be printed on the plot and used for the filename.
        :param isInteractive: Whether the plot should be interactive or written to file.
        """
        # self._figure = plt.figure()
        plt.rc('xtick', labelsize=4)  # fontsize of the tick labels
        plt.rc('ytick', labelsize=4)  # fontsize of the tick labels
        plt.rc('legend', frameon=False)

        self._specimens = specimens
        self._title = analysisTitle
        self._interactive = isInteractive
        self._autoLegend = True


    @property
    def title(self) -> str:
        return self._title

    # @property
    # def figure(self):
    #     return self._figure

    def writeOrShowFigure(self, plotfolder: str=None):
        """
        :param plotfolder: Folder to place the plot in. If not set, use reportFolder from nemere.utils.evaluationHelpers

        If isInteractive was set to true, show the plot in a window, else write it to a file,
        if none of the same name already exists. Closes all figures afterwards.
        """
        if plotfolder is None:
            plotfolder = reportFolder
        pcapName = splitext(basename(self._specimens.pcapFileName))[0]
        plotfile = join(plotfolder, '{}_{}.pdf'.format(self._title, pcapName))

        if self._autoLegend:
            plt.legend()
        plt.suptitle('{} | {}'.format(pcapName, self._title))
        plt.tight_layout(rect=[0,0,1,1])

        if not self._interactive and not exists(plotfile):
            plt.savefig(plotfile)
            print('plot written to file', plotfile)
        else:
            plt.show()
        plt.close('all')
        # del self._figure


    @staticmethod
    def color_y_axis(ax, color):
        """
        Change color of y axis.

        :param ax: The ax to change the color of.
        :param color: The color to change the ax to.
        """
        for t in ax.get_yticklabels():
            t.set_color(color)
        return None

    @staticmethod
    def fillDiffToCompare(ax: plt.Axes, analysisResult: List[float], compareValue: List[float]):
        """
        Fill the difference between analysisResults and compareValue in the plot.

        Call only after first plot into the figure has been done.

        :param ax: The ax to plot the difference to
        :param analysisResult: The first list of values of an analysis result
        :param compareValue: The second list of values of another analysis result
        """
        ax.fill_between(range(len(analysisResult)), analysisResult, compareValue, color='b', alpha=.4)

    @property
    def ax(self) -> plt.Axes:
        """Convenience property for future change of the class to use something else then pyplot."""
        return plt.gca()

    @property
    def fig(self) -> plt.Figure:
        """Convenience property for future change of the class to use something else then pyplot."""
        return plt.gcf()

