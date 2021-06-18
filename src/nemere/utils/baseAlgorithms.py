import numpy
from typing import List, Tuple

def ngrams(sequence, n: int):
    """
    :return: The ngrams of the message in order
    """
    assert isinstance(n, int)
    mlen = len(sequence)
    ngramlist = ( sequence[start:end] for start, end in
               zip( range(mlen - n + 1),
                  range(n, mlen + 1)))
    return ngramlist


def sad(v, u):
    """
    Sum of absolute differences between vectors v and u.
    Both vectors must be of the same length.
    Sinmple and cheap similarity measure from image/movie processing.

    :param v: Vector v.
    :param u: Vector u.
    :return: The SAD.
    """
    if len(v) != len(u):
        raise ValueError("Vectors need to be of equal length.")
    return numpy.sum(numpy.abs(numpy.subtract(v, u)))

def tril(arrayIn: numpy.ndarray) -> numpy.ndarray:
    """
    >>> a = numpy.array([[1,2,3,4],[2,3,4,5],[3,4,5,6],[4,5,6,7]])
    >>> tril(a)
    array([2, 3, 4, 4, 5, 6])

    :param arrayIn: a symmetrical matrix
    :return: Value list of the lower triangle of arrayIn without the identity (diagonal).
    """
    premask = numpy.full_like(arrayIn, True, bool)
    mask = numpy.tril(premask, k=-1)  # mask also including the first diagonal
    return arrayIn[mask]

def trilNaN(distances: numpy.ndarray):
    """
    :param distances: a symmetrical matrix
    :return: lower triangle in original shape. Upper triangle and identity (diagonal) set to nan.
    """
    mask = numpy.tril(numpy.full_like(distances, True, bool), k=-1)
    dist = distances.copy()
    dist[~mask] = numpy.nan
    return dist

def generateTestSegments():
    from netzob.Model.Vocabulary.Messages.RawMessage import RawMessage
    from nemere.inference.analyzers import Value
    from nemere.inference.segments import MessageSegment

    bytedata = [
        bytes([1, 2, 3, 4]),
        bytes([2, 3, 4]),
        bytes([1, 3, 4]),
        bytes([2, 4]),
        bytes([2, 3]),
        bytes([20, 30, 37, 50, 69, 2, 30]),
        bytes([37, 5, 69]),
        bytes([0, 0, 0, 0]),
        bytes([3, 2, 3, 4])
    ]
    messages = [RawMessage(bd) for bd in bytedata]
    analyzers = [Value(message) for message in messages]
    segments = [MessageSegment(analyzer, 0, len(analyzer.message.data)) for analyzer in analyzers]
    return segments


def autoconfigureDBSCAN(neighbors: List[List[Tuple[int, float]]]):
    """
    Auto configure the clustering parameters epsilon and minPts regarding the input data

    :param neighbors: List of (ordered) samples and their neighbors
        (with id in order of outer list and distance) sorted from nearest to furtherst
    :return: epsilon, min_samples, k
    """
    from scipy.ndimage.filters import gaussian_filter1d
    from math import log, ceil

    sigma = log(len(neighbors))
    knearest = dict()
    smoothknearest = dict()
    seconddiff = dict()
    seconddiffMax = (0, 0, 0)
    for k in range(0, ceil(log(len(neighbors)**2))):  # first log(n^2)   alt.: // 10 first 10% of k-neighbors
        knearest[k] = sorted([nfori[k][1] for nfori in neighbors])
        smoothknearest[k] = gaussian_filter1d(knearest[k], sigma)
        # max of second difference (maximum positive curvature) as knee (this not actually the knee!)
        seconddiff[k] = numpy.diff(smoothknearest[k], 2)
        seconddiffargmax = seconddiff[k].argmax()
        if smoothknearest[k][seconddiffargmax] > 0:
            # noinspection PyArgumentList
            diffrelmax = seconddiff[k].max() / smoothknearest[k][seconddiffargmax]
            if 2*sigma < seconddiffargmax < len(neighbors) - 2*sigma and diffrelmax > seconddiffMax[2]:
                seconddiffMax = (k, seconddiffargmax, diffrelmax)

    k = seconddiffMax[0]
    x = seconddiffMax[1] + 1

    # if epsilon is 0, i.e. there is no slope in the change of the neighbor distances whatsoever, most probably all
    #   samples are evenly distributed and we either have only evenly distributed noise, or - much more reasonable -
    #   all the samples are the same. A very low epsilon is appropriate for both situations.
    epsilon = smoothknearest[k][x] if smoothknearest[k][x] > 0 else 0.001
    min_samples = round(sigma)
    return epsilon, min_samples, k


def ecdf(data: List, evenlySpaced: bool = True):
    """
    Empirical Cumulative Distribution Function (ECDF)
    from https://www.codementor.io/kripanshubharga/calculate-ecdf-in-python-gycltzxi3

    :param data: The data for which to calculate the ECDF.
    :param evenlySpaced: typically a CDF evaluates the occurance of samples (x-values) at evenly spaced intervals,
        we sometimes need the exact discrete jump in the frequency, then set this to False
    :return: x-values, y-values of the ECDF
    """
    raw_data = numpy.array(data)

    # create a sorted series of unique data
    cdfx = numpy.sort(list(set(data)))
    if evenlySpaced:
        # x-data for the ECDF: evenly spaced sequence of the uniques
        x_values = numpy.linspace(start=min(cdfx), stop=max(cdfx), num=len(cdfx))
    else:
        x_values = cdfx
    # size of the x_values
    size_data = raw_data.size
    # y-data for the ECDF:
    y_values = []
    for i in x_values:
        # all the values in raw data less than the ith value in x_values
        temp = raw_data[raw_data <= i]
        # fraction of that value with respect to the size of the x_values
        value = temp.size / size_data
        # pushing the value in the y_values
        y_values.append(value)
    # return both x and y values
    return x_values, y_values

