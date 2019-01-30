import numpy

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
    :return: lower triangle values of arrayIn removing the identity (diagonal).
    """
    premask = numpy.full_like(arrayIn, True, bool)
    mask = numpy.tril(premask, k=-1)  # mask including the first diagonal
    return arrayIn[mask]


def generateTestSegments():
    from netzob.Model.Vocabulary.Messages.RawMessage import RawMessage
    from inference.analyzers import Value
    from inference.segments import MessageSegment

    bytedata = [
        bytes([1, 2, 3, 4]),
        bytes([2, 3, 4]),
        bytes([1, 3, 4]),
        bytes([2, 4]),
        bytes([2, 3]),
        bytes([20, 30, 37, 50, 69, 2, 30]),
        bytes([37, 5, 69]),
        bytes([70, 2, 3, 4]),
        bytes([3, 2, 3, 4])
    ]
    messages = [RawMessage(bd) for bd in bytedata]
    analyzers = [Value(message) for message in messages]
    segments = [MessageSegment(analyzer, 0, len(analyzer.message.data)) for analyzer in analyzers]
    return segments