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


def tril(arrayIn: numpy.ndarray):
    mask = numpy.tril(numpy.full_like(arrayIn, True, bool))
    return arrayIn[mask]