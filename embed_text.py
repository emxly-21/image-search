import re, string
import numpy as np
from collections import Counter

punc_regex = re.compile('[{}]'.format(re.escape(string.punctuation)))

def strip_punc(corpus):
    """ Removes all punctuation from a string.

        Parameters
        ----------
        corpus : str

        Returns
        -------
        str
            the corpus with all punctuation removed"""
    # substitute all punctuation marks with ""
    return punc_regex.sub('', corpus)


def to_counter(doc):
    """
    Produce word-count of document, removing all punctuation
    and making all the characters lower-cased.

    Parameters
    ----------
    doc : str

    Returns
    -------
    collections.Counter
        lower-cased word -> count"""
    doc = str.split(strip_punc(doc).lower())
    doc = sorted(doc)
    return Counter(doc)


def to_vocab(counters):
    """
    Takes in an iterable of multiple counters, and returns a sorted list of unique words
    accumulated across all the counters

    [word_counter0, word_counter1, ...] -> sorted list of unique words

    Parameters
    ----------
    counters : Iterable[collections.Counter]
        An iterable containing {word -> count} counters for respective
        documents.

    Returns
    -------
    List[str]
        An alphabetically-sorted list of all of the unique words in `counters`"""
    vocab = []
    for counter in counters:
        for word in counter:
            if word not in vocab:
                vocab.append(word)
    vocab = sorted(vocab)
    return vocab


def to_idf(vocab, counters):
    """
    Given the vocabulary, and the word-counts for each document, computes
    the inverse document frequency (IDF) for each term in the vocabulary.

    Parameters
    ----------
    vocab : Sequence[str]
        Ordered list of words that we care about.

    counters : Iterable[collections.Counter]
        The word -> count mapping for each document.

    Returns
    -------
    Dict[str, float]
        Dictionary mapping each word in vocab to its IDF
    """
    li = []
    dict = {}
    for word in vocab:
        count = 0
        for counter in counters:
            if word in counter:
                count += 1
        li.append(len(counters) / count)
    logged = np.log10(np.array(li))
    for word in range(len(vocab)):
        dict[vocab[word]] = logged[word]
    return dict

def se_text(caption, glove50, idfs):
    '''
    Returns a 50-dimensional vector embedding the text from the caption into the semantic space.

    Parameters
    ----------
    caption : str
        Caption for an image acting as a search query

    glove50 : Dict[str, numpy.ndarray]
        GloVe dataset with encodings for each word

    idfs : Dict[str, float]
        Dictionary with IDFs for each word

    Returns
    -------
    numpy.ndarray
        The 50-dimensional encoding for the caption in the semantic space
    '''
    encoding = np.zeros((50,))
    caption = strip_punc(caption).lower()
    words = caption.split()
    for word in words:
        encoding += idfs[word] * glove50[word]
    encoding /= np.linalg.norm(encoding)
    return encoding
