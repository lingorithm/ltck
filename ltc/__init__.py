
import sys
from .tc import TC


def ltc(lang, data, **kwargs):
    """[summary]

    Args:
        lang (str): ISO language naming
        processors (list): List for processing processors
        options (dict, optional): options used by the processores. Defaults to {}.

    Returns:
        class: class corosponding to the language
    """
    return TC(lang, data, **kwargs)


sys.modules[__name__] = ltc
