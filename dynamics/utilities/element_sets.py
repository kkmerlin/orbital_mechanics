"""Created on Wed Dec 28 2016 16:55.

@author: Nathan Budd
"""


class ElementSets():
    """Lists current and possible orbital element sets.

    Stores the orbital element set currently in use, provides a list
    of allowed element sets, and provides boolean output functions to test the
    current element set against the possible element sets

    Members
    -------
    current : string
        The current element set.
    allowed_sets : list of strings
        List of allowed element sets.
    """

    allowed_sets = ['rv', 'coe', 'mee']

    def __init__(self, current_set):
        self.current = current_set

    def str(self):
        return self.current

    @classmethod
    def list(self):
        return self.allowed_sets

    def is_rv(self):
        return self.current == 'rv'

    def is_coe(self):
        return self.current == 'coe'

    def is_mee(self):
        return self.current == 'mee'
