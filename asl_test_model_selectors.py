from unittest import TestCase

from asl_data import AslDb
from my_model_selectors import (
    SelectorConstant, SelectorBIC, SelectorDIC, SelectorCV,
)

FEATURES = ['right-y', 'right-x']


class TestSelectors(TestCase):
    def setUp(self):
        asl = AslDb()
        self.training = asl.build_training(FEATURES)
        self.sequences = self.training.get_all_sequences()
        self.xlengths = self.training.get_all_Xlengths()

  

    def test_select_dic_interface(self):
        model = SelectorDIC(self.sequences, self.xlengths, 'MARY').select()
        self.assertGreaterEqual(model.n_components, 2)
        model = SelectorDIC(self.sequences, self.xlengths, 'TOY').select()
        self.assertGreaterEqual(model.n_components, 2)
