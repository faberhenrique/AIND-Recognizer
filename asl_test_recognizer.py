from unittest import TestCase

from asl_data import AslDb
from asl_utils import train_all_words
from my_model_selectors import SelectorConstant
from my_recognizer import recognize

FEATURES = ['right-y', 'right-x']

class TestRecognize(TestCase):
    def setUp(self):
        self.asl = AslDb()
        self.training_set = self.asl.build_training(FEATURES)
        self.test_set = self.asl.build_test(FEATURES)
        self.models = train_all_words(self.training_set, SelectorConstant)

 
