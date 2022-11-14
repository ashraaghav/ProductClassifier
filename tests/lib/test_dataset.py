import unittest
from lib.dataset import clean, clean_and_truncate


class TestDataset(unittest.TestCase):

    def test_clean_simple(self):
        """ test text clean """
        s = 'asd asdf'
        out = clean(s)
        self.assertEqual(s, out)

    def test_clean_delims(self):
        """ test text clean """
        s = 'asd+asd ll-ll 4/4'
        out = clean(s)
        self.assertEqual(len(out.split(' ')), 3)

    def test_clean_and_truncate(self):
        """ test clean_and_truncate truncating """
        s = 'asd qwe rt/y cv_b'
        out = clean_and_truncate(s, max_len=3)
        self.assertEqual(len(out.split(' ')), 3)

    def test_clean_and_pad(self):
        """ test clean_and_truncate truncating """
        s = ''
        out = clean_and_truncate(s, max_len=3)
        self.assertEqual(out.split(' ')[0], '[UNK]')
        self.assertEqual(len(out.split(' ')), 3)


if __name__ == '__main__':
    unittest.main()
