import glob
import os
import unittest

import kaldi_io


class TestArkReactor(unittest.TestCase):
    def test_modified_arks(self):
        original_ark_paths = glob.glob('data/kaldi/*.ark')
        for original_ark_path in original_ark_paths:
            modified_ark_paths = glob.glob('data/kaldi/modified/*' + os.path.basename(original_ark_path))
            original_ark = kaldi_io.read_vec_flt_ark(original_ark_path)
            modified_arks = [kaldi_io.read_vec_flt_ark(ark_path) for ark_path in modified_ark_paths]
            for original_key, original_vector in original_ark:
                for modified_ark in modified_arks:
                    modified_key, modified_vector = next(modified_ark)
                    self.assertEqual(original_key, modified_key)


if __name__ == '__main__':
    unittest.main()
