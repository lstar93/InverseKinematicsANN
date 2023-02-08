""" ANN unit tests """
#!/usr/bin/env python

# pylint: disable=C0103
# pylint: disable=C0116
# pylint: disable=C0115

import os
import unittest
from glob import glob
from tests.ann_unit import ann_test_suite


def clean():
    for h5model in glob('saved_model*.h5'):
        os.remove(h5model)
    for scaler in glob('saved_model*.bin'):
        os.remove(scaler)

def main():
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(ann_test_suite())
    clean()

if __name__ == '__main__':
    main()
