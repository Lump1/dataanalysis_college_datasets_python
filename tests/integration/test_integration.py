import sys
import unittest
import os
import tempfile
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))  # noqa
import loaddata as ld  # noqa
import graphics as gp  # noqa


class TestIntegration(unittest.TestCase):

    def setUp(self):
        self.test_data = pd.DataFrame({
            'Date': ['2000-01-01', '2005-05-12', '2010-07-19'],
            'Sex': ['Male', 'Female', 'Male'],
            'Value': [100, 200, 300]
        })
        self.temp_dir = tempfile.TemporaryDirectory()
        self.file_path = os.path.join(self.temp_dir.name, "test_data.csv")
        self.test_data.to_csv(self.file_path, index=False)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_task_1_integration(self):
        """
        Download -> analysis -> graphics
        """
        df_loaded = ld.loadDataset(self.file_path)
        self.assertFalse(df_loaded.empty)
        self.assertListEqual(list(df_loaded.columns), ['Date', 'Sex', 'Value'])

        image_path = gp.task_1(self.file_path)

        self.assertTrue(os.path.exists(image_path))
        self.assertTrue(image_path.endswith(".png"))

        self.assertGreater(os.path.getsize(image_path), 0)


if __name__ == '__main__':
    unittest.main()
