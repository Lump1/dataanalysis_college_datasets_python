import unittest
import os
import sys
import tempfile
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))  # noqa
import loaddata as ld  # noqa
import graphics as gp  # noqa


class TestAcceptance(unittest.TestCase):

    def setUp(self):
        self.test_data = pd.DataFrame({
            'Date': ['1990-01-01', '2000-05-12', '2010-07-19', '2020-09-09'],
            'Sex': ['Male', 'Female', 'Male', 'Female'],
            'Value': [50, 150, 250, 300]
        })
        self.temp_dir = tempfile.TemporaryDirectory()
        self.file_path = os.path.join(self.temp_dir.name, "user_data.csv")
        self.test_data.to_csv(self.file_path, index=False)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_user_can_generate_chart_from_csv(self):
        df = ld.loadDataset(self.file_path)
        self.assertFalse(df.empty)

        chart_path = gp.task_2(self.file_path)

        self.assertTrue(os.path.exists(chart_path))
        self.assertTrue(chart_path.endswith(".png"))

        self.assertGreater(os.path.getsize(chart_path), 0)

    def test_user_input_handling(self):
        bad_data = pd.DataFrame({
            'WrongColumn': ['A', 'B', 'C']
        })
        bad_path = os.path.join(self.temp_dir.name, "bad_data.csv")
        bad_data.to_csv(bad_path, index=False)

        with self.assertRaises(Exception):
            _ = gp.task_1(bad_path)


if __name__ == '__main__':
    unittest.main()
