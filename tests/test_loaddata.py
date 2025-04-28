import sys
import os
import pytest
import tempfile
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))  # noqa
from loaddata import loadDataset, loadDatasetLocal, getDatasetName  # noqa


@pytest.fixture
def temp_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


def test_getDatasetName_short_url():
    url = "http://example.com/data.csv"
    name = getDatasetName(url)
    assert name == "data.csv"


def test_getDatasetName_long_url():
    url = "http://example.com/" + "a" * 25
    name = getDatasetName(url)
    assert name.endswith(".csv")
    assert len(name) == 16


def test_loadDataset(temp_dir):
    from http.server import HTTPServer, SimpleHTTPRequestHandler
    import threading

    csv_content = "col1,col2\n1,2\n3,4\n"
    file_path = os.path.join(temp_dir, "sample.csv")
    with open(file_path, "w") as f:
        f.write(csv_content)

    os.chdir(temp_dir)
    server = HTTPServer(('localhost', 8000), SimpleHTTPRequestHandler)
    thread = threading.Thread(target=server.serve_forever)
    thread.start()

    try:
        df = loadDataset(
            "http://localhost:8000/sample.csv",
            workspace=temp_dir)
        assert isinstance(df, pd.DataFrame)
        assert df.shape == (2, 2)
    finally:
        server.shutdown()
        thread.join()


def test_loadDatasetLocal(temp_dir):
    csv_content = "col1,col2\n5,6\n7,8\n"
    file_path = os.path.join(temp_dir, "local.csv")
    with open(file_path, "w") as f:
        f.write(csv_content)

    df = loadDatasetLocal("local.csv", workspace=temp_dir)
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (2, 2)
    assert list(df.columns) == ["col1", "col2"]
