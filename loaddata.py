import requests
import pandas as pd
import random
import string
import os


def loadDataset(
        url: str,
        datasetName: str = None,
        workspace: str = "./data",
        columnsToRender=None,
        colNames=None,
        skip=0) -> pd.DataFrame:
    print(datasetName)

    if datasetName is None or datasetName == "None":
        datasetName = getDatasetName(url)

    if os.path.isfile(url):
        fullpath = url
    else:
        res = requests.get(url, allow_redirects=True)
        fullpath = os.path.join(workspace, datasetName)

        with open(fullpath, 'wb') as file:
            file.write(res.content)

    return pd.read_csv(
        fullpath,
        on_bad_lines='skip',
        usecols=columnsToRender,
        names=colNames,
        skiprows=skip,
        engine="python")


def loadDatasetLocal(datasetName: str, workspace: str = "./data"):
    return (
        pd.read_csv(
            os.path.join(
                workspace,
                datasetName),
            on_bad_lines='skip'))


def getDatasetName(url: str) -> str:
    splitedArray = url.split('/')
    print(len(splitedArray[len(splitedArray) - 1]))

    if len(splitedArray[len(splitedArray) - 1]) > 20:
        dataSetName = ''.join([random.choice(string.ascii_letters)
                              for n in range(12)]) + ".csv"
    else:
        dataSetName = splitedArray[len(splitedArray) - 1]

    print("csv name: " + dataSetName)

    return dataSetName
