import requests
import pandas as pd

def loadDataset(url: str) -> pd.DataFrame : 
    datasetName = getDatasetName(url)
    res = requests.get(url, allow_redirects=True)

    with open(datasetName,'wb') as file:
        file.write(res.content)

    return(pd.read_csv(datasetName, on_bad_lines='skip'))

def getDatasetName(url: str) -> str : 
    splitedArray = url.split('/')
    dataSetName = splitedArray[len(splitedArray) - 1]

    print("csv name: " + dataSetName)

    return dataSetName
