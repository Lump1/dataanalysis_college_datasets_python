FROM python:3.13-slim

WORKDIR /dataanalysis_college_datasets_python

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENTRYPOINT ["python", "main.py"]
