## Better run and download dependencies in VM
```python3
python3 -m venv .venv
source .venv/bin/activate

# install dependencies
python3 -m pip install -r requirements.txt
```

```python3
deactivate
```
## Please download dataset on:
Dataset 1: Amazon Electronic Reviews(1.48 GB)：
https://www.kaggle.com/code/shivamparab/amazon-electronic-reviews/input
Dataset 2: Amazon Electronics Metadata(621.86 MB)：
https://www.kaggle.com/datasets/gvaldata/amazon-electronics-metadata
Dataset 3: Best Buy & Amazon Electronics Reviews(62 MB)
https://www.kaggle.com/datasets/rishidamarla/electronics-from-best-buy-and-amazon-reviews/data

## Please rename your path of dataset in Part2Merge.py if needed before you run it

## If you face issues with environment on Mac, and see error message:
PySparkRuntimeError: [JAVA_GATEWAY_EXITED] Java gateway process exited before sending its port number.
you're encountering is caused by Spark failing to download required dependencies httpcore-4.4.13.jar and javax.annotation-api-1.3.2.jar from Maven.
## Download the below two links:
https://repo1.maven.org/maven2/org/apache/httpcomponents/httpcore/4.4.13/httpcore-4.4.13.jar
https://repo1.maven.org/maven2/javax/annotation/javax.annotation-api/1.3.2/javax.annotation-api-1.3.2.jar
```bash
# Run Command:
mkdir -p ~/.m2/repository/org/apache/httpcomponents/httpcore/4.4.13/
mkdir -p ~/.m2/repository/javax/annotation/javax.annotation-api/1.3.2/
mv ~/Downloads/httpcore-4.4.13.jar ~/.m2 repository/org/apache/httpcomponents/httpcore/4.4.13/
mv ~/Downloads/javax.annotation-api-1.3.2.jar ~/.m2/repository/javax/annotation/javax.annotation-api/1.3.2/
```

## How to run database
```bash
# Run Command:
Python3 Part2Merge.py
```

## How to run Streamlit App
```bash
# Run Command:
streamlit run app.py
```


