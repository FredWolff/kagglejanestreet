import polars as pl

from janestreet.analysis import FeatureAnalysis

train = pl.read_parquet("data/train.parquet/partition_id=0")

analysis = FeatureAnalysis(train, target="responder_6", exclude=[
    "date_id", 
    "time_id", 
    "weight",
    "responder_0",
    "responder_1",
    "responder_2",
    "responder_3",
    "responder_4",
    "responder_5",
    "responder_7",
    "responder_8",
])
results = analysis.analyze()

with pl.Config(tbl_rows=-1):
    print(results["correlation"])
    print(results["rolling_correlation"])
