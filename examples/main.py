from deduplication import find_near_duplicates
from featurizer import custom_featurizer
from issues import find_issues
from pipeline import make_step, run_pipeline
import pandas as pd
from tqdm.auto import tqdm

bar = tqdm(total=100, leave=True)

steps = [
    make_step(find_near_duplicates, name="dedup")(progress=bar),
    make_step(custom_featurizer, name="featurize")(
        label=None,                 # optional; only used to drop NaN label rows
        nan_strategy="impute",
        on_pipeline_error="drop",
        progress=bar
    ),
    make_step(find_issues, name="find_label_issues")(label="HARDSHIP_INDEX", progress=bar)
]

df = pd.read_csv("./data/Lisette.csv")
results = run_pipeline(steps, df=df)

bar.close()
print(results)


