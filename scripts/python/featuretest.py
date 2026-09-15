"""Run model on CV.
"""

from janestreet.setup_env import setup_environment
from janestreet.pipeline import FullPipeline, PipelineCV
from janestreet.models.nn import NN
from janestreet.tracker import WandbTracker
from janestreet.transformers import PolarsTransformer
from janestreet.feature_testing import build_tests1_processor, build_tests2_processor

TRACK = True
COMMENT = ""
CATEGORY = "model_ver27_ts"

MODEL_TYPE = "gru"
NUM = 1
LOAD_MODEL = False
REFIT = True
N_SPLITS = 2
START = 700
TRAIN_SIZE = None

setup_environment(TRACK)

params_nn = {
    "model_type": "gru",

    # ### Model 1
    # "hidden_sizes": [250, 150, 150],
    # "dropout_rates": [0.0, 0.0, 0.0],
    # "hidden_sizes_linear": [],
    # "dropout_rates_linear": [],
    # ###

    ### Model 2
    "hidden_sizes": [500],
    "dropout_rates": [0.3, 0.0, 0.0],
    "hidden_sizes_linear": [500, 300],
    "dropout_rates_linear": [0.2, 0.1],
    ###

    "batch_size": 1,
    "early_stopping_patience": 1,
    "lr_refit": 0.0003,
    "lr": 0.0005,

    "epochs": 1000,
    "early_stopping": True,
    "lr_patience": 1000,
    "lr_factor": 0.5,
}

print(params_nn)

SEED = 5

# the methods are used to select features to add averages, rolling stats etc. in order to make their info more obtainable
tests1 = {
    0: "base",
    1: "correlation",
    2: "rolling_stats",
    3: "information_trans",
    4: "mutual_information",
}

# here all combinations are made to all features, but the methods are used to select features the 125 features with the most information
tests2 = {
    5: "autofeat",
    6: "correlation",
    7: "rolling_stats",
    8: "information_trans",
    9: "mutual_information",
}

for SIM in range(10):
    if SIM < 5:
        prefix = "develop16"
        method = tests1[SIM]
        data_processor, df = build_tests1_processor(
            method,
            name=f"{MODEL_TYPE}_{prefix}.{SIM}_{START}",
            skip_days=START,
        )
    else:
        prefix = "keep125"
        method = tests2[SIM]
        data_processor, df = build_tests2_processor(
            method,
            name=f"{MODEL_TYPE}_{prefix}.{SIM}_{START}",
            skip_days=START,
        )

    features = data_processor.features
    print(f"SIM {SIM} ({method}): {features}")
    print(f"Number of features: {len(features)}")

    MODEL_NAME = f"{MODEL_TYPE}_{prefix}.{SIM}_{700}_featuretesting"

    model = NN(**params_nn, random_seed=SEED)

    pipeline = FullPipeline(
        model,
        preprocessor=PolarsTransformer(features),
        run_name="full",
        name=MODEL_NAME,
        load_model=LOAD_MODEL,
        features=features,
        refit=REFIT,
        change_lr=True,
    )

    wandb_tracker = None
    if TRACK:
        params = dict(params_nn)
        params["n_splits"] = N_SPLITS
        params["seed"] = SEED
        params["start"] = START
        wandb_tracker = WandbTracker(
            MODEL_NAME,
            params,
            category=CATEGORY,
            comment=COMMENT
        )
        wandb_tracker.init_run(features)
        if data_processor.feature_ranking_ is not None:
            wandb_tracker.save_data(data_processor.feature_ranking_.to_pandas(), name="feature_ranking")

    cv = PipelineCV(pipeline, wandb_tracker, n_splits=N_SPLITS, train_size=TRAIN_SIZE)
    scores = cv.fit(df, verbose=True)
    
    if wandb_tracker is not None:
        wandb_tracker.finish()
