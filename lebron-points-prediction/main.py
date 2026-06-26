"""
CLI entry point for the LeBron Points Prediction pipeline.

Usage:
    python main.py fetch-data
    python main.py build-features
    python main.py train
    python main.py evaluate
    python main.py predict-next
    python main.py run-all
"""

import sys
import logging
from pathlib import Path

# Ensure src is on the path
sys.path.insert(0, str(Path(__file__).parent))

from src.config import get_config
from src.utils import setup_logging


def cmd_fetch_data():
    """Fetch raw NBA and odds data."""
    from src.data_pipeline import DataPipeline
    logger = logging.getLogger("lebron.main")
    logger.info("=== FETCH DATA ===")
    pipeline = DataPipeline()
    df = pipeline.run()
    logger.info(f"Data fetched: {len(df)} games, {df.shape[1]} columns")
    print(f"[OK] Data fetched: {len(df)} games")


def cmd_build_features():
    """Build feature set from processed data."""
    from src.data_pipeline import DataPipeline
    from src.feature_engineering import FeatureEngineer
    from src.prospect_theory import ProspectTheoryFeatures
    from src.pressure_index import PressureIndexFeatures
    logger = logging.getLogger("lebron.main")
    logger.info("=== BUILD FEATURES ===")

    pipeline = DataPipeline()
    df = pipeline.load_master()
    if df is None:
        logger.info("No master dataset found — running data fetch first...")
        df = pipeline.run()

    df = FeatureEngineer().run(df)
    df = ProspectTheoryFeatures().run(df)
    df = PressureIndexFeatures().run(df)

    cfg = get_config()
    out = cfg.processed_path / "features.parquet"
    df.to_parquet(out, index=False)
    logger.info(f"Features saved: {out} ({df.shape[1]} columns)")
    print(f"[OK] Features built: {df.shape[1]} columns, {len(df)} rows")


def cmd_train():
    """Train all models and select the best."""
    from src.modeling import ModelTrainer
    logger = logging.getLogger("lebron.main")
    logger.info("=== TRAIN ===")

    cfg = get_config()
    features_path = cfg.processed_path / "features.parquet"
    if not features_path.exists():
        logger.info("Features not found — running build-features first...")
        cmd_build_features()

    import pandas as pd
    df = pd.read_parquet(features_path)
    trainer = ModelTrainer()
    results = trainer.train(df)
    print(f"[OK] Best model: {results['best_model_name']}")
    print(f"     CV MAE summary:")
    for model_name, mae in sorted(results["cv_summary"].items(), key=lambda x: x[1]):
        print(f"       {model_name:<20} MAE={mae:.2f}")


def cmd_evaluate():
    """Evaluate model performance and generate charts."""
    from src.modeling import ModelTrainer
    from src.evaluate import Evaluator
    logger = logging.getLogger("lebron.main")
    logger.info("=== EVALUATE ===")

    cfg = get_config()
    features_path = cfg.processed_path / "features.parquet"
    if not features_path.exists():
        print("[ERROR] Run 'build-features' and 'train' first.")
        sys.exit(1)

    import pandas as pd
    df = pd.read_parquet(features_path)
    trainer = ModelTrainer()
    trainer.best_model, trainer.feature_cols = trainer.load_best_model()
    # Set best_model_name from meta file
    import json
    meta_path = cfg.model_artifacts_path / "model_meta.json"
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
        trainer.best_model_name = meta.get("model_name", "unknown")

    evaluator = Evaluator()
    metrics = evaluator.run(df, trainer)
    print(f"[OK] Evaluation complete")
    print(f"     MAE:  {metrics.get('MAE', 'N/A'):.2f}")
    print(f"     RMSE: {metrics.get('RMSE', 'N/A'):.2f}")
    print(f"     R2:   {metrics.get('R2', 'N/A'):.3f}")
    print(f"     Charts: {cfg.charts_path}")
    print(f"     Report: {cfg.reports_path / 'model_metrics.csv'}")


def cmd_predict_next():
    """Predict LeBron's points in the next game."""
    logger = logging.getLogger("lebron.main")
    logger.info("=== PREDICT NEXT GAME ===")
    from src.predict_next_game import predict_next_game
    predict_next_game()


def cmd_run_all():
    """Run the full pipeline end-to-end."""
    logger = logging.getLogger("lebron.main")
    logger.info("=== RUN ALL ===")
    cmd_fetch_data()
    cmd_build_features()
    cmd_train()
    cmd_evaluate()
    cmd_predict_next()


COMMANDS = {
    "fetch-data": cmd_fetch_data,
    "build-features": cmd_build_features,
    "train": cmd_train,
    "evaluate": cmd_evaluate,
    "predict-next": cmd_predict_next,
    "run-all": cmd_run_all,
}


def main():
    cfg = get_config()
    setup_logging(level=cfg.log_level, log_file=cfg.log_file)

    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
        print(__doc__)
        print("Available commands:")
        for cmd in COMMANDS:
            print(f"  python main.py {cmd}")
        sys.exit(1)

    command = sys.argv[1]
    COMMANDS[command]()


if __name__ == "__main__":
    main()
