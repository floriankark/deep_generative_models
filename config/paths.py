from pathlib import Path

project_root = Path(__file__).parent.parent.absolute()
STORAGE = project_root / "storage"
IMAGES = project_root / "images"
GLOBAL_STATS = project_root / "data" / "global_stats.csv"
CELL_DATA = project_root / "data" / "cell_data.h5"
CNN_MODEL_CONFIG = project_root / "deep_generative_models" / "config.toml"
