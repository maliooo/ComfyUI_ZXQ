import omegaconf
from pathlib import Path

config_path = Path(__file__).parents[2] / "config" / "llm.yaml"
LLM_CONFIG = omegaconf.OmegaConf.load(config_path)