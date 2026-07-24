from pathlib import Path
import yaml


class ConfigLoader:
    """Loads project configuration from YAML."""

    def __init__(self):
        self.config_path = (
            Path(__file__).resolve().parent.parent
            / "configs"
            / "project_config.yaml"
        )

    def load_config(self):
        with open(self.config_path, "r", encoding="utf-8") as file:
            return yaml.safe_load(file)


config = ConfigLoader().load_config()