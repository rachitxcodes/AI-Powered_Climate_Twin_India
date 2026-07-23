from pathlib import Path
import yaml

class ConfigLoader:
    """Loads project configuration from YAML."""

    def __init__(self):
        self.config_path = (
            Path(__file__).resolve().parent.parent
            / "config"  
            / "config.yaml"
        )

    def LoadConfig(self):
        """Loads the configuration from the YAML file."""
        with open(self.config_path, "r") as file:
            return yaml.safe_load(file)
        
config = ConfigLoader().LoadConfig()

