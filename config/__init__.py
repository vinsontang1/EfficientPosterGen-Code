import yaml
from pathlib import Path

_config_dir = Path(__file__).parent

_config_file = _config_dir / 'config.yaml'

if _config_file.exists():
    with open(_config_file, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
else:
    config = {}  
    print(f"Warning: Config file not found at {_config_file}")