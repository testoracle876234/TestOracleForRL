from configparser import ConfigParser
import os

def parserConfig():
    cfg = ConfigParser()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, 'config.ini')

    cfg.read(config_path)
    config = {}
    config['root_dir'] = cfg.get('param', 'root_dir')

    return config

if __name__ == '__main__':
    parserConfig()