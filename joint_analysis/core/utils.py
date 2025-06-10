from pathlib import Path


def get_module_path():
    return Path(__file__).parent


def get_data_path():
    return get_module_path().parent / "data"