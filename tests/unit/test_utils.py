from matplotlib.font_manager import json_load
import pytest
from cnn_classifier.utils import read_yaml, create_directories, save_json, load_json
from pathlib import Path
from box import ConfigBox
from ensure.main import EnsureError


class Test_utils:
    yaml_files = [
        "tests/unit/data/empty.yaml",
        "tests/unit/data/demo.yaml"
    ]
    
    folder = [
        "tests/unit/data/test_folder1",
        "tests/unit/data/test_folder2"
    ]
    
    json_dict = {
        "a" : 1,
        "b" : 2
    }
    
    def test_read_yaml_empty(self):
        with pytest.raises(ValueError):
            read_yaml(Path(self.yaml_files[0]))
            
    def test_read_yaml_content(self):
        response = read_yaml(Path(self.yaml_files[1]))
        assert isinstance(response, ConfigBox) # assert tells if condition is true or false

    @pytest.mark.parametrize("path_to_yaml", yaml_files)    
    def test_read_yaml_bad_type(self, path_to_yaml):
        with pytest.raises(EnsureError):
            read_yaml(path_to_yaml)

    def test_single_create_directories(self):
        with pytest.raises(EnsureError):
            create_directories(self.folder[0])

    @pytest.mark.parametrize("path_of_dir", folder)  
    def test_multiple_directories(self, path_of_dir):
        with pytest.raises(EnsureError):
            create_directories(path_of_dir)

    def test_save_json(self):
        with pytest.raises(Exception):
            assert save_json(Path("tests/unit/data/test.json"), self.json_dict)
