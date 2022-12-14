from pathlib import Path
import os
import logging

logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: %(message)s: ')

package_name = "cnn_classifier" # Write name of package

list_of_files = [
    ".github/workflows/.gitkeep",
   f"src/{package_name}/__init__.py", 
   f"src/{package_name}/components/__init__.py", 
   f"src/{package_name}/utils/__init__.py", 
   f"src/{package_name}/config/__init__.py", 
   f"src/{package_name}/pipeline/__init__.py", 
   f"src/{package_name}/entity/__init__.py", 
   f"src/{package_name}/constants/__init__.py",
   "tests/__init__.py",
   "tests/unit/__init__.py",
   "tests/integration/__init__.py",
   "configs/config.yaml",
   "dvc.yaml",
   "params.yaml",
   "init_setup.sh",
   "requirements.txt", 
   "requirements_dev.txt",
   "setup.py",
   "setup.cfg",
   "pyproject.toml",
   "tox.ini",
   "research/trials.ipynb",
]

for filepath in list_of_files:
    filepath = Path(filepath)
    filedir, filename = os.path.split(filepath) # Spliting file directory and file name
    if filedir != "":
        os.makedirs(filedir, exist_ok=True)
        logging.info(f"Creating directory: {filedir} for file: {filename}")
    
    # Checking file exist in directory or file is empty then only create the file
    if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
        with open(filepath, "w") as f:
            pass # create an empty file
            logging.info(f"Creating empty file: {filepath}")

    else:
        # Skipping file already exist 
        logging.info(f"{filename} already exists")