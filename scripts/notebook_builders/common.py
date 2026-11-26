import nbformat as nbf
md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
SETUP = code("""import os, sys
from pathlib import Path
ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
os.chdir(ROOT); sys.path.insert(0, str(ROOT))
import pandas as pd
from IPython.display import Image
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)""")
def write(path, cells):
    nb = nbf.v4.new_notebook(); nb.cells = cells; nbf.write(nb, path)
