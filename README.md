# Setup Instructions

The projection is implemented in Python. All required libraries can be found in `requirements.txt`.

The project is developed in a virtual environment. To replicate the envrionment, run the following commands (make sure `uv` is installed):

1. `uv venv cgv`
2. `source cgv/Scripts/activate`
3. `uv pip install -r requirements.txt`

# Run instructions

Launch application by running `grid_visualization.py`. In `uv`, this can be done (once virtual environment is activated) by `uv run grid_visualization.py`.

# Project structure

* `schwarzchild2D.py` contains a blackhole physics simulator
* `environment_map.py` implements 1D environment map
* `grid.py` implements uniform & adaptive grid
* `cell.py` contains utility class and functions for adaptive grid
* `grid_visualization.py` is the GUI
* `utils.py` contains utility functions for drawing

