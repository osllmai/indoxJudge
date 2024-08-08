# Visualization
A class to create and manage visualizations for model evaluation metrics.

# Initialization
The `Visualization` class is initialized with `data` and a `mode`.

```python
class Visualization:
    def __init__(self, data: Union[List[Dict], Dict], mode: str = 'llm'):
        """
        Initializes the Visualization object with data and mode.

        Args:
            data (Union[List[Dict], Dict]): The data for the models.
            mode (str): The mode of the visualization ('llm', 'safety', or 'rag').
        """
        self.mode = mode
        self.light_template = "plotly_white"
        self.dark_template = "plotly_dark"
        self.current_template = self.light_template
        self.models = data if isinstance(data, list) else [data]
        self.plots = self.determine_plots()
```
Parameters Explanation

- **data**: The data for the models, either as a list of dictionaries or a single dictionary.
- **mode**: The mode of the visualization, which can be `llm`, `safety`, or `rag`.
# Usage Example
Here is an example of how to use the `Visualization` class:
```python
from visualization import Visualization

# Define the model data
model_data = [
    {'name': 'Model A', 'metrics': {'Metric 1': 0.8, 'Metric 2': 0.6, 'Metric 3': 0.7}, 'score': 0.7},
    {'name': 'Model B', 'metrics': {'Metric 1': 0.7, 'Metric 2': 0.5, 'Metric 3': 0.8}, 'score': 0.6}
]

# Initialize the Visualization object
visualizer = Visualization(model_data, mode='llm')

# Run the Dash application
visualizer.plot()
```