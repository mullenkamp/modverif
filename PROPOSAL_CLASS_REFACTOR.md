# Proposal: Refactoring `evaluate.py` to a Class-Based API

To reduce code duplication (currently ~70% shared between cell and domain functions) and improve maintainability, I propose introducing a `WRFEvaluator` class.

## Current Pain Points
1.  **Redundant Setup:** Every function independently performs file discovery, date matching, lat/lon slicing, and metadata extraction.
2.  **Brittle Parameter Passing:** Users must pass 6+ identical configuration parameters to every function call.
3.  **IO Boilerplate:** Setting up the `NetCDF4Writer` and defining dimensions is repeated in every orchestrator.

---

## Proposed Class Interface

The `WRFEvaluator` class will handle the "Context" (where the data is, what region we care about), while specific methods handle the "Action" (calculating metrics).

```python
class WRFEvaluator:
    def __init__(
        self,
        source_folder: Union[str, Path],
        test_folder: Union[str, Path],
        domain: int,
        region: Union[tuple, np.ndarray, None] = None,
        start_date: Union[str, date, None] = None,
        end_date: Union[str, date, None] = None,
    ):
        """
        Initializes the evaluation context.
        Performs file discovery and spatial/temporal alignment once.
        """
        self.source_folder = Path(source_folder)
        self.test_folder = Path(test_folder)
        self.domain = domain
        self.region = region
        
        # 1. Discover and match files
        self.file_map = self._match_files(start_date, end_date)
        
        # 2. Extract shared metadata (Proj4, DX, DY, Dimensions) from first file
        self.metadata = self._load_metadata()
        
        # 3. Resolve region into slices and masks
        self.spatial_context = self._resolve_spatial_context()

    def evaluate_cell(self, output_path, variables, metrics, threshold=None):
        """High-resolution spatial evaluation."""
        return self._run_engine(output_path, variables, metrics, threshold, agg_type='cell')

    def evaluate_domain(self, output_path, variables, metrics, threshold=None):
        """Aggregate domain evaluation."""
        return self._run_engine(output_path, variables, metrics, threshold, agg_type='domain')

    def _run_engine(self, output_path, variables, metrics, threshold, agg_type):
        """The core processing loop using rechunkit, unified for both types."""
        # Setup NetCDF output once using self.metadata
        # Iterate through common timesteps...
        # Dispatch to model_eval.metrics functions...
```

---

## Benefits

### 1. Unified Processing Engine
We can unify `evaluate_models_cell` and `evaluate_models_domain` into a single internal `_run_engine`. The only difference between them is the **shape of the output dataset** and **which metric function is called**. 

### 2. Reduced Complexity for Users
Users can define their "Experiment" once and run multiple types of analysis without re-specifying paths.

**Example Usage:**
```python
# 1. Setup the experiment
evaluator = WRFEvaluator(
    "run_baseline/", 
    "run_test/", 
    domain=4, 
    region=SOUTHLAND_BOUNDS
)

# 2. Perform various analyses using the same context
evaluator.evaluate_cell("spatial_error.nc", variables=["T2"], metrics=["ne", "bias"])
evaluator.evaluate_domain("rainfall_skill.nc", variables=["RAINNC"], metrics=["pod", "gss"], threshold=1.0)
```

### 3. Easier to Extend
Adding a new model type (e.g., `GRIB2Evaluator`) becomes a matter of subclassing and overriding the `_match_files` and `_load_metadata` methods, while keeping the core math engine identical.

---

## Implementation TODO
1.  **Create `model_eval/evaluator.py`**: Define the class and migration logic.
2.  **Consolidate IO**: Move NetCDF initialization logic into a shared helper or the `NetCDF4Writer` itself.
3.  **Deprecation Path**: Keep the existing standalone functions in `evaluate.py` as thin wrappers around the new class to avoid breaking existing user scripts.
