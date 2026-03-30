# Installation

Requires Python >= 3.10.

```bash
pip install modverif
# or
uv add modverif
```

## Optional Dependencies

**cartopy** is recommended for geographic map projections in station maps and composite plots. Without it, plots fall back to plain matplotlib axes.

```bash
pip install cartopy
# or
uv add cartopy
```

See the [Cartopy Projection Notes](../guide/composite-plots.md#cartopy-projection-notes) in the composite plots guide for important caveats when working with Southern Hemisphere and antimeridian-crossing domains.
