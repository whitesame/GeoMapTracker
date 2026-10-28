"""Complete nuScenes oldsplit: N1 inference configuration."""
from tools.geomaptracker.config import make_config as _make_config
globals().update(_make_config('inference'))
del _make_config
