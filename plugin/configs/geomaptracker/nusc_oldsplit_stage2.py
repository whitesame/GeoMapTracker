"""Complete nuScenes oldsplit: N1 stage2 configuration."""
from tools.geomaptracker.config import make_config as _make_config
globals().update(_make_config('stage2'))
del _make_config
