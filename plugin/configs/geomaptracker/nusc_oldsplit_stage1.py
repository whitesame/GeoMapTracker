"""Complete nuScenes oldsplit: N1 stage1 configuration."""
from tools.geomaptracker.config import make_config as _make_config
globals().update(_make_config('stage1'))
del _make_config
