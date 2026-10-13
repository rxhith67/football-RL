"""Record tactical action IDs from the installed GRF Python package."""

import json

from gfootball.env import football_action_set
from tactical_translator_py39 import ACTION


names = ("idle", "left", "top_left", "top", "top_right", "right",
         "bottom_right", "bottom", "bottom_left", "sprint",
         "release_direction", "release_sprint", "builtin_ai")
print(json.dumps({"installed_action_source": football_action_set.__file__,
                  "action_set": "v2",
                  "action_set_size": len(football_action_set.action_set_v2),
                  "ids": {name: ACTION[name] for name in names}}, indent=2))
