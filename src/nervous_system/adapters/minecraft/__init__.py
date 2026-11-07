"""Minecraft adapter: Mineflayer bridge (bridge/minecraft/) over a localhost socket.

Run `scripts/run_minecraft.py` to start a local flying-squid test server, the bridge and the
nervous system together. Config: configs/games/minecraft.yaml.
"""
from .adapter import SCENARIOS, MinecraftAdapter  # noqa: F401
from .playbook import MinecraftPlaybook  # noqa: F401
