from rich import print  # pip install rich
# __all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]

from .src import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

print(f"[yellow]已经加载： ZXQ NODES LOADED, 一共有{len(NODE_CLASS_MAPPINGS)}个节点[/yellow]")
print(f"[red]NODE_CLASS_MAPPINGS: {NODE_DISPLAY_NAME_MAPPINGS}[/red]")