# Nvidia Isaac Sim - Tutorials
Isaac Sim - related Tutorials

## Fix .VSCode json
- in launch.json (Change)
```json
"python": "${workspaceFolder}/kit/python/bin/python3",   
"envFile": "${workspaceFolder}/.vscode/.standalone_examples.env",   

==>

"python": "${workspaceFolder}\\kit\\python\\python.exe",   
"envFile": "${workspaceFolder}\\.vscode\\.standalone_examples.env",
```

- in settings.json (Add)
```json
"python.defaultInterpreterPath": "${workspaceFolder}\\kit\\python\\python.exe",
```

- in tasks.json (Add)
```json
"windows": {
	"command": "set CARB_APP_PATH=${workspaceFolder}\\kit && set ISAAC_PATH=${workspaceFolder} && set EXP_PATH=${workspaceFolder}\\apps && ${workspaceFolder}\\setup_python_env.bat && set >${workspaceFolder}\\.vscode\\.standalone_examples.env"
}
```

## Set Default Terminal
1. Open Visual Studio Code
2. Press CTRL + SHIFT + P to open the Command Palette
3. Search for “Terminal: Select Default Profile” (previously “Terminal: Select Default Shell”)
4. Select “Command Prompt”


## Fix viewport_legacy Error
[RuntimeError: Failed to acquire interface: omni::kit::IViewport (pluginName: nullptr)]   
https://forums.developer.nvidia.com/t/runtimeerror-failed-to-acquire-interface-omni-iviewport-pluginname-nullptr/240671/9


- in the folder
```
_viewport_legacy.cp37-win_amd64.pyd ==to==> _viewport_legacy.cp37_win_amd64.pyd 
_viewport_legacy.cp37-win_amd64.lib ==to==> _viewport_legacy.cp37_win_amd64.lib
```

- in the scripts\viewport.py
```python
from .._viewport_legacy import \* ==to==> from .._viewport_legacy.cp37_win_amd64 import \*
```
