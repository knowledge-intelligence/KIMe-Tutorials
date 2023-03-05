# Because ExtensionWindowHandle is used, we should get rid of it really:
import omni.kit.extensionwindow
import asyncio
import carb
import copy

#from .._viewport_legacy import *
from .._viewport_legacy.cp37_win_amd64 import *
from omni.hydra.engine.stats import HydraEngineStats, get_mem_stats
import warnings

def get_viewport_interface():
    """Returns cached :class:`omni.kit.viewport_legacy.IViewport` interface"""

    if not hasattr(get_viewport_interface, "viewport"):
        get_viewport_interface.viewport = acquire_viewport_interface()
    return get_viewport_interface.viewport


def get_default_viewport_window():
    """Returns default (first) Viewport Window if available"""

    viewport = get_viewport_interface()
    if viewport:
        return viewport.get_viewport_window()
    return None


def menu_update(menu_path, visible):
    # get the correct viewport window as there can be multiple
    vp_iface = omni.kit.viewport_legacy.get_viewport_interface()
    viewports = vp_iface.get_instance_list()
    for viewport in viewports:
        if menu_path.endswith(vp_iface.get_viewport_window_name(viewport)):
            viewport_window = vp_iface.get_viewport_window(viewport)
            viewport_window.show_hide_window(visible)
            omni.kit.ui.get_editor_menu().set_value(menu_path, visible)


async def _query_next_picked_world_position_async(self) -> carb.Double3:
    """Asynchronous version of :func:`IViewportWindow.query_next_picked_world_position`. Return a ``carb.Double3``. If
    no position is sampled, the returned object is None."""
    f = asyncio.Future()

    def cb(pos):
        f.set_result(copy.deepcopy(pos))

    self.query_next_picked_world_position(cb)
    return await f


IViewportWindow.query_next_picked_world_position_async = _query_next_picked_world_position_async


def get_nested_gpu_profiler_result(vw: IViewportWindow, max_indent: int = 1):
    warnings.warn(
        "IViewportWindow.get_nested_gpu_profiler_result is deprecated, use omni.hydra.engine.stats.HydraEngineStats instead",
        DeprecationWarning
    )
    return HydraEngineStats(vw.get_usd_context_name(), vw.get_active_hydra_engine()).get_nested_gpu_profiler_result(max_indent)

def get_gpu_profiler_result(vw: IViewportWindow):
    warnings.warn(
        "IViewportWindow.get_gpu_profiler_result is deprecated, use omni.hydra.engine.stats.HydraEngineStats instead",
        DeprecationWarning
    )
    return HydraEngineStats(vw.get_usd_context_name(), vw.get_active_hydra_engine()).get_gpu_profiler_result()

def save_gpu_profiler_result_to_json(vw: IViewportWindow, file_name: str):
    warnings.warn(
        "IViewportWindow.save_gpu_profiler_result_to_json is deprecated, use omni.hydra.engine.stats.HydraEngineStats instead",
        DeprecationWarning
    )
    return HydraEngineStats(vw.get_usd_context_name(), vw.get_active_hydra_engine()).save_gpu_profiler_result_to_json(file_name)

def reset_gpu_profiler_containers(vw: IViewportWindow):
    warnings.warn(
        "IViewportWindow.reset_gpu_profiler_containers is deprecated, use omni.hydra.engine.stats.HydraEngineStats instead",
        DeprecationWarning
    )
    return HydraEngineStats(vw.get_usd_context_name(), vw.get_active_hydra_engine()).reset_gpu_profiler_containers()

def get_mem_stats_result(vw: IViewportWindow, detailed: bool = False):
    warnings.warn(
        "IViewportWindow.get_mem_stats_result is deprecated, use omni.hydra.engine.stats.get_mem_stats instead",
        DeprecationWarning
    )
    return get_mem_stats(detailed)

IViewportWindow.get_nested_gpu_profiler_result = get_nested_gpu_profiler_result
IViewportWindow.get_gpu_profiler_result = get_gpu_profiler_result
IViewportWindow.save_gpu_profiler_result_to_json = save_gpu_profiler_result_to_json
IViewportWindow.reset_gpu_profiler_containers = reset_gpu_profiler_containers
IViewportWindow.get_mem_stats_result = get_mem_stats_result
