"""Memory information utilities."""

import torch


def _gather_gpu_info(memory_info: dict, gpu_idx: int) -> None:
    """Populate ``memory_info`` with VRAM stats for one GPU.

    Reports three different views of memory because they answer
    different questions:

    - ``gpu{i}_allocated`` (``torch.cuda.memory_allocated``): bytes
      currently held by live PyTorch tensors. Useful for spotting
      tensor-level leaks.
    - ``gpu{i}_reserved`` (``torch.cuda.memory_reserved``): bytes the
      PyTorch caching allocator is holding (live + cached free
      blocks). Always >= allocated, often by a wide margin.
    - ``gpu{i}_actual_used`` (``torch.cuda.mem_get_info``): the GPU
      driver's view of bytes in use across the *whole device* -
      every CUDA context, every process. This matches what
      ``nvidia-smi`` shows and is the only number that captures the
      CUDA context overhead, the cuDNN workspace, and any other
      processes sharing the device. For single-process workloads
      this is the most honest "how much VRAM are we using" answer.

    Dashboard / terminal callers should prefer ``actual_used`` for
    the user-facing VRAM line; the others stay available for
    diagnostics and for backward compatibility with code that hasn't
    been updated yet.
    """
    allocated = torch.cuda.memory_allocated(gpu_idx) / (1024**3)
    reserved = torch.cuda.memory_reserved(gpu_idx) / (1024**3)
    total_bytes = torch.cuda.get_device_properties(gpu_idx).total_memory
    total = total_bytes / (1024**3)

    # ``mem_get_info`` returns (free, total) in bytes for the device,
    # querying the driver directly. Available since PyTorch 1.10; on
    # older builds we fall back to the allocator-only reading.
    try:
        free_bytes, _total_bytes = torch.cuda.mem_get_info(gpu_idx)
        actual_used = (total_bytes - free_bytes) / (1024**3)
    except Exception:
        actual_used = reserved

    memory_info[f"gpu{gpu_idx}_used"] = f"{allocated:.1f}GB"
    memory_info[f"gpu{gpu_idx}_allocated"] = f"{allocated:.1f}GB"
    memory_info[f"gpu{gpu_idx}_reserved"] = f"{reserved:.1f}GB"
    memory_info[f"gpu{gpu_idx}_actual_used"] = f"{actual_used:.1f}GB"
    memory_info[f"gpu{gpu_idx}_total"] = f"{total:.1f}GB"
    memory_info[f"gpu{gpu_idx}_percent"] = f"{(actual_used/total)*100:.1f}%"
    memory_info[f"gpu{gpu_idx}_alloc_percent"] = f"{(allocated/total)*100:.1f}%"


def get_memory_info(target_device=None):
    """Get current RAM and VRAM usage information.

    See :func:`_gather_gpu_info` for the three GPU memory views and
    when to prefer each. Dashboard callers should read
    ``gpu{i}_actual_used`` for the displayed VRAM line.
    """
    memory_info = {}

    try:
        # Get RAM information
        import psutil

        ram = psutil.virtual_memory()
        memory_info["ram_used"] = f"{ram.used / (1024**3):.1f}GB"
        memory_info["ram_total"] = f"{ram.total / (1024**3):.1f}GB"
        memory_info["ram_percent"] = f"{ram.percent:.1f}%"
    except ImportError:
        memory_info["ram_used"] = "N/A"
        memory_info["ram_total"] = "N/A"
        memory_info["ram_percent"] = "N/A"

    try:
        # Get VRAM information if CUDA is available
        if torch.cuda.is_available():
            # If target_device is specified (e.g., "cuda:1"), extract the device index
            if target_device and target_device.startswith("cuda:"):
                target_gpu_idx = int(target_device.split(":")[1])
                if target_gpu_idx < torch.cuda.device_count():
                    _gather_gpu_info(memory_info, target_gpu_idx)
                else:
                    memory_info["gpu_status"] = f"Invalid device {target_device}"
            else:
                # Default behavior: check all GPUs
                for i in range(torch.cuda.device_count()):
                    _gather_gpu_info(memory_info, i)
        else:
            memory_info["gpu_status"] = "No CUDA"
    except Exception:
        memory_info["gpu_status"] = "N/A"

    return memory_info
