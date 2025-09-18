"""Memory information utilities."""

import torch


def get_memory_info(target_device=None):
    """Get current RAM and VRAM usage information."""
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
                    allocated = torch.cuda.memory_allocated(target_gpu_idx) / (1024**3)
                    reserved = torch.cuda.memory_reserved(target_gpu_idx) / (1024**3)
                    total = torch.cuda.get_device_properties(
                        target_gpu_idx
                    ).total_memory / (1024**3)

                    memory_info[f"gpu{target_gpu_idx}_used"] = f"{allocated:.1f}GB"
                    memory_info[f"gpu{target_gpu_idx}_reserved"] = f"{reserved:.1f}GB"
                    memory_info[f"gpu{target_gpu_idx}_total"] = f"{total:.1f}GB"
                    memory_info[f"gpu{target_gpu_idx}_percent"] = (
                        f"{(reserved/total)*100:.1f}%"
                    )
                    # Also add allocated percentage for debugging
                    memory_info[f"gpu{target_gpu_idx}_alloc_percent"] = (
                        f"{(allocated/total)*100:.1f}%"
                    )
                else:
                    memory_info["gpu_status"] = f"Invalid device {target_device}"
            else:
                # Default behavior: check all GPUs
                for i in range(torch.cuda.device_count()):
                    device_name = f"cuda:{i}"
                    allocated = torch.cuda.memory_allocated(i) / (1024**3)
                    reserved = torch.cuda.memory_reserved(i) / (1024**3)
                    total = torch.cuda.get_device_properties(i).total_memory / (1024**3)

                    memory_info[f"gpu{i}_used"] = f"{allocated:.1f}GB"
                    memory_info[f"gpu{i}_reserved"] = f"{reserved:.1f}GB"
                    memory_info[f"gpu{i}_total"] = f"{total:.1f}GB"
                    memory_info[f"gpu{i}_percent"] = f"{(reserved/total)*100:.1f}%"
        else:
            memory_info["gpu_status"] = "No CUDA"
    except Exception:
        memory_info["gpu_status"] = "N/A"

    return memory_info
