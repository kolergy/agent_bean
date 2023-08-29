import psutil
import GPUtil
from typing import List, Dict, Any

# SystemInfo class collects system information related to CPU and GPU
class SystemInfo:
    # Constructor initializes CPU and GPU information
    def __init__(self):
        self.cpu_info: Dict[str, Any] = self.get_cpu_info()
        self.gpu_info: List[Dict[str, Any]] = self.get_gpu_info()

    # Method to get CPU information
    def get_cpu_info(self) -> Dict[str, Any]:
        cpu_info = {
            "brand": psutil.cpu_freq().current,
            "cores": psutil.cpu_count(),
            "ram_total": psutil.virtual_memory().total,
            "ram_used": psutil.virtual_memory().used
        }
        return cpu_info

    # Method to get GPU information
    def get_gpu_info(self) -> List[Dict[str, Any]]:
        gpus = GPUtil.getGPUs()
        gpu_info = []
        for gpu in gpus:
            info = {
                "brand": gpu.name,
                "vram_total": gpu.memoryTotal,
                "vram_used": gpu.memoryUsed
            }
            gpu_info.append(info)
        return gpu_info
