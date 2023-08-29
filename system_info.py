import psutil
import GPUtil
from typing import List, Dict, Any

class SystemInfo:
    """
    SystemInfo class collects system information related to CPU and GPU.
    """
    def __init__(self):
        """
        Constructor initializes CPU and GPU information.
        """
        self.cpu_info: Dict[str, Any] = self.get_cpu_info()
        self.gpu_info: List[Dict[str, Any]] = self.get_gpu_info()

    def get_cpu_info(self) -> Dict[str, Any]:
        """
        Method to get CPU information.
        Returns a dictionary with CPU brand, number of cores, total RAM, and used RAM.
        """
        cpu_info = {
            "brand": psutil.cpu_freq().current,
            "cores": psutil.cpu_count(),
            "ram_total": psutil.virtual_memory().total,
            "ram_used": psutil.virtual_memory().used
        }
        return cpu_info

    def get_gpu_info(self) -> List[Dict[str, Any]]:
        """
        Method to get GPU information.
        Returns a list of dictionaries, each containing GPU brand, total VRAM, and used VRAM.
        """
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
