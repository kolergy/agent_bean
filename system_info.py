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
        cpu_info        = self.get_cpu_info()
        self.cpu_brand  = cpu_info["brand"]
        self.cpu_cores  = cpu_info["cores"]
        self.ram_total  = cpu_info["ram_total"]
        self.ram_used   = cpu_info["ram_used"]

        gpu_info        = self.get_gpu_info()
        self.gpu_brand  = gpu_info[0]["brand"] if gpu_info else None
        self.vram_total_gb = gpu_info[0]["vram_total_gb"] if gpu_info else None
        self.vram_used_gb  = gpu_info[0]["vram_used_gb"] if gpu_info else None

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
        Returns a list of dictionaries, each containing GPU brand, total VRAM in GB, and used VRAM in GB.
        """
        gpus = GPUtil.getGPUs()
        gpu_info = []
        for gpu in gpus:
            info = {
                "brand": gpu.name,
                "vram_total_gb": gpu.memoryTotal / (1024 ** 3),
                "vram_used_gb": gpu.memoryUsed / (1024 ** 3)
            }
            gpu_info.append(info)
        return gpu_info

    def get_cpu_brand(self) -> Any:
        return self.cpu_brand

    def get_cpu_cores(self) -> Any:
        return self.cpu_cores

    def get_ram_total(self) -> Any:
        return self.ram_total

    def get_ram_used(self) -> Any:
        return self.ram_used

    def get_gpu_brand(self) -> Any:
        return self.gpu_brand

    def get_vram_total(self) -> Any:
        return self.vram_total

    def get_vram_used(self) -> Any:
        return self.vram_used
    
    
