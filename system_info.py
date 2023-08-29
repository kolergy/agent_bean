import psutil
import GPUtil

class SystemInfo:
    def __init__(self):
        self.cpu_info = self.get_cpu_info()
        self.gpu_info = self.get_gpu_info()

    def get_cpu_info(self):
        cpu_info = {
            "brand": psutil.cpu_freq().current,
            "cores": psutil.cpu_count(),
            "ram_total": psutil.virtual_memory().total,
            "ram_used": psutil.virtual_memory().used
        }
        return cpu_info

    def get_gpu_info(self):
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
