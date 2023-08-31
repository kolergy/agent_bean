import psutil
import GPUtil
import cpuinfo
import torch

from   typing    import List, Dict, Any

class SystemInfo:
    """
    SystemInfo class collects system information related to CPU and GPU.
    """
    def __init__(self):
        """
        Constructor initializes CPU and GPU information.
        """
        cpu_info           = self.get_cpu_info()
        self.cpu_brand_raw = cpu_info["brand_raw"]
        self.cpu_brand     = cpu_info["brand_raw"].split(" ")[0].strip()
        self.cpu_cores     = cpu_info["cores"]
        self.ram_total_gb  = cpu_info["ram_total_gb"]
        self.ram_used_gb   = cpu_info["ram_used_gb"]

        gpu_info                = self.get_gpu_info()
        self.gpu_brand_raw      = gpu_info[0]["brand_raw"]                       if gpu_info else None
        self.gpu_brand          = gpu_info[0]["brand_raw"].split(" ")[0].strip() if gpu_info else None

        vram                    = torch.cuda.mem_get_info()                 if torch.cuda.is_available() else None
        self.vram_total_gb      = vram[1]/(1024 ** 3)                       if vram else None
        self.vram_used_gb       = self.vram_total_gb - vram[0]/(1024 ** 3)                       if vram else None
        self.vram_free_gb       = vram[0]/(1024 ** 3)                       if vram else None
        self.GPU_current_device = torch.cuda.current_device()               if torch.cuda.is_available() else None


    def get_cpu_info(self) -> Dict[str, Any]:
        """
        Method to get CPU information.
        Returns a dictionary with CPU brand, number of cores, total RAM in GB, and used RAM in GB.
        """
        cpu_info = {
            "brand_raw": str(cpuinfo.get_cpu_info()['brand_raw']),
            "cores": int(psutil.cpu_count()),
            "ram_total_gb": float(psutil.virtual_memory().total / (1024 ** 3)), # convert bytes to GB
            "ram_used_gb": float(psutil.virtual_memory().used / (1024 ** 3))
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
                "brand_raw": str(gpu.name),

            }
            gpu_info.append(info)
        return gpu_info


    def get_cpu_brand(self) -> str:
        return self.cpu_brand

    def get_cpu_cores(self) -> int:
        return self.cpu_cores

    def get_ram_total(self) -> float:
        return self.ram_total_gb

    def get_ram_used(self) -> float:
        return self.ram_used_gb

    def get_gpu_brand(self) -> str:
        return self.gpu_brand

    def get_vram_total(self) -> float:
        return self.vram_total_gb

    def get_vram_used(self) -> float:
        self.vram_used_gb = torch.cuda.mem_get_info()[1]/(1024 ** 3) if torch.cuda.is_available() else None
        return self.vram_used_gb

    def get_vram_free(self) -> float:
        vram  = torch.cuda.mem_get_info()   if torch.cuda.is_available() else None
        print(f"XXX vram XXX: {vram}")
        self.vram_free_gb = (vram[0]) / (1024 ** 3) if vram else None
        return(self.vram_free_gb)

    def get_GPU_current_device(self) -> int:
        return self.GPU_current_device
    
    def print_GPU_info(self) -> None:
        vram  = torch.cuda.mem_get_info()   if torch.cuda.is_available() else None
        self.vram_used_gb       = self.vram_total_gb - vram[0]/(1024 ** 3)                       if vram else None
        self.vram_free_gb       = vram[0]/(1024 ** 3)                       if vram else None
        print(f"GPU device: {self.GPU_current_device}, brand: {self.gpu_brand}, VRAM [used: {self.vram_used_gb}, free: {self.vram_free_gb}, total: {self.vram_total_gb} GB")

    

    
