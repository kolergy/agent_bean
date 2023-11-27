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
        self.ERROR_VALUE   = -9999
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
        ci = cpuinfo.get_cpu_info()
        #print(ci)
        if 'brand_raw' in ci:
            br = str(ci['brand_raw'])
        else:
            br = "unknown"
        cpu_info = {
            "brand_raw": str(br),
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
            info = { "brand_raw": str(gpu.name),}
            gpu_info.append(info)
        return gpu_info

    def get_ram_free(self) -> float:
        cpu_info = self.get_cpu_info()
        ram_free = cpu_info["ram_total_gb"] - cpu_info["ram_used_gb"]
        if ram_free != None: 
            return ram_free
        else:
            return self.ERROR_VALUE

    def get_cpu_brand(self) -> str:
        if self.cpu_brand != None: 
            return self.cpu_brand
        else:
            return self.ERROR_VALUE

    def get_cpu_cores(self) -> int:
        if self.cpu_cores != None: 
            return self.cpu_cores
        else:
            return self.ERROR_VALUE

    def get_ram_total(self) -> float:
        if self.ram_total_gb != None: 
            return self.ram_total_gb
        else:
            return self.ERROR_VALUE

    def get_ram_used(self) -> float:
        cpu_info = self.get_cpu_info()
        if cpu_info != None: 
            return cpu_info["ram_used_gb"]
        else:
            return self.ERROR_VALUE

    def get_gpu_brand(self) -> str:
        if self.gpu_brand != None: 
            return self.gpu_brand
        else:
            return self.ERROR_VALUE

    def get_v_ram_total(self) -> float:
        if self.vram_total_gb != None: 
            return self.vram_total_gb
        else:
            return self.ERROR_VALUE

    def get_v_ram_used(self) -> float:
        if torch.cuda.is_available():
            self.vram_used_gb = self.vram_total_gb - torch.cuda.mem_get_info()[0]/(1024 ** 3) if torch.cuda.is_available() else None
        else:
            self.vram_used_gb = self.ERROR_VALUE
        #print(f"UUU vram UUU: {self.vram_used_gb}")
        return self.vram_used_gb

    def get_v_ram_free(self) -> float:
        vram  = torch.cuda.mem_get_info()   if torch.cuda.is_available() else None
        #print(f"XXX vram XXX: {vram}")
        self.vram_free_gb = (vram[0]) / (1024 ** 3) if vram else self.ERROR_VALUE
        #print(f"FFF vram FFF: {self.vram_free_gb}")
        return(self.vram_free_gb)

    def get_GPU_current_device(self) -> int:
        if self.GPU_current_device != None: 
            return self.GPU_current_device
        else:
            return self.ERROR_VALUE
        
    def print_GPU_info(self) -> None:
        vram               = torch.cuda.mem_get_info()                 if torch.cuda.is_available() else None
        self.vram_used_gb  = self.vram_total_gb - vram[0]/(1024 ** 3)  if vram else None
        self.vram_free_gb  = vram[0]/(1024 ** 3)                       if vram else None
        print(f"GPU device: {self.GPU_current_device}, brand: {self.gpu_brand}, VRAM [used: {self.vram_used_gb}, free: {self.vram_free_gb}, total: {self.vram_total_gb} GB")

    

    
