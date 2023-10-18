import torch

class MS_Torch:
    def __init__(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.cpu = torch.device("cpu")
        self.vram_total_mb = torch.cuda.get_device_properties(self.device).total_memory / (1024**2)
        
    def info(self):
        info = dict()
        info['count'] = torch.cuda.device_count()
        
        if self.device.type == 'cuda':
            free, total = torch.cuda.mem_get_info(0)
            info['device'] = (torch.cuda.get_device_name(0))
            info['Total Vram'] = round(torch.cuda.get_device_properties(self.device).total_memory/1024**3,1)
            info['Free Vram'] = round(free/1024**3,1)
            info['Vram Allocated'] = round(torch.cuda.memory_allocated(0)/1024**3,1)
            info['Vram Cached'] = round(torch.cuda.memory_reserved(0)/1024**3,1)

        return info
    
    def torch_gc(self):
        if torch.cuda.is_available():
            with torch.cuda.device(self.device):
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()

cuda_device = MS_Torch()
# print (cuda_device.device, cuda_device.info())
