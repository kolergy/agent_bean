import unittest
from system_info import SystemInfo

class TestSystemInfo(unittest.TestCase):
    def setUp(self):
        self.system_info = SystemInfo()

    def test_cpu_info(self):
        cpu_info = self.system_info.get_cpu_info()
        print(f"CPU INFO: {cpu_info}")
        self.assertIsNotNone(cpu_info)
        self.assertIn('brand', cpu_info)
        self.assertIn('cores', cpu_info)
        self.assertIn('ram_total', cpu_info)
        self.assertIn('ram_used', cpu_info)

    def test_gpu_info(self):
        gpu_info = self.system_info.get_gpu_info()
        print(f"GPU INFO: {gpu_info}")
        self.assertIsNotNone(gpu_info)
        if gpu_info:
            self.assertIn('brand', gpu_info[0])
            self.assertIn('vram_total', gpu_info[0])
            self.assertIn('vram_used', gpu_info[0])

if __name__ == '__main__':
    unittest.main()
