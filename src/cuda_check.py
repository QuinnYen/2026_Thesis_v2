#!/usr/bin/env python3
"""
CUDA Detection and Testing Script
獨立的CUDA檢測程式

This script provides comprehensive CUDA/GPU detection and testing
independent of the main system.
"""

import sys
import torch
import platform
from typing import Dict, List, Any
import subprocess
import os


def print_separator(title: str = ""):
    """Print a separator line with optional title"""
    if title:
        print(f"\n{'='*20} {title} {'='*20}")
    else:
        print("="*60)


def get_system_info() -> Dict[str, Any]:
    """Get basic system information"""
    return {
        'platform': platform.platform(),
        'python_version': sys.version,
        'pytorch_version': torch.__version__,
        'pytorch_compiled_with_cuda': torch.version.cuda,
        'cudnn_version': torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None
    }


def check_cuda_availability() -> Dict[str, Any]:
    """Check CUDA availability and basic info"""
    cuda_info = {
        'cuda_available': torch.cuda.is_available(),
        'cuda_device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
        'current_device': torch.cuda.current_device() if torch.cuda.is_available() else None,
        'cuda_version': torch.version.cuda if torch.version.cuda else None
    }
    
    if cuda_info['cuda_available']:
        devices = []
        for i in range(cuda_info['cuda_device_count']):
            device_props = torch.cuda.get_device_properties(i)
            devices.append({
                'device_id': i,
                'name': device_props.name,
                'total_memory': device_props.total_memory,
                'memory_gb': round(device_props.total_memory / (1024**3), 2),
                'compute_capability': f"{device_props.major}.{device_props.minor}",
                'multiprocessor_count': device_props.multi_processor_count
            })
        cuda_info['devices'] = devices
    
    return cuda_info


def check_cudnn_support() -> Dict[str, Any]:
    """Check cuDNN support"""
    return {
        'cudnn_available': torch.backends.cudnn.is_available(),
        'cudnn_enabled': torch.backends.cudnn.enabled,
        'cudnn_version': torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
        'cudnn_benchmark': torch.backends.cudnn.benchmark,
        'cudnn_deterministic': torch.backends.cudnn.deterministic
    }


def test_gpu_memory() -> Dict[str, Any]:
    """Test GPU memory allocation and operations"""
    if not torch.cuda.is_available():
        return {'status': 'CUDA not available'}
    
    try:
        device = torch.cuda.current_device()
        
        # Get memory info before test
        memory_before = torch.cuda.memory_allocated(device)
        memory_reserved_before = torch.cuda.memory_reserved(device)
        
        # Test tensor creation and operations
        test_size = 1000
        x = torch.randn(test_size, test_size, device='cuda')
        y = torch.randn(test_size, test_size, device='cuda')
        z = torch.matmul(x, y)
        
        # Get memory info after test
        memory_after = torch.cuda.memory_allocated(device)
        memory_reserved_after = torch.cuda.memory_reserved(device)
        
        # Clean up
        del x, y, z
        torch.cuda.empty_cache()
        
        return {
            'status': 'SUCCESS',
            'test_tensor_size': f"{test_size}x{test_size}",
            'memory_used_mb': round((memory_after - memory_before) / (1024**2), 2),
            'memory_reserved_mb': round((memory_reserved_after - memory_reserved_before) / (1024**2), 2),
            'operation': 'Matrix multiplication successful'
        }
        
    except Exception as e:
        return {
            'status': 'FAILED',
            'error': str(e)
        }


def test_cuda_operations() -> Dict[str, Any]:
    """Test various CUDA operations"""
    if not torch.cuda.is_available():
        return {'status': 'CUDA not available'}
    
    results = {}
    
    try:
        # Test 1: Basic tensor operations
        a = torch.tensor([1.0, 2.0, 3.0], device='cuda')
        b = torch.tensor([4.0, 5.0, 6.0], device='cuda')
        c = a + b
        results['basic_ops'] = 'SUCCESS'
        
        # Test 2: Neural network operations
        linear = torch.nn.Linear(10, 5).cuda()
        input_tensor = torch.randn(32, 10, device='cuda')
        output = linear(input_tensor)
        results['nn_ops'] = 'SUCCESS'
        
        # Test 3: Gradient computation
        x = torch.randn(10, requires_grad=True, device='cuda')
        y = x.sum()
        y.backward()
        results['gradient_ops'] = 'SUCCESS' if x.grad is not None else 'FAILED'
        
        # Test 4: Data transfer
        cpu_tensor = torch.randn(100, 100)
        gpu_tensor = cpu_tensor.cuda()
        back_to_cpu = gpu_tensor.cpu()
        results['data_transfer'] = 'SUCCESS'
        
    except Exception as e:
        results['error'] = str(e)
    
    return results


def get_nvidia_smi_info() -> Dict[str, Any]:
    """Get nvidia-smi information if available"""
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            return {
                'nvidia_smi_available': True,
                'output': result.stdout
            }
        else:
            return {
                'nvidia_smi_available': False,
                'error': result.stderr
            }
    except (subprocess.TimeoutExpired, FileNotFoundError, Exception) as e:
        return {
            'nvidia_smi_available': False,
            'error': str(e)
        }


def main():
    """Main function to run all CUDA checks"""
    print("CUDA Detection and Testing Script")
    print("CUDA檢測與測試程式")
    print_separator()
    
    # System Information
    print_separator("系統資訊 / System Information")
    sys_info = get_system_info()
    for key, value in sys_info.items():
        print(f"{key:25}: {value}")
    
    # CUDA Availability
    print_separator("CUDA 可用性檢查 / CUDA Availability Check")
    cuda_info = check_cuda_availability()
    
    print(f"CUDA Available: {cuda_info['cuda_available']}")
    print(f"CUDA Version: {cuda_info['cuda_version']}")
    print(f"GPU Count: {cuda_info['cuda_device_count']}")
    
    if cuda_info['cuda_available'] and 'devices' in cuda_info:
        print("\nGPU Devices:")
        for device in cuda_info['devices']:
            print(f"  Device {device['device_id']}: {device['name']}")
            print(f"    Memory: {device['memory_gb']} GB")
            print(f"    Compute Capability: {device['compute_capability']}")
            print(f"    Multiprocessors: {device['multiprocessor_count']}")
    
    # cuDNN Support
    print_separator("cuDNN 支援檢查 / cuDNN Support Check")
    cudnn_info = check_cudnn_support()
    for key, value in cudnn_info.items():
        print(f"{key:20}: {value}")
    
    # GPU Memory Test
    print_separator("GPU 記憶體測試 / GPU Memory Test")
    memory_test = test_gpu_memory()
    for key, value in memory_test.items():
        print(f"{key:20}: {value}")
    
    # CUDA Operations Test
    print_separator("CUDA 運算測試 / CUDA Operations Test")
    ops_test = test_cuda_operations()
    for key, value in ops_test.items():
        print(f"{key:20}: {value}")
    
    # NVIDIA-SMI
    print_separator("NVIDIA-SMI 資訊 / NVIDIA-SMI Information")
    smi_info = get_nvidia_smi_info()
    
    if smi_info['nvidia_smi_available']:
        print("NVIDIA-SMI Output:")
        print(smi_info['output'])
    else:
        print(f"NVIDIA-SMI not available: {smi_info.get('error', 'Unknown error')}")
    
    # Summary
    print_separator("總結 / Summary")
    
    if cuda_info['cuda_available']:
        print("✅ CUDA is available and functional")
        print(f"✅ Found {cuda_info['cuda_device_count']} GPU(s)")
        
        if cudnn_info['cudnn_available']:
            print("✅ cuDNN is available")
        else:
            print("⚠️  cuDNN is not available")
        
        if memory_test.get('status') == 'SUCCESS':
            print("✅ GPU memory operations successful")
        else:
            print("❌ GPU memory operations failed")
        
        if 'error' not in ops_test:
            print("✅ CUDA operations successful")
        else:
            print("❌ CUDA operations failed")
            
        print("\n🎉 Your system is ready for GPU-accelerated deep learning!")
        
    else:
        print("❌ CUDA is not available")
        print("Please check:")
        print("  - NVIDIA GPU drivers are installed")
        print("  - PyTorch was installed with CUDA support")
        print("  - CUDA toolkit is properly installed")
    
    print_separator()


if __name__ == "__main__":
    main()