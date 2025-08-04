#!/usr/bin/env python3
"""
Test script for TA ULS Training System with Julia Integration
"""

import sys
import time
import numpy as np
import logging
import subprocess
import requests
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_julia_server():
    """Test Julia server startup and basic functionality"""
    logger.info("Testing Julia server...")
    
    try:
        # Start Julia server in background
        julia_process = subprocess.Popen([
            "julia", "-e", 
            """
            include("Al-uLS.jl")
            start_http_server(8001)
            """
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Wait for server to start
        logger.info("Waiting for Julia server to start...")
        time.sleep(5)
        
        # Test connection
        test_matrix = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]
        payload = {
            "function": "optimize_matrix",
            "args": [test_matrix, "sparsity"]
        }
        
        response = requests.post(
            "http://localhost:8001",
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=10
        )
        
        if response.status_code == 200:
            result = response.json()
            if "error" not in result:
                logger.info("✓ Julia server test passed")
                logger.info(f"Optimization result: {result.get('method', 'unknown')}")
                return True, julia_process
            else:
                logger.error(f"✗ Julia server returned error: {result['error']}")
                return False, julia_process
        else:
            logger.error(f"✗ Julia server HTTP error: {response.status_code}")
            return False, julia_process
            
    except Exception as e:
        logger.error(f"✗ Julia server test failed: {e}")
        return False, None

def test_python_imports():
    """Test Python module imports"""
    logger.info("Testing Python imports...")
    
    try:
        import torch
        import torch.nn as nn
        import numpy as np
        import requests
        logger.info("✓ All Python dependencies imported successfully")
        
        # Test torch functionality
        x = torch.randn(2, 3)
        y = torch.nn.Linear(3, 1)(x)
        logger.info(f"✓ PyTorch test passed, tensor shape: {y.shape}")
        return True
        
    except ImportError as e:
        logger.error(f"✗ Import error: {e}")
        return False
    except Exception as e:
        logger.error(f"✗ PyTorch test failed: {e}")
        return False

def test_ta_uls_components():
    """Test TA ULS components without Julia integration"""
    logger.info("Testing TA ULS components...")
    
    try:
        # Import after sys.path modification
        sys.path.append('.')
        import importlib.util
        
        # Load the module dynamically
        spec = importlib.util.spec_from_file_location("al_uls", "al-ULs.Py")
        al_uls = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(al_uls)
        
        KFPLayer = al_uls.KFPLayer
        TAULSControlUnit = al_uls.TAULSControlUnit
        StabilityAwareLoss = al_uls.StabilityAwareLoss
        TAULSTrainingConfig = al_uls.TAULSTrainingConfig
        import torch
        
        # Test KFPLayer
        kfp_layer = KFPLayer(dim=64, stability_weight=0.1)
        test_input = torch.randn(2, 10, 64)
        output, stability = kfp_layer(test_input)
        logger.info(f"✓ KFPLayer test passed, output shape: {output.shape}")
        
        # Test TAULSControlUnit
        control_unit = TAULSControlUnit(input_dim=64, hidden_dim=128, control_dim=64)
        control_output = control_unit(test_input)
        logger.info(f"✓ TAULSControlUnit test passed, control output keys: {list(control_output.keys())}")
        
        # Test StabilityAwareLoss
        loss_fn = StabilityAwareLoss(entropy_weight=0.05, stability_weight=0.1)
        logits = torch.randn(2, 10, 1000)
        targets = torch.randint(0, 1000, (2, 10))
        hidden_states = [torch.randn(2, 10, 64) for _ in range(3)]
        stability_metrics = [{'stability_info': torch.randn(64)} for _ in range(2)]
        
        loss_info = loss_fn(logits, targets, hidden_states, stability_metrics)
        logger.info(f"✓ StabilityAwareLoss test passed, loss: {loss_info['total_loss'].item():.4f}")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ TA ULS components test failed: {e}")
        return False

def test_julia_integration():
    """Test Julia-Python integration"""
    logger.info("Testing Julia integration...")
    
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location("al_uls", "al-ULs.Py")
        al_uls = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(al_uls)
        JuliaClient = al_uls.JuliaClient
        
        # Create client
        client = JuliaClient("http://localhost:8001")
        
        # Test matrix optimization
        test_matrix = np.random.rand(5, 5)
        result = client.optimize_matrix(test_matrix, method="kfp")
        
        if "error" not in result:
            logger.info("✓ Julia matrix optimization test passed")
            logger.info(f"Method: {result.get('method', 'unknown')}")
            logger.info(f"Compression ratio: {result.get('compression_ratio', 0.0):.4f}")
        else:
            logger.error(f"✗ Julia optimization error: {result['error']}")
            return False
        
        # Test polynomial creation
        test_data = np.random.rand(3, 4)
        variables = ["x1", "x2", "x3", "x4"]
        poly_result = client.create_polynomials(test_data, variables)
        
        if "error" not in poly_result:
            logger.info("✓ Julia polynomial creation test passed")
            logger.info(f"Polynomials created: {poly_result.get('total_polynomials', 0)}")
        else:
            logger.error(f"✗ Julia polynomial error: {poly_result['error']}")
            return False
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Julia integration test failed: {e}")
        return False

def test_training_demo():
    """Test training demonstration without full training"""
    logger.info("Testing training demo...")
    
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location("al_uls", "al-ULs.Py")
        al_uls = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(al_uls)
        
        TAULSTrainer = al_uls.TAULSTrainer
        TAULSTrainingConfig = al_uls.TAULSTrainingConfig
        create_dummy_dataset = al_uls.create_dummy_dataset
        
        # Create small configuration for testing
        config = TAULSTrainingConfig(
            vocab_size=100,
            d_model=32,
            batch_size=2,
            use_julia_optimization=False,  # Disable for quick test
            optimization_frequency=10
        )
        
        # Create trainer
        trainer = TAULSTrainer(config)
        
        # Initialize without Julia
        trainer.start_training()
        
        # Create small dataset
        dataset = create_dummy_dataset(config, num_samples=5)
        
        # Test single training step
        batch = dataset[0]
        batch_data = {
            'input_ids': batch['input_ids'][:10].unsqueeze(0),
            'targets': batch['targets'][:10].unsqueeze(0)
        }
        
        result = trainer.train_step(batch_data)
        loss_value = result['loss']['total_loss'].item()
        
        logger.info(f"✓ Training demo test passed, loss: {loss_value:.4f}")
        
        # Test stability evaluation
        stability = trainer.evaluate_stability()
        logger.info(f"✓ Stability evaluation passed: {stability}")
        
        # Cleanup
        trainer.cleanup()
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Training demo test failed: {e}")
        return False

def main():
    """Run all tests"""
    logger.info("=" * 60)
    logger.info("TA ULS Training System Test Suite")
    logger.info("=" * 60)
    
    test_results = []
    julia_process = None
    
    # Test 1: Python imports
    test_results.append(("Python Imports", test_python_imports()))
    
    # Test 2: TA ULS components
    test_results.append(("TA ULS Components", test_ta_uls_components()))
    
    # Test 3: Training demo (without Julia)
    test_results.append(("Training Demo", test_training_demo()))
    
    # Test 4: Julia server (if available)
    try:
        julia_available, julia_process = test_julia_server()
        test_results.append(("Julia Server", julia_available))
        
        if julia_available:
            # Test 5: Julia integration
            test_results.append(("Julia Integration", test_julia_integration()))
    except Exception as e:
        logger.warning(f"Julia tests skipped: {e}")
        test_results.append(("Julia Server", False))
    
    # Print results
    logger.info("=" * 60)
    logger.info("Test Results:")
    logger.info("=" * 60)
    
    passed = 0
    total = len(test_results)
    
    for test_name, result in test_results:
        status = "PASS" if result else "FAIL"
        logger.info(f"{test_name:<20}: {status}")
        if result:
            passed += 1
    
    logger.info("=" * 60)
    logger.info(f"Summary: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All tests passed! System is ready to use.")
    elif passed >= total - 2:  # Allow Julia tests to fail
        logger.info("✅ Core system tests passed. Julia integration may need setup.")
    else:
        logger.error("❌ Some core tests failed. Please check dependencies.")
    
    # Cleanup Julia process
    if julia_process:
        try:
            julia_process.terminate()
            julia_process.wait(timeout=5)
            logger.info("Julia server stopped")
        except:
            julia_process.kill()
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)