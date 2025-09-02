#!/usr/bin/env python3
"""
Full Agent Benchmark Runner - No Subsetting Issues

This script runs the full benchmark with ALL tools (catalog size 50)
to avoid the artificial failures caused by limited tool catalogs.
"""

import os
import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from agentbench.eval.run_matrix import TestMatrixRunner
from agentbench.core.adapters.crewai import CrewAIAdapter
from agentbench.core.runner import MockOrchestratorAdapter

def check_environment():
    """Check if environment is ready for testing."""
    print("🔍 Environment Check")
    print("=" * 30)
    
    # Check virtual environment
    if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print("✅ Virtual environment active")
    else:
        print("❌ Not running in virtual environment")
        return False
    
    # Check OpenAI API key
    api_key = os.getenv('OPENAI_API_KEY')
    if api_key:
        print(f"✅ OpenAI API key found: {api_key[:8]}...")
    else:
        print("❌ OPENAI_API_KEY not set")
        return False
    
    # Check CrewAI installation
    try:
        import crewai
        print(f"✅ CrewAI version: {crewai.__version__}")
    except ImportError:
        print("❌ CrewAI not installed")
        return False
    
    print("✅ Environment check passed")
    return True

def run_full_benchmark():
    """Run the full benchmark with all tools and all test cases."""
    print("\n🚀 Full Agent Benchmark - All Tools, All Tests")
    print("=" * 60)
    
    # Create runner with full configuration
    runner = TestMatrixRunner()
    
    # Register adapters
    print("📋 Registering platform adapters...")
    
    # Mock adapter (for comparison)
    mock_adapter = MockOrchestratorAdapter(
        tools={},
        system_prompt="You are a mock agent for testing.",
        llm_params={"temperature": 0.0, "top_p": 0}
    )
    runner.register_adapter("mock", mock_adapter)
    print("✅ Mock adapter registered")
    
    # CrewAI adapter (if environment is ready)
    try:
        crewai_adapter = CrewAIAdapter()
        crewai_adapter.register_tools({})  # Will be set during execution
        runner.register_adapter("crewai", crewai_adapter)
        print("✅ CrewAI adapter registered")
    except Exception as e:
        print(f"⚠️  CrewAI adapter registration failed: {e}")
        print("   Running with mock adapter only")
    
    # Run full matrix with catalog size 50 only
    print("\n🎯 Running full benchmark matrix...")
    print("   Catalog size: 50 (ALL tools)")
    print("   Task complexities: K=1, K=2, K=3")
    print("   Platforms: All registered")
    
    # Create configuration for full test
    config = {
        "platforms": list(runner.adapters.keys()),
        "catalog_sizes": [50],  # Only full catalog
        "task_complexities": ["K=1", "K=2", "K=3"],
        "max_steps": 20,
        "timeout_seconds": 300
    }
    
    print(f"\n📊 Test Configuration:")
    print(f"   Platforms: {config['platforms']}")
    print(f"   Catalog sizes: {config['catalog_sizes']}")
    print(f"   Task complexities: {config['task_complexities']}")
    print(f"   Max steps per task: {config['max_steps']}")
    print(f"   Timeout per task: {config['timeout_seconds']}s")
    
    # Run the benchmark
    try:
        results = runner.run_full_matrix(config)
        
        print("\n🎉 Benchmark completed successfully!")
        print("📊 Results summary:")
        print(f"   Total configurations: {len(results)}")
        
        # Print summary for each platform
        for platform, platform_results in results.items():
            print(f"\n🔧 {platform.upper()} Results:")
            for catalog_size, catalog_results in platform_results.items():
                print(f"   Catalog size {catalog_size}:")
                for complexity, complexity_results in catalog_results.items():
                    total = len(complexity_results)
                    successful = sum(1 for r in complexity_results if r['success'])
                    success_rate = (successful / total * 100) if total > 0 else 0
                    avg_time = sum(r['wall_time_ms'] for r in complexity_results) / total if total > 0 else 0
                    print(f"     {complexity}: {successful}/{total} ({success_rate:.1f}%) - Avg: {avg_time:.1f}ms")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Benchmark failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main function."""
    print("Agent Orchestration Benchmark Framework")
    print("Full Test Runner - No Subsetting")
    print("=" * 50)
    
    # Check environment
    if not check_environment():
        print("\n❌ Environment check failed. Cannot run benchmark.")
        return False
    
    # Run full benchmark
    success = run_full_benchmark()
    
    if success:
        print("\n🎉 Full benchmark completed successfully!")
        print("📁 Check the results/ directory for detailed logs and CSV data.")
    else:
        print("\n⚠️  Benchmark had issues. Check the errors above.")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
