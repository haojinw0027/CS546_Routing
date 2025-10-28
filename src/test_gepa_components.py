#!/usr/bin/env python3
"""
Test script to verify all components work before running full GEPA training.
"""

import sys
import json
import requests
from pathlib import Path

# Add gepa to path
GEPA_PATH = Path(__file__).parent.parent / "gepa" / "src"
sys.path.insert(0, str(GEPA_PATH))


def test_gepa_import():
    """Test that GEPA can be imported from local directory"""
    print("Testing GEPA import...")
    try:
        import gepa
        from gepa.core.adapter import GEPAAdapter
        print("✓ GEPA import successful")
        return True
    except Exception as e:
        print(f"✗ GEPA import failed: {e}")
        return False


def test_vllm_connection(host="localhost", port=8088):
    """Test connection to vLLM server"""
    print(f"\nTesting vLLM connection at {host}:{port}...")
    try:
        # Test models endpoint
        models_url = f"http://{host}:{port}/v1/models"
        response = requests.get(models_url, timeout=5)
        response.raise_for_status()
        models = response.json()

        print(f"✓ vLLM server is running")
        if "data" in models:
            print(f"  Available models: {[m['id'] for m in models['data']]}")

        # Test simple completion
        completion_url = f"http://{host}:{port}/v1/completions"
        payload = {
            "model": models["data"][0]["id"] if models.get("data") else "default",
            "prompt": "What is 2+2?",
            "max_tokens": 10,
            "temperature": 0.0,
        }
        response = requests.post(completion_url, json=payload, timeout=10)
        response.raise_for_status()
        result = response.json()

        if "choices" in result and len(result["choices"]) > 0:
            print(f"✓ vLLM completion test successful")
            print(f"  Response: {result['choices'][0]['text'][:50]}...")
            return True
        else:
            print("✗ vLLM returned unexpected response format")
            return False

    except requests.exceptions.ConnectionError:
        print(f"✗ Cannot connect to vLLM server at {host}:{port}")
        print("  Make sure vLLM is running on port 8088")
        return False
    except Exception as e:
        print(f"✗ vLLM test failed: {e}")
        return False


def test_claude_connection(host="http://172.31.13.66:8080"):
    """Test connection to local Claude HTTP service (web.py /reward endpoint)"""
    print(f"\nTesting Claude connection at {host}...")
    try:
        claude_url = f"{host}/reward"

        payload = {
            "prompt": "Say 'test successful' and nothing else.",
            "max_tokens": 50,
            "temperature": 0.0,
        }

        response = requests.post(claude_url, json=payload, timeout=30)
        response.raise_for_status()
        result = response.json()

        # Response format from Bedrock
        if "content" in result and len(result["content"]) > 0:
            content = result["content"][0].get("text", "")
            print(f"✓ Claude connection successful")
            print(f"  Response: {content[:100]}")
            return True
        else:
            print("✗ Claude returned unexpected response format")
            print(f"  Response: {result}")
            return False

    except requests.exceptions.ConnectionError:
        print(f"✗ Cannot connect to Claude service at {host}")
        print("  Make sure the Claude service is running (python src/web.py)")
        return False
    except Exception as e:
        print(f"✗ Claude connection failed: {e}")
        return False


def test_data_loading():
    """Test loading the MATH dataset"""
    print("\nTesting data loading...")
    try:
        train_file = "data/MATH_adaptive_demo/train.json"
        with open(train_file, 'r') as f:
            data = json.load(f)

        print(f"✓ Dataset loaded successfully")
        print(f"  Total problems: {len(data)}")
        print(f"  First problem preview: {data[0]['problem'][:100]}...")
        print(f"  Required fields present: {all(k in data[0] for k in ['problem', 'answer', 'solution'])}")
        return True

    except Exception as e:
        print(f"✗ Data loading failed: {e}")
        return False


def main():
    """Run all tests"""
    print("="*80)
    print("GEPA Component Testing")
    print("="*80)

    results = {
        "GEPA Import": test_gepa_import(),
        "vLLM Connection": test_vllm_connection(),
        "Claude Connection": test_claude_connection(),
        "Data Loading": test_data_loading(),
    }

    print("\n" + "="*80)
    print("Test Results Summary")
    print("="*80)

    for test_name, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{test_name:.<40} {status}")

    all_passed = all(results.values())

    print("="*80)
    if all_passed:
        print("✓ All tests passed! Ready to run gepa_traj.py")
    else:
        print("✗ Some tests failed. Please fix issues before running gepa_traj.py")
        print("\nTroubleshooting:")
        if not results["vLLM Connection"]:
            print("  - Start vLLM server: vllm serve Qwen/Qwen2.5-Math-7B-Instruct --port 8088")
        if not results["Claude Connection"]:
            print("  - Make sure Claude service is running on http://172.31.13.66:8080")
            print("  - Check that the endpoint is accessible from this machine")
    print("="*80)

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
