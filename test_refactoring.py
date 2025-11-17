"""
Test script to verify the refactored imports work correctly.

This script tests that all the newly created modules can be imported
without errors and that basic functionality works.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")

    try:
        from src.data.storage import DataStorage
        print("✓ src.data.storage")
    except Exception as e:
        print(f"✗ src.data.storage: {e}")
        return False

    try:
        from src.features.data_preparer import build_full_processed_dataset
        print("✓ src.features.data_preparer")
    except Exception as e:
        print(f"✗ src.features.data_preparer: {e}")
        return False

    try:
        from src.models.base import BazaarModel
        print("✓ src.models.base")
    except Exception as e:
        print(f"✗ src.models.base: {e}")
        return False

    try:
        from src.models.tabular.xgb_model import XGBEdgeModel
        print("✓ src.models.tabular.xgb_model")
    except Exception as e:
        print(f"✗ src.models.tabular.xgb_model: {e}")
        return False

    try:
        from src.models.seq.tcn_model import TCNSequenceModel
        print("✓ src.models.seq.tcn_model")
    except Exception as e:
        print(f"✗ src.models.seq.tcn_model: {e}")
        return False

    try:
        from src.ensemble.conductor import Conductor
        print("✓ src.ensemble.conductor")
    except Exception as e:
        print(f"✗ src.ensemble.conductor: {e}")
        return False

    try:
        from src.portfolio.allocator import alloc_greedy
        print("✓ src.portfolio.allocator")
    except Exception as e:
        print(f"✗ src.portfolio.allocator: {e}")
        return False

    try:
        from src.execution.slippage_model import SimpleCostModel
        print("✓ src.execution.slippage_model")
    except Exception as e:
        print(f"✗ src.execution.slippage_model: {e}")
        return False

    try:
        from src.backtest.contracts import BacktestConfig, BacktestResult
        from src.backtest.engine import run_backtest
        print("✓ src.backtest")
    except Exception as e:
        print(f"✗ src.backtest: {e}")
        return False

    try:
        from src.pipeline.live_loop import run_offline_cycle
        print("✓ src.pipeline.live_loop")
    except Exception as e:
        print(f"✗ src.pipeline.live_loop: {e}")
        return False

    return True


def test_basic_functionality():
    """Test basic functionality of key components."""
    print("\nTesting basic functionality...")

    try:
        # Test Conductor
        from src.ensemble.conductor import Conductor
        import numpy as np

        conductor = Conductor(n_experts=3, edge_buffer_bp=1.0)
        expert_edges = np.array([5.0, 3.0, 4.0])
        net_edge = conductor.expected_net_edge_bp(expert_edges, cost_bp=2.0)
        assert isinstance(net_edge, float)
        print(f"✓ Conductor: net_edge={net_edge:.2f}bp")
    except Exception as e:
        print(f"✗ Conductor: {e}")
        return False

    try:
        # Test allocator
        from src.portfolio.allocator import alloc_greedy
        import numpy as np

        items = ['ITEM_A', 'ITEM_B', 'ITEM_C']
        exp_edge = np.array([5.0, 3.0, 7.0])
        risk = np.array([10.0, 5.0, 15.0])

        positions = alloc_greedy(
            items=items,
            exp_edge_bp=exp_edge,
            risk_bp=risk,
            max_gross_exposure=100.0,
            per_item_cap=50.0
        )
        assert isinstance(positions, dict)
        assert len(positions) == 3
        print(f"✓ Allocator: {len(positions)} positions allocated")
    except Exception as e:
        print(f"✗ Allocator: {e}")
        return False

    try:
        # Test cost model
        from src.execution.slippage_model import SimpleCostModel

        cost_model = SimpleCostModel(fee_bp=10.0)
        cost = cost_model.predict_bp(
            spread_bp=5.0,
            rel_size=0.5,
            queue_util=0.3
        )
        assert isinstance(cost, (float, int))
        print(f"✓ Cost Model: estimated_cost={cost:.2f}bp")
    except Exception as e:
        print(f"✗ Cost Model: {e}")
        return False

    return True


def main():
    """Run all tests."""
    print("="*60)
    print("Symphony Architecture - Import & Functionality Test")
    print("="*60)

    # Test imports
    imports_ok = test_imports()

    if not imports_ok:
        print("\n❌ Import tests FAILED")
        return 1

    print("\n✅ All imports successful!")

    # Test basic functionality
    functionality_ok = test_basic_functionality()

    if not functionality_ok:
        print("\n❌ Functionality tests FAILED")
        return 1

    print("\n✅ All functionality tests passed!")
    print("="*60)
    print("🎉 Refactoring validation complete!")
    print("="*60)

    return 0


if __name__ == "__main__":
    exit(main())

