"""
Example script demonstrating the new Symphony API.

This shows how to use the refactored modules for a complete workflow.
Run this to see the new architecture in action (requires processed data).
"""

import sys
from pathlib import Path

# Add project root
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

def example_1_data_access():
    """Example 1: Data Access Layer"""
    print("\n" + "="*60)
    print("Example 1: Data Access Layer")
    print("="*60)

    from src.data.storage import DataStorage

    try:
        storage = DataStorage()
        df = storage.load_full_table()
        print(f"✓ Loaded {len(df):,} rows with {len(df.columns)} columns")
        print(f"  Columns: {', '.join(df.columns[:5])}...")

        # Show window iteration capability
        print("\n  Window iteration example:")
        count = 0
        for item_id, window in storage.iter_item_windows(window_size=10, step=10):
            count += 1
            if count <= 3:
                print(f"    Item {item_id}: window with {len(window)} rows")
            if count >= 3:
                break
        print(f"  ✓ Window iterator working ({count}+ windows available)")

    except FileNotFoundError:
        print("  ⚠ No processed data found. Run data_preparer.py first.")
    except Exception as e:
        print(f"  ✗ Error: {e}")


def example_2_models():
    """Example 2: Model Wrappers"""
    print("\n" + "="*60)
    print("Example 2: Model Wrappers")
    print("="*60)

    try:
        from src.models.base import BazaarModel
        from src.models.tabular.xgb_model import XGBEdgeModel

        print("✓ Model base class imported")
        print("✓ XGBEdgeModel available")

        # Show model creation (without data)
        print("\n  Creating XGB model instance:")
        model = XGBEdgeModel(
            feature_cols=['feat1', 'feat2', 'feat3'],
            label_col='label',
            n_estimators=10,
            max_depth=3
        )
        print(f"  ✓ Model created: {type(model).__name__}")
        print(f"  ✓ Features: {len(model.feature_cols)}")
        print(f"  ✓ Params: n_estimators={model.get_params()['n_estimators']}")

    except Exception as e:
        print(f"  ✗ Error: {e}")


def example_3_ensemble():
    """Example 3: Ensemble Conductor"""
    print("\n" + "="*60)
    print("Example 3: Ensemble Conductor")
    print("="*60)

    try:
        from src.ensemble.conductor import Conductor
        import numpy as np

        # Create conductor for 3 experts
        conductor = Conductor(n_experts=3, edge_buffer_bp=1.5)
        print(f"✓ Conductor created for {conductor.n_experts} experts")
        print(f"  Initial weights: {conductor.get_weights()}")

        # Simulate expert predictions
        expert_edges = np.array([5.0, 3.5, 4.2])  # Edge in bp from each expert
        cost_bp = 2.0

        net_edge = conductor.expected_net_edge_bp(expert_edges, cost_bp=cost_bp)
        print(f"\n  Expert edges: {expert_edges} bp")
        print(f"  Cost: {cost_bp} bp")
        print(f"  Buffer: {conductor.edge_buffer_bp} bp")
        print(f"  ✓ Net edge: {net_edge:.2f} bp")

    except Exception as e:
        print(f"  ✗ Error: {e}")


def example_4_portfolio():
    """Example 4: Portfolio Allocation"""
    print("\n" + "="*60)
    print("Example 4: Portfolio Allocation")
    print("="*60)

    try:
        from src.portfolio.allocator import alloc_greedy
        import numpy as np

        # Sample items with edge and risk
        items = ['ENCHANTED_BOOK', 'HYPERION', 'NECRON_BLADE', 'WITHER_HELMET']
        exp_edge_bp = np.array([8.0, 5.5, 12.0, 3.0])
        risk_bp = np.array([15.0, 10.0, 25.0, 8.0])

        positions = alloc_greedy(
            items=items,
            exp_edge_bp=exp_edge_bp,
            risk_bp=risk_bp,
            max_gross_exposure=1000.0,
            per_item_cap=400.0
        )

        print("✓ Greedy allocation complete")
        print(f"\n  {'Item':<20} {'Edge(bp)':<12} {'Risk(bp)':<12} {'Allocation':<12}")
        print("  " + "-"*56)
        for i, item in enumerate(items):
            if positions[item] > 0:
                print(f"  {item:<20} {exp_edge_bp[i]:<12.1f} {risk_bp[i]:<12.1f} {positions[item]:<12.2f}")

        total = sum(positions.values())
        print(f"\n  ✓ Total exposure: {total:.2f}")

    except Exception as e:
        print(f"  ✗ Error: {e}")


def example_5_execution():
    """Example 5: Execution Cost Model"""
    print("\n" + "="*60)
    print("Example 5: Execution Cost Model")
    print("="*60)

    try:
        from src.execution.slippage_model import SimpleCostModel

        cost_model = SimpleCostModel(fee_bp=10.0)
        print(f"✓ Cost model created (fee={cost_model.fee_bp}bp)")

        # Estimate costs for different scenarios
        scenarios = [
            ("Narrow spread, small size", 3.0, 0.2, 0.1),
            ("Wide spread, large size", 15.0, 0.8, 0.6),
            ("Medium spread, medium size", 8.0, 0.5, 0.3),
        ]

        print(f"\n  {'Scenario':<30} {'Spread':<10} {'Size':<10} {'Queue':<10} {'Cost':<10}")
        print("  " + "-"*70)
        for name, spread, size, queue in scenarios:
            cost = cost_model.predict_bp(spread, size, queue)
            print(f"  {name:<30} {spread:<10.1f} {size:<10.2f} {queue:<10.2f} {cost:<10.2f}")

    except Exception as e:
        print(f"  ✗ Error: {e}")


def example_6_backtest():
    """Example 6: Backtest Contracts"""
    print("\n" + "="*60)
    print("Example 6: Backtest Framework")
    print("="*60)

    try:
        from src.backtest.contracts import BacktestConfig, BacktestResult
        import pandas as pd

        # Create a backtest configuration
        cfg = BacktestConfig(
            label_col='label_up',
            prediction_col='prob_up',
            timestamp_col='timestamp',
            item_id_col='item',
            fee_bp=10.0,
            min_edge_bp=2.0,
            slippage_bp=5.0
        )

        print("✓ Backtest configuration created")
        print(f"  Fee: {cfg.fee_bp}bp")
        print(f"  Min edge: {cfg.min_edge_bp}bp")
        print(f"  Slippage: {cfg.slippage_bp}bp")

        # Show result structure
        print("\n  BacktestResult structure:")
        print("    - trades: DataFrame")
        print("    - equity_curve: DataFrame")
        print("    - summary: Dict[str, float]")
        print("    - config: BacktestConfig")

    except Exception as e:
        print(f"  ✗ Error: {e}")


def main():
    """Run all examples."""
    print("="*60)
    print("Symphony Architecture - API Examples")
    print("="*60)
    print("\nThese examples demonstrate the new modular API.")
    print("No data required - just showing the interfaces.\n")

    try:
        example_1_data_access()
        example_2_models()
        example_3_ensemble()
        example_4_portfolio()
        example_5_execution()
        example_6_backtest()

        print("\n" + "="*60)
        print("✅ All examples completed successfully!")
        print("="*60)
        print("\nNext steps:")
        print("  1. Prepare data: python scripts/data_preparer.py")
        print("  2. Run pipeline: python -m src.pipeline.live_loop")
        print("  3. See REFACTORING_GUIDE.md for more details")
        print("="*60 + "\n")

    except Exception as e:
        print(f"\n❌ Error running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

