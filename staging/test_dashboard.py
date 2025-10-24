#!/usr/bin/env python3
"""Test script for dashboard visualization without training."""

import math
import random
import time

from praxis.interface import TerminalDashboard


def run_dashboard():
    """Run the dashboard with simulated data."""
    # Create dashboard
    dashboard = TerminalDashboard(seed=42, arg_hash="TEST123")

    # Set some initial values
    dashboard.update_params(1234567890)  # 1.2B params
    dashboard.set_mode("test")
    dashboard.update_status("Testing dashboard visualization...")

    # Start the dashboard
    dashboard.start()

    try:
        # Simulate training loop
        step = 0
        batch = 0
        total_tokens = 0

        while True:
            # Update metrics with simulated data
            step += 1
            if step % 10 == 0:
                batch += 1

            # Simulate loss curve
            base_loss = 2.5 + 1.5 * math.exp(-step * 0.01)
            noise = random.uniform(-0.1, 0.1)
            train_loss = max(0.1, base_loss + noise)

            dashboard.update_step(step)
            dashboard.update_batch(batch)
            dashboard.update_loss(train_loss)

            # Update validation loss occasionally
            if step % 50 == 0:
                val_loss = train_loss * random.uniform(0.95, 1.05)
                dashboard.update_val(val_loss)

            # Update other metrics
            if step % 20 == 0:
                acc0 = min(0.99, 0.3 + step * 0.001 + random.uniform(-0.05, 0.05))
                acc1 = min(0.99, 0.5 + step * 0.0005 + random.uniform(-0.03, 0.03))
                dashboard.update_accuracy(acc0, acc1)

            if step % 30 == 0:
                fitness = min(99, 20 + step * 0.1 + random.uniform(-5, 5))
                dashboard.update_fitness(fitness)

            if step % 15 == 0:
                memory_churn = max(
                    0, 50 + 30 * math.sin(step * 0.05) + random.uniform(-10, 10)
                )
                dashboard.update_memory(memory_churn)

            # Update token count
            tokens_per_step = random.randint(1000, 5000)
            total_tokens += tokens_per_step
            dashboard.update_tokens(total_tokens / 1e9)  # Convert to billions

            # Update context tokens
            context_tokens = random.randint(512, 4096)
            dashboard.update_context_tokens(context_tokens)

            # Update rate
            rate = 0.5 + random.uniform(-0.1, 0.1)
            dashboard.update_rate(rate)

            # Add some log messages
            if step % 25 == 0:
                messages = [
                    f"Step {step}: Processing batch {batch}",
                    f"Current learning rate: {1e-4 * math.exp(-step * 0.0001):.6f}",
                    f"GPU memory: {random.randint(60, 90)}%",
                    f"Temperature: {random.randint(65, 85)}°C",
                    "Checkpoint saved successfully",
                    "Validation metrics computed",
                    f"Gradient norm: {random.uniform(0.5, 2.5):.3f}",
                ]
                dashboard.add_log(random.choice(messages))

            # Update status occasionally
            if step % 100 == 0:
                statuses = [
                    f"Training epoch {step // 100}...",
                    f"Processing batch {batch} of epoch {step // 100}",
                    "Computing validation metrics...",
                    "Saving checkpoint...",
                    "Adjusting learning rate...",
                ]
                dashboard.update_status(random.choice(statuses))

            # Update info dict
            if step % 40 == 0:
                info = {
                    "lr": f"{1e-4 * math.exp(-step * 0.0001):.2e}",
                    "grad_norm": f"{random.uniform(0.5, 2.5):.3f}",
                    "gpu_mem": f"{random.randint(60, 90)}%",
                    "temp": f"{random.randint(65, 85)}°C",
                    "epoch": str(step // 100),
                    "samples": str(batch * 16),
                }
                dashboard.update_info(info)

            # Sleep to control update rate
            time.sleep(0.05)  # 20 updates per second

    except KeyboardInterrupt:
        print("\nStopping dashboard test...")
    finally:
        dashboard.stop()
        print("Dashboard test completed!")


if __name__ == "__main__":
    run_dashboard()
