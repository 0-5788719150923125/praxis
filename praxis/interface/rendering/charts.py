"""Chart rendering for dashboard display."""

import asciichartpy


class ChartRenderer:
    """Renders charts for dashboard display."""

    def draw_chart(self, data, width, height):
        """Draw a chart from data points."""
        if len(data) > 1:
            # Use logarithmic time scale to show more recent data in detail
            # while still including historical context
            num_points = min(1000, len(data))
            all_data = list(data)[-num_points:]

            if len(all_data) <= width:
                # If we have fewer points than display width, just use them all
                plot_data = all_data
            else:
                # Create logarithmic sampling: more samples from recent data
                plot_data = []

                # We'll use a power function to distribute sample indices
                # Higher power = more bias towards recent data
                power = 2.0  # Adjust this to control the bias (1.0 = linear, higher = more recent bias)

                for i in range(width):
                    # Map display position to data index using power function
                    # i/width goes from 0 to 1, we transform it to sample more from the end
                    normalized_pos = i / (width - 1) if width > 1 else 0
                    # Apply power function to bias towards recent data
                    biased_pos = pow(normalized_pos, power)
                    # Map to data index
                    data_idx = int(biased_pos * (len(all_data) - 1))

                    # For smoother visualization, average a small window around this point
                    window_size = max(1, len(all_data) // width)
                    start_idx = max(0, data_idx - window_size // 2)
                    end_idx = min(len(all_data), start_idx + window_size)

                    window_data = all_data[start_idx:end_idx]
                    if window_data:
                        plot_data.append(sum(window_data) / len(window_data))

            # Add slight smoothing to make trends more visible
            if len(plot_data) > 3:
                smoothed_data = []
                for i in range(len(plot_data)):
                    # Simple moving average with small window
                    start = max(0, i - 1)
                    end = min(len(plot_data), i + 2)
                    smoothed_data.append(sum(plot_data[start:end]) / (end - start))
                plot_data = smoothed_data

            chart = asciichartpy.plot(
                plot_data,
                {
                    "height": height - 2,
                    "width": width - 2,
                    "format": "{:8.2f}",
                    "min": min(plot_data),
                    "max": max(plot_data),
                },
            )
            lines = chart.split("\n")
            return [line.ljust(width)[:width] for line in lines]
        return [" " * width for _ in range(height)]
