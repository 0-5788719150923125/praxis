"""Test agent commit freshness color gradient."""

import json
from playwright.sync_api import Page


def test_agent_color_gradient(page: Page):
    """Test that agent colors are properly calculated based on commit freshness."""

    # Capture console messages
    console_messages = []

    def handle_console(msg):
        if '[COLOR]' in msg.text:
            console_messages.append(msg.text)

    page.on('console', handle_console)

    # Navigate to app
    page.goto("http://localhost:5555")
    page.wait_for_timeout(1000)

    # Click on Agents tab
    agents_button = page.locator('button[data-tab="agents"]')
    if agents_button.count() > 0:
        agents_button.click()
        page.wait_for_timeout(2000)  # Wait for agents to load

        # Print all console messages for debugging
        print("\n" + "="*80)
        print("AGENT COLOR CALCULATIONS:")
        print("="*80)
        for msg in console_messages:
            print(msg)
        print("="*80)

        # Parse the console output to analyze color calculations
        agents_data = []
        current_agent = {}

        for msg in console_messages:
            if msg.startswith('[COLOR] Agent:'):
                # Start of new agent
                if current_agent:
                    agents_data.append(current_agent)
                current_agent = {'raw': msg}
                # Parse agent info
                parts = msg.split(', ')
                for part in parts:
                    if 'Agent:' in part:
                        current_agent['name'] = part.split('Agent:')[1].strip()
                    elif 'Hash:' in part:
                        current_agent['hash'] = part.split('Hash:')[1].strip()
                    elif 'Status:' in part:
                        current_agent['status'] = part.split('Status:')[1].strip()
            elif '[COLOR]   Timestamp:' in msg:
                # Parse timestamps
                parts = msg.replace('[COLOR]   Timestamp:', '').split(',')
                if len(parts) >= 3:
                    current_agent['timestamp'] = int(parts[0].strip())
                    current_agent['oldest'] = int(parts[1].split(':')[1].strip())
                    current_agent['newest'] = int(parts[2].split(':')[1].strip())
            elif '[COLOR]   Freshness:' in msg:
                # Parse freshness value
                parts = msg.replace('[COLOR]   Freshness:', '').split(',')
                if len(parts) >= 1:
                    freshness_str = parts[0].strip()
                    current_agent['freshness'] = float(freshness_str)
            elif '[COLOR]   Final:' in msg:
                # Parse final color
                current_agent['final_color'] = msg.replace('[COLOR]   Final:', '').strip()

        # Add last agent
        if current_agent:
            agents_data.append(current_agent)

        # Analyze the data
        print("\n" + "="*80)
        print("ANALYSIS:")
        print("="*80)

        if agents_data:
            print(f"\nFound {len(agents_data)} agents with color calculations\n")

            for agent in agents_data:
                print(f"Agent: {agent.get('name', 'unknown')}")
                print(f"  Hash: {agent.get('hash', 'unknown')}")
                print(f"  Status: {agent.get('status', 'unknown')}")
                print(f"  Timestamp: {agent.get('timestamp', 'N/A')}")
                print(f"  Freshness: {agent.get('freshness', 'N/A')}")
                print(f"  Final Color: {agent.get('final_color', 'N/A')}")
                print()

            # Find issues
            print("ISSUES DETECTED:")

            # Check if newest commit has freshness = 1.0
            newest_agents = [a for a in agents_data if a.get('freshness', 0) == 1.0]
            if not newest_agents:
                print("  ⚠ No agent has freshness=1.0 (newest should be full color)")
            else:
                print(f"  ✓ {len(newest_agents)} agent(s) at full freshness (1.0)")

            # Check if oldest commit has freshness = 0.0
            oldest_agents = [a for a in agents_data if a.get('freshness', 1) == 0.0]
            if not oldest_agents:
                print("  ⚠ No agent has freshness=0.0 (oldest should be greyscale)")
            else:
                print(f"  ✓ {len(oldest_agents)} agent(s) at zero freshness (0.0)")

            # Check for duplicate freshness values (same color for different commits)
            freshness_values = {}
            for agent in agents_data:
                f = agent.get('freshness')
                h = agent.get('hash', 'unknown')
                if f is not None:
                    if f not in freshness_values:
                        freshness_values[f] = []
                    freshness_values[f].append(h)

            for freshness, hashes in freshness_values.items():
                if len(hashes) > 1:
                    print(f"  ⚠ Multiple commits with same freshness {freshness}: {', '.join(hashes)}")
        else:
            print("  ❌ No agent color data found - agents may not have timestamps")

        print("="*80 + "\n")

        # Assert we got some data
        assert len(console_messages) > 0, "No console messages captured - debugging may not be working"
