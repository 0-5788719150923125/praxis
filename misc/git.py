import subprocess
import sys


def check_for_updates():
    try:
        # First, fetch the latest changes from remote
        subprocess.run(["git", "fetch"], check=True, capture_output=True)

        # Try to get the current branch name
        branch = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

        # Get commit counts ahead and behind
        status = (
            subprocess.run(
                [
                    "git",
                    "rev-list",
                    "--left-right",
                    "--count",
                    f"HEAD...origin/{branch}",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            .stdout.strip()
            .split()
        )

        commits_ahead = int(status[0])
        commits_behind = int(status[1])

        if commits_behind > 0:
            return f"Update available: Your repository is {commits_behind} commit(s) behind the remote."
        elif commits_ahead > 0:
            return f"Local changes: Your repository is {commits_ahead} commit(s) ahead of the remote."
        else:
            return "Up to date: Your repository is synchronized with the remote."

    except subprocess.CalledProcessError as e:
        return f"Error checking for updates: {str(e)}"
    except Exception as e:
        return f"Unexpected error: {str(e)}"


# Usage example
if __name__ == "__main__":
    print(check_for_updates())


# # At the start of your application
# def main():
#     # Check for updates
#     update_status = check_for_updates()
#     if "Update available" in update_status:
#         print("\nWarning:", update_status)
#         print("Please pull the latest changes to ensure you're running the latest version.\n")

#     # Rest of your application code...
