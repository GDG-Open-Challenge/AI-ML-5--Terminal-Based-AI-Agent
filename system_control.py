import os
import platform
import subprocess

def confirm_action(action_name: str) -> bool:
    """Ask the user for explicit confirmation before a system operation."""
    print(f"\n⚠  SYSTEM ACTION: {action_name}")
    answer = input(f"Are you sure you want to {action_name}? (yes/no): ").strip().lower()
    return answer in ("yes", "y")

def system_control(action: str) -> str:
    """Perform a system-level operation.
    Supported actions: shutdown, reboot, sleep, lock, cancel_shutdown.
    """
    action = action.strip().lower()
    os_name = platform.system()

    commands = {}
    if os_name == "Windows":
        commands = {
            "shutdown": "shutdown /s /t 60",
            "reboot": "shutdown /r /t 60",
            "sleep": "rundll32.exe powrprof.dll,SetSuspendState 0,1,0",
            "lock": "rundll32.exe user32.dll,LockWorkStation",
            "cancel_shutdown": "shutdown /a",
        }
    elif os_name == "Linux":
        commands = {
            "shutdown": "shutdown -h +1",
            "reboot": "shutdown -r +1",
            "sleep": "systemctl suspend",
            "lock": "loginctl lock-session",
            "cancel_shutdown": "shutdown -c",
        }
    elif os_name == "Darwin":  # macOS
        commands = {
            "shutdown": "sudo shutdown -h +1",
            "reboot": "sudo shutdown -r +1",
            "sleep": "pmset sleepnow",
            "lock": "/System/Library/CoreServices/Menu\\ Extras/User.menu/Contents/Resources/CGSession -suspend",
            "cancel_shutdown": "sudo killall shutdown",
        }
    else:
        return f"Unsupported operating system: {os_name}"

    if action not in commands:
        return f"Unknown action '{action}'."

    safe_actions = {"cancel_shutdown", "lock"}
    if action not in safe_actions:
        if not confirm_action(action):
            return f"{action} cancelled by user."

    try:
        subprocess.Popen(commands[action], shell=True)
        return f"System {action} initiated."
    except Exception as exc:
        return f"System control error: {exc}"
