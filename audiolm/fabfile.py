from fabric import Connection, task

# Configuration
WORKSTATION_IP = "172.21.52.246"
WORKSTATION_USER = "meraki"
SCRIPT_PATH = "/path/to/your_script.py"
DEST_PATH = "/path/to/destination/"

@task
def build_job(c):


@task
def upload_script(c):
    """Upload the training script to the workstation."""
    c.put(SCRIPT_PATH, DEST_PATH)

@task
def run_script(c):
    """Run the training script on the workstation."""
    c.run(f"cd {DEST_PATH} && nohup python {SCRIPT_PATH} > output.log 2>&1 &", pty=False)

@task
def check_status(c):
    """Check the status of the running script."""
    c.run("ps aux | grep python")

@task
def kill_script(c):
    """Kill the running script."""
    c.run("pkill -f your_script.py")

# Connection to the workstation
workstation = Connection(host=WORKSTATION_IP, user=WORKSTATION_USER)
local = Connection(host='localhost')

# Example usage
if __name__ == "__main__":
    # Upload the script
    upload_script(workstation)
    # Run the script
    run_script(workstation)
