from mpi4py import MPI
import os
import time
import threading
from flask import Flask, request, jsonify

# Initialize Flask for HTTP server
app = Flask(__name__)

# Initialize MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Global variable to store the child processes' communicators
intercomms = {}  # Store intercommunicators by rank

@app.route('/spawn', methods=['POST'])
def spawn():
    global intercomms
    data = request.json
    program = data.get('program', '')  # Path to the program
    args = data.get('args', [])        # Arguments for the program
    nprocs = data.get('nprocs', 1)     # Number of processes to spawn
    print(data)
    
    # Spawn child processes using MPI_Comm_spawn
    intercomm = comm.Spawn(program, args=args, maxprocs=nprocs)
    intercomms[rank] = intercomm  # Store the intercomm for the current rank

    return jsonify({'status': 'Processes spawned', 'nprocs': nprocs}), 200

def heartbeat_monitor(interval=1, timeout=5):
    time.sleep(10)
    global intercomms
    while True:
        for rank, intercomm in intercomms.items():
            if intercomm is not None:
                try:
                    status = MPI.Status()
                    # Probe to check if a message is available
                    flag = intercomm.iprobe(source=MPI.ANY_SOURCE, tag=0)
                    
                    if flag:
                        intercomm.recv(source=MPI.ANY_SOURCE, tag=0, status=status)
                        print(f"Rank {rank}: Received heartbeat from rank {status.source}")
                    else:
                        # Wait for the timeout period
                        start_time = time.time()
                        while time.time() - start_time < timeout:
                            time.sleep(0.1)  # Sleep a little to prevent busy-waiting
                        
                        # If we reach here, it means no message was received in the timeout period
                        print(f"Rank {rank}: No heartbeat received within timeout.")

                except Exception as e:
                    print(f"Rank {rank}: Child process failed: {e}")
                    handle_failure(rank)
        time.sleep(interval)

def handle_failure(rank):
    global intercomms
    if rank in intercomms and intercomms[rank] is not None:
        print(f"Handling failure for rank {rank}, aborting and respawning processes...")
        intercomms[rank].Abort()
        intercomms[rank] = None  # Clear the communicator
        # You can implement logic to respawn processes or take other recovery steps here

# Start the heartbeat monitoring thread

if __name__ == '__main__':
    # Run the Flask app to handle curl requests
    threading.Thread(target=heartbeat_monitor, daemon=True).start()
    app.run(host='0.0.0.0', port=5000)

