import socket
import threading
import time
import ctypes

# 保存活跃worker连接及其rank
active_workers = []
lock = threading.Lock()

def handle_worker(conn, addr):
    global active_workers
    with lock:
        rank = len(active_workers)
        active_workers.append((conn, rank))
        print(f"Worker {addr} connected and assigned rank {rank}")

    # 启动心跳监测线程
    threading.Thread(target=heartbeat_monitor, args=(conn, rank), daemon=True).start()

    # 通知所有worker更新rank
    broadcast_ranks()

def heartbeat_monitor(conn, rank):
    try:
        while True:
            conn.settimeout(10)
            heartbeat = conn.recv(1024)
            if heartbeat.decode('utf-8') != 'heartbeat':
                raise ConnectionResetError
            time.sleep(5)
    except (ConnectionResetError, socket.timeout):
        print(f"Worker {rank} disconnected.")
        remove_worker(conn)
        broadcast_ranks()

def remove_worker(disconnected_conn):
    global active_workers
    with lock:
        active_workers = [(conn, r) for conn, r in active_workers if conn != disconnected_conn]
        active_workers = [(conn, i) for i, (conn, _) in enumerate(active_workers)]
        print(f"Updated ranks: {[r for _, r in active_workers]}")

def broadcast_ranks():
    global active_workers
    with lock:
        for conn, rank in active_workers:
            try:
                # 通知所有worker更新rank信息
                conn.sendall(f"RANK_UPDATE {rank} {len(active_workers)}".encode('utf-8'))
            except (ConnectionResetError, BrokenPipeError):
                pass

        # Rank 0 需要生成 NCCL unique ID，并发送给 controller
        if active_workers:
            rank0_conn = active_workers[0][0]
            try:
                unique_id_buffer = rank0_conn.recv(128)  # Rank 0 发送 NCCL unique ID
                # Controller 转发 unique ID 给所有非 rank 0 的 worker
                for conn, rank in active_workers:
                    if rank != 0:
                        conn.sendall(unique_id_buffer)
            except (ConnectionResetError, BrokenPipeError):
                pass

def start_controller(host='localhost', port=5000):
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((host, port))
    server_socket.listen()
    print(f"Controller listening on {host}:{port}")

    try:
        while True:
            conn, addr = server_socket.accept()
            threading.Thread(target=handle_worker, args=(conn, addr), daemon=True).start()
    except KeyboardInterrupt:
        print("Controller shutting down.")
    finally:
        server_socket.close()

if __name__ == "__main__":
    start_controller()
