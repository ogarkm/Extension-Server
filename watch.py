import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import subprocess
import sys
import os
import signal

class FileChangeHandler(FileSystemEventHandler):
    def __init__(self, app_process):
        self.app_process = app_process
        self.last_reload = time.time()
        self.cooldown = 1  # Cooldown in seconds to prevent multiple reloads

    def on_modified(self, event):
        if event.is_directory:
            return
        
        # Only watch Python files and module files
        if not (event.src_path.endswith('.py') or event.src_path.endswith('.module')):
            return

        # Implement cooldown to prevent multiple reloads
        current_time = time.time()
        if current_time - self.last_reload < self.cooldown:
            return

        print(f"\n🔄 Detected change in {os.path.basename(event.src_path)}, restarting server...")
        
        # Kill the current process group
        try:
            os.killpg(os.getpgid(self.app_process.pid), signal.SIGTERM)
        except:
            pass
        
        # Start a new process
        self.app_process = start_app()
        self.last_reload = current_time

def start_app():
    return subprocess.Popen(
        ['uvicorn', 'app:app', '--host', '0.0.0.0', '--port', '7275', '--log-level', 'info'],
        preexec_fn=os.setsid
    )

def main():
    app_process = start_app()
    
    # Set up file watching
    event_handler = FileChangeHandler(app_process)
    observer = Observer()
    
    # Watch the current directory and subdirectories
    observer.schedule(event_handler, '.', recursive=True)
    observer.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
        try:
            os.killpg(os.getpgid(app_process.pid), signal.SIGTERM)
        except:
            pass
        
    observer.join()

if __name__ == '__main__':
    main()