"""Dev launcher with AUTO-RELOAD for the browser-use web UI.

Runs webui.py as a child process and restarts it whenever webui.py changes, so
edits take effect after a simple browser refresh — no manual server restart.
Used by `npm run dev`. (`npm start` runs webui.py directly, no reload.)
"""

import os
import subprocess
import sys
import time

TARGET = 'webui.py'
PYTHON = os.path.join('.venv', 'Scripts', 'python.exe')
if not os.path.exists(PYTHON):
	PYTHON = sys.executable


def _mtime() -> float:
	try:
		return os.stat(TARGET).st_mtime
	except OSError:
		return 0.0


def _spawn() -> subprocess.Popen:
	return subprocess.Popen([PYTHON, TARGET])


def main() -> None:
	print(f'[dev] auto-reload watching {TARGET} — edit + refresh browser to see changes', flush=True)
	proc = _spawn()
	last = _mtime()
	try:
		while True:
			time.sleep(1.0)
			m = _mtime()
			if m != last:
				last = m
				print('[dev] change detected → restarting server…', flush=True)
				proc.terminate()
				try:
					proc.wait(timeout=5)
				except subprocess.TimeoutExpired:
					proc.kill()
				# small pause so the OS releases the port before rebinding
				time.sleep(0.5)
				proc = _spawn()
			elif proc.poll() is not None:
				# server exited on its own (crash / port issue) — relaunch
				print('[dev] server exited → restarting…', flush=True)
				time.sleep(0.5)
				proc = _spawn()
				last = _mtime()
	except KeyboardInterrupt:
		pass
	finally:
		try:
			proc.terminate()
		except Exception:
			pass


if __name__ == '__main__':
	main()
