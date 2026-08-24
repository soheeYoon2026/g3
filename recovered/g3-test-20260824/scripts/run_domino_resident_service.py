"""Start the resident DoMINO API without mixing the two venv NumPy builds."""
import sys

import numpy  # Bind the PhysicsNeMo-compatible NumPy before adding API packages.

sys.path.insert(0, "/home/ubuntu/g3-v2")
sys.path.append("/home/ubuntu/venv_g3/lib/python3.12/site-packages")

import uvicorn


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8005
    uvicorn.run(
        "scripts.domino_stl_service:app",
        host="127.0.0.1",
        port=port,
        workers=1,
    )
