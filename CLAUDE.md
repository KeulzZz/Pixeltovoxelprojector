# CLAUDE.md

Guide for AI assistants working on the Pixel To Voxel Projector codebase.

## Project Overview

A C++17/Python toolkit for reconstructing 3D objects from multiple camera views. It detects motion in 2D image streams and maps changed pixels into a 3D voxel grid via ray casting. Key capabilities:

- **Real-time RTSP processing**: Multi-camera live stream ingestion, motion detection, and voxel accumulation
- **Batch processing**: Offline 3D reconstruction from pre-recorded images
- **Drone detection**: Voxel-based clustering to detect/track drones, with ATAK integration via Cursor-on-Target (CoT) messaging
- **Visualization**: OpenGL and PyVista 3D voxel viewers

## Repository Structure

```
├── CMakeLists.txt              # CMake build config (C++17, OpenCV, nlohmann_json, zlib, OpenMP, ZeroMQ)
├── setup.sh                    # Automated dependency install + build (Debian/Fedora/Arch)
├── setup.py                    # Python C++ extension build (pybind11)
├── config/
│   └── camera_config.json      # Camera RTSP URLs, positions, orientations, FOV
├── src/
│   ├── main.cpp                # Drone detector entry point
│   ├── processing/
│   │   ├── rtsp_processor.cpp  # Real-time RTSP stream processor (main executable)
│   │   ├── ray_voxel.cpp       # Batch ray-casting voxel accumulation
│   │   ├── process_image.cpp   # Python C++ extension (pybind11)
│   │   ├── voxel_drone_detector.h
│   │   └── voxel_drone_detector.cpp  # BFS clustering + drone tracking
│   └── cot/
│       ├── cot_message_generator.h/.cpp  # CoT XML message generation
│       ├── cot_network_handler.h/.cpp    # TCP send to ATAK server
│       └── cot_message.cpp               # CoT message utilities
├── scripts/
│   ├── visualize_voxel_grid.py       # Real-time OpenGL voxel viewer
│   ├── voxelmotionviewer.py          # PyVista interactive 3D viewer
│   ├── spacevoxelviewer.py           # FITS/astronomy voxel viewer
│   └── examplebuildvoxelgridfrommotion.bat  # Windows batch example
├── PixelationDecensorer.py     # Super-resolution recovery from pixelated video
├── blenderrenderscript.py      # Blender multi-camera render automation
└── blenderrestarteraftercrashing.py  # Blender watchdog/restart utility
```

## Build & Run

### Prerequisites

System packages: C++ compiler (g++), CMake >= 3.10, OpenCV with videoio, zlib, OpenGL/GLUT, Python 3, pip. Optional: ZeroMQ (for live voxel publishing).

### Building

```bash
# Automated (installs deps + builds):
sudo ./setup.sh

# Manual:
mkdir -p build && cd build
cmake ..
make -j$(nproc)
```

This produces two executables in `build/`:
- `rtsp_processor` - real-time RTSP stream voxel processor
- `drone_detector` - voxel-based drone detection with ATAK integration

### Python extension

```bash
pip install pybind11 numpy
python setup.py build_ext --inplace
```

Builds the `process_image_cpp` Python module from `src/processing/process_image.cpp`.

### Running

```bash
# Real-time processing
./build/rtsp_processor config/camera_config.json output_dir [save_interval_seconds]

# Visualization
python3 scripts/visualize_voxel_grid.py output_dir [threshold]

# Batch processing
./build/ray_voxel <metadata.json> <image_folder> <output_voxel.bin>

# Drone detector
./build/drone_detector [config_file.conf]
```

## Architecture & Data Flow

```
RTSP Streams → Frame Capture → Motion Detection (frame diff)
    → Ray Casting (DDA algorithm) → Voxel Grid Accumulation
    → [Optional] ZeroMQ Publish / Binary File Save
    → Drone Detection (BFS clustering) → CoT Message → ATAK Server (TCP)
```

### Key algorithms

- **Motion detection**: Frame differencing between consecutive frames per camera
- **Ray casting**: DDA (Digital Differential Analyzer) traversal from camera through changed pixels into the voxel grid
- **Voxel accumulation**: Overlapping rays from multiple cameras increase voxel intensity
- **Drone detection**: BFS-based voxel clustering with distance thresholds, centroid/velocity estimation, UUID-based tracking

### Binary voxel grid format

`[grid_size: int32] [voxel_size: float32] [grid_data: float32[grid_size^3]]`

## Code Conventions

### C++ (src/)

- **Standard**: C++17
- **Naming**: `camelCase` for variables and functions, `PascalCase` for structs/classes, trailing underscore for private members (`member_`)
- **Threading**: `std::thread` + `std::mutex` + `std::condition_variable` for thread safety; `std::atomic` / `volatile sig_atomic_t` for control flags
- **Patterns**: Producer-consumer queues (frame capture → processing), RAII for resource management
- **Headers**: `.h` files for declarations, `.cpp` for implementations; headers use `#pragma once` or include guards
- **Comments**: Block comments at file top describing purpose; inline comments for non-obvious logic

### Python (scripts/, root .py files)

- **Style**: PEP 8, snake_case naming
- **Dependencies**: numpy, PyOpenGL, pyvista (optional), astropy (optional), OpenCV

### General

- No formal test suite exists; testing is done via test modes in executables (e.g., simulated drone in `main.cpp`) and visual verification through the viewer scripts
- No CI/CD pipeline
- No linter configuration

## Dependencies

| Dependency | Version | Source | Purpose |
|---|---|---|---|
| OpenCV | system | apt/dnf/pacman | Image processing, RTSP capture |
| nlohmann_json | 3.11.2 | System or CMake FetchContent | JSON parsing |
| zlib | system | apt/dnf/pacman | Voxel grid compression |
| OpenMP | system (optional) | Compiler | Parallel processing |
| ZeroMQ | system (optional) | pkg-config | Live voxel data publishing |
| pybind11 | pip | Python package | C++ Python bindings |
| numpy | pip | Python package | Array operations |
| PyOpenGL | pip | Python package | Voxel visualization |

## Configuration

Camera config (`config/camera_config.json`) is a JSON array of camera objects:

```json
{
  "camera_index": 0,
  "rtsp_url": "rtsp://user:pass@ip:554/stream",
  "camera_position": [x, y, z],
  "yaw": 0.0,
  "pitch": 0.0,
  "roll": 0.0,
  "fov_degrees": 87.0
}
```

Drone detector config is a text file with `key=value` pairs for grid size, voxel size, detection thresholds, ATAK server IP/port, and origin lat/lon.

## Important Notes

- `config/camera_config.json` contains real RTSP credentials -- do not commit updated credentials
- The project is licensed under Apache License 2.0 (LICENSE file), though README.md mentions MIT -- the LICENSE file governs
- No unit tests exist; changes should be verified by building successfully and testing with the visualization tools
- The `setup.sh` script requires root privileges and auto-detects the Linux package manager
