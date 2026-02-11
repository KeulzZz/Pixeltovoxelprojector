// rtsp_processor.cpp
//
// Real-time RTSP stream processor for voxel-based 3D reconstruction
// This code:
//   1) Connects to multiple RTSP streams
//   2) Processes frames in real-time
//   3) Detects motion between consecutive frames for each camera
//   4) Casts rays (voxel DDA) for changed pixels
//   5) Accumulates in a shared 3D voxel grid
//   6) Optionally saves or visualizes the voxel grid

#include <iostream>
#include <fstream>
#include <vector>
#include <map>
#include <string>
#include <thread>
#include <mutex>
#include <atomic>
#include <chrono>
#include <cmath>
#include <memory>
#include <queue>
#include <condition_variable>
#ifdef HAVE_ZMQ
#include <zmq.hpp>
#include <zlib.h>
#endif

// External libraries for JSON, OpenCV for stream handling
#include "nlohmann/json.hpp"
#include <opencv2/opencv.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/imgproc.hpp>

// For convenience
using json = nlohmann::json;

//----------------------------------------------
// 1) Data Structures
//----------------------------------------------
struct Vec3 {
    float x, y, z;
};

struct Mat3 {
    float m[9];
};

struct CameraInfo {
    int camera_index;
    std::string rtsp_url;
    Vec3 camera_position;
    float yaw, pitch, roll;
    float fov_degrees;
};

struct GridConfig {
    int Nx = 500, Ny = 500, Nz = 500;
    float voxel_size = 6.0f;
    Vec3 center = {0.f, 0.f, 500.f};
    float decay_rate = 1.0f;   // fraction of voxel value remaining after 1 second (1.0 = no decay)
    float motion_threshold = 5.0f;
    float distance_attenuation = 0.1f;
    bool show_debug = false;
    std::string output_dir = ".";
    int save_interval_seconds = 30;
};

struct FrameData {
    int camera_index;
    cv::Mat frame;  // The actual frame from the camera
    std::chrono::steady_clock::time_point timestamp; // monotonic, immune to clock jumps
};

// Thread-safe frame queue
class FrameQueue {
private:
    std::queue<FrameData> queue;
    std::mutex mutex;
    std::condition_variable cond;
    static constexpr std::size_t MAX_Q = 30; // total frames kept in queue

public:
    void push(const FrameData& frame) {
        std::unique_lock<std::mutex> lock(mutex);
        // Drop oldest frames when queue is full to keep latency bounded
        while (queue.size() >= MAX_Q) {
            queue.pop();
        }
        queue.push(frame);
        cond.notify_one();
    }

    bool pop(FrameData& frame) {
        std::unique_lock<std::mutex> lock(mutex);
        if (queue.empty()) {
            return false;
        }
        frame = queue.front();
        queue.pop();
        return true;
    }

    bool waitAndPop(FrameData& frame, int timeout_ms) {
        std::unique_lock<std::mutex> lock(mutex);
        if (cond.wait_for(lock, std::chrono::milliseconds(timeout_ms), [this] { return !queue.empty(); })) {
            frame = queue.front();
            queue.pop();
            return true;
        }
        return false;
    }

    std::size_t size() {
        std::unique_lock<std::mutex> lock(mutex);
        return queue.size();
    }
};

//----------------------------------------------
// 2) Basic Math Helpers
//----------------------------------------------
static inline float deg2rad(float deg) {
    return deg * 3.14159265358979323846f / 180.0f;
}

static inline Vec3 normalize(const Vec3 &v) {
    float len = std::sqrt(v.x*v.x + v.y*v.y + v.z*v.z);
    if(len < 1e-12f) {
        return {0.f, 0.f, 0.f};
    }
    return { v.x/len, v.y/len, v.z/len };
}

// Multiply 3x3 matrix by Vec3
static inline Vec3 mat3_mul_vec3(const Mat3 &M, const Vec3 &v) {
    Vec3 r;
    r.x = M.m[0]*v.x + M.m[1]*v.y + M.m[2]*v.z;
    r.y = M.m[3]*v.x + M.m[4]*v.y + M.m[5]*v.z;
    r.z = M.m[6]*v.x + M.m[7]*v.y + M.m[8]*v.z;
    return r;
}

//----------------------------------------------
// 3) Euler -> Rotation Matrix
//----------------------------------------------
Mat3 rotation_matrix_yaw_pitch_roll(float yaw_deg, float pitch_deg, float roll_deg) {
    float y = deg2rad(yaw_deg);   // yaw (Z)
    float p = deg2rad(pitch_deg); // pitch (Y)
    float r = deg2rad(roll_deg);  // roll (X)
    
    // Rz(yaw)
    float cy = std::cos(y), sy = std::sin(y);
    float Rz[9] = { cy,-sy,0,  sy,cy,0,  0,0,1 };
    
    // Ry(pitch)
    float cp = std::cos(p), sp = std::sin(p);
    float Ry[9] = {  cp,0,sp,  0,1,0,  -sp,0,cp };
    
    // Rx(roll)
    float cr = std::cos(r), sr = std::sin(r);
    float Rx[9] = { 1,0,0,  0,cr,-sr,  0,sr,cr };

    // Helper to multiply 3x3
    auto matmul3x3 = [&](const float A[9], const float B[9], float C[9]){
        for(int row=0; row<3; ++row) {
            for(int col=0; col<3; ++col) {
                C[row*3+col] =
                    A[row*3+0]*B[0*3+col] +
                    A[row*3+1]*B[1*3+col] +
                    A[row*3+2]*B[2*3+col];
            }
        }
    };

    float Rtemp[9], Rfinal[9];
    matmul3x3(Rz, Ry, Rtemp);    // Rz * Ry
    matmul3x3(Rtemp, Rx, Rfinal); // (Rz*Ry)*Rx

    Mat3 out;
    for(int i=0; i<9; i++){
        out.m[i] = Rfinal[i];
    }
    return out;
}

//----------------------------------------------
// 4) Load Camera Configuration
//----------------------------------------------

static CameraInfo parse_camera_entry(const json &entry) {
    CameraInfo ci;
    ci.camera_index = entry.value("camera_index", 0);
    ci.rtsp_url = entry.value("rtsp_url", "");
    ci.yaw = entry.value("yaw", 0.f);
    ci.pitch = entry.value("pitch", 0.f);
    ci.roll = entry.value("roll", 0.f);
    ci.fov_degrees = entry.value("fov_degrees", 60.f);
    if (entry.contains("camera_position") && entry["camera_position"].is_array()) {
        auto arr = entry["camera_position"];
        if (arr.size() >= 3) {
            ci.camera_position.x = arr[0].get<float>();
            ci.camera_position.y = arr[1].get<float>();
            ci.camera_position.z = arr[2].get<float>();
        }
    }
    return ci;
}

static GridConfig parse_grid_config(const json &g) {
    GridConfig gc;
    // "size" can be a single int (cubic) or an array of 3 ints
    if (g.contains("size")) {
        if (g["size"].is_array() && g["size"].size() >= 3) {
            gc.Nx = g["size"][0].get<int>();
            gc.Ny = g["size"][1].get<int>();
            gc.Nz = g["size"][2].get<int>();
        } else if (g["size"].is_number()) {
            gc.Nx = gc.Ny = gc.Nz = g["size"].get<int>();
        }
    }
    gc.voxel_size = g.value("voxel_size", gc.voxel_size);
    if (g.contains("center") && g["center"].is_array() && g["center"].size() >= 3) {
        gc.center.x = g["center"][0].get<float>();
        gc.center.y = g["center"][1].get<float>();
        gc.center.z = g["center"][2].get<float>();
    }
    gc.decay_rate = g.value("decay_rate", gc.decay_rate);
    gc.motion_threshold = g.value("motion_threshold", gc.motion_threshold);
    gc.distance_attenuation = g.value("distance_attenuation", gc.distance_attenuation);
    gc.show_debug = g.value("show_debug", gc.show_debug);
    gc.save_interval_seconds = g.value("save_interval_seconds", gc.save_interval_seconds);
    return gc;
}

struct AppConfig {
    std::vector<CameraInfo> cameras;
    GridConfig grid;
};

AppConfig load_config(const std::string &json_path) {
    AppConfig cfg;

    std::ifstream ifs(json_path);
    if(!ifs.is_open()){
        std::cerr << "ERROR: Cannot open " << json_path << std::endl;
        return cfg;
    }

    json j;
    ifs >> j;

    if (j.is_array()) {
        // Legacy format: bare array of cameras, grid uses defaults
        for (const auto &entry : j) {
            cfg.cameras.push_back(parse_camera_entry(entry));
        }
    } else if (j.is_object()) {
        // New format: { "grid": {...}, "cameras": [...] }
        if (j.contains("cameras") && j["cameras"].is_array()) {
            for (const auto &entry : j["cameras"]) {
                cfg.cameras.push_back(parse_camera_entry(entry));
            }
        } else {
            // Single camera object (legacy)
            cfg.cameras.push_back(parse_camera_entry(j));
        }
        if (j.contains("grid") && j["grid"].is_object()) {
            cfg.grid = parse_grid_config(j["grid"]);
        }
    } else {
        std::cerr << "ERROR: JSON format not recognized.\n";
    }

    return cfg;
}

//----------------------------------------------
// 5) Image Processing & Motion Detection
//----------------------------------------------
struct ImageGray {
    int width;
    int height;
    std::vector<float> pixels;  // grayscale float
};

// Convert OpenCV Mat to our ImageGray structure
ImageGray convert_mat_to_gray(const cv::Mat &frame) {
    ImageGray img;

    // Fast path: convert in OpenCV vectorised code
    cv::Mat gray8;
    if (frame.channels() == 1) {
        gray8 = frame; // no copy
    } else {
        cv::cvtColor(frame, gray8, cv::COLOR_BGR2GRAY);
    }

    cv::Mat gray32f;
    gray8.convertTo(gray32f, CV_32F); // 0..255 floats

    img.width  = gray32f.cols;
    img.height = gray32f.rows;
    const float* startPtr = gray32f.ptr<float>();
    const float* endPtr   = startPtr + gray32f.total();
    img.pixels.assign(startPtr, endPtr);
    return img;
}

// Detect motion by absolute difference
// Returns a boolean mask + the difference for each pixel
struct MotionMask {
    int width;
    int height;
    std::vector<bool> changed;
    std::vector<float> diff; // absolute difference
};

MotionMask detect_motion(const ImageGray &prev, const ImageGray &next, float threshold) {
    MotionMask mm;
    if(prev.width != next.width || prev.height != next.height) {
        std::cerr << "Images differ in size. Can't do motion detection!\n";
        mm.width = 0;
        mm.height = 0;
        return mm;
    }
    mm.width = prev.width;
    mm.height = prev.height;
    mm.changed.resize(mm.width * mm.height, false);
    mm.diff.resize(mm.width * mm.height, 0.f);

    for(int i=0; i < mm.width*mm.height; i++){
        float d = std::fabs(prev.pixels[i] - next.pixels[i]);
        mm.diff[i] = d;
        mm.changed[i] = (d > threshold);
    }
    return mm;
}

//----------------------------------------------
// 6) Voxel DDA
//----------------------------------------------
struct RayStep {
    int ix, iy, iz;
    int step_count;
    float distance;
};

static inline float safe_div(float num, float den) {
    float eps = 1e-12f;
    if(std::fabs(den) < eps) {
        return std::numeric_limits<float>::infinity();
    }
    return num / den;
}

std::vector<RayStep> cast_ray_into_grid(
    const Vec3 &camera_pos,
    const Vec3 &dir_normalized,
    int Nx, int Ny, int Nz,
    float voxel_size,
    const Vec3 &grid_center)
{
    std::vector<RayStep> steps;
    steps.reserve(64);

    float half_x = 0.5f * (Nx * voxel_size);
    float half_y = 0.5f * (Ny * voxel_size);
    float half_z = 0.5f * (Nz * voxel_size);
    Vec3 grid_min = { grid_center.x - half_x,
                      grid_center.y - half_y,
                      grid_center.z - half_z };
    Vec3 grid_max = { grid_center.x + half_x,
                      grid_center.y + half_y,
                      grid_center.z + half_z };

    float t_min = 0.f;
    float t_max = std::numeric_limits<float>::infinity();

    // 1) Ray-box intersection
    for(int i=0; i<3; i++){
        float origin = (i==0)? camera_pos.x : ((i==1)? camera_pos.y : camera_pos.z);
        float d      = (i==0)? dir_normalized.x : ((i==1)? dir_normalized.y : dir_normalized.z);
        float mn     = (i==0)? grid_min.x : ((i==1)? grid_min.y : grid_min.z);
        float mx     = (i==0)? grid_max.x : ((i==1)? grid_max.y : grid_max.z);

        if(std::fabs(d) < 1e-12f){
            if(origin < mn || origin > mx){
                return steps; // no intersection
            }
        } else {
            float t1 = (mn - origin)/d;
            float t2 = (mx - origin)/d;
            float t_near = std::fmin(t1, t2);
            float t_far  = std::fmax(t1, t2);
            if(t_near > t_min) t_min = t_near;
            if(t_far  < t_max) t_max = t_far;
            if(t_min > t_max){
                return steps;
            }
        }
    }

    if(t_min < 0.f) t_min = 0.f;

    // 2) Start voxel
    Vec3 start_world = { camera_pos.x + t_min*dir_normalized.x,
                         camera_pos.y + t_min*dir_normalized.y,
                         camera_pos.z + t_min*dir_normalized.z };
    float fx = (start_world.x - grid_min.x)/voxel_size;
    float fy = (start_world.y - grid_min.y)/voxel_size;
    float fz = (start_world.z - grid_min.z)/voxel_size;

    int ix = int(fx);
    int iy = int(fy);
    int iz = int(fz);
    if(ix<0 || ix>=Nx || iy<0 || iy>=Ny || iz<0 || iz>=Nz) {
        return steps;
    }

    // 3) Step direction
    int step_x = (dir_normalized.x >= 0.f)? 1 : -1;
    int step_y = (dir_normalized.y >= 0.f)? 1 : -1;
    int step_z = (dir_normalized.z >= 0.f)? 1 : -1;

    auto boundary_in_world_x = [&](int i_x){ return grid_min.x + i_x*voxel_size; };
    auto boundary_in_world_y = [&](int i_y){ return grid_min.y + i_y*voxel_size; };
    auto boundary_in_world_z = [&](int i_z){ return grid_min.z + i_z*voxel_size; };

    int nx_x = ix + (step_x>0?1:0);
    int nx_y = iy + (step_y>0?1:0);
    int nx_z = iz + (step_z>0?1:0);

    float next_bx = boundary_in_world_x(nx_x);
    float next_by = boundary_in_world_y(nx_y);
    float next_bz = boundary_in_world_z(nx_z);

    float t_max_x = safe_div(next_bx - camera_pos.x, dir_normalized.x);
    float t_max_y = safe_div(next_by - camera_pos.y, dir_normalized.y);
    float t_max_z = safe_div(next_bz - camera_pos.z, dir_normalized.z);

    float t_delta_x = safe_div(voxel_size, std::fabs(dir_normalized.x));
    float t_delta_y = safe_div(voxel_size, std::fabs(dir_normalized.y));
    float t_delta_z = safe_div(voxel_size, std::fabs(dir_normalized.z));

    float t_current = t_min;
    int step_count = 0;

    // 4) Walk
    while(t_current <= t_max){
        RayStep rs;
        rs.ix = ix; 
        rs.iy = iy; 
        rs.iz = iz;
        rs.step_count = step_count;
        rs.distance = t_current;

        steps.push_back(rs);

        if(t_max_x < t_max_y && t_max_x < t_max_z){
            ix += step_x;
            t_current = t_max_x;
            t_max_x += t_delta_x;
        } else if(t_max_y < t_max_z){
            iy += step_y;
            t_current = t_max_y;
            t_max_y += t_delta_y;
        } else {
            iz += step_z;
            t_current = t_max_z;
            t_max_z += t_delta_z;
        }
        step_count++;
        if(ix<0 || ix>=Nx || iy<0 || iy>=Ny || iz<0 || iz>=Nz){
            break;
        }
    }

    return steps;
}

//----------------------------------------------
// 7) Camera Stream Thread
//----------------------------------------------
void camera_stream_thread(
    const CameraInfo& camera_info,
    std::shared_ptr<FrameQueue> frame_queue,
    std::atomic<bool>& running
) {
    // Force OpenCV to use the FFmpeg backend instead of GStreamer for RTSP.
    cv::VideoCapture cap(camera_info.rtsp_url, cv::CAP_FFMPEG);
    if (!cap.isOpened()) {
        std::cerr << "ERROR: Failed to open RTSP stream: " << camera_info.rtsp_url << std::endl;
        return;
    }

    // Keep latency down (supported on recent OpenCV/FFmpeg builds)
    cap.set(cv::CAP_PROP_BUFFERSIZE, 1);
    cap.set(cv::CAP_PROP_OPEN_TIMEOUT_MSEC, 5000);
    cap.set(cv::CAP_PROP_READ_TIMEOUT_MSEC, 5000);

    // Set capture properties if needed
    // cap.set(cv::CAP_PROP_BUFFERSIZE, 1);  // Reduce buffer size for lower latency

    std::cout << "Camera " << camera_info.camera_index << " connected to: " << camera_info.rtsp_url << std::endl;

    cv::Mat frame;
    while (running) {
        if (cap.read(frame)) {
            if (frame.empty()) {
                std::cerr << "WARNING: Empty frame received from camera " << camera_info.camera_index << std::endl;
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
                continue;
            }

            // Create frame data and push to queue
            FrameData frameData;
            frameData.camera_index = camera_info.camera_index;
            frameData.frame = frame.clone();
            frameData.timestamp = std::chrono::steady_clock::now();
            
            frame_queue->push(frameData);
        } else {
            std::cerr << "ERROR: Failed to read frame from camera " << camera_info.camera_index << std::endl;
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            
            // Attempt to reconnect
            cap.release();
            std::this_thread::sleep_for(std::chrono::seconds(1));
            cap.open(camera_info.rtsp_url);
            if (!cap.isOpened()) {
                std::cerr << "ERROR: Failed to reconnect to RTSP stream: " << camera_info.rtsp_url << std::endl;
            } else {
                std::cout << "Camera " << camera_info.camera_index << " reconnected." << std::endl;
            }
        }
    }
    
    cap.release();
    std::cout << "Camera " << camera_info.camera_index << " thread stopped." << std::endl;
}

//----------------------------------------------
// 8) Debug frame sharing
//----------------------------------------------
struct DebugState {
    std::map<int, cv::Mat> frames;   // latest motion overlay per camera
    std::mutex mutex;
};

//----------------------------------------------
// 9) Processing Thread
//----------------------------------------------
void processing_thread(
    std::map<int, CameraInfo> camera_info_map,
    std::shared_ptr<FrameQueue> frame_queue,
    std::mutex& voxel_grid_mutex,
    std::vector<float>& voxel_grid,
    const GridConfig& gc,
    std::atomic<bool>& running,
#ifdef HAVE_ZMQ
    zmq::socket_t* pub_socket,
#else
    void* /*unused*/,
#endif
    DebugState& debug_state
) {
    const int Nx = gc.Nx, Ny = gc.Ny, Nz = gc.Nz;
    const float voxel_size = gc.voxel_size;
    const Vec3 grid_center = gc.center;
    const float motion_threshold = gc.motion_threshold;
    const float alpha = gc.distance_attenuation;

    // Map to store the previous frame for each camera
    std::map<int, ImageGray> prev_frames;
    std::map<int, cv::Mat> prev_raw_frames; // for debug overlay
    std::map<int, std::chrono::steady_clock::time_point> last_frame_times;

    auto last_save_time = std::chrono::system_clock::now();
    auto last_decay_time = std::chrono::steady_clock::now();

    while (running) {
        FrameData frame_data;
        if (frame_queue->waitAndPop(frame_data, 100)) {
            int camera_index = frame_data.camera_index;

            // Convert to our image format
            ImageGray curr_img = convert_mat_to_gray(frame_data.frame);

            // Check if we have a previous frame for this camera
            if (prev_frames.find(camera_index) == prev_frames.end()) {
                prev_frames[camera_index] = curr_img;
                if (gc.show_debug) prev_raw_frames[camera_index] = frame_data.frame.clone();
                last_frame_times[camera_index] = frame_data.timestamp;
                continue;
            }

            // Compute time difference between frames
            auto time_diff = std::chrono::duration_cast<std::chrono::milliseconds>(
                frame_data.timestamp - last_frame_times[camera_index]).count();

            if (time_diff < 30) {
                continue;
            }

            // --- Voxel decay (time-based, independent of camera count/FPS) ---
            if (gc.decay_rate < 1.0f && gc.decay_rate > 0.f) {
                auto now_steady = std::chrono::steady_clock::now();
                float dt = std::chrono::duration<float>(now_steady - last_decay_time).count();
                if (dt >= 0.033f) { // apply at ~30Hz max
                    float factor = std::pow(gc.decay_rate, dt);
                    {
                        std::lock_guard<std::mutex> lock(voxel_grid_mutex);
                        for (auto& v : voxel_grid) v *= factor;
                    }
                    last_decay_time = now_steady;
                }
            }

            // Detect motion
            auto& prev_img = prev_frames[camera_index];
            MotionMask mm = detect_motion(prev_img, curr_img, motion_threshold);

            // --- Debug visualization: motion mask overlaid on camera feed ---
            if (gc.show_debug && mm.width > 0 && mm.height > 0) {
                cv::Mat overlay;
                // Resize current frame to a reasonable debug window size
                int dbg_w = std::min(mm.width, 640);
                float scale = float(dbg_w) / float(mm.width);
                int dbg_h = int(mm.height * scale);
                cv::resize(frame_data.frame, overlay, cv::Size(dbg_w, dbg_h));
                if (overlay.channels() == 1)
                    cv::cvtColor(overlay, overlay, cv::COLOR_GRAY2BGR);

                // Draw motion pixels as red overlay
                for (int row = 0; row < dbg_h; row++) {
                    for (int col = 0; col < dbg_w; col++) {
                        int src_row = int(row / scale);
                        int src_col = int(col / scale);
                        if (src_row < mm.height && src_col < mm.width) {
                            if (mm.changed[src_row * mm.width + src_col]) {
                                auto& pix = overlay.at<cv::Vec3b>(row, col);
                                pix[0] = 0;                          // B
                                pix[1] = 0;                          // G
                                pix[2] = std::min(255, pix[2] + 180); // R
                            }
                        }
                    }
                }

                // Count changed pixels for display
                int changed_count = 0;
                for (int i = 0; i < mm.width * mm.height; i++)
                    if (mm.changed[i]) changed_count++;
                std::string label = "Cam " + std::to_string(camera_index)
                    + " motion: " + std::to_string(changed_count) + " px";
                cv::putText(overlay, label, cv::Point(10, 25),
                            cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 0), 2);

                {
                    std::lock_guard<std::mutex> lock(debug_state.mutex);
                    debug_state.frames[camera_index] = overlay;
                }
            }

            if (mm.width > 0 && mm.height > 0) {
                // Get camera info
                const auto& cam_info = camera_info_map[camera_index];
                Vec3 cam_pos = cam_info.camera_position;
                Mat3 cam_rot = rotation_matrix_yaw_pitch_roll(cam_info.yaw, cam_info.pitch, cam_info.roll);

                float fov_h_rad = deg2rad(cam_info.fov_degrees);
                float fx = (mm.width * 0.5f) / std::tan(fov_h_rad * 0.5f);
                float aspect = float(mm.width) / float(mm.height);
                float fov_v_rad = 2.f * std::atan((1.f / aspect) * std::tan(fov_h_rad * 0.5f));
                float fy = (mm.height * 0.5f) / std::tan(fov_v_rad * 0.5f);

                bool any_voxel_written = false;
                {
                    std::lock_guard<std::mutex> lock(voxel_grid_mutex);

                    for (int v = 0; v < mm.height; v++) {
                        for (int u = 0; u < mm.width; u++) {
                            if (!mm.changed[v * mm.width + u])
                                continue;

                            float pix_val = mm.diff[v * mm.width + u];
                            if (pix_val < 1e-3f)
                                continue;

                            float x = (float(u) - 0.5f * mm.width)  / fx;
                            float y = -(float(v) - 0.5f * mm.height) / fy;
                            Vec3 ray_cam = { x, y, -1.f };
                            ray_cam = normalize(ray_cam);

                            Vec3 ray_world = mat3_mul_vec3(cam_rot, ray_cam);
                            ray_world = normalize(ray_world);

                            std::vector<RayStep> steps = cast_ray_into_grid(
                                cam_pos, ray_world, Nx, Ny, Nz, voxel_size, grid_center);

                            for (const auto& rs : steps) {
                                float dist = rs.distance;
                                float attenuation = 1.f / (1.f + alpha * dist);
                                float val = pix_val * attenuation;
                                int idx = rs.ix * Ny * Nz + rs.iy * Nz + rs.iz;
                                voxel_grid[idx] += val;
                                any_voxel_written = true;
                            }
                        }
                    }
                }

                if (any_voxel_written) {
                    auto t = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
                    std::cout << "[" << std::put_time(std::localtime(&t), "%H:%M:%S")
                              << "] Camera " << camera_index << " added voxels." << std::endl;
                }

#ifdef HAVE_ZMQ
                if (pub_socket && pub_socket->handle() != nullptr) {
                    std::lock_guard<std::mutex> grid_lock(voxel_grid_mutex);

                    struct Meta { int nx; int ny; int nz; float vs; } meta{Nx, Ny, Nz, voxel_size};

                    uLong src_len = voxel_grid.size() * sizeof(float);
                    uLong dst_len = compressBound(src_len);
                    std::vector<uint8_t> compressed(dst_len);
                    if (compress2(compressed.data(), &dst_len,
                                  reinterpret_cast<const Bytef*>(voxel_grid.data()),
                                  src_len, Z_BEST_SPEED) == Z_OK) {
                        compressed.resize(dst_len);

                        zmq::message_t m_meta(sizeof(meta));
                        memcpy(m_meta.data(), &meta, sizeof(meta));

                        zmq::message_t m_data(compressed.size());
                        memcpy(m_data.data(), compressed.data(), compressed.size());

                        pub_socket->send(m_meta, zmq::send_flags::sndmore);
                        pub_socket->send(m_data, zmq::send_flags::dontwait);
                    }
                }
#endif
            }

            prev_frames[camera_index] = curr_img;
            if (gc.show_debug) prev_raw_frames[camera_index] = frame_data.frame.clone();
            last_frame_times[camera_index] = frame_data.timestamp;
        }

        // Periodic voxel grid save
        auto now = std::chrono::system_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - last_save_time).count();

        if (elapsed >= gc.save_interval_seconds) {
            std::lock_guard<std::mutex> lock(voxel_grid_mutex);

            std::string output_bin = gc.output_dir + "/voxel_grid_" +
                std::to_string(std::chrono::system_clock::to_time_t(now)) + ".bin";

            std::ofstream ofs(output_bin, std::ios::binary);
            if (!ofs) {
                std::cerr << "Cannot open output file: " << output_bin << "\n";
            } else {
                ofs.write(reinterpret_cast<const char*>(&Nx), sizeof(int));
                ofs.write(reinterpret_cast<const char*>(&Ny), sizeof(int));
                ofs.write(reinterpret_cast<const char*>(&Nz), sizeof(int));
                ofs.write(reinterpret_cast<const char*>(&voxel_size), sizeof(float));
                ofs.write(reinterpret_cast<const char*>(voxel_grid.data()), voxel_grid.size() * sizeof(float));
                ofs.close();
                std::cout << "Saved voxel grid to " << output_bin << "\n";
            }

            last_save_time = now;
        }
    }
}

//----------------------------------------------
// 10) Save voxel grid helper
//----------------------------------------------
static void save_voxel_grid(
    const std::vector<float>& voxel_grid,
    std::mutex& voxel_grid_mutex,
    const GridConfig& gc,
    const std::string& tag)
{
    std::lock_guard<std::mutex> lock(voxel_grid_mutex);
    auto now = std::chrono::system_clock::now();
    std::string output_bin = gc.output_dir + "/voxel_grid_" + tag + "_" +
        std::to_string(std::chrono::system_clock::to_time_t(now)) + ".bin";

    std::ofstream ofs(output_bin, std::ios::binary);
    if (!ofs) {
        std::cerr << "Cannot open output file: " << output_bin << "\n";
        return;
    }
    ofs.write(reinterpret_cast<const char*>(&gc.Nx), sizeof(int));
    ofs.write(reinterpret_cast<const char*>(&gc.Ny), sizeof(int));
    ofs.write(reinterpret_cast<const char*>(&gc.Nz), sizeof(int));
    float vs = gc.voxel_size;
    ofs.write(reinterpret_cast<const char*>(&vs), sizeof(float));
    ofs.write(reinterpret_cast<const char*>(voxel_grid.data()), voxel_grid.size() * sizeof(float));
    ofs.close();
    std::cout << "Saved voxel grid to " << output_bin << "\n";
}

//----------------------------------------------
// 11) Main Function
//----------------------------------------------
int main(int argc, char** argv) {
    setenv("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp|stimeout;5000000|max_delay;0", 1);

    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <camera_config.json> [output_dir]\n";
        std::cerr << "  camera_config.json: JSON config with grid parameters and cameras\n";
        std::cerr << "  output_dir: Override output directory (default: from config or '.')\n";
        return 1;
    }

    std::string config_path = argv[1];

    //------------------------------------------
    // Load configuration
    //------------------------------------------
    AppConfig app = load_config(config_path);
    if (app.cameras.empty()) {
        std::cerr << "No cameras configured.\n";
        return 1;
    }

    GridConfig& gc = app.grid;

    // CLI override for output dir
    if (argc > 2) gc.output_dir = argv[2];

    // Create output directory
    std::string mkdir_cmd = "mkdir -p " + gc.output_dir;
    system(mkdir_cmd.c_str());

    std::map<int, CameraInfo> camera_info_map;
    for (const auto& camera : app.cameras) {
        camera_info_map[camera.camera_index] = camera;
    }

    std::cout << "Loaded " << app.cameras.size() << " camera(s)\n";
    std::cout << "Grid: " << gc.Nx << "x" << gc.Ny << "x" << gc.Nz
              << "  voxel_size=" << gc.voxel_size << "m"
              << "  center=(" << gc.center.x << "," << gc.center.y << "," << gc.center.z << ")"
              << "  decay=" << gc.decay_rate
              << "  debug=" << (gc.show_debug ? "ON" : "OFF") << "\n";
    std::cout << "Grid memory: "
              << (size_t(gc.Nx) * gc.Ny * gc.Nz * sizeof(float)) / (1024*1024) << " MB\n";

    //------------------------------------------
    // Create voxel grid
    //------------------------------------------
    std::vector<float> voxel_grid(size_t(gc.Nx) * gc.Ny * gc.Nz, 0.f);
    std::mutex voxel_grid_mutex;

    //------------------------------------------
    // Start threads
    //------------------------------------------
    std::atomic<bool> running(true);
    auto frame_queue = std::make_shared<FrameQueue>();

    std::vector<std::thread> camera_threads;
    for (const auto& camera : app.cameras) {
        camera_threads.emplace_back(camera_stream_thread,
                                   camera,
                                   frame_queue,
                                   std::ref(running));
    }

#ifdef HAVE_ZMQ
    zmq::context_t zmq_ctx(1);
    zmq::socket_t  zmq_pub(zmq_ctx, zmq::socket_type::pub);

    const char* portEnv = std::getenv("ZMQ_PORT");
    std::string port    = portEnv ? portEnv : "5556";
    std::string address = "tcp://127.0.0.1:" + port;

    try {
        zmq_pub.bind(address);
        std::cout << "ZeroMQ publisher bound to " << address << '\n';
    } catch(const zmq::error_t& e) {
        std::cerr << "ERROR: cannot bind ZeroMQ socket: " << e.what()
                  << "  (" << address << ")\n";
        return 2;
    }
#endif

    DebugState debug_state;

    std::thread processor(processing_thread,
                         camera_info_map,
                         frame_queue,
                         std::ref(voxel_grid_mutex),
                         std::ref(voxel_grid),
                         std::cref(gc),
                         std::ref(running),
#ifdef HAVE_ZMQ
                         &zmq_pub,
#else
                         nullptr,
#endif
                         std::ref(debug_state));

    std::cout << "Real-time processing started.\n";
    std::cout << "Press 'q' to quit, 's' to save current voxel grid.\n";
    if (gc.show_debug) {
        std::cout << "Debug windows enabled - press 'q' in any OpenCV window to quit.\n";
    }

    //------------------------------------------
    // Main loop: show debug windows + handle input
    //------------------------------------------
    if (gc.show_debug) {
        // GUI event loop: show debug frames and poll keyboard via OpenCV
        while (running) {
            {
                std::lock_guard<std::mutex> lock(debug_state.mutex);
                for (auto& [cam_id, frame] : debug_state.frames) {
                    if (!frame.empty()) {
                        cv::imshow("Camera " + std::to_string(cam_id) + " Motion", frame);
                    }
                }
            }
            int key = cv::waitKey(30);
            if (key == 'q' || key == 'Q') {
                running = false;
            } else if (key == 's' || key == 'S') {
                save_voxel_grid(voxel_grid, voxel_grid_mutex, gc, "manual");
            }
        }
        cv::destroyAllWindows();
    } else {
        // Text-only mode: read from stdin
        char cmd;
        while (running) {
            cmd = std::cin.get();
            if (cmd == 'q' || cmd == 'Q') {
                running = false;
            } else if (cmd == 's' || cmd == 'S') {
                save_voxel_grid(voxel_grid, voxel_grid_mutex, gc, "manual");
            }
        }
    }

    // Wait for threads to finish
    for (auto& thread : camera_threads) {
        if (thread.joinable()) thread.join();
    }
    if (processor.joinable()) processor.join();

    std::cout << "All threads stopped. Exiting.\n";
    return 0;
} 