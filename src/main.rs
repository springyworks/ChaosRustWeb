#![recursion_limit = "256"]
//! # ChaosRustWeb - 4D Wave Equation Ripple Tank
//!
//! A browser-based 3D visualization of a 4-dimensional wave propagation system using
//! Bevy game engine and Burn tensor library, compiled to WebAssembly.
//!
//! ## Overview
//!
//! This application simulates wave propagation across three coupled 3D tensor grids
//! (12×12×12 particles each) connected in a ring topology, creating a "4D" system where
//! waves can travel through spatial dimensions AND between tensor layers.
//!
//! ## Physics Model
//!
//! - **Wave Equation**: Solved using Finite Difference Time Domain (FDTD) method
//! - **Intra-tensor coupling**: Laplacian stencil operator for wave propagation
//! - **Inter-tensor coupling**: Ring topology (Tensor 0 → 1 → 2 → 0)
//! - **Pinger system**: Periodic impulses create observable wave phenomena
//!
//! ## Architecture
//!
//! - **Rendering**: Bevy ECS with WebGL2 backend
//! - **Physics**: Custom wave solver with ~5,184 particles
//! - **Platform**: WASM target for browser deployment via Trunk
//!
//! Adapted from ChaosRust wed04-1730.rs for web deployment.

use bevy::prelude::*;
use bevy::input::mouse::{MouseMotion, MouseWheel, MouseScrollUnit};
use bevy::input::touch::Touches;

/// Helper macro for logging to browser console on WASM
#[cfg(target_arch = "wasm32")]
macro_rules! web_log {
    ($($arg:tt)*) => {
        web_sys::console::log_1(&format!($($arg)*).into());
    }
}

#[cfg(not(target_arch = "wasm32"))]
macro_rules! web_log {
    ($($arg:tt)*) => {
        println!($($arg)*);
    }
}

/// Detect if running on a mobile device (via canvas/screen size heuristic)
#[cfg(target_arch = "wasm32")]
fn is_mobile() -> bool {
    let window = web_sys::window().unwrap();
    let screen = window.screen().unwrap();
    let w = screen.width().unwrap_or(1920);
    let h = screen.height().unwrap_or(1080);
    let min_dim = w.min(h);
    // Mobile screens are typically < 800px on the smaller dimension
    // Also check user agent for Android/iPhone
    let ua = window.navigator().user_agent().unwrap_or_default();
    let is_touch = ua.contains("Android") || ua.contains("iPhone") || ua.contains("iPad") || ua.contains("Mobile");
    web_log!("[ChaosRust] Screen: {}x{}, UA mobile: {}, min_dim: {}", w, h, is_touch, min_dim);
    is_touch || min_dim < 800
}

#[cfg(not(target_arch = "wasm32"))]
fn is_mobile() -> bool {
    false
}

/// Application entry point - configures Bevy app and launches the simulation
fn main() {
    // Install panic hook for better error messages in browser console
    #[cfg(target_arch = "wasm32")]
    std::panic::set_hook(Box::new(console_error_panic_hook::hook));

    web_log!("[ChaosRust] ===== App Starting =====");
    web_log!("[ChaosRust] Detecting platform...");

    let mobile = is_mobile();
    let grid_dim = if mobile { 6 } else { 12 };
    web_log!("[ChaosRust] Mobile: {}, Grid dim: {} ({}³ = {} particles per tensor, {} total)",
        mobile, grid_dim, grid_dim, grid_dim*grid_dim*grid_dim, grid_dim*grid_dim*grid_dim*3);

    web_log!("[ChaosRust] Configuring Bevy plugins...");

    let mut physics = PhysicsConfig::default();
    physics.dim = grid_dim;
    if mobile {
        // Reduce physics intensity for mobile
        physics.pinger_strength = 1500.0;
    }

    // On mobile with high DPR, fit_canvas_to_parent causes surface > GPU max texture (4096).
    // Instead, compute a safe logical resolution from viewport dimensions such that
    // logical × DPR < 4096 for both dimensions.
    #[cfg(target_arch = "wasm32")]
    let (fit_canvas, resolution) = if mobile {
        let window = web_sys::window().unwrap();
        let vw = window.inner_width().unwrap().as_f64().unwrap_or(980.0) as f32;
        let vh = window.inner_height().unwrap().as_f64().unwrap_or(1811.0) as f32;
        let dpr = window.device_pixel_ratio() as f32;
        let max_safe = 4000.0 / dpr; // stay under GPU max 4096
        let w = vw.min(max_safe);
        let h = vh.min(max_safe);
        web_log!("[ChaosRust] Safe resolution: {}x{} (viewport {}x{}, DPR {}, max_safe {})",
            w, h, vw, vh, dpr, max_safe);
        (false, bevy::window::WindowResolution::new(w, h))
    } else {
        (true, bevy::window::WindowResolution::default())
    };
    #[cfg(not(target_arch = "wasm32"))]
    let (fit_canvas, resolution) = (true, bevy::window::WindowResolution::default());

    App::new()
        .insert_resource(ClearColor(Color::srgb(0.02, 0.02, 0.03)))
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                title: "ChaosRust Web - 4D Ripple Tank".into(),
                canvas: Some("#bevy".into()),
                fit_canvas_to_parent: fit_canvas,
                prevent_default_event_handling: false,
                resolution,
                ..default()
            }),
            ..default()
        }))
        .insert_resource(physics)
        .insert_resource(SimStats::default())
        .insert_resource(CameraController::default())
        .insert_resource(TouchState::default())
        .insert_resource(SimSpeed::default())
        .insert_resource(PendingTap::default())
        .insert_resource(TensorEnergy::default())
        .add_systems(Startup, (setup_scene, setup_physics))
        .add_systems(Update, camera_controls)
        .add_systems(Update, touch_camera_controls)
        .add_systems(Update, read_ui_sliders)
        .add_systems(Update, physics_step)
        .add_systems(Update, pinger_system)
        .add_systems(Update, update_visuals)
        .add_systems(Update, tap_shoot_system)
        .add_systems(Update, audio_bridge_system)
        .add_systems(Update, update_stats_ui)
        .run();
}

/// Physics simulation configuration controlling wave behavior and coupling
#[derive(Resource)]
struct PhysicsConfig {
    /// Time step for numerical integration (seconds)
    dt: f32,
    /// Wave propagation speed (controls frequency response)
    wave_speed: f32,
    /// Energy loss factor per frame (0.0 = instant decay, 1.0 = no decay)
    damping: f32,
    /// Strength of coupling between adjacent tensor grids (4D connection)
    inter_coupling: f32,
    /// Restoring force pulling particles back to grid positions
    anchor_strength: f32,
    /// Whether the pinger system is active
    pinger_active: bool,
    /// Internal timer for pinger system
    pinger_timer: f32,
    /// Time between ping impulses (seconds)
    pinger_interval: f32,
    /// Magnitude of velocity impulse applied by pinger
    pinger_strength: f32,
    /// Grid dimension per axis (12 = 12×12×12 = 1,728 particles per tensor)
    dim: usize,
}

impl Default for PhysicsConfig {
    fn default() -> Self {
        Self {
            dt: 0.008,           // 8ms time step
            wave_speed: 17.0,    // Tuned for visible wave propagation
            damping: 0.95,       // 5% energy loss per frame
            inter_coupling: 150.0, // Strong 4D coupling
            anchor_strength: 0.5,  // Weak anchoring allows large swings
            pinger_active: true,
            pinger_timer: 0.0,
            pinger_interval: 2.0,  // Ping every 2 seconds
            pinger_strength: 3000.0, // Large impulse for dramatic effect
            dim: 12,
        }
    }
}

/// Mass-spring-damper particle with position, velocity, and physics state
#[derive(Component, Clone)]
struct MassPoint {
    /// Current 3D position in world space
    position: Vec3,
    /// Current velocity vector
    velocity: Vec3,
    /// Particle mass (affects inertia)
    mass: f32,
    /// Which tensor grid this particle belongs to (0, 1, or 2)
    tensor_id: usize,
    /// Grid coordinates within the tensor (x, y, z ∈ [0, 11])
    grid_index: (usize, usize, usize),
    /// Rest position in grid (used for anchor forces)
    base_pos: Vec3,
}

/// Marker component for identifying grid cells and their tensor membership
#[derive(Component)]
struct GridCell {
    /// Tensor ID (0 = red/left, 1 = green/center, 2 = blue/right)
    tensor_id: usize,
    /// X coordinate in grid [0, 11]
    x: usize,
    /// Y coordinate in grid [0, 11]
    y: usize,
    /// Z coordinate in grid [0, 11]
    z: usize,
}

/// Camera controller for interactive navigation
#[derive(Resource)]
struct CameraController {
    /// Orbital angle around Y axis (radians)
    angle: f32,
    /// Pitch angle (elevation)
    pitch: f32,
    /// Distance from focus point
    radius: f32,
    /// Point camera orbits around
    focus: Vec3,
    /// Auto-rotation speed (radians/sec)
    auto_rotate_speed: f32,
    /// Whether auto-rotation is enabled
    auto_rotate: bool,
}

impl Default for CameraController {
    fn default() -> Self {
        Self {
            angle: 0.0,
            pitch: 0.52, // ~30 degrees
            radius: 45.0, // Closer default for better visibility on mobile
            focus: Vec3::ZERO,
            auto_rotate_speed: 0.2,
            auto_rotate: true,
        }
    }
}

/// Marker component for the controllable camera
#[derive(Component)]
struct MainCamera;

/// Marker component for the stats text UI element
#[derive(Component)]
struct StatsText;

/// State tracking for two-finger touch gestures
#[derive(Resource, Default)]
struct TouchState {
    /// Previous distance between two touches (for pinch zoom)
    prev_pinch_dist: Option<f32>,
    /// Previous midpoint of two touches (for two-finger pan)
    prev_midpoint: Option<Vec2>,
}

/// Simulation speed multiplier controlled by UI slider
#[derive(Resource)]
struct SimSpeed(f32);

impl Default for SimSpeed {
    fn default() -> Self {
        Self(1.0)
    }
}

/// Stores pending tap screen position for raycast shooting
#[derive(Resource, Default)]
struct PendingTap(Option<Vec2>);

/// Per-tensor kinetic energy for audio bridge
#[derive(Resource, Default)]
struct TensorEnergy([f32; 3]);

/// Runtime simulation statistics for display and monitoring
#[derive(Resource, Default)]
struct SimStats {
    /// Total simulation steps executed
    steps: usize,
    /// Total kinetic energy in system (Joules)
    total_ke: f32,
    /// Total potential energy in system (Joules)
    total_pe: f32,
    /// Maximum particle velocity this frame (for debugging)
    max_vel: f32,
}

/// Sets up the 3D scene with camera, lights, and particle grids
///
/// Creates:
/// - 1 perspective camera
/// - 2 lights (directional + point)
/// - 3 tensor grids (red, green, blue) of 12×12×12 spheres each
/// - UI text overlay for statistics
fn setup_scene(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    config: Res<PhysicsConfig>,
) {
    web_log!("[ChaosRust] setup_scene: Starting scene setup...");

    // Spawn main camera with perspective projection
    commands.spawn((
        Camera3d::default(),
        Transform::from_xyz(0.0, 20.0, 45.0).looking_at(Vec3::ZERO, Vec3::Y),
        MainCamera,
    ));
    web_log!("[ChaosRust] setup_scene: Camera spawned");

    // Directional light (sun-like, illuminates entire scene)
    commands.spawn((
        DirectionalLight {
            illuminance: 15000.0,
            shadows_enabled: false,
            ..default()
        },
        Transform::from_xyz(5.0, 10.0, 5.0).looking_at(Vec3::ZERO, Vec3::Y),
    ));

    // Point light for additional fill lighting
    commands.spawn((
        PointLight {
            intensity: 30000.0,
            range: 1000.0,
            shadows_enabled: false,
            ..default()
        },
        Transform::from_xyz(50.0, 50.0, 50.0),
    ));

    // Create materials for the three tensor grids
    // Red (left), Green (center), Blue (right)
    // High emissive values ensure visibility even with limited WebGL2 lighting
    let mat1 = materials.add(StandardMaterial {
        base_color: Color::srgb(1.0, 0.3, 0.3),
        emissive: LinearRgba::rgb(2.0, 0.2, 0.2),
        unlit: false,
        ..default()
    });

    let mat2 = materials.add(StandardMaterial {
        base_color: Color::srgb(0.3, 1.0, 0.3),
        emissive: LinearRgba::rgb(0.2, 2.0, 0.2),
        unlit: false,
        ..default()
    });

    let mat3 = materials.add(StandardMaterial {
        base_color: Color::srgb(0.3, 0.3, 1.0),
        emissive: LinearRgba::rgb(0.2, 0.2, 2.0),
        unlit: false,
        ..default()
    });

    web_log!("[ChaosRust] setup_scene: Lights spawned, creating grid mesh...");

    // Create 3D grid mesh (shared by all particles)
    // ico(1) = 42 verts per sphere — light enough for mobile GPUs
    let dim = config.dim;
    let sphere_detail = if dim <= 6 { 1 } else { 1 }; // ico subdivision level
    let sphere_radius = if dim <= 6 { 0.4 } else { 0.15 }; // larger spheres for smaller grids
    let grid_mesh = meshes.add(Sphere::new(sphere_radius).mesh().ico(sphere_detail).unwrap());
    let spacing = if dim <= 6 { 2.5 } else { 1.5 };
    let offset = (dim as f32 * spacing) / 2.0;
    web_log!("[ChaosRust] setup_scene: Grid config: dim={}, spacing={}, sphere_r={}", dim, spacing, sphere_radius);

    // Spawn three tensor grids in a row
    for tensor_id in 0..3 {
        let (base_x, mat) = match tensor_id {
            0 => (-30.0, mat1.clone()),
            1 => (0.0, mat2.clone()),
            _ => (30.0, mat3.clone()),
        };

        // Create dim³ particles per tensor
        web_log!("[ChaosRust] setup_scene: Spawning tensor {} ({} particles)...", tensor_id, dim*dim*dim);
        for x in 0..dim {
            for y in 0..dim {
                for z in 0..dim {
                    let pos = Vec3::new(
                        base_x + (x as f32 * spacing) - offset,
                        (y as f32 * spacing) - offset,
                        (z as f32 * spacing) - offset,
                    );

                    commands.spawn((
                        Mesh3d(grid_mesh.clone()),
                        MeshMaterial3d(mat.clone()),
                        Transform::from_translation(pos),
                        MassPoint {
                            position: pos,
                            velocity: Vec3::ZERO,
                            mass: 1.0,
                            tensor_id,
                            grid_index: (x, y, z),
                            base_pos: pos,
                        },
                        GridCell {
                            tensor_id,
                            x,
                            y,
                            z,
                        },
                    ));
                }
            }
        }
    }

    web_log!("[ChaosRust] setup_scene: All {} particles spawned!", dim*dim*dim*3);

    // UI text overlay showing simulation stats
    commands.spawn((
        Text::new("Initializing..."),
        TextFont {
            font_size: 18.0,
            ..default()
        },
        TextColor(Color::WHITE),
        Node {
            position_type: PositionType::Absolute,
            top: Val::Px(10.0),
            left: Val::Px(10.0),
            ..default()
        },
        StatsText,
    ));
}

/// Initializes physics system (currently minimal setup)
fn setup_physics(config: Res<PhysicsConfig>) {
    let dim = config.dim;
    web_log!("[ChaosRust] Physics system initialized for 3x{}x{}x{} wave grids ({} total particles)",
        dim, dim, dim, dim*dim*dim*3);
    web_log!("[ChaosRust] Physics config: dt={}, wave_speed={}, damping={}, inter_coupling={}",
        config.dt, config.wave_speed, config.damping, config.inter_coupling);
}

/// Main physics simulation step using wave equation solver
///
/// Implements:
/// 1. Laplacian operator for wave propagation within each tensor
/// 2. Inter-tensor coupling forces (4D connectivity)
/// 3. Anchor forces (restoring to grid positions)
/// 4. Damping (energy dissipation)
/// 5. Numerical integration (velocity Verlet-like)
///
/// Runs at ~60 FPS, computing forces for all 5,184 particles each frame
fn physics_step(
    mut query: Query<(&mut MassPoint, &GridCell)>,
    config: Res<PhysicsConfig>,
    sim_speed: Res<SimSpeed>,
    mut stats: ResMut<SimStats>,
    mut tensor_energy: ResMut<TensorEnergy>,
    _time: Res<Time>,
) {
    let dt = config.dt * sim_speed.0;
    let c2 = config.wave_speed.powi(2);
    let coupling = config.inter_coupling;
    let dim = config.dim;

    // Build spatial lookup structure for fast neighbor queries
    // grid_map[tensor_id][x][y][z] = particle_index
    let mut grid_map = vec![vec![vec![vec![None; dim]; dim]; dim]; 3];
    let mut particles = Vec::with_capacity(dim * dim * dim * 3);

    for (mass_point, cell) in query.iter() {
        grid_map[cell.tensor_id][cell.x][cell.y][cell.z] = Some(particles.len());
        particles.push((
            mass_point.position,
            mass_point.velocity,
            mass_point.base_pos,
            mass_point.mass,
        ));
    }

    let mut accelerations = vec![Vec3::ZERO; particles.len()];
    let mut total_ke = 0.0;
    let mut total_pe = 0.0;
    let mut max_vel: f32 = 0.0;

    // Compute forces for all particles using wave equation
    for t in 0..3 {
        for x in 0..dim {
            for y in 0..dim {
                for z in 0..dim {
                    if let Some(idx) = grid_map[t][x][y][z] {
                        let (pos, vel, base, mass) = particles[idx];
                        let disp = pos - base;

                        // Laplacian operator: ∇²ψ = Σ(neighbor_disp - current_disp)
                        // Implements discrete wave equation: ∂²ψ/∂t² = c² ∇²ψ
                        let mut laplacian = Vec3::ZERO;

                        let neighbors = [
                            (x.wrapping_sub(1), y, z, x > 0),
                            (x + 1, y, z, x < dim - 1),
                            (x, y.wrapping_sub(1), z, y > 0),
                            (x, y + 1, z, y < dim - 1),
                            (x, y, z.wrapping_sub(1), z > 0),
                            (x, y, z + 1, z < dim - 1),
                        ];

                        for (nx, ny, nz, valid) in neighbors {
                            if valid {
                                if let Some(nidx) = grid_map[t][nx][ny][nz] {
                                    let n_disp = particles[nidx].0 - particles[nidx].2;
                                    laplacian += n_disp - disp;
                                }
                            } else {
                                // Fixed wall boundary condition
                                laplacian -= disp;
                            }
                        }

                        // Wave equation force: F = c² ∇²ψ
                        let wave_force = laplacian * c2;

                        // Inter-tensor coupling (4D ring topology: 0→1→2→0)
                        let next_t = (t + 1) % 3;
                        let prev_t = (t + 2) % 3;

                        let mut coupling_force = Vec3::ZERO;
                        if let Some(next_idx) = grid_map[next_t][x][y][z] {
                            let next_disp = particles[next_idx].0 - particles[next_idx].2;
                            coupling_force += (next_disp - disp) * coupling;
                        }
                        if let Some(prev_idx) = grid_map[prev_t][x][y][z] {
                            let prev_disp = particles[prev_idx].0 - particles[prev_idx].2;
                            coupling_force += (prev_disp - disp) * coupling;
                        }

                        // Anchor force (Klein-Gordon-like term)
                        let anchor_force = -disp * config.anchor_strength;

                        // Damping force (energy dissipation)
                        let damp_force = -vel * (1.0 - config.damping);

                        let total_force = wave_force + coupling_force + anchor_force + damp_force;
                        accelerations[idx] = total_force / mass;

                        // Accumulate potential energy
                        total_pe += 0.5 * c2 * disp.length_squared();
                    }
                }
            }
        }
    }

    // Integrate equations of motion (Euler method)
    let mut tensor_ke = [0.0f32; 3];
    for (mut mass_point, cell) in query.iter_mut() {
        if let Some(idx) = grid_map[cell.tensor_id][cell.x][cell.y][cell.z] {
            let accel = accelerations[idx];
            mass_point.velocity += accel * dt;
            mass_point.velocity *= config.damping;
            
            let vel = mass_point.velocity;
            mass_point.position += vel * dt;

            let speed = mass_point.velocity.length();
            max_vel = max_vel.max(speed);
            let ke = 0.5 * mass_point.mass * speed.powi(2);
            total_ke += ke;
            tensor_ke[cell.tensor_id] += ke;
        }
    }

    tensor_energy.0 = tensor_ke;
    stats.steps += 1;
    stats.total_ke = total_ke;
    stats.total_pe = total_pe;
    stats.max_vel = max_vel;
}

/// Pinger system: Periodically injects energy into the system
///
/// Creates impulses in the center of each tensor grid in sequence,
/// allowing observation of wave propagation and inter-tensor coupling.
fn pinger_system(
    mut query: Query<&mut MassPoint>,
    mut config: ResMut<PhysicsConfig>,
    time: Res<Time>,
) {
    if !config.pinger_active {
        return;
    }

    config.pinger_timer += time.delta_secs();

    if config.pinger_timer >= config.pinger_interval {
        config.pinger_timer = 0.0;

        // Rotate through tensors: 0 -> 1 -> 2 -> 0
        let target_tensor = ((time.elapsed_secs() / config.pinger_interval) as usize) % 3;
        let center = (config.dim / 2) as i32;

        // Apply radial impulse from center of target tensor
        for mut mass_point in query.iter_mut() {
            if mass_point.tensor_id == target_tensor {
                let x = mass_point.grid_index.0 as i32;
                let y = mass_point.grid_index.1 as i32;
                let z = mass_point.grid_index.2 as i32;

                let dist_sq = (x - center).pow(2) + (y - center).pow(2) + (z - center).pow(2);

                // Apply impulse with radial falloff (3D Gaussian-like kernel)
                if dist_sq <= 4 {
                    let factor = 1.0 - (dist_sq as f32 / 5.0);
                    mass_point.velocity += Vec3::Y * config.pinger_strength * factor;
                }
            }
        }

        web_log!("🔔 Ping on tensor {}", target_tensor);
    }
}

/// Updates visual representation to match physics state
///
/// - Synchronizes mesh transforms with particle positions
/// - Scales particles based on velocity (faster = bigger)
fn update_visuals(mut query: Query<(&mut Transform, &MassPoint)>) {
    for (mut transform, mass_point) in query.iter_mut() {
        transform.translation = mass_point.position;

        // Dynamic scaling: faster particles appear larger
        let speed = mass_point.velocity.length();
        let scale = 0.5 + (speed * 0.05).min(1.5);
        transform.scale = Vec3::splat(scale);
    }
}

/// Interactive camera controls with mouse support
///
/// Features:
/// - Auto-orbit: Automatic rotation around scene
/// - Left drag: Manual orbit control
/// - Right drag: Pan camera focus point
/// - Scroll: Zoom in/out
/// - Space: Toggle auto-rotation
fn camera_controls(
    time: Res<Time>,
    mut cam_controller: ResMut<CameraController>,
    mut query: Query<&mut Transform, With<MainCamera>>,
    mouse_buttons: Res<ButtonInput<MouseButton>>,
    mut mouse_motion: EventReader<MouseMotion>,
    mut mouse_wheel: EventReader<MouseWheel>,
    keyboard: Res<ButtonInput<KeyCode>>,
    mut pending_tap: ResMut<PendingTap>,
    windows: Query<&Window>,
) {
    if keyboard.just_pressed(KeyCode::Space) {
        cam_controller.auto_rotate = !cam_controller.auto_rotate;
    }

    // Auto-rotation
    if cam_controller.auto_rotate {
        cam_controller.angle += time.delta_secs() * cam_controller.auto_rotate_speed;
    }

    // Manual camera controls
    let mut delta = Vec2::ZERO;
    for motion in mouse_motion.read() {
        delta += motion.delta;
    }

    // Mouse click without drag = tap to shoot
    if mouse_buttons.just_released(MouseButton::Left) && delta.length() < 2.0 {
        let win = windows.single();
        if let Some(cursor) = win.cursor_position() {
            pending_tap.0 = Some(cursor);
        }
    }

    // Orbit control (left mouse button)
    if mouse_buttons.pressed(MouseButton::Left) {
        cam_controller.angle -= delta.x * 0.005;
        cam_controller.pitch = (cam_controller.pitch - delta.y * 0.005).clamp(-1.5, 1.5);
        cam_controller.auto_rotate = false; // Disable auto-rotate when manually controlling
    }

    // Pan control (right mouse button)
    if mouse_buttons.pressed(MouseButton::Right) {
        // Calculate right and up vectors
        let rot = Quat::from_rotation_y(cam_controller.angle) * Quat::from_rotation_x(cam_controller.pitch);
        let right = rot * Vec3::X;
        let up = Vec3::Y;
        
        cam_controller.focus -= right * delta.x * 0.1;
        cam_controller.focus += up * delta.y * 0.1;
    }

    // Zoom control (mouse wheel)
    for wheel in mouse_wheel.read() {
        let zoom_delta = match wheel.unit {
            MouseScrollUnit::Line => wheel.y * 3.0,
            MouseScrollUnit::Pixel => wheel.y * 0.1,
        };
        cam_controller.radius = (cam_controller.radius - zoom_delta).clamp(20.0, 150.0);
    }

    // Update camera transform
    let mut transform = query.single_mut();
    let rot = Quat::from_rotation_y(cam_controller.angle) * Quat::from_rotation_x(cam_controller.pitch);
    transform.translation = cam_controller.focus + rot * Vec3::new(0.0, 0.0, cam_controller.radius);
    transform.look_at(cam_controller.focus, Vec3::Y);
}

/// Touch-based camera controls for mobile devices
///
/// - One finger drag: orbit camera
/// - Two finger pinch: zoom in/out
/// - Two finger drag: pan camera
/// - Single tap: shoot impulse at nearest particle
fn touch_camera_controls(
    touches: Res<Touches>,
    mut cam: ResMut<CameraController>,
    mut touch_state: ResMut<TouchState>,
    mut pending_tap: ResMut<PendingTap>,
) {
    // Detect single-finger tap (just released with minimal movement)
    for t in touches.iter_just_released() {
        let start = t.start_position();
        let end = t.position();
        if start.distance(end) < 10.0 {
            pending_tap.0 = Some(end);
        }
    }

    let active: Vec<_> = touches.iter().collect();

    match active.len() {
        1 => {
            // Single finger: orbit
            let t = active[0];
            let delta = t.delta();
            if delta.length() > 0.0 {
                cam.angle -= delta.x * 0.008;
                cam.pitch = (cam.pitch - delta.y * 0.008).clamp(-1.5, 1.5);
                cam.auto_rotate = false;
            }
            // Reset two-finger state
            touch_state.prev_pinch_dist = None;
            touch_state.prev_midpoint = None;
        }
        2 => {
            let p0 = active[0].position();
            let p1 = active[1].position();
            let dist = p0.distance(p1);
            let mid = (p0 + p1) * 0.5;

            // Pinch zoom
            if let Some(prev_dist) = touch_state.prev_pinch_dist {
                let zoom_delta = (dist - prev_dist) * 0.15;
                cam.radius = (cam.radius - zoom_delta).clamp(10.0, 150.0);
            }

            // Two-finger pan
            if let Some(prev_mid) = touch_state.prev_midpoint {
                let pan_delta = mid - prev_mid;
                let rot = Quat::from_rotation_y(cam.angle) * Quat::from_rotation_x(cam.pitch);
                let right = rot * Vec3::X;
                cam.focus -= right * pan_delta.x * 0.08;
                cam.focus += Vec3::Y * pan_delta.y * 0.08;
            }

            touch_state.prev_pinch_dist = Some(dist);
            touch_state.prev_midpoint = Some(mid);
            cam.auto_rotate = false;
        }
        _ => {
            touch_state.prev_pinch_dist = None;
            touch_state.prev_midpoint = None;
        }
    }
}

/// Reads slider values from JS globals and applies them to physics config
#[allow(unused_variables, unused_mut)]
fn read_ui_sliders(
    mut config: ResMut<PhysicsConfig>,
    mut sim_speed: ResMut<SimSpeed>,
) {
    #[cfg(target_arch = "wasm32")]
    {
        use wasm_bindgen::JsValue;
        let window = web_sys::window().unwrap();

        fn read_f64(window: &web_sys::Window, key: &str) -> Option<f64> {
            js_sys::Reflect::get(window, &JsValue::from_str(key))
                .ok()
                .and_then(|v| v.as_f64())
        }

        // Damping: slider 0 = no decay (1.00), slider 100 = max decay (0.90)
        if let Some(f) = read_f64(&window, "__chaos_damping") {
            config.damping = 1.00 - (f as f32 / 100.0) * 0.10;
        }
        // Pulse interval: 0..100 → 5.0..0.3 seconds
        if let Some(f) = read_f64(&window, "__chaos_pulse") {
            config.pinger_interval = 0.3 + (1.0 - f as f32 / 100.0) * 4.7;
        }
        // Sim speed: 0..100 → 0.1..3.0 multiplier
        if let Some(f) = read_f64(&window, "__chaos_speed") {
            sim_speed.0 = 0.1 + (f as f32 / 100.0) * 2.9;
        }
        // Wave speed: 0..100 → 5..35
        if let Some(f) = read_f64(&window, "__chaos_wavespd") {
            config.wave_speed = 5.0 + (f as f32 / 100.0) * 30.0;
        }
        // Coupling: 0..100 → 10..300
        if let Some(f) = read_f64(&window, "__chaos_coupling") {
            config.inter_coupling = 10.0 + (f as f32 / 100.0) * 290.0;
        }
        // Pulse strength: 0..100 → 500..6000
        if let Some(f) = read_f64(&window, "__chaos_pstr") {
            config.pinger_strength = 500.0 + (f as f32 / 100.0) * 5500.0;
        }
    }
}

/// Updates on-screen statistics display
///
/// Shows:
/// - Simulation steps and FPS
/// - Wave speed and damping parameters
/// - Energy metrics (kinetic, potential, total)
/// - Maximum particle velocity
fn update_stats_ui(
    stats: Res<SimStats>,
    config: Res<PhysicsConfig>,
    time: Res<Time>,
    mut query: Query<&mut Text, With<StatsText>>,
) {
    let mut text = query.single_mut();
    let total_e = stats.total_ke + stats.total_pe;
        **text = format!(
            "🌊 4D Ripple Tank (3x{}³)\n\
            Steps: {} | FPS: {:.0}\n\
            Wave: {:.1} | Damp: {:.3} | Coup: {:.0}\n\
            E: {:.1} J (KE: {:.1}, PE: {:.1})\n\
            Vmax: {:.2}",
            config.dim,
            stats.steps,
            1.0 / time.delta_secs(),
            config.wave_speed,
            config.damping,
            config.inter_coupling,
            total_e,
            stats.total_ke,
            stats.total_pe,
            stats.max_vel
        );
}

/// Raycast from screen tap position to find and shoot nearest particle
fn tap_shoot_system(
    mut pending_tap: ResMut<PendingTap>,
    camera_q: Query<(&Camera, &GlobalTransform), With<MainCamera>>,
    mut particles: Query<&mut MassPoint>,
) {
    let screen_pos = match pending_tap.0.take() {
        Some(p) => p,
        None => return,
    };

    let (camera, cam_gt) = camera_q.single();
    let Ok(ray) = camera.viewport_to_world(cam_gt, screen_pos) else {
        return;
    };

    // Find nearest particle to ray
    let mut best_dist = f32::MAX;
    let mut best_entity = None;
    let ray_origin = ray.origin;
    let ray_dir = ray.direction.as_vec3();

    for mass_point in particles.iter() {
        let to_point = mass_point.position - ray_origin;
        let t = to_point.dot(ray_dir);
        if t < 0.0 { continue; } // behind camera
        let closest = ray_origin + ray_dir * t;
        let dist = (mass_point.position - closest).length();
        if dist < best_dist {
            best_dist = dist;
            best_entity = Some(mass_point.position);
        }
    }

    // If hit is close enough (within ~3 units of ray), apply impulse
    if best_dist < 3.0 {
        if let Some(hit_pos) = best_entity {
            let impulse_dir = (hit_pos - ray_origin).normalize();
            let impulse = impulse_dir * 2000.0;
            for mut mp in particles.iter_mut() {
                let dist = (mp.position - hit_pos).length();
                if dist < 4.0 {
                    let falloff = 1.0 - (dist / 4.0);
                    mp.velocity += impulse * falloff;
                }
            }
            web_log!("🎯 Tap-shoot hit at {:?}, dist={:.2}", hit_pos, best_dist);

            // Trigger shoot sound in JS
            #[cfg(target_arch = "wasm32")]
            {
                use wasm_bindgen::JsCast;
                let window = web_sys::window().unwrap();
                if let Ok(func) = js_sys::Reflect::get(&window, &wasm_bindgen::JsValue::from_str("__chaos_shoot_sound")) {
                    if let Some(f) = func.dyn_ref::<js_sys::Function>() {
                        let _ = f.call0(&wasm_bindgen::JsValue::NULL);
                    }
                }
            }
        }
    }
}

/// Writes per-tensor kinetic energy and camera angle to JS for audio engine
fn audio_bridge_system(
    particles: Query<&MassPoint>,
    cam: Res<CameraController>,
    tensor_energy: Res<TensorEnergy>,
) {
    #[cfg(target_arch = "wasm32")]
    {
        use wasm_bindgen::JsValue;
        let window = web_sys::window().unwrap();

        // Write tensor KE array
        let ke = &tensor_energy.0;
        let arr = js_sys::Array::new_with_length(3);
        arr.set(0, JsValue::from_f64(ke[0] as f64));
        arr.set(1, JsValue::from_f64(ke[1] as f64));
        arr.set(2, JsValue::from_f64(ke[2] as f64));
        let _ = js_sys::Reflect::set(&window, &JsValue::from_str("__chaos_tensor_ke"), &arr);

        // Write camera angle
        let _ = js_sys::Reflect::set(
            &window,
            &JsValue::from_str("__chaos_cam_angle"),
            &JsValue::from_f64(cam.angle as f64),
        );
    }
    let _ = (&particles, &cam, &tensor_energy); // suppress unused warnings on non-wasm
}
