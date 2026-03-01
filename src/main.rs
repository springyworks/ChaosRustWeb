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

/// Application entry point - configures Bevy app and launches the simulation
fn main() {
    // Install panic hook for better error messages in browser console
    #[cfg(target_arch = "wasm32")]
    std::panic::set_hook(Box::new(console_error_panic_hook::hook));

    App::new()
        .insert_resource(ClearColor(Color::srgb(0.02, 0.02, 0.03)))
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                title: "ChaosRust Web - 4D Ripple Tank".into(),
                canvas: Some("#bevy".into()),
                fit_canvas_to_parent: true,
                ..default()
            }),
            ..default()
        }))
        .insert_resource(PhysicsConfig::default())
        .insert_resource(SimStats::default())
        .insert_resource(CameraController::default())
        .add_systems(Startup, (setup_scene, setup_physics))
        .add_systems(Update, camera_controls)
        .add_systems(Update, physics_step)
        .add_systems(Update, pinger_system)
        .add_systems(Update, update_visuals)
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
            radius: 70.0,
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
) {
    // Spawn main camera with perspective projection
    commands.spawn((
        Camera3d::default(),
        Transform::from_xyz(0.0, 30.0, 70.0).looking_at(Vec3::ZERO, Vec3::Y),
        MainCamera,
    ));

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
    let mat1 = materials.add(StandardMaterial {
        base_color: Color::srgb(1.0, 0.2, 0.2),
        emissive: LinearRgba::rgb(0.3, 0.0, 0.0),
        ..default()
    });

    let mat2 = materials.add(StandardMaterial {
        base_color: Color::srgb(0.2, 1.0, 0.2),
        emissive: LinearRgba::rgb(0.0, 0.3, 0.0),
        ..default()
    });

    let mat3 = materials.add(StandardMaterial {
        base_color: Color::srgb(0.2, 0.2, 1.0),
        emissive: LinearRgba::rgb(0.0, 0.0, 0.3),
        ..default()
    });

    // Create 3D grid mesh (shared by all particles)
    let grid_mesh = meshes.add(Sphere::new(0.15).mesh().ico(2).unwrap());
    let dim = 12;
    let spacing = 1.5;
    let offset = (dim as f32 * spacing) / 2.0;

    // Spawn three tensor grids in a row
    for tensor_id in 0..3 {
        let (base_x, mat) = match tensor_id {
            0 => (-30.0, mat1.clone()),
            1 => (0.0, mat2.clone()),
            _ => (30.0, mat3.clone()),
        };

        // Create 12×12×12 = 1,728 particles per tensor
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
fn setup_physics(mut _config: ResMut<PhysicsConfig>) {
    #[cfg(target_arch = "wasm32")]
    {
        use web_sys::console;
        console::log_1(&"Physics system initialized for 3x12x12x12 wave grids".into());
    }
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
    mut stats: ResMut<SimStats>,
    _time: Res<Time>,
) {
    let dt = config.dt;
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
    for (mut mass_point, cell) in query.iter_mut() {
        if let Some(idx) = grid_map[cell.tensor_id][cell.x][cell.y][cell.z] {
            let accel = accelerations[idx];
            mass_point.velocity += accel * dt;
            mass_point.velocity *= config.damping;
            
            let vel = mass_point.velocity;
            mass_point.position += vel * dt;

            let speed = mass_point.velocity.length();
            max_vel = max_vel.max(speed);
            total_ke += 0.5 * mass_point.mass * speed.powi(2);
        }
    }

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
        let center = 6;

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

        #[cfg(target_arch = "wasm32")]
        {
            use web_sys::console;
            console::log_1(&format!("🔔 Ping on tensor {}", target_tensor).into());
        }
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
            "🌊 4D Ripple Tank (3x12³ grids)\n\
            Steps: {} | FPS: {:.0}\n\
            Wave Speed: {:.1} | Damping: {:.3}\n\
            Energy: {:.1} J (KE: {:.1}, PE: {:.1})\n\
            Max Velocity: {:.2}\n\
            \n\
            Controls:\n\
            Left Drag: Orbit | Right Drag: Pan\n\
            Scroll: Zoom | Space: Auto-Rotate",
            stats.steps,
            1.0 / time.delta_secs(),
            config.wave_speed,
            config.damping,
            total_e,
            stats.total_ke,
            stats.total_pe,
            stats.max_vel
        );
}
