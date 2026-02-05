use bevy::prelude::*;

fn main() {
    #[cfg(target_arch = "wasm32")]
    std::panic::set_hook(Box::new(console_error_panic_hook::hook));

    App::new()
        .insert_resource(ClearColor(Color::srgb(0.1, 0.1, 0.15)))
        .insert_resource(AmbientLight {
            color: Color::WHITE,
            brightness: 1000.0,
        })
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                title: "Bevy Web Test".into(),
                canvas: Some("#bevy".into()),
                fit_canvas_to_parent: true,
                ..default()
            }),
            ..default()
        }))
        .add_systems(Startup, setup)
        .add_systems(Update, (rotate_objects, rotate_camera))
        .run();
}

#[derive(Component)]
struct Rotator {
    speed: f32,
}

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    // Camera
    commands.spawn(Camera3dBundle {
        transform: Transform::from_xyz(0.0, 5.0, 10.0).looking_at(Vec3::ZERO, Vec3::Y),
        ..default()
    });

    // Directional Light
    commands.spawn(DirectionalLightBundle {
        directional_light: DirectionalLight {
            illuminance: 20000.0,
            shadows_enabled: false,
            ..default()
        },
        transform: Transform::from_xyz(4.0, 8.0, 4.0).looking_at(Vec3::ZERO, Vec3::Y),
        ..default()
    });

    // Red Cube
    commands.spawn((
        PbrBundle {
            mesh: meshes.add(Cuboid::new(1.0, 1.0, 1.0)),
            material: materials.add(StandardMaterial {
                base_color: Color::srgb(1.0, 0.2, 0.2),
                emissive: Color::srgb(0.3, 0.0, 0.0).into(),
                ..default()
            }),
            transform: Transform::from_xyz(-2.5, 0.0, 0.0),
            ..default()
        },
        Rotator { speed: 1.0 },
    ));

    // Green Sphere
    commands.spawn((
        PbrBundle {
            mesh: meshes.add(Sphere::new(0.6)),
            material: materials.add(StandardMaterial {
                base_color: Color::srgb(0.2, 1.0, 0.2),
                emissive: Color::srgb(0.0, 0.3, 0.0).into(),
                ..default()
            }),
            transform: Transform::from_xyz(0.0, 0.0, 0.0),
            ..default()
        },
        Rotator { speed: -1.5 },
    ));

    // Blue Cube
    commands.spawn((
        PbrBundle {
            mesh: meshes.add(Cuboid::new(1.0, 1.0, 1.0)),
            material: materials.add(StandardMaterial {
                base_color: Color::srgb(0.2, 0.2, 1.0),
                emissive: Color::srgb(0.0, 0.0, 0.3).into(),
                ..default()
            }),
            transform: Transform::from_xyz(2.5, 0.0, 0.0),
            ..default()
        },
        Rotator { speed: 0.8 },
    ));

    // Ground plane
    commands.spawn(PbrBundle {
        mesh: meshes.add(Plane3d::default().mesh().size(10.0, 10.0)),
        material: materials.add(StandardMaterial {
            base_color: Color::srgb(0.3, 0.3, 0.4),
            ..default()
        }),
        transform: Transform::from_xyz(0.0, -1.0, 0.0),
        ..default()
    });
}

fn rotate_objects(
    time: Res<Time>,
    mut query: Query<(&mut Transform, &Rotator)>,
) {
    for (mut transform, rotator) in &mut query {
        transform.rotate_y(rotator.speed * time.delta_seconds());
    }
}

fn rotate_camera(
    time: Res<Time>,
    mut query: Query<&mut Transform, With<Camera>>,
) {
    for mut transform in &mut query {
        let radius = 10.0;
        let angle = time.elapsed_seconds() * 0.3;
        transform.translation.x = angle.cos() * radius;
        transform.translation.z = angle.sin() * radius;
        transform.look_at(Vec3::ZERO, Vec3::Y);
    }
}
