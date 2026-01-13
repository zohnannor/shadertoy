use std::{
    fs::File,
    io::{self, Read, Seek},
    sync::{Arc, mpsc},
    thread,
    time::{Duration, Instant, SystemTime},
};

use tracing::level_filters::LevelFilter;
use tracing_subscriber::EnvFilter;
use wgpu::{
    BindGroup, BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindingResource, BindingType, Buffer, BufferBinding, BufferBindingType,
    BufferDescriptor, BufferUsages, ColorTargetState, ColorWrites, CommandEncoderDescriptor,
    Device, DeviceDescriptor, Features, FragmentState, Instance, InstanceDescriptor, Limits,
    MultisampleState, Operations, PipelineCompilationOptions, PipelineLayoutDescriptor,
    PrimitiveState, Queue, RenderPassColorAttachment, RenderPassDescriptor, RenderPipeline,
    RenderPipelineDescriptor, RequestAdapterOptionsBase, ShaderModule, ShaderModuleDescriptor,
    ShaderSource, ShaderStages, Surface, SurfaceConfiguration, TextureViewDescriptor, VertexState,
};
use winit::{
    application::ApplicationHandler,
    dpi::PhysicalSize,
    event::WindowEvent,
    event_loop::{ActiveEventLoop, EventLoop},
    window::{Window, WindowId},
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::builder()
                .with_default_directive(LevelFilter::INFO.into())
                .from_env_lossy(),
        )
        .with_file(false)
        .compact()
        .init();
    tracing::info!("Starting application...");
    let el = EventLoop::new()?;
    el.run_app(&mut App::default())?;
    Ok(())
}

#[derive(Debug)]
struct AppState {
    window: Arc<Window>,
    device: Device,
    queue: Queue,
    surface: Surface<'static>,
    render_pipeline: RenderPipeline,
    config: SurfaceConfiguration,
    buffer: Buffer,
    fragment_source_rx: mpsc::Receiver<String>,
    bind_group_layout: BindGroupLayout,
    bind_group: BindGroup,
    time: Instant,
    alignment: u64,
    fallback_shader: ShaderModule,
}

#[derive(Debug, Default)]
struct App {
    state: Option<AppState>,
}

impl AppState {
    #[tracing::instrument(skip_all)]
    async fn new(window: Arc<Window>) -> Result<Self, Box<dyn std::error::Error>> {
        tracing::info!("Initializing renderer...");

        let (width, height) = window.inner_size().into();
        tracing::debug!("Window size: {}x{}", width, height);

        let instance = Instance::new(&InstanceDescriptor::default());

        let surface = instance.create_surface(window.clone())?;
        tracing::trace!("Surface created");

        let adapter = instance
            .request_adapter(&RequestAdapterOptionsBase::default())
            .await?;
        tracing::debug!("Adapter: {:?}", adapter.get_info().name);

        let (device, queue) = adapter
            .request_device(&DeviceDescriptor {
                label: Some("device"),
                required_features: Features::SHADER_F64,
                required_limits: Limits {
                    min_uniform_buffer_offset_alignment: 64,
                    ..Default::default()
                },
                ..Default::default()
            })
            .await?;
        tracing::trace!("Device and queue created");

        let config = surface.get_default_config(&adapter, width, height).unwrap();
        surface.configure(&device, &config);
        tracing::debug!("Surface format: {:?}", config.format);

        let alignment = u64::from(device.limits().min_uniform_buffer_offset_alignment);
        tracing::debug!("Buffer alignment: {} bytes", alignment);

        let (buffer, bind_group_layout, bind_group) = Self::create_bindings(&device, alignment);

        let fallback_shader = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("shader.wgsl"),
            source: ShaderSource::Wgsl(INITIAL_FRAGMENT_SHADER.into()),
        });

        let fragment_source_rx = Self::spawn_watcher_thread()?;
        tracing::info!("Shader hot reload enabled");

        let fragment_source = fragment_source_rx.try_recv().ok();

        let render_pipeline = Self::create_pipeline(
            &device,
            &config,
            &fallback_shader,
            fragment_source.as_deref(),
            &bind_group_layout,
        );

        tracing::info!("Renderer ready");
        Ok(Self {
            window,
            device,
            queue,
            surface,
            render_pipeline,
            config,
            buffer,
            fragment_source_rx,
            bind_group_layout,
            bind_group,
            time: Instant::now(),
            alignment,
            fallback_shader,
        })
    }

    #[tracing::instrument]
    fn create_bindings(device: &Device, alignment: u64) -> (Buffer, BindGroupLayout, BindGroup) {
        let buffer_size = alignment * 2;

        let buffer = device.create_buffer(&BufferDescriptor {
            label: Some("uniform buffer"),
            size: buffer_size,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("bind group layout"),
            entries: &[
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::FRAGMENT,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::default(),
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: ShaderStages::FRAGMENT,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::default(),
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("bind group"),
            layout: &bind_group_layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::Buffer(BufferBinding {
                        buffer: &buffer,
                        offset: 0,
                        size: None,
                    }),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: BindingResource::Buffer(BufferBinding {
                        buffer: &buffer,
                        offset: alignment,
                        size: None,
                    }),
                },
            ],
        });
        (buffer, bind_group_layout, bind_group)
    }

    #[tracing::instrument(skip_all)]
    fn create_pipeline(
        device: &Device,
        config: &SurfaceConfiguration,
        fallback_shader: &ShaderModule,
        fragment_source: Option<&str>,
        bind_group_layout: &BindGroupLayout,
    ) -> RenderPipeline {
        let vertex_shader = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("vertex shader"),
            source: ShaderSource::Wgsl(VERTEX_SHADER.into()),
        });

        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[bind_group_layout],
            immediate_size: 0,
        });

        let create_render_pipeline = |fragment_shader| {
            device.create_render_pipeline(&RenderPipelineDescriptor {
                label: Some("render pipeline"),
                layout: Some(&pipeline_layout),
                vertex: VertexState {
                    module: &vertex_shader,
                    entry_point: None,
                    compilation_options: PipelineCompilationOptions::default(),
                    buffers: &[],
                },
                fragment: Some(FragmentState {
                    module: &fragment_shader,
                    entry_point: None,
                    compilation_options: PipelineCompilationOptions::default(),
                    targets: &[Some(ColorTargetState {
                        format: config.format,
                        blend: None,
                        write_mask: ColorWrites::default(),
                    })],
                }),
                primitive: PrimitiveState::default(),
                depth_stencil: None,
                multisample: MultisampleState::default(),
                multiview_mask: None,
                cache: None,
            })
        };

        let error_scope_guard = device.push_error_scope(wgpu::ErrorFilter::Validation);
        let fallback = || {
            tracing::warn!("Using initial fragment shader");
            fallback_shader.clone()
        };
        let t = create_render_pipeline(fragment_source.map_or_else(fallback, |fragment_source| {
            tracing::debug!("Fragment shader module created successfully");
            device.create_shader_module(ShaderModuleDescriptor {
                label: Some("shader.wgsl"),
                source: ShaderSource::Wgsl(fragment_source.into()),
            })
        }));
        let ef = error_scope_guard.pop();
        pollster::block_on(ef).map_or_else(
            || t,
            |error| {
                tracing::error!("Fragment shader module creation failed: {error}");
                create_render_pipeline(fallback())
            },
        )
    }

    #[tracing::instrument]
    fn spawn_watcher_thread() -> Result<mpsc::Receiver<String>, io::Error> {
        tracing::trace!("Spawning shader watcher thread");
        let (tx, rx) = mpsc::channel();

        thread::spawn(move || -> io::Result<()> {
            tracing::debug!("Shader watcher thread started");

            let mut f = loop {
                match File::open("shader.wgsl") {
                    Ok(file) => break file,
                    Err(err) => {
                        tracing::error!("Failed to open shader file: {err}. Retrying in 1 second");
                        tracing::error!(
                            "Create a file named `shader.wgsl` in the same directory as the executable"
                        );
                        thread::sleep(Duration::from_millis(1000));
                    }
                }
            };
            let mut buf = String::new();
            let mut last = SystemTime::UNIX_EPOCH;

            loop {
                let modified = match f.metadata()?.modified() {
                    Ok(time) => time,
                    Err(e) => {
                        tracing::error!("Failed to get file metadata: {:?}", e);
                        thread::sleep(Duration::from_millis(1000));
                        continue;
                    }
                };

                if modified > last {
                    match f.read_to_string(&mut buf) {
                        Ok(bytes_read) => {
                            tracing::info!("Shader file modified, read {} bytes", bytes_read);
                            if tx.send(buf.clone()).is_ok() {
                                tracing::trace!("Shader source sent to main thread");
                                last = modified;
                                f.rewind()?;
                                buf.clear();
                            } else {
                                tracing::warn!(
                                    "Failed to send shader source, channel disconnected"
                                );
                            }
                        }
                        Err(e) => {
                            tracing::error!("Failed to read shader file: {:?}", e);
                        }
                    }
                }

                thread::sleep(Duration::from_millis(500));
            }
        });
        Ok(rx)
    }

    #[tracing::instrument(skip(self))]
    fn resize(&mut self, size: PhysicalSize<u32>) {
        let (width, height): (u32, u32) = size.into();
        tracing::debug!("Resized to {}x{}", width, height);
        self.config.width = width.max(1);
        self.config.height = height.max(1);
        self.surface.configure(&self.device, &self.config);

        let resolution: [f32; 2] = size.into();
        tracing::trace!(?resolution, "Updating resolution uniform");
        self.queue.write_buffer(
            &self.buffer,
            self.alignment,
            bytemuck::bytes_of(&resolution),
        );
    }

    #[tracing::instrument(skip_all)]
    fn update(&mut self) {
        if let Ok(fragment_source) = self.fragment_source_rx.try_recv() {
            self.time = Instant::now();
            self.render_pipeline = Self::create_pipeline(
                &self.device,
                &self.config,
                &self.fallback_shader,
                Some(&fragment_source),
                &self.bind_group_layout,
            );
            tracing::info!("Shader reloaded");
        }

        let elapsed = self.time.elapsed();
        tracing::trace!(?elapsed, "Updating time uniform");
        self.queue
            .write_buffer(&self.buffer, 0, bytemuck::bytes_of(&elapsed.as_secs_f32()));
    }

    #[tracing::instrument(skip_all)]
    fn render(&self) -> Result<(), Box<dyn std::error::Error>> {
        let frame = self.surface.get_current_texture()?;
        let view = frame.texture.create_view(&TextureViewDescriptor {
            label: Some("view"),
            ..Default::default()
        });

        let mut encoder = self
            .device
            .create_command_encoder(&CommandEncoderDescriptor {
                label: Some("command encoder"),
            });

        let mut render_pass = encoder.begin_render_pass(&RenderPassDescriptor {
            label: Some("render pass"),
            color_attachments: &[Some(RenderPassColorAttachment {
                view: &view,
                depth_slice: None,
                resolve_target: None,
                ops: Operations::default(),
            })],
            ..Default::default()
        });

        render_pass.set_pipeline(&self.render_pipeline);
        render_pass.set_bind_group(0, &self.bind_group, &[]);
        render_pass.draw(0..3, 0..1);
        drop(render_pass);

        self.queue.submit([encoder.finish()]);
        frame.present();
        self.window.request_redraw();

        Ok(())
    }
}

const VERTEX_SHADER: &str = "
@vertex
fn main(@builtin(vertex_index) vertex_index: u32) -> @builtin(position) vec4<f32> {
    let vert = array(
        vec2<f32>(0.0, 0.0),
        vec2<f32>(0.0, 1.0),
        vec2<f32>(1.0, 0.0),
    );
    return vec4<f32>(vert[vertex_index] * 4 - 1, 0.0, 1.0);
}
";

const INITIAL_FRAGMENT_SHADER: &str = "
@fragment
fn main(@builtin(position) p: vec4<f32>) -> @location(0) vec4<f32> {
    return vec4(1.0, 0.0, 1.0, 1.0);
}
";

impl ApplicationHandler for App {
    #[tracing::instrument(skip_all)]
    fn resumed(&mut self, el: &ActiveEventLoop) {
        let window = Arc::new(
            el.create_window(Window::default_attributes().with_title("Shadertoy"))
                .expect("Failed to create window"),
        );
        tracing::trace!("Window created");

        let state = match pollster::block_on(AppState::new(window)) {
            Ok(state) => state,
            Err(err) => {
                tracing::error!("Failed to init app: {err}");
                el.exit();
                return;
            }
        };
        tracing::trace!("AppState initialized successfully");
        self.state = Some(state);
    }

    #[tracing::instrument(skip_all)]
    fn window_event(&mut self, el: &ActiveEventLoop, _: WindowId, event: WindowEvent) {
        let Some(state) = &mut self.state else { return };

        match event {
            WindowEvent::Resized(physical_size) => state.resize(physical_size),
            WindowEvent::CloseRequested | WindowEvent::Destroyed => {
                tracing::info!("Closing app");
                el.exit();
            }
            WindowEvent::RedrawRequested => {
                state.update();
                if let Err(e) = state.render() {
                    tracing::error!("Render error: {}", e);
                }
            }
            _ => { /* ignore */ }
        }
    }
}
