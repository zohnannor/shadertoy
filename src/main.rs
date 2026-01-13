use std::{
    fs::File,
    io::{self, Read, Seek},
    sync::{Arc, mpsc},
    thread,
    time::{Duration, Instant, SystemTime},
};

use wgpu::{
    BindGroup, BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutEntry,
    BindingResource, BindingType, Buffer, BufferBinding, BufferBindingType, BufferDescriptor,
    BufferUsages, ColorTargetState, ColorWrites, CommandEncoderDescriptor, Device,
    DeviceDescriptor, Features, FragmentState, Instance, InstanceDescriptor, Limits,
    MultisampleState, Operations, PipelineCompilationOptions, PipelineLayoutDescriptor,
    PrimitiveState, Queue, RenderPassColorAttachment, RenderPassDescriptor, RenderPipeline,
    RenderPipelineDescriptor, RequestAdapterOptionsBase, ShaderModuleDescriptor, ShaderSource,
    ShaderStages, Surface, SurfaceConfiguration, TextureViewDescriptor, VertexState,
};
use winit::{
    application::ApplicationHandler,
    dpi::PhysicalSize,
    event::WindowEvent,
    event_loop::{ActiveEventLoop, EventLoop},
    window::{Window, WindowId},
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt().with_file(false).compact().init();
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
}

#[derive(Debug, Default)]
struct App {
    state: Option<AppState>,
}

impl AppState {
    #[tracing::instrument]
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

        let fragment_source_rx = Self::spawn_watcher_thread()?;
        tracing::info!("Shader hot reload enabled");

        let fragment_source = fragment_source_rx
            .try_recv()
            .inspect(|source| {
                tracing::debug!("Loaded fragment shader from file ({} chars)", source.len());
            })
            .inspect_err(|_| {
                tracing::debug!("Using initial fragment shader");
            })
            .unwrap_or_else(|_| INITIAL_FRAGMENT_SHADER.into());

        let render_pipeline =
            Self::create_pipeline(&device, &config, &fragment_source, &bind_group_layout);

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

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
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
        fragment_source: &str,
        bind_group_layout: &BindGroupLayout,
    ) -> RenderPipeline {
        let vertex_shader = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("vertex shader"),
            source: ShaderSource::Wgsl(VERTEX_SHADER.into()),
        });

        let fragment_shader = {
            let error_scope_guard = device.push_error_scope(wgpu::ErrorFilter::Validation);

            let fragment_shader_try = device.create_shader_module(ShaderModuleDescriptor {
                label: Some(fragment_source),
                source: ShaderSource::Wgsl(fragment_source.into()),
            });

            let ef = error_scope_guard.pop();
            pollster::block_on(ef).map_or_else(
                || {
                    tracing::debug!("Fragment shader module created successfully");
                    fragment_shader_try
                },
                |error| {
                    tracing::error!("Fragment shader module creation failed: {error}");
                    tracing::debug!("Using initial fragment shader");
                    device.create_shader_module(ShaderModuleDescriptor {
                        label: Some("shader.wgsl"),
                        source: ShaderSource::Wgsl(INITIAL_FRAGMENT_SHADER.into()),
                    })
                },
            )
        };

        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[bind_group_layout],
            immediate_size: 0,
        });

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
    }

    #[tracing::instrument]
    fn spawn_watcher_thread() -> Result<mpsc::Receiver<String>, io::Error> {
        tracing::trace!("Spawning shader watcher thread");
        let (tx, rx) = mpsc::channel();
        let mut f = File::open("shader.wgsl")?;
        let mut buf = String::new();

        let mut last = SystemTime::UNIX_EPOCH;

        thread::spawn(move || -> io::Result<()> {
            tracing::debug!("Shader watcher thread started");

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
                &fragment_source,
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
            el.create_window(Window::default_attributes())
                .expect("Failed to create window"),
        );
        tracing::trace!("Window created");
        let Ok(state) = pollster::block_on(AppState::new(window)) else {
            tracing::error!("Failed to init app");
            el.exit();
            return;
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
