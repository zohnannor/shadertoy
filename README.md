# WGPU Shader Hot Reload Demo

A minimal graphics application demonstrating live shader reloading using [`wgpu`] and [`winit`]. The application watches for changes to a shader file and recompiles the pipeline automatically without restarting.

## Features

-   Fragment shader hot reloading
-   Time and resolution uniform buffers

## Usage

Create a file named `shader.wgsl` in the same directory as the executable. The application will automatically load and watch this file for changes. Any modifications trigger an immediate pipeline rebuild with the updated shader.

If the shader file is missing or contains errors, the application falls back to a default magenta shader.

## Shader Uniforms

The fragment shader receives two uniform buffers:

-   `@group(0) binding(0)`: Elapsed time in seconds (`f32`)
-   `@group(0) binding(1)`: Screen resolution as `[width, height]` (`vec2<f32>`)

## Dependencies

-   [`wgpu`] for graphics API abstraction
-   [`winit`] for window management

[`wgpu`]: https://docs.rs/wgpu
[`winit`]: https://docs.rs/winit

## License

Licensed under either of

-   Apache License, Version 2.0, ([LICENSE-APACHE](LICENSE-APACHE))
-   MIT license ([LICENSE-MIT](LICENSE-MIT))

at your option.
