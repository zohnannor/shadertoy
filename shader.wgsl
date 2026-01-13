@group(0) @binding(0)
var<uniform> time: f32;

@group(0) @binding(1)
var<uniform> resolution: vec2f;

@fragment
fn main(@builtin(position) pos: vec4f) -> @location(0) vec4f {
    // let uv = (2.0 * pos.xy - resolution) / resolution.y;
    let uv = pos.xy / resolution;

    return vec4f(uv, 1.0, 1.0);
}
