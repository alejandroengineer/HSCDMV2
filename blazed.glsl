#version 120

uniform sampler2D tex;
uniform sampler1D lut;

//blazed grating digital hologram
//all phase angels are normalized

void main() {
    vec2 data = texture2D(tex, gl_TexCoord[0].xy).xy;
    float mag = sqrt((data.x*data.x) + (data.y*data.y));
    float arg = atan(data.y, data.x)/6.28318530718;                 
    float phiG = (gl_FragCoord.x/19.0f) + (gl_FragCoord.y/13.0f); //blazed grating phase
    float phase = texture1D(lut, mag * mod(arg + phiG, 1.0)).x;
    gl_FragColor = vec4(vec3(phase), 1);
}