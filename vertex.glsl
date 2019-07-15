#version 120

//ignore this code, it only generates stuff for the fragment shader

void main() {
    gl_Position = ftransform();
    gl_TexCoord[0] = gl_MultiTexCoord0;
}