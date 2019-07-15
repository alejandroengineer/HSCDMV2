#version 120    //version 120 of glsl is used as it was the last version to contain built in texture coordinates
                //we don't need any of the functionality offered by higher versions

//blazed grating digital hologram shader

//this shader automatically generates a blazed gratting for displaying both amplitude and phase modulated images

uniform sampler2D tex;  //field we want to generate (red - real, greaan - imaginary, blue = nothing, alpha - nothing)
uniform sampler1D Alut; //lookup table for inverse sinc (used for modulating the amplitude)

uniform sampler2D calA, celB;   //calirbation polynomial coefficients stored in the color information of these two textures

uniform vec2 dir;   //direction vector of the grating (1.0/nx, 1.0/ny where nx and ny are ideally coprime)
                    //It is the k momentum vector multiplied by the pixel dimensions

uniform vec2 screen_size;   //the size of the screen is needed for the calibration texture

float phaseCal(float in, vec4 A, vec4 B)    {   //converts between phase and pixel values using a calibration polynomial
    return A.x + in*(A.y + in*(A.z + in*(A.w + in*(B.x + in*(B.y + in*(B.z + in*B.w))))));  //the polynomial is evaluated using Horner's method
}

void main() {
    vec2 data = texture2D(tex, gl_TexCoord[0].xy).xy;       //retrieve pixel data

    float mag = sqrt((data.x*data.x) + (data.y*data.y));    //convert to magnitude and phase
    float arg = atan(data.y, data.x);

    vec2 coords = floor(gl_FragCoord.xy);   //fragment coordinates are pixel center justified (or worse, random when AA is enabled)
                                            //the floor must be taken to obtain the actual pixel location

    float phig = dot(gl_FragCoord.xy, dir); //blazed grating phase is obtained by taking a dot product between the pixel location and the grating direction
                                            //grating speed is determined by the length of the direction vector

    float phase = texture1D(Alut, mag).x  * mod(arg + phig, 6.28318530718); //apply the inverse sinc and obtain the blazed grating phase

    vec2 screen_loc = coords/screen_size;   //the calibration textures require global normalized coordinates

    vec4 A = texture2D(calA, screen_loc);   //calibration coeffs are obtained from the two calibration textures
    vec4 B = texture2D(calB, screen_loc);

    gl_FragColor = vec4(vec3(phaseCal(phase, A, B)), 1);    //display the pixel
}