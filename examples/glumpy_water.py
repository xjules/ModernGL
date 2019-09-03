from glumpy import app, gl, gloo, glm
from moderngl.ext import obj
import numpy as np
from pyrr import Matrix44, Vector3, vector, Quaternion
from PIL import Image

vertex_shader = '''
                #version 450
                uniform sampler2D normal_map;

                uniform float dist;
                uniform float time;

                uniform mat4 projMat;
                uniform mat4 viewMat;

                in vec3 in_vert;

                out vec4 orig_view_pos;
                out vec4 rfrc_view_pos;

                vec3 sample_normal(vec2 view_pos) {
                    vec2 t0 = vec2(time * 0.1);
                    vec2 t1 = vec2(time * 0.1, 0.0);
                    vec2 uv0 = fract((view_pos + 15.0) / 60.0 + t0);
                    vec2 uv1 = fract((view_pos + 15.0) / 60.0 + t1);
                    vec3 nm0 = texture(normal_map, uv0).xyz;
                    vec3 nm1 = texture(normal_map, uv1).xyz;
                    return ((nm0 + nm1) * 0.5 * 2.0) - vec3(1.0);
                }

                void main() {
                    orig_view_pos = viewMat * vec4(in_vert, 1.0);

                    vec3 incident = vec3(0.0, 0.0, -1.0);
                    vec3 normal = sample_normal(orig_view_pos.xy);
                    rfrc_view_pos = vec4(
                        orig_view_pos.xy + refract(incident, normal, 1.0 / 1.33).xy * dist,
                        orig_view_pos.z, 1.0);

                    gl_Position = projMat * rfrc_view_pos;
                }
'''
fragment_shader = '''
                #version 450

                in vec4 orig_view_pos;
                in vec4 rfrc_view_pos;               

                out vec4 f_color;

                void main() {


                    float s0x = length(dFdx(orig_view_pos.xy));
                    float s0y = length(dFdy(orig_view_pos.xy));
                    float s1x = length(dFdx(rfrc_view_pos.xy));
                    float s1y = length(dFdy(rfrc_view_pos.xy));
                    //float s0y = length(ddy_fine(orig_view_pos.xy));
                    //float s0x = length(ddx_fine(orig_view_pos.xy));
                    //float s1x = length(ddx_fine(rfrc_view_pos.xy));
                    //float s1y = length(ddy_fine(rfrc_view_pos.xy));
                    float delta = (s0x * s0y) / (s1x * s1y);
                    f_color = delta * vec4(0.2, 0.66, 0.9, 1.0) * 0.6;
                }
'''

#get main window
window = app.Window(width=800, height=800, color=(1,1,1,1))

#get normal map
water_img = Image.open("water_normal.jpeg").convert('RGB').transpose(Image.FLIP_TOP_BOTTOM)
model = obj.Obj.open('plane256.obj')

print('Model size:',len(model.vert))
program = gloo.Program(vertex_shader, fragment_shader)
view = np.eye(4, dtype=np.float32)
proj = np.eye(4, dtype=np.float32)
glm.translate(view, 0, 0, -40)

program['viewMat'] = view
program['projMat'] = proj
program['normal_map'] = water_img
program['normal_map'].wrapping = gl.GL_TEXTURE_WRAP_S
program['in_vert'] = model.vert
program['dist'] = 3.14

@window.event
def on_draw(dt):
    global theta, phi, translate
    window.clear()
    program.draw(gl.GL_TRIANGLES)
    # model = np.eye(4, dtype=np.float32)
    # glm.rotate(model, theta, 0, 0, 1)
    # glm.rotate(model, phi, 0, 1, 0)
    # program['model'] = model

@window.event
def on_resize(width, height):
    program['projection'] = glm.perspective(45.0, width / float(height), 1.0, 1000.0)

gl.glDisable(gl.GL_DEPTH_TEST)
gl.glEnable(gl.GL_BLEND)
gl.glDisable(gl.GL_CULL_FACE)
app.run()
