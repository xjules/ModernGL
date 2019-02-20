'''
    Added a simple camera class to an existing example.
    The camera class is built using following tutorials:
       https://learnopengl.com/Getting-started/Camera
       http://in2gpu.com/2016/03/14/opengl-fps-camera-quaternion/
   
    Controls:
        Move:
            Forward - "w"
            Backwards - "s"
    
        Strafe:
            Up - left "shift", "up arrow"
            Down - left "control", "down arrow"
            Left - "a", "left arrow"
            Right - "d", "right arrow"

        Rotate:
            Left - "q"
            Right - "e"
        
        Zoom:
            In - numpad "+"
            Out - numpad "-"
    
    adopted by: Alex Zakrividoroga
'''

import moderngl
from ModernGL.ext import obj
import numpy as np
from pyrr import Matrix44, Vector3, vector, Quaternion
from example_window import Example, run_example
from PIL import Image
import time

class Camera():
    
    def __init__(self, ratio):
        self._zoom_step = 0.1
        self._move_vertically = 0.1
        self._move_horizontally = 0.1
        self._rotate_horizontally = 0.1
        self._rotate_vertically = 0.1

        self._field_of_view_degrees = 60.0
        self._z_near = 0.1
        self._z_far = 100
        self._ratio = ratio
        self.build_projection()

        self._camera_position = Vector3 ([0.0, 0.0, -40.0])
        self._camera_front = Vector3 ([0.0, 0.0, 1.0])
        self._camera_up = Vector3 ([0.0, 1.0, 0.0])
        self._cameras_target = (self._camera_position + self._camera_front)
        self.build_look_at()

    def zoom_in(self):
        self._field_of_view_degrees = self._field_of_view_degrees - self._zoom_step
        self.build_projection()

    def zoom_out(self):
        self._field_of_view_degrees = self._field_of_view_degrees + self._zoom_step
        self.build_projection()

    def move_forward(self):
        self._camera_position = self._camera_position + self._camera_front * self._move_horizontally
        self.build_look_at()

    def move_backwards(self):
        self._camera_position = self._camera_position - self._camera_front * self._move_horizontally
        self.build_look_at()

    def strafe_left(self):
        self._camera_position = self._camera_position - vector.normalize(self._camera_front ^ self._camera_up) * self._move_horizontally
        self.build_look_at()

    def strafe_right(self):
        self._camera_position = self._camera_position + vector.normalize(self._camera_front ^ self._camera_up) * self._move_horizontally
        self.build_look_at()

    def strafe_up(self):
        self._camera_position = self._camera_position + self._camera_up * self._move_vertically
        self.build_look_at()

    def strafe_down(self):
        self._camera_position = self._camera_position - self._camera_up * self._move_vertically
        self.build_look_at()

    def rotate_left(self):
        rotation = Quaternion.from_y_rotation(2 * float (self._rotate_horizontally) * np.pi / 180)
        self._camera_front = rotation * self._camera_front
        self.build_look_at()

    def rotate_right(self):
        rotation = Quaternion.from_y_rotation(- 2 * float (self._rotate_horizontally) * np.pi / 180)
        self._camera_front = rotation * self._camera_front
        self.build_look_at()

    def build_look_at(self):
        self._cameras_target = (self._camera_position + self._camera_front)   
        self.mat_lookat = Matrix44.look_at(
            self._camera_position,
            self._cameras_target,
            self._camera_up)

    def build_projection(self):
        self.mat_projection = Matrix44.perspective_projection(
            self._field_of_view_degrees,
            self._ratio,
            self._z_near,
            self._z_far)

class QTKeyDecoder():

    def __init__(self, keys):
        self._keys = keys

    def up(self):
        return self._keys[38]

    def down(self):
        return self._keys[40]

    def left(self):
        return self._keys[37]

    def right(self):
        return self._keys[39]

    def num_plus(self):
        return self._keys[107]

    def num_minus(self):
        return self._keys[109]

    def key_w(self):
        return self._keys[87]

    def key_s(self):
        return self._keys[83]

    def key_a(self):
        return self._keys[65]

    def key_d(self):
        return self._keys[68]

    def left_shift(self):
        return self._keys[16]

    def left_control(self):
        return self._keys[17]

    def key_e(self):
        return self._keys[69]
        
    def key_q(self):
        return self._keys[81]

def grid(size, steps):
    u = np.repeat(np.linspace(-size, size, steps), 2)
    v = np.tile([-size, size], steps)
    w = np.zeros(steps * 2)
    return np.concatenate([np.dstack([u, v, w]), np.dstack([v, u, w])])
 
class PerspectiveProjection(Example):
    def __init__(self):

        self.water_img = Image.open("water_normal.jpeg").convert('RGB').transpose(Image.FLIP_TOP_BOTTOM)

        self.ctx = moderngl.create_context()

        self.prog = self.ctx.program(
            vertex_shader='''
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
                    vec2 t0 = vec2(0.0, time * 100.0);
                    vec2 t1 = vec2(time * 100.0, 0.0);
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
            ''',
            fragment_shader='''
                #version 450
                
                in vec4 orig_view_pos;
                in vec4 rfrc_view_pos;               
                
                out vec4 f_color;

                void main() {
                    float s0x = length(ddx_fine(orig_view_pos.xy));
                    float s0y = length(ddy_fine(orig_view_pos.xy));
                    float s1x = length(ddx_fine(rfrc_view_pos.xy));
                    float s1y = length(ddy_fine(rfrc_view_pos.xy));
                    float delta = (s0x * s0y) / (s1x * s1y);
                    f_color = delta * vec4(0.2, 0.66, 0.9, 1.0) * 0.6;
                }
            ''',
        )

        #build texture
        tex0 = self.ctx.texture(self.water_img.size, 3, self.water_img.tobytes())
        self.camera = Camera(self.wnd.ratio)
        tex0.use(0)
        self.prog['normal_map'].value = 0
        self.dist = self.prog['dist']
        self.time = self.prog['time']
        self.viewMat = self.prog['viewMat']
        self.projMat = self.prog['projMat']

        model = obj.Obj.open('plane256.obj')
        self.vbo = self.ctx.buffer(data = model.pack('vx vy vz'))
        self.vao = self.ctx.simple_vertex_array(self.prog, self.vbo, 'in_vert')

    def input_update(self):
        #Keyboard Processing
        qt_keys = QTKeyDecoder(self.wnd.keys)
        
        if qt_keys.up() == True:
            self.camera.strafe_up()

        if qt_keys.down() == True:
            self.camera.strafe_down()

        if qt_keys.left() == True:
            self.camera.strafe_left()

        if qt_keys.right() == True:
            self.camera.strafe_right()

        if qt_keys.num_plus() == True:
           self.camera.zoom_in()

        if qt_keys.num_minus() == True:
            self.camera.zoom_out()

        if qt_keys.key_w() == True:
            self.camera.move_forward()

        if qt_keys.key_s() == True:
            self.camera.move_backwards()

        if qt_keys.key_a() == True:
            self.camera.strafe_left()

        if qt_keys.key_d() == True:
            self.camera.strafe_right()

        if qt_keys.left_shift() == True:
            self.camera.strafe_up()

        if qt_keys.left_control() == True:
            self.camera.strafe_down()

        if qt_keys.key_q() == True:
            self.camera.rotate_left()

        if qt_keys.key_e() == True:
            self.camera.rotate_right()
      
    def render(self):
        self.input_update()
        
        self.ctx.viewport = self.wnd.viewport
        self.ctx.clear(0.0, 0.0, 0.0)
        # self.ctx.enable(moderngl.DEPTH_TEST)
        self.ctx.disable(moderngl.DEPTH_TEST)
        self.ctx.enable(moderngl.BLEND)
        self.ctx.disable(moderngl.CULL_FACE)
        # self.ctx.blend_func(moderngl.ONE,moderngl.ONE)

        self.dist.write(np.array(3.14).astype('f4'))
        # self.time.write(np.array(time.time()).astype('f4'))
        self.time.write(np.array(self.wnd.time).astype('f4'))
        self.projMat.write(self.camera.mat_projection.astype('f4').tobytes())
        self.viewMat.write(self.camera.mat_lookat.astype('f4').tobytes())
        # self.vao.render(moderngl.TRIANGLE_STRIP)
        self.vao.render(moderngl.TRIANGLES)

run_example(PerspectiveProjection)
