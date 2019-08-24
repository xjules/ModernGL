# ModernGL Examples

Before running examples make sure you install the requirmenets:

```bash
pip install -r examples/requirements.txt
```


## Running Examples

All examples should use [moderngl_window] to make window creation and resource loading as simple
as possible and ensures all supported platforms and covered.

**Examples should work out of the box on Windows, Linux and OS X.**

## Options

```
optional arguments:
  -h, --help            show this help message and exit
  -w {pyglet,pyside2,glfw,sdl2,pyqt5}, --window {glfw,pyglet,pyqt5,sdl2}
                        Name for the window type to use
  -fs, --fullscreen     Open the window in fullscreen mode
  -vs VSYNC, --vsync VSYNC
                        Enable or disable vsync
  -s SAMPLES, --samples SAMPLES
                        Specify the desired number of samples to use for
                        multisampling
  -c CURSOR, --cursor CURSOR
                        Enable or disable displaying the mouse cursor
```

For example:

```bash
# Run example in fullscreen with 8 x MSAA, vsync and no visible mouse cursor
python 03_alpha_blending.py --fullscreen --samples 8 --cursor off --vsync on

# Run example using SDL2
python 03_alpha_blending.py --window sdl2

# Run example using PySide2
python 03_alpha_blending.py --window pyside2

# Run example using glfw
python 03_alpha_blending.py --window glfw
```

## Basic Examples

Hello world :tada:

[![](preview_images/hello_world.png)](01_a_hello_world.py)

Simple color triangle

[![](preview_images/color_triangle.png)](01_b_simple_color_triangle.py)

Uniforms and attributes

[![](preview_images/uniforms_and_attributes.png)](02_uniforms_and_attributes.py)

Alpha blending

[![](preview_images/alpha_blending.png)](03_alpha_blending.py)

Colors and texture

[![](preview_images/colors_and_texture.png)](04_colors_and_texture.py)

Perspective projection

[![](preview_images/perspective_projection.png)](05_perspective_projection.py)

Index buffer

[![](preview_images/index_buffer.png)](06_index_buffer.py)

Multiple instance rendering

[![](preview_images/multiple_instance_rendering.png)](07_multiple_instance_rendering.py)

## Other Examples

The examples directory also contains a **large amount of examples** contributed
by varous people over time. Browse around and see what you can find :)

## Dependencies

```
pip install -r requirements.txt
```

## Old examples

Also see the [old examples](old-examples).


[moderngl_window]: (https://github.com/moderngl/moderngl_window)
