import glfw
import moderngl
import numpy as np
import time as tm
from physics import BlackHole, Path
from config import screenDimensions

red = (0.5,0.5,0.5,1.0)
bhl = []

# Initialize GLFW
if not glfw.init():
    raise RuntimeError("Failed to initialize GLFW")

window = glfw.create_window(screenDimensions[0], screenDimensions[1], "black hole integrator", None, None)
if not window:
    glfw.terminate()
    raise RuntimeError("Failed to create GLFW window")

glfw.make_context_current(window)
ctx = moderngl.create_context()
ctx.line_width = 3.0
ctx.point_size = 3.0

bh = BlackHole(64, 0, 0, red, 8.54e36, ctx)
bhl.append(bh)

rays = []
n = 10
for i in (lambda n, half_span: np.array([0.0]) if n == 1 else np.linspace(-half_span, half_span, n))(n, 1.35e11*0.9):
    rays.append(Path(5000, -2.3e11, i, bhl, ctx))


while not glfw.window_should_close(window):
    ctx.clear(1,1,1,1)
    bh.render()
    for r in rays:
        r.step()
    glfw.swap_buffers(window)
    glfw.poll_events()
    tm.sleep(.0)

glfw.terminate()
