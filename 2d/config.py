screenDimensions = [1800,1000]
WORLD_WIDTH  = screenDimensions[0]/8e-9 # meters
WORLD_HEIGHT = screenDimensions[1]/8e-9
c = 299792458
G = 6.6743*10**-11












def ortho_projection(left, right, bottom, top, near=-1.0, far=1.0):
    rl = right - left
    tb = top - bottom
    fn = far - near

    # Avoid division by zero
    if rl == 0 or tb == 0 or fn == 0:
        raise ValueError("Invalid orthographic bounds")

    return [
        2.0 / rl, 0.0,      0.0,       -(right + left) / rl,
        0.0,      2.0 / tb, 0.0,       -(top + bottom) / tb,
        0.0,      0.0,     -2.0 / fn,  -(far + near) / fn,
        0.0,      0.0,      0.0,        1.0
    ]

proj = ortho_projection(
    -WORLD_WIDTH, WORLD_WIDTH,
    -WORLD_HEIGHT, WORLD_HEIGHT
)
