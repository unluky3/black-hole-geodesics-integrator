from config import proj
import moderngl
import numpy as np
from numpy import sin, cos

class circle:
    def __init__(
            self,
            res,
            radius,
            tr_x,
            tr_y,
            color,
            context
            ):
        self.res = res
        self.radius = radius
        self.tr_x = tr_x
        self.tr_y = tr_y
        self.col = color
        self.ctx = context
        self.pos = np.array((self.tr_x, self.tr_y))
        self.prog = self.ctx.program(
                vertex_shader="""
                    #version 330
                    in vec2 in_vert;
                    uniform mat4 proj;
                    void main() {
                        gl_Position = proj * vec4(in_vert, 0.0, 1.0);
                    }
                """,
                fragment_shader="""
                    #version 330
                    out vec4 f_color;
                    uniform vec4 col;
                    void main() {
                        f_color = col;
                    }
                """
                )
        self.prog['proj'].write(np.array(proj, dtype='f4').tobytes())
        self.angles = np.linspace(0, 360, self.res, endpoint=False)
        self.verts = np.empty((self.res, 2), dtype='f4')
        self.verts[:,0] = self.radius * cos(np.radians(self.angles)) + self.tr_x
        self.verts[:,1] = self.radius * sin(np.radians(self.angles)) + self.tr_y
        self.prog['col'].value = self.col
        self.vbo = self.ctx.buffer(self.verts.tobytes())
        self.vao = self.ctx.simple_vertex_array(self.prog, self.vbo, 'in_vert')
    def reload(self):
        self.verts[:,0] = self.radius * cos(np.radians(self.angles)) + self.tr_x
        self.verts[:,1] = self.radius * sin(np.radians(self.angles)) + self.tr_y
        self.vbo.write(self.verts.tobytes())
        self.prog['col'].value = self.col
    def render(self):
        self.prog['col'].value = self.col
        self.vao.render(mode=moderngl.TRIANGLE_FAN)
    def translate(self, tr_x, tr_y):
        self.tr_x = self.tr_x+tr_x
        self.tr_y = self.tr_y+tr_y
        self.reload()
    def scale(self, radius):
        self.radius = radius
        self.reload()


def project_cpu(verts, proj):
    P = np.array(proj, dtype='f8').reshape((4,4))
    verts_ndc = np.empty((len(verts), 2), dtype='f4')
    for i in range(len(verts)):
        vec = np.array([verts[i,0], verts[i,1], 0.0, 1.0], dtype='f8')
        t = P @ vec
        verts_ndc[i] = (t[:2] / t[3]).astype('f4')
    return verts_ndc


class ray:
    def __init__(
            self,
            res,
            or_x,
            or_y,
            context,
            ):
        self.res = res
        self.tr_x = or_x
        self.tr_y = or_y
        self.posa = np.array((self.tr_x, self.tr_y))
        self.ctx = context
        self.prog = self.ctx.program(
                vertex_shader="""
                    #version 330
                    in vec2 in_vert;
                    flat out int vert_id;
                    void main() {
                        gl_Position = vec4(in_vert, 0.0, 1.0);
                        vert_id = gl_VertexID;
                    }
                """,
                fragment_shader="""
                    #version 330
                    flat in int vert_id;
                    out vec4 f_color;
                    uniform float total_vcs;
                    void main() {
                        float prog = (total_vcs-vert_id)/total_vcs;
                        f_color = vec4(1.0-(1.0*prog), 1.0-(1.0*prog), 1.0-(1.0*prog), 1.0-(1.0*prog));
                    }
                """
                )
        self.verts = np.empty((self.res, 2), dtype='f8')
        for i in range(self.res):
            self.verts[i, 0] = self.tr_x
            self.verts[i, 1] = self.tr_y
        self.proj = proj
        self.prog['total_vcs'].value = self.res
        verts_ndc = project_cpu(self.verts, self.proj)
        self.vbo = self.ctx.buffer(verts_ndc.tobytes())
        self.vao = self.ctx.simple_vertex_array(self.prog, self.vbo, 'in_vert')

    def translate(self, tr):
        # shift old verts
        for i in range(self.res-1, 0, -1):
            self.verts[i] = self.verts[i-1]
        # set new head
        self.verts[0] = np.array((tr[0], tr[1]), dtype='f8')
        self.posa = self.verts[0]
        # CPU-side projection to NDC
        verts_ndc = project_cpu(self.verts, self.proj)
        # write float32 to GPU
        self.vbo.write(verts_ndc.tobytes())

    def render(self):
        self.vao.render(mode=moderngl.LINE_STRIP)
