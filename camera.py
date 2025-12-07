import taichi as ti
from taichi.math import sqrt, cos, sin
import taichi.math as tim
from config import c, SZR, r_s
from typing import Tuple, Any



@ti.data_oriented
class Camera:

    dt: float = 1.0
    max_steps: int = int(6e1)
    object_count: int = 3 # how many objects do you have
    max_dist: float = 30.0
    smtpEnabled: int = 1

    # main functionality
    def __init__(self, position: ti.math.vec3, rotation: ti.math.vec3, resolution: Tuple[int, int], screenRes: Tuple[int, int], dimensions: ti.math.vec3):
        
        # Taichi Fields (Instance variables)
        self.offset: ti.Field = ti.Vector.field(3, dtype=ti.f32, shape=())
        self.position: ti.Field = ti.Vector.field(3, dtype=ti.f32, shape=())
        self.rotation: ti.Field = ti.Vector.field(3, dtype=ti.f32, shape=())
        self.dimensions: ti.Field = ti.Vector.field(3, dtype=ti.f32, shape=())
        self.gravEnable: ti.Field = ti.field(dtype=ti.i32, shape=())
        self.temp: ti.Field = ti.field(dtype=ti.f32, shape=(3,))

        self.temp.fill(0)

        self.position[None] = position
        self.rotation[None] = rotation

        self.resolution: Tuple[int, int] = tuple(resolution)
        self.dimensions[None] = dimensions
        self.gravEnable[None] = 1

        self.object_count += self.temp.shape[0]
        
        self.dimage: ti.Field = ti.Vector.field(n=3, dtype=ti.f32, shape=self.resolution)
        self.temp_blur: ti.Field = ti.Vector.field(n=3, dtype=ti.f32, shape=self.resolution)
        self.temp_bloom: ti.Field = ti.Vector.field(n=3, dtype=ti.f32, shape=self.resolution)
        self.image: ti.Field = ti.Vector.field(n=3, dtype=ti.f32, shape=screenRes)

        # Ray variables ------------ |
        self.rays: ti.StructField = ti.Struct.field({
            "pos"      : ti.math.vec3, # cartesian
            "prev_pos" : ti.math.vec3, # previous pos value
            "rot"      : ti.math.vec3, # angles
            "step"     : ti.math.vec3, # step vector (dx, dy, dz)
            "acc"      : ti.math.vec3, # unused in new logic
            "vel"      : ti.math.vec3, # cartesian (dx/dλ, dy/dλ, dz/dλ)
            "E"        : ti.f32,       # conserved energy
            "hit"      : ti.i32,       # is hit(0=not hit; 1=hit from an object; 2=timed out; 3=passed through volume)
            "col"      : ti.math.vec3, # color
            "abd"      : ti.math.vec4  # albedo color + alpha chanel
            }, shape=self.resolution)

        self.distances: ti.MatrixField = ti.Matrix.field(
                n=self.object_count,
                m=1,
                dtype=ti.f32,
                shape=self.rays.shape
                )
        self.prev_dist: ti.Field = ti.field(
                dtype=ti.f32,
                shape=(self.distances.shape[0], self.distances.shape[1], self.object_count)
                )

        self.rk4: ti.Field = ti.Vector.field(3, dtype=ti.f32, shape=(self.rays.shape[0],self.rays.shape[1], 2, 4))

    @ti.kernel
    def init(self):

        f = self.rotation[None]

        rd = tim.normalize(tim.cross(f, ti.Vector([0.0,1.0,0.0])))
        ud = tim.cross(rd, f)

        for i,j in self.image:
            self.image[i,j] = ti.Vector([0.0,1.0,1.0])

        # Initialize rays ---------- |
        for x,y in self.rays:
            # x, y are ti.i32 indices

            u = (x / (self.rays.shape[0]-1) - 0.5) * self.dimensions[None].x
            v = (y / (self.rays.shape[1]-1) - 0.5) * self.dimensions[None].y


            self.rays[x,y].pos       = self.position[None]
            self.rays[x,y].rot       = tim.normalize(f + u*rd + v*ud)

            r_pos = self.rays[x,y].pos
            r = r_pos.norm()
            F = 1.0 - SZR / r

            r_c = (c * self.rays[x,y].rot) # c is a float constant

            dt_dl = c / ti.sqrt(F) 
            self.rays[x,y].E         = F * dt_dl

            self.rays[x,y].step      = r_c * self.dt # (dx, dy, dz)
            self.rays[x,y].vel       = r_c # (dx/dλ, dy/dλ, dz/dλ)
            
            self.rays[x,y].hit         = 0
            self.rays[x,y].col     = ti.Vector([1.0,1.0,1.0])
            self.rays[x,y].abd    = ti.Vector([0.0,0.0,0.0,0.0])

        self.prev_dist.fill(-1.0)
        self.dimage.fill(0.0)
        self.image.fill(0.0)
        self.temp_bloom.fill(0.0)
        self.temp_blur.fill(0.0)

    @ti.func
    def geodesic(self, x: ti.i32, y: ti.i32, r: ti.f32, vel_cart: ti.math.vec3, r_pos: ti.math.vec3, k: ti.i32):
        f = 1.0 - SZR / r
        E = self.rays[x,y].E
        
        dr_dl = (r_pos.dot(vel_cart)) / r
        
        dt_dl = E / f

        # Ensure all literals are floats
        term1 = -(SZR / (2.0 * r**3.0)) * ( (dt_dl**2.0 / f) - dr_dl**2.0 )
        term2 = -(SZR / r**3.0) * (1.0/f - 1.0) * dr_dl

        accel = term1 * r_pos + term2 * vel_cart

        self.rk4[x,y,0,k] = vel_cart
        self.rk4[x,y,1,k] = accel

    @ti.func
    def rk4step(self, x: ti.i32, y: ti.i32):

        r_pos_0 = self.rays[x,y].pos
        r0 = r_pos_0.norm()
        vel_cart_0 = self.rays[x,y].vel # (dx/dλ, dy/dλ, dz/dλ)

        # k1
        self.geodesic(x,y, r0, vel_cart_0, r_pos_0, 0)

        # k2
        r_pos_1 = r_pos_0 + 0.5 * self.dt * self.rk4[x,y,0,0]
        r1 = r_pos_1.norm()
        vel_cart_1 = vel_cart_0 + 0.5 * self.dt * self.rk4[x,y,1,0]
        self.geodesic(x,y, r1, vel_cart_1, r_pos_1, 1)

        # k3
        r_pos_2 = r_pos_0 + 0.5 * self.dt * self.rk4[x,y,0,1]
        r2 = r_pos_2.norm()
        vel_cart_2 = vel_cart_0 + 0.5 * self.dt * self.rk4[x,y,1,1]
        self.geodesic(x,y, r2, vel_cart_2, r_pos_2, 2)

        # k4
        r_pos_3 = r_pos_0 + self.dt * self.rk4[x,y,0,2]
        r3 = r_pos_3.norm()
        vel_cart_3 = vel_cart_0 + self.dt * self.rk4[x,y,1,2]
        self.geodesic(x,y, r3, vel_cart_3, r_pos_3, 3)

        r_pos_new = r_pos_0 + (self.dt/6.0) * (self.rk4[x,y,0,0] + 2.0*self.rk4[x,y,0,1] + 2.0*self.rk4[x,y,0,2] + self.rk4[x,y,0,3])

        vel_cart_new = vel_cart_0 + (self.dt/6.0) * (self.rk4[x,y,1,0] + 2.0*self.rk4[x,y,1,1] + 2.0*self.rk4[x,y,1,2] + self.rk4[x,y,1,3])

        self.rays[x,y].vel = vel_cart_new

        self.rays[x,y].step = r_pos_new - r_pos_0

    
    @ti.func
    def step(self, x: ti.i32, y: ti.i32):
        if self.gravEnable[None]:
            self.rk4step(x,y)
        else:
            self.rays[x,y].step = (c * self.rays[x,y].rot) * self.dt
        self.rays[x,y].prev_pos = self.rays[x,y].pos
        self.rays[x,y].pos += self.rays[x,y].step

    @ti.func
    def box_blur_1d(self, x: ti.i32, y: ti.i32, is_horizontal: ti.template(), src: ti.template(), dst: ti.template(), radius: ti.i32):
        color_sum = ti.Vector([0.0, 0.0, 0.0])
        count = 0

        if ti.static(is_horizontal):
            # Horizontal Pass
            for k in range(-radius, radius + 1):
                sample_x = ti.max(0, ti.min(x + k, self.resolution[0] - 1))
                color_sum += src[sample_x, y]
                count += 1
        else:
            # Vertical Pass
            for k in range(-radius, radius + 1):
                sample_y = ti.max(0, ti.min(y + k, self.resolution[1] - 1))
                color_sum += src[x, sample_y]
                count += 1

        dst[x, y] = color_sum / ti.cast(count, ti.f32)

    @ti.kernel
    def frame(self):
        # x, y are ti.i32 indices
        for x,y in self.rays:

            for _ in range(self.max_steps):

                # step
                if not self.rays[x,y].hit:
                    self.step(x,y)
                
                # early exit for performance
                if not self.rays[x,y].hit:
                    
                    # collision and distance check
                    self.blackHole_df(x,y,sdf=1)
                    self.accretionDisk_df(x,y,sdf=2)

                    self.planet_df(x,y, # hour planet
                                   ti.Vector([17.5,self.temp[0]]),
                                   ti.Vector([.7,.6,.4]),
                                   r_s*1.0,
                                   sdf=3
                                   )
                    
                    self.planet_df(x,y, # minute planet
                                   ti.Vector([13.5,self.temp[1]]),
                                   ti.Vector([.3,.4,.6]),
                                   r_s*.6,
                                   sdf=3
                                   )

                    self.planet_df(x,y, # second planet
                                   ti.Vector([9.0,self.temp[2]]),
                                   ti.Vector([.5,.4,.5]),
                                   r_s*0.4,
                                   sdf=4
                                   )
                    self.grid_df(x,y, sdf=5)

                    d = ti.cast(self.max_steps * c, ti.f32)
                    all_away = 1
                    for i in range(self.object_count):
                        d = self.distances[x,y][i,0]

                        # for the first step
                        if self.prev_dist[x,y,i] < 0.0:
                            self.prev_dist[x,y,i] = d

                        if self.prev_dist[x,y,i] >= d:
                            all_away = 0

                        self.prev_dist[x,y,i] = d
                    
                    dist = (self.rays[x,y].pos.x**2) + (self.rays[x,y].pos.y**2) + (self.rays[x,y].pos.z**2)
                    if all_away and not self.rays[x,y].hit or dist > self.max_dist**2:
                        self.rays[x,y].hit = 2

            # change the color if the ray had missed
            if self.rays[x,y].hit != 1:
                topB = .5
                botB = .25
                biasB = 0.0

                topG = .0
                botG = .0
                biasG = .1
                
                topR = .3
                botR = .15
                biasR = 0.0
                
                # rays[x,y].rot is vec3 (x,y,z)
                colB = (botB+((self.rays[x,y].rot.x+1.0)/2.0) * (topB-botB))+biasB
                colG = (botG+((self.rays[x,y].rot.x+1.0)/2.0) * (topG-botG))+biasG
                colR = (botR+((self.rays[x,y].rot.x+1.0)/2.0) * (topR-botR))+biasR

                self.rays[x,y].col *= ti.Vector([colR,colG,colB])
                self.rays[x,y].col /= 4.0

            # tim.mix arguments should be of the same type (vec3)
            self.rays[x,y].col = tim.mix(self.rays[x,y].col, self.rays[x,y].abd.rgb, self.rays[x,y].abd.a)

        # compile rays into an image
        for x,y in self.rays:
            self.dimage[x,y] = self.rays[x,y].col

        # get the bright areas
        treshold = .75
        for x,y in self.dimage:
            col = self.dimage[x,y]
            lum = col.x * 0.2126 + col.y * 0.7152 + col.z * 0.0722
            if lum >= treshold:
                scale = (lum - treshold) / lum 
                self.temp_bloom[x,y] = self.dimage[x,y] * scale

        # blur the bright areas
        blrr = 15
        for x, y in self.dimage:
            self.box_blur_1d(x, y, ti.static(True), self.temp_bloom, self.temp_blur, blrr)
        self.temp_bloom.fill(0.0)
        for x, y in self.dimage:
            self.box_blur_1d(x, y, ti.static(False), self.temp_blur, self.temp_bloom, blrr)
        self.temp_blur.fill(0.0)
        
        # add the bloom to the main image
        bloom = 2.0
        for x,y in self.dimage:         
            self.dimage[x,y] += self.temp_bloom[x,y]*bloom
        
        # blur a little for aa
        blrr = 0
        for x, y in self.dimage:
            self.box_blur_1d(x, y, ti.static(True), self.dimage, self.temp_blur, blrr)
        self.dimage.fill(0.0)
        for x, y in self.dimage:
            self.box_blur_1d(x, y, ti.static(False), self.temp_blur, self.dimage, blrr)
        self.temp_blur.fill(0.0)

        # upscaling
        if self.smtpEnabled:
            self.sstp_interp()
        else:
            self.nn_interp()

    @ti.func
    def nn_interp(self):
        # x, y are ti.i32 indices for self.image
        for x,y in self.image:
            fx = x * self.dimage.shape[0] / self.image.shape[0]
            fy = y * self.dimage.shape[1] / self.image.shape[1]
            rx = ti.cast(tim.round(fx), ti.i32)
            ry = ti.cast(tim.round(fy), ti.i32)
            rx = ti.max(0, ti.min(rx, self.dimage.shape[0] - 1))
            ry = ti.max(0, ti.min(ry, self.dimage.shape[1] - 1))

            self.image[x,y] = self.dimage[rx, ry]

    @ti.func
    def sstp_interp(self):
        # x, y are ti.i32 indices for self.image
        for x,y in self.image:
            fx = x * self.dimage.shape[0] / self.image.shape[0]
            fy = y * self.dimage.shape[1] / self.image.shape[1]
            x0 = ti.cast(fx, ti.i32) # int(fx) implies truncation
            y0 = ti.cast(fy, ti.i32) # int(fy) implies truncation
            x1 = ti.min(x0 + 1, self.dimage.shape[0] - 1)
            y1 = ti.min(y0 + 1, self.dimage.shape[1] - 1)
            tx = fx - ti.cast(x0, ti.f32)
            ty = fy - ti.cast(y0, ti.f32)
            sx = tim.smoothstep(0.0, 1.0, tx)
            sy = tim.smoothstep(0.0, 1.0, ty)
            c00 = self.dimage[x0, y0]
            c10 = self.dimage[x1, y0]
            c01 = self.dimage[x0, y1]
            c11 = self.dimage[x1, y1]
            cx0 = c00 * (1.0 - sx) + c10 * sx
            cx1 = c01 * (1.0 - sx) + c11 * sx
            self.image[x,y] = cx0 * (1.0 - sy) + cx1 * sy


    # distance functions(colliders)
    @ti.func
    def planet_df(self, x: ti.i32, y: ti.i32, pol: ti.math.vec2, col: ti.math.vec3, rad: ti.f32, sdf: ti.i32):
        # pol.x is radius on xz-plane, pol.y is angle
        center = ti.Vector([0.0,0.0,0.0])
        center.x = pol.x * ti.cos(pol.y)
        center.z = pol.x * ti.sin(pol.y)
        radius = rad

        p0 = self.rays[x, y].prev_pos - center
        p1 = self.rays[x, y].pos - center
        d = p1 - p0
        m = p0

        if sdf > 0:
            self.distances[x, y][sdf-1,0] = (self.rays[x, y].pos - center).norm() - radius

        a = d.dot(d)
        b = 2.0 * m.dot(d)
        c_val = m.dot(m) - radius * radius # Renamed c to c_val to avoid conflict with imported 'c'

        ds = b * b - 4.0 * a * c_val

        if ds >= 0.0:
            sqrt_disc = ti.sqrt(ds)
            t0 = (-b - sqrt_disc) / (2.0 * a)
            t1 = (-b + sqrt_disc) / (2.0 * a)

            t_hit = 1e9
            if 0.0 <= t0 <= 1.0:
                t_hit = t0
            elif 0.0 <= t1 <= 1.0:
                t_hit = t1

            if t_hit <= 1.0:
                hit_pos = p0 + t_hit * d # Relative hit position
                self.rays[x, y].pos = self.rays[x, y].prev_pos + t_hit * d # Absolute hit position

                normal = hit_pos.normalized() 
                light_dir = (center-ti.Vector([0.0,0.0,0.0])).normalized() 
                tint = ti.Vector([.75**1.75,.75**2.0,.75**1.1])
                diffuse_factor = ti.max(0.0, -normal.dot(light_dir)) 
                factor = 0.1 + 0.9 * diffuse_factor * tint 

                self.rays[x, y].col *= col * factor
                self.rays[x, y].hit = 1

    @ti.func
    def blackHole_df(self, x: ti.i32, y: ti.i32, sdf: ti.i32):
        center = ti.Vector([0.0,0.0,0.0])
        radius = r_s

        p0 = self.rays[x, y].prev_pos - center
        p1 = self.rays[x, y].pos - center
        d = p1 - p0
        m = p0

        if sdf > 0:
            self.distances[x, y][sdf-1,0] = (self.rays[x,y].pos - center).norm() - radius

        a = d.dot(d)
        b = 2.0 * m.dot(d)
        c_val = m.dot(m) - radius * radius

        ds = b * b - 4.0 * a * c_val

        if ds >= 0.0:
            sqrt_disc = ti.sqrt(ds)
            t0 = (-b - sqrt_disc) / (2.0 * a)
            t1 = (-b + sqrt_disc) / (2.0 * a)

            t_hit = 1e9
            if 0.0 <= t0 <= 1.0:
                t_hit = t0
            elif 0.0 <= t1 <= 1.0:
                t_hit = t1

            if t_hit <= 1.0:
                hit_pos = p0 + t_hit * d # Relative hit position
                self.rays[x, y].pos = hit_pos + center # Absolute hit position
                self.rays[x, y].col *= ti.Vector([0.0,0.0,0.0])
                self.rays[x, y].hit = 1

    @ti.func
    def accretionDisk_df(self, x: ti.i32, y: ti.i32, sdf: ti.i32):
        coord = ti.Vector([0.0,0.0,0.0])
        pos = self.rays[x, y].pos - coord
        prev_pos = self.rays[x, y].prev_pos

        outer_rad = r_s * 4.5
        inner_rad = r_s * 1.45
        outer_radius2 = outer_rad**2
        inner_radius2 = inner_rad**2

        dist_xy = tim.sqrt(pos.x**2 + pos.z**2)

        if sdf > 0:
            self.distances[x,y][sdf-1,0] = ti.max(
                                 ti.abs(pos.y),
                                 ti.max(
                                     inner_rad - dist_xy,
                                     dist_xy - outer_rad
                                 )
                             )


        if prev_pos.y * pos.y < 0.0:
            t = -prev_pos.y / (pos.y - prev_pos.y)
            hit_pos = prev_pos + t * (pos - prev_pos)

            dist2 = hit_pos.x**2 + hit_pos.z**2

            if dist2 <= outer_radius2 and dist2 >= inner_radius2:
                self.rays[x, y].pos = hit_pos
                
                r_hit = ti.sqrt(dist2)
                t_norm = (r_hit - inner_rad) / (outer_rad - inner_rad)
                
                brightness = 1.0 - t_norm

                v = (t_norm)+0.57418
                j = 1.0-(v+(0.4560479-(sqrt((v*6.1)-0.3))))**100
                alpha = 1.7224+((j-6.11768)/(6.11768*1.158))

                color = ti.Vector((brightness**1.5, brightness**2.5, (brightness**1.1)/0.9))

                if self.rays[x,y].abd.r != 0 and self.rays[x,y].abd.g != 0 and self.rays[x,y].abd.b != 0:
                    self.rays[x,y].abd.rgb = tim.mix(
                            color,
                            self.rays[x,y].abd.rgb + color*0.5,
                            self.rays[x,y].abd.a
                            )
                else:
                    self.rays[x,y].abd.a = alpha
                    self.rays[x,y].abd.rgb = color
                
    @ti.func
    def schwarzschild_warp(self, r_xz: ti.f32) -> ti.f32:
        rs = SZR
        visual_scale = 2.0
        y_offset = -rs * 5.0
        deltaY_warp = 2.0 * ti.sqrt(ti.max(0.0, rs * (r_xz - rs)))
        height_warp = visual_scale * deltaY_warp + y_offset
        height_pit = visual_scale * 2.0 * rs + y_offset
        # ti.select is used for ternary operator
        return (ti.select(r_xz > rs, height_warp, height_pit))-(visual_scale*9.0)

    @ti.func
    def grid_df(self, x: ti.i32, y: ti.i32, sdf: ti.i32):
        pos = self.rays[x, y].pos
        prev_pos = self.rays[x, y].prev_pos
        grid_spacing = r_s
        epsilon = 0.05
        cutoff = 20.0

        r_xz = ti.sqrt(pos.x**2 + pos.z**2)
        r_xz_prev = ti.sqrt(prev_pos.x**2 + prev_pos.z**2)

        warped_plane_y = self.schwarzschild_warp(r_xz)

        f_prev = prev_pos.y - self.schwarzschild_warp(r_xz_prev)
        f_curr = pos.y - warped_plane_y

        if sdf > 0:
            dx = ti.max(ti.abs(pos.x) - cutoff, 0.0)
            dz = ti.max(ti.abs(pos.z) - cutoff, 0.0)
            dy = ti.max(ti.min(pos.y - warped_plane_y, 0.1), 0.0)
            self.distances[x, y][sdf-1, 0] = ti.sqrt(dx*dx + dy*dy + dz*dz)

        if f_prev * f_curr <= 0.0 and f_prev != 0.0:
            t = ti.abs(f_prev) / (ti.abs(f_prev) + ti.abs(f_curr))
            if 0.0 <= t <= 1.0:
                hit_pos = prev_pos + t * (pos - prev_pos)
                hit_r_xz = ti.sqrt(hit_pos.x**2 + hit_pos.z**2)

                dist_x = ti.abs(tim.fract(hit_pos.x / grid_spacing) - 0.5) * grid_spacing
                dist_z = ti.abs(tim.fract(hit_pos.z / grid_spacing) - 0.5) * grid_spacing

                if (dist_x < epsilon or dist_z < epsilon) and hit_r_xz > SZR and ((prev_pos.x**2) < cutoff**2 and (prev_pos.z**2) < cutoff**2):
                    self.rays[x, y].pos = hit_pos
                    self.rays[x, y].col *= ti.Vector([.8, .8, .9])
                    self.rays[x, y].hit = 1

    def Render(self):
        self.init()
        self.frame()