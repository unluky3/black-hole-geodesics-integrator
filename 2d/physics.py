import numpy as np
from rendering import circle, ray
from config import c, G
from numpy import sin, cos, sqrt

def cart_pol(x, y):
    r = sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)  # radians
    return r, theta

def pol_cart(r, theta):
    x = r * cos(theta)
    y = r * sin(theta)
    return x, y

class BlackHole(circle):
    def __init__(
        self,
        res,
        tr_x,
        tr_y,
        color,
        mass,
        context
    ):
        self.m = mass
        self.szr = ((2*G*self.m)/c**2)
        super().__init__(res, self.szr, tr_x, tr_y, color, context)

    def get_params(self):
        out = {
                "radius":self.szr,
                "position":self.pos
                }
        return out

class Path(ray):
    dl = .5

    def bh(self):
        return self.bhl[
                np.argmin(
                    [
                        np.linalg.norm(
                            bh.get_params()["position"] - self.position
                            )
                        for bh in self.bhl
                        ]
                    )
                ].get_params()
    def __init__(
            self,
            res,
            or_x,
            or_y,
            bhl,
            context,
            ):
        super().__init__(res, or_x, or_y, context)
        self.bhl = bhl
        
        self.velocity     = np.array((c, 0.0), dtype="f8")
        self.position     = np.array((or_x, or_y), dtype="f8")
        self.bh_position  = self.bh()["position"]
        self.rel_Position = np.array(self.position - self.bh_position, dtype="f8")

        self.szr          = self.bh()["radius"]
        
        self.r            = np.hypot(self.rel_Position[0], self.rel_Position[1]) # get the hypot of relative x and y
        self.phi          = np.arctan2(self.rel_Position[1], self.rel_Position[0]) # get the atan2 of relative y and x

        # in m/s
        self.dr           = self.velocity[0] * cos(self.phi) + self.velocity[1] * sin(self.phi)
        self.dphi         = ( ( - self.velocity[0] * sin(self.phi) ) + ( self.velocity[1] * cos(self.phi) ) ) / self.r

        # constants
        self.f            = 1 - (self.szr/self.r)
        self.dt_dl        = sqrt( self.dr**2 / self.f**2 + (self.r**2 * self.dphi**2) / self.f )
        self.E            = self.f * self.dt_dl
        self.hit          = False

    def reload(self):
        # stuff that needs reloading, like position and all that
        self.rel_Position = np.array(self.position - self.bh_position, dtype="f8")

        self.r            = np.hypot(self.rel_Position[0], self.rel_Position[1])
        self.phi          = np.arctan2(self.rel_Position[1], self.rel_Position[0])

        # resample the velocity
        self.dr           = ( self.velocity[0] * cos(self.phi) ) + ( self.velocity[1] * sin(self.phi) )
        self.dphi         = ( ( - self.velocity[0] * sin(self.phi) ) + ( self.velocity[1] * cos(self.phi) ) ) / self.r
        
        self.rk4()

    def geodesic(self, r, dr, dphi):
        E    = self.E
        f    = 1 - (self.szr/self.r)
        rs   = self.szr
        if rs <= 0:
            return 0,0

        dt_dl = E / f

        d2r   = -(rs / (2 * r**2 )) * (dt_dl**2) + (rs / (2 * r**2 * f)) * (dr**2) + (r-rs) * (dphi**2)
        d2phi = -2.0 * dr * dphi / r
        
        return dr, dphi, d2r, d2phi
    
    def rk4(self):
        args = np.array((self.r, self.phi, self.dr, self.dphi))
        def addState(a, b, fac): # a, b are np arrays
            return a + b * fac

        k1 = self.geodesic(self.r, self.dr, self.dphi)
        
        temp   = addState(args, np.array(k1), self.dl/2)
        k2 = self.geodesic(temp[0], temp[2], temp[3])

        temp   = addState(args, np.array(k2), self.dl/2)
        k3 = self.geodesic(temp[0], temp[2], temp[3])

        temp   = addState(args, np.array(k3), self.dl)
        k4 = self.geodesic(temp[0], temp[2], temp[3])

        self.r    += (self.dl/6)*(k1[0] + 2*k2[0] + 2*k3[0] + k4[0])
        self.phi  += (self.dl/6)*(k1[1] + 2*k2[1] + 2*k3[1] + k4[1])
        self.dr   += (self.dl/6)*(k1[2] + 2*k2[2] + 2*k3[2] + k4[2])
        self.dphi += (self.dl/6)*(k1[3] + 2*k2[3] + 2*k3[3] + k4[3])

    def step(self):
        if self.hit:
            self.render()
            return
        self.reload()
        if self.r < self.szr:
            self.render()
            self.hit = True
            return


        self.velocity[0] = self.dr * cos(self.phi) - self.r * self.dphi * sin(self.phi)
        self.velocity[1] = self.dr * sin(self.phi) + self.r * self.dphi * cos(self.phi)
        
        self.position[0] += self.velocity[0] * self.dl
        self.position[1] += self.velocity[1] * self.dl

        self.translate(self.position)
        self.render()