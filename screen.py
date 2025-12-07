import time as tm
import taichi as ti
import numpy as np
from colorama import Fore
from colorama import init as colorama_init

from config import screenDimensions, camRes, screen_size 
from camera import Camera

class Screen:
    def __init__(self):
        # Start initialization timer and colorama
        start = tm.perf_counter()
        colorama_init()
        print(f"{Fore.YELLOW}initializing...")

        # Initialize taichi backend
        ti.init(arch=ti.vulkan) 
        print(f"{Fore.GREEN}> initialized taichi")

        self.cam_resolution = tuple(camRes)
        self.resolution = tuple(screenDimensions)
        
        # Calculate aspect ratio
        g = np.gcd(self.cam_resolution[0], self.cam_resolution[1])
        ax = self.cam_resolution[0] // g
        ay = self.cam_resolution[1] // g

        # Create the camera instance
        self.cam = Camera(
                ti.Vector((15, 15, 15), dt=ti.f32),
                ti.Vector((0,0,0), dt=ti.f32),
                self.cam_resolution,
                self.resolution,
                ti.Vector((screen_size*ax/10,screen_size*ay/10, 0), dt=ti.f32),
                )
        
        end = tm.perf_counter()
        print(f"initialized with: aspect ratio{ax}/{ay}; camera resolution:{self.cam_resolution}; screen resolution {self.resolution}")
        print(f"{Fore.GREEN}time to initialize screen: start:{start}, end:{end}, elapsed:{end-start}")

    def Start(self):
        gui = ti.GUI("black hole integrator", self.resolution, fast_gui=True)
    
        # Camera orbit control state
        rot = [np.pi/6, -np.pi/16]
        target = ti.Vector([0.0, 0.0, 0.0], dt=ti.f32)
        rotation = np.array(rot)    # yaw, pitch
        # Use .to_numpy() to get a copy of the camera position vector for distance calculation
        distance = np.linalg.norm((self.cam.position[None] - target).to_numpy())
        default_dis = distance
        default_tgt = target.to_numpy().copy() 

        last_mouse = None
        rotate_speed = 0.005
        pan_speed = 0.2
        zoom_speed = 1

        while gui.running:
            start = tm.perf_counter()
            
            # Process GUI events (Press, Release, Wheel)
            for e in gui.get_events(ti.GUI.PRESS, ti.GUI.RELEASE, ti.GUI.WHEEL):
                
                # Handle scroll wheel (Zoom)
                if e.key == ti.GUI.WHEEL and e.delta[1] != 0:
                    distance -= zoom_speed * np.sign(e.delta[1])
                    distance = max(0.1, min(distance, self.cam.max_dist)) 
                    
                if e.type == ti.GUI.PRESS:
                    # Quit
                    if e.key == 'q': 
                        gui.running = False
                    
                    # Reset camera position/target
                    if e.key == 'r':
                        # FIX: Reassign the local ti.Vector 'target' using the default NumPy array.
                        # This resolves the 'AttributeError' because 'from_numpy' only exists on Taichi fields.
                        target = ti.Vector(default_tgt, dt=ti.f32) 
                        rotation = np.array(rot)
                        distance = default_dis
                        
                    # Toggle gravity
                    if e.key == 'g':
                        self.cam.gravEnable[None] = 1 - self.cam.gravEnable[None]

            # Mouse input for rotation and panning
            mpos = np.array(gui.get_cursor_pos())
            mpos[0] *= self.resolution[0]
            mpos[1] *= self.resolution[1]

            if gui.is_pressed(ti.GUI.LMB) or gui.is_pressed(ti.GUI.RMB):
                if last_mouse is not None:
                    delta = mpos - last_mouse
                    
                    if gui.is_pressed(ti.GUI.LMB):
                        # Orbit rotation
                        rotation[0] -= delta[0] * rotate_speed
                        rotation[1] += delta[1] * rotate_speed
                        rotation[1] = np.clip(rotation[1], -np.pi/2+0.01, np.pi/2-0.01)
                        
                    elif gui.is_pressed(ti.GUI.RMB):
                        # Pan the target point
                        right = np.array([np.sin(rotation[0] - np.pi/2), 0.0, np.cos(rotation[0] - np.pi/2)])
                        forward = np.array([np.sin(rotation[0]), 0.0, np.cos(rotation[0])])
                        
                        move = -delta[0] * pan_speed * right*.1 - delta[1] * pan_speed * forward*.1
                        
                        target_np = target.to_numpy() + move
                        target = ti.Vector(target_np, dt=ti.f32)
                        
                last_mouse = mpos.copy()
            else:
                last_mouse = None
                
            # Keyboard movement (WASD/ZX)
            yaw, pitch = rotation
            
            # Calculate direction vectors
            forward = np.array([
                np.cos(pitch) * np.sin(yaw),
                np.sin(pitch),
                np.cos(pitch) * np.cos(yaw)
            ])
            right = np.array([np.sin(yaw - np.pi/2), 0.0, np.cos(yaw - np.pi/2)])
            up = np.array([0.0, 1.0, 0.0]) # Global up

            move = np.zeros(3, dtype=np.float32)

            # Accumulate movement
            if gui.is_pressed('w'): move += forward
            if gui.is_pressed('s'): move -= forward
            if gui.is_pressed('a'): move -= right
            if gui.is_pressed('d'): move += right
            if gui.is_pressed('z'): move += up
            if gui.is_pressed('x'): move -= up

            # Apply Keyboard Movement (Walk style)
            if np.linalg.norm(move) > 0:
                move = move / np.linalg.norm(move)
                move *= pan_speed
                
                # Update both target and camera position
                target_np = target.to_numpy() + move
                self.cam.position[None] += ti.Vector(move, dt=ti.f32)
                target = ti.Vector(target_np, dt=ti.f32)
                
                # Resynchronize distance after manual movement
                cam_pos_np = self.cam.position[None].to_numpy()
                distance = np.linalg.norm(cam_pos_np - target_np) 
                
            # Update time-based variables (for animation/simulation)
            p = tm.perf_counter()
            self.cam.temp[0] = np.radians(p*4)
            self.cam.temp[1] = np.radians(p*8)
            self.cam.temp[2] = np.radians(p*12)

            # Calculate final camera position based on target, rotation, and distance
            yaw, pitch = rotation
            direction = np.array([
                np.cos(pitch) * np.sin(yaw),
                np.sin(pitch),
                np.cos(pitch) * np.cos(yaw)
            ])
            cam_pos = target.to_numpy() - direction * distance

            # Update camera fields in Taichi
            self.cam.position[None] = ti.Vector(cam_pos, dt=ti.f32)
            self.cam.rotation[None] = ti.Vector(direction, dt=ti.f32)

            # Render and display
            self.cam.Render()
            gui.set_image(self.cam.image)
            gui.show()

            end = tm.perf_counter()
            print(f"{Fore.YELLOW}frame time:{(end-start):.5f}; fps={int(1/(end-start))}          ", end="\r")