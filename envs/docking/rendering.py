'''
Rendering for 2D Spacecraft Docking Simulation

Created by Kai Delsing
Mentor: Kerianne Hobbs

Description:
	A class for rendering the SpacecraftDocking environment.
 
 visuals

renderSim:
    Create, run, and update the rendering
create_particle:
    Instantiate and initialize a particle object in the necessary lists
clean_particles:
    Delete particles past their ttl or all at once
close:
    Close the viewer and rendering
'''


#5/10
#   removed green circle (removed force arrows - line 120, 218, 312)
#   added phase circles (changed ellipses to circles - line 125 docking)


import math
import random
import numpy as np
import gym
from gym.envs.classic_control import rendering
# import matplotlib.pyplot as plt 
class DockingRender():
    #define class called DockingRender

    def renderSim(self, mode='human'):
        #define method (function) inside class
        #create scale-adjusted variables
        # if self.rH>=1000: 
        #     again = 0
        #     holderx = self.x_threshold 
        #     holdery = self.y_threshold 
        #     x_thresh = self.x_threshold * self.scale_factor
        #     y_thresh = self.y_threshold * self.scale_factor
        #     screen_width, screen_height = int(x_thresh * 2), int(y_thresh * 2)

        #     #create dimensions of satellites
        #     bodydim =  300 * self.scale_factor
        #     panelwid = 500 * self.scale_factor
        #     panelhei = 200 * self.scale_factor
        if  self.rH >= 100: 
            # if self.scale_factor != .6 * 500 / 1000:
            #     again=1
            #     self.scale_factor = .6 * 500 / 1000
            # else: 
            #     again = 0 
            again = 0
            holderx = self.x_threshold/(self.position_deputy/1000) 
            holdery = self.y_threshold/(self.position_deputy/1000)

            x_thresh = holderx * self.scale_factor
            y_thresh = holdery * self.scale_factor
            screen_width, screen_height = int(x_thresh * 2), int(y_thresh * 2)

            #create dimensions of satellites
            bodydim =  30 * self.scale_factor
            panelwid = 50 * self.scale_factor
            panelhei = 20 * self.scale_factor
        else: 
            if self.scale_factor != .6 * 500 / 100:
                again=1
                self.scale_factor = .6 * 500 / 100
            else: 
                again = 0 
            holderx = self.x_threshold/(self.position_deputy/100)
            holdery = self.y_threshold/(self.position_deputy/100)

            x_thresh = holderx * self.scale_factor
            y_thresh = holdery * self.scale_factor
            screen_width, screen_height = int(x_thresh * 2), int(y_thresh * 2)

            #create dimensions of satellites
            bodydim =  3 * self.scale_factor
            panelwid = 5 * self.scale_factor
            panelhei = 2 * self.scale_factor
            
        #changes size of window that opens for simulation
        if self.showRes:
            #print height and width
            print("Height: ", screen_height)
            print("Width: ", screen_width)
            self.showRes = False

        if self.viewer is None or again == 1:
            if again==0: 
                self.viewer = rendering.Viewer(screen_width, screen_height) #create render viewer object

            #SKY
            b, t, l, r = 0, y_thresh * 2, 0, x_thresh * 2  #creates sky dimensions
            sky = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])  #creates sky polygon
            self.skytrans = rendering.Transform()  #allows sky to be moved
            sky.add_attr(self.skytrans)
            # sky.set_color(self.bg_color[0], self.bg_color[1], self.bg_color[2])  #sets color of sky
            self.viewer.add_geom(sky)  #adds sky to viewer
   
            #CHIEF BODY
            b, t, l, r = -bodydim, bodydim, -bodydim, bodydim
            chief_body = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])  #creates body polygon
            self.chief_bodytrans = rendering.Transform()  #allows body to be moved
            chief_body.add_attr(self.chief_bodytrans)
            chief_body.set_color(.6, .6, .6)  #sets color of body

            #LOS
            s = self.scale_factor
            chief_los = rendering.FilledPolygon([(0, t), (-1*((self.ellipse_a1*s-t)*np.sin(self.theta_los)), (self.ellipse_a1*s-t)*np.cos(self.theta_los)+t), ((self.ellipse_a1*s-t)*np.sin(self.theta_los), (self.ellipse_a1*s-t)*np.cos(self.theta_los)+t), (0, t)])  #creates solar panel polygon
            self.chief_los = rendering.Transform()  #allows panel to be moved
            chief_los.add_attr(self.chief_los)
            chief_los.add_attr(self.chief_bodytrans) #sets panel as part of chief object
            chief_los.set_color(.2, .7, .3)  #sets color of panel

            # CHIEF DOCKING PORT 
            chief_dock_port = rendering.FilledPolygon([(0,t), (-t, 3+t), (t, 3+t), (0,t)])  
            self.chief_dock_port = rendering.Transform()  #allows body to be moved
            chief_dock_port.add_attr(self.chief_bodytrans)
            chief_dock_port.add_attr(self.chief_dock_port)
            chief_dock_port.set_color(.6, .6, .6)  #sets color of body

            #CHIEF SOLAR PANEL
            b, t, l, r = -panelhei, panelhei, -panelwid * 2, panelwid * 2
            chief_panel = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])  #creates solar panel polygon
            self.chief_panel_trans = rendering.Transform()  #allows panel to be moved
            chief_panel.add_attr(self.chief_panel_trans)
            chief_panel.add_attr(self.chief_bodytrans) #sets panel as part of chief object
            chief_panel.set_color(.2, .2, .6)  #sets color of panel

            #DEPUTY BODY
            b, t, l, r = -bodydim, bodydim, -bodydim, bodydim
            deputy_body = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])  #creates deputy body polygon
            self.deputy_bodytrans = rendering.Transform()  #allows body to be moved
            deputy_body.add_attr(self.deputy_bodytrans)
            deputy_body.set_color(.6, .6, .6)  #sets color of body

            #DEPUTY SOLAR PANEL
            b, t, l, r = -panelhei, panelhei, -panelwid * 2, panelwid * 2
            #can change by factor of 10 for actual
            deputy_panel = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])  #creates solar panel polygon
            self.deputy_panel_trans = rendering.Transform()  #allows panel to be moved
            deputy_panel.add_attr(self.deputy_panel_trans) #sets panel as part of deputy object
            deputy_panel.add_attr(self.deputy_bodytrans)
            deputy_panel.set_color(.2, .2, .6)  #sets color of panel

            # DEPUTY DOCKING PORT 
            deputy_dock_port = rendering.FilledPolygon([(0,t), (-t, 5+t), (t, 5+t), (0,t)])  

            deputy_dock_port.add_attr(self.deputy_bodytrans)
            deputy_dock_port.set_color(.6, .6, .6)  #sets color of body

            
            #VELOCITY ARROW
            if self.velocityArrow:
                velocityArrow = rendering.Line((0, 0),(panelwid * 3, 0)) #length of arrow
                self.velocityArrowTrans = rendering.Transform()
                velocityArrow.add_attr(self.velocityArrowTrans)
                velocityArrow.add_attr(self.deputy_bodytrans)
                velocityArrow.set_color(.8, .1, .1) #arrow is red           
                
            #THRUST BLOCKS
            if self.thrustVis == 'Block':
                b, t, l, r = -bodydim / 2, bodydim / 2, -panelwid, panelwid #half the panel dimensions
                L_thrust = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])  #creates thrust polygon
                self.L_thrust_trans = rendering.Transform()  #allows thrust to be moved
                L_thrust.add_attr(self.L_thrust_trans)
                L_thrust.add_attr(self.deputy_bodytrans)
                L_thrust.set_color(.7, .3, .1)  #sets color of thrust
                R_thrust = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])  #creates thrust polygon
                self.R_thrust_trans = rendering.Transform()  #allows thrust to be moved
                R_thrust.add_attr(self.R_thrust_trans)
                R_thrust.add_attr(self.deputy_bodytrans)
                R_thrust.set_color(.7, .3, .1)  #sets color of thrust

                b, t, l, r = -bodydim / 2, bodydim / 2, -bodydim / 2, bodydim / 2
                T_thrust = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])  #creates thrust polygon
                self.T_thrust_trans = rendering.Transform()  #allows thrust to be moved
                T_thrust.add_attr(self.T_thrust_trans)
                T_thrust.add_attr(self.deputy_bodytrans)
                T_thrust.set_color(.7, .3, .1)  #sets color of thrust
                B_thrust = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])  #creates thrust polygon
                self.B_thrust_trans = rendering.Transform()  #allows thrust to be moved
                B_thrust.add_attr(self.B_thrust_trans)
                B_thrust.add_attr(self.deputy_bodytrans)
                B_thrust.set_color(.7, .3, .1)  #sets color of thrust

            #STARS
            if self.stars > 0:
                for i in range(self.stars):
                    x, y = random.random() * (x_thresh * 2), random.random() * (y_thresh * 2)
                    dim = bodydim / 10
                    if dim <= 0:
                        dim = 1
                    star = rendering.make_circle(dim)  #creates trace dot
                    self.startrans = rendering.Transform()  #allows trace to be moved
                    star.add_attr(self.startrans)
                    star.set_color(.9, .9, .9)  #sets color of stars
                    self.viewer.add_geom(star)  #adds stars into render
                    self.startrans.set_translation(x,y)

            #ELLIPSES
            #docking program has values of ellipse_a1, etc.
            if self.ellipse_quality > 0:
                thetaList = []
                i = 0
                while i <= math.pi * 2:
                    thetaList.append(i)
                    i += (1 / 100) * math.pi
                dotsize = int(self.scale_factor) + 1
                if dotsize < 0:
                    dotsize += 1

                for i in range(0, len(thetaList)): #ellipse 1
                    x, y = self.ellipse_a1 * math.cos(thetaList[i]), self.ellipse_a1 * math.sin(thetaList[i])
                    x = (x * self.scale_factor) + x_thresh
                    y = (y * self.scale_factor) + y_thresh
                    dot1 = rendering.make_circle(dotsize)  #creates dot
                    self.dot1trans = rendering.Transform()  #allows dot to be moved
                    dot1.add_attr(self.dot1trans)
                    dot1.set_color(.1, .9, .1)  #sets color of larger ellipse
                    self.dot1trans.set_translation(x, y)
                    self.viewer.add_geom(dot1)  #adds dot into render

                for i in range(0, len(thetaList)): #ellipse 2
                    x, y = self.ellipse_a2 * math.cos(thetaList[i]), self.ellipse_a2 * math.sin(thetaList[i])
                    x = (x * self.scale_factor) + x_thresh
                    y = (y * self.scale_factor) + y_thresh
                    dot2 = rendering.make_circle(dotsize)  #creates dot
                    self.dot2trans = rendering.Transform()  #allows dot to be moved
                    dot2.add_attr(self.dot2trans)
                    dot2.set_color(.8, .9, .1)  #sets color of smaller ellipse
                    self.dot2trans.set_translation(x, y)
                    self.viewer.add_geom(dot2)  #adds dot into render
                    
                for i in range(0, len(thetaList)): #ellipse 3
                    #staring distance
                    x, y = self.ellipse_a3 * math.cos(thetaList[i]), self.ellipse_a3 * math.sin(thetaList[i])
                    x = (x * self.scale_factor) + x_thresh
                    y = (y * self.scale_factor) + y_thresh
                    dot3 = rendering.make_circle(dotsize)  #creates dot
                    self.dot3trans = rendering.Transform()  #allows dot to be moved
                    dot3.add_attr(self.dot3trans)
                    dot3.set_color(.6, .9, .9)  #sets color of outer ellipse
                    self.dot3trans.set_translation(x, y)
                    self.viewer.add_geom(dot3)  #adds dot into render

            self.viewer.add_geom(chief_panel)  #adds solar panel to viewer
            self.viewer.add_geom(chief_los)
            self.viewer.add_geom(chief_dock_port)
            self.viewer.add_geom(chief_body)  #adds satellites to viewer
  
            if self.thrustVis == 'Block':
                self.viewer.add_geom(L_thrust)  #adds solar panel to viewer
                self.viewer.add_geom(R_thrust)  #adds solar panel to viewer
                self.viewer.add_geom(T_thrust)  #adds solar panel to viewer
                self.viewer.add_geom(B_thrust)  #adds thrust into viewer

            self.viewer.add_geom(deputy_panel)  #adds solar panel to viewer

            if self.velocityArrow:
                self.viewer.add_geom(velocityArrow)  #adds velocityArrow to viewer

            # if self.forceArrow:
            #     self.viewer.add_geom(forceArrow)  #adds forceArrow to viewer

            self.viewer.add_geom(deputy_body)  #adds body to viewer
            self.viewer.add_geom(deputy_dock_port)

        if self.state is None:  #if there is no state (either the simulation has not begun or it has ended), end
            print('No state')
            return None

        x = self.state
        tx, ty = (x[0] + holderx) * self.scale_factor, (x[1] + holdery) * self.scale_factor  #pulls the state of the x and y coordinates
        self.deputy_bodytrans.set_translation(tx, ty)  #translate deputy
        self.chief_bodytrans.set_translation(self.x_chief + x_thresh, self.y_chief + y_thresh)  #translate chief
        #(tx,ty) gives speed in x and y direction

        #PARTICLE DYNAMICS
        if self.thrustVis == 'Particle':
            lx, ly = (x[0]) * self.scale_factor, (x[1]) * self.scale_factor
            v = random.randint(-self.p_var, self.p_var)
            if self.x_force > 0:
                DockingRender.create_particle(self, self.p_velocity, 180 + v, lx, ly, self.p_ttl)
            elif self.x_force < 0:
                DockingRender.create_particle(self, self.p_velocity, 0 + v, lx, ly, self.p_ttl)
            if self.y_force > 0:
                DockingRender.create_particle(self, self.p_velocity, 270 + v, lx, ly, self.p_ttl)
            elif self.y_force < 0:
                DockingRender.create_particle(self, self.p_velocity, 90 + v, lx, ly, self.p_ttl)
    

            for i in range(0, len(self.particles)):
                #velocity, theta, x, y, ttl
                self.particles[i][4] -= 1 #decrement the ttl
                r = (self.particles[i][1] * math.pi) / 180
                
                #gives speed of force particles in x and y directions
                self.particles[i][2] += (self.particles[i][0] * math.cos(r))
                self.particles[i][3] += (self.particles[i][0] * math.sin(r))

            DockingRender.clean_particles(self, False)

            #translate & rotate all particles
            for i in range(0, len(self.p_obj)):
                self.trans[i].set_translation(x_thresh + self.particles[i][2], y_thresh + self.particles[i][3])  #translate particle
                self.trans[i].set_rotation(self.particles[i][1])

        #TRACE DOTS
        if self.trace != 0: #if trace enabled, draw trace
            if self.tracectr == self.trace: #if time to draw a trace, draw, else increment counter
                if self.traceMin:
                    tracewidth = 1
                else:
                    tracewidth = int(bodydim / 8) + 1

                trace = rendering.make_circle(tracewidth)  #creates trace dot
                self.tracetrans = rendering.Transform()  #allows trace to be moved
                trace.add_attr(self.tracetrans)
                trace.set_color(.9, .1, .9)  #sets color of trace as pink
                self.viewer.add_geom(trace)  #adds trace into render
                self.tracectr = 0
            else:
                self.tracectr += 1
                
        self.tracetrans.set_translation(tx, ty)  #translate trace to follow chaser

        #BLOCK THRUSTERS
        if self.thrustVis == 'Block':
            inc_l, inc_r, inc_b, inc_t = -25, 25, -5, 5 #create block dimensions
            #calculate block translations
            if self.x_force > 0:
                inc_l = -65 * self.scale_factor
                inc_r = 25 * self.scale_factor
            elif self.x_force < 0:
                inc_r = 65 * self.scale_factor
                inc_l = -25 * self.scale_factor
            if self.y_force > 0:
                inc_b = -35 * self.scale_factor
                inc_t = 5 * self.scale_factor
            elif self.y_force < 0:
                inc_t = 35 * self.scale_factor
                inc_b = -5 * self.scale_factor

            #translate blocks
            self.L_thrust_trans.set_translation(inc_l, 0)
            self.R_thrust_trans.set_translation(inc_r, 0)
            self.T_thrust_trans.set_translation(0, inc_t)
            self.B_thrust_trans.set_translation(0, inc_b)

        #VELOCITY ARROW
        if self.velocityArrow:
            tv = math.atan(x[3] / x[2]) #angle of velocity arrow that points in direction of motion
            if x[2] < 0: #arctan adjustment
                tv += math.pi
            self.velocityArrowTrans.set_rotation(tv)

        # #FORCE ARROW#
        # if self.forceArrow:
        #     if self.x_force == 0:
        #         tf = math.atan(0) #angle of velocity
        #     else:
        #         tf = math.atan(self.y_force / self.x_force) #angle of velocity
        #     if self.x_force < 0: #arctan adjustment
        #         tf += math.pi
        #     self.forceArrowTrans.set_rotation(tf)

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def create_particle(self, velocity, theta, x, y, ttl):
        p = [velocity, theta, x, y, ttl]
        obj_len = len(self.p_obj) #position of particle in list
        p_len = len(self.particles) #position of particle in list
        trans_len = len(self.trans) #position of particle in list

        self.particles.append(p)
        self.p_obj.append(self.particles[p_len])
        self.p_obj[obj_len] = rendering.make_circle(1)  #creates particle dot
        self.trans.append(rendering.Transform())  #allows particle to be moved
        self.p_obj[obj_len].add_attr(self.trans[trans_len])
        self.p_obj[obj_len].set_color(.9, .9, .6)  #sets color of particle

        self.trans[trans_len].set_translation(self.particles[p_len][2], self.particles[p_len][3])  #translate particle
        self.trans[trans_len].set_rotation(self.particles[p_len][1])
        self.viewer.add_geom(self.p_obj[obj_len])  #adds particle into render

        DockingRender.clean_particles(self, False)
        return p

    def clean_particles(self, all):
        while self.particles and (all or self.particles[0][4] < 0): #if all or if the first particle has reached its ttl
            self.p_obj[0].set_color(self.bg_color[0], self.bg_color[1], self.bg_color[2])  #sets color of particle
            self.particles.pop(0) #delete particle at beginning of list
            self.p_obj.pop(0) #position of particle in list
            self.trans.pop(0) #position of particle in list

    def close(self):  #if a viewer exists, close and kill it
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

