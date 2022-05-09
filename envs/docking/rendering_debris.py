'''
Rendering for 2D Spacecraft Docking Simulation

Created by Kai Delsing
Mentor: Kerianne Hobbs

Description:
	A class for rendering the SpacecraftDocking environment.

renderSim:
    Create, run, and update the rendering
create_particle:
    Instantiate and initialize a particle object in the necessary lists
clean_particles:
    Delete particles past their ttl or all at once
close:
    Close the viewer and rendering
'''


import math
import random
import gym
from gym.envs.classic_control import rendering


class DockingRender():

    def renderSim(self, mode='human'):
        # create scale-adjusted variables
        x_thresh = self.x_threshold * self.scale_factor
        y_thresh = self.y_threshold * self.scale_factor
        screen_width, screen_height = int(x_thresh * 2), int(y_thresh * 2)

        if self.showRes:
            # print height and width
            print("Height: ", screen_height)
            print("Width: ", screen_width)
            self.showRes = False

        # create dimensions of satellites
        bodydim = 30 * self.scale_factor
        panelwid = 50 * self.scale_factor
        panelhei = 20 * self.scale_factor

        # create dimensions of obstacles
        obdim = 10 * self.scale_factor

        if self.viewer is None:
            # create render viewer object
            self.viewer = rendering.Viewer(screen_width, screen_height)

            #SKY#
            b, t, l, r = 0, y_thresh * 2, 0, x_thresh * 2  # creates sky dimensions
            sky = rendering.FilledPolygon(
                [(l, b), (l, t), (r, t), (r, b)])  # creates sky polygon
            self.skytrans = rendering.Transform()  # allows sky to be moved
            sky.add_attr(self.skytrans)
            # sets color of sky
            sky.set_color(self.bg_color[0], self.bg_color[1], self.bg_color[2])
            self.viewer.add_geom(sky)  # adds sky to viewer

            #DEPUTY BODY#
            b, t, l, r = -bodydim, bodydim, -bodydim, bodydim
            deputy_body = rendering.FilledPolygon(
                [(l, b), (l, t), (r, t), (r, b)])  # creates deputy body polygon
            self.deputy_bodytrans = rendering.Transform()  # allows body to be moved
            deputy_body.add_attr(self.deputy_bodytrans)
            deputy_body.set_color(.5, .5, .5)  # sets color of body

            #DEPUTY SOLAR PANEL#
            b, t, l, r = -panelhei, panelhei, -panelwid * 2, panelwid * 2
            deputy_panel = rendering.FilledPolygon(
                [(l, b), (l, t), (r, t), (r, b)])  # creates solar panel polygon
            self.deputy_panel_trans = rendering.Transform()  # allows panel to be moved
            # sets panel as part of deputy object
            deputy_panel.add_attr(self.deputy_panel_trans)
            deputy_panel.add_attr(self.deputy_bodytrans)
            deputy_panel.set_color(.2, .2, .5)  # sets color of panel

            # OBSTACLES
            b, t, l, r = -obdim, obdim, -obdim, obdim
            obstacle_body = rendering.FilledPolygon(
                [(l, b), (l, t), (r, t), (r, b)])  # creates deputy body polygon
            self.obstacle_bodytrans = rendering.Transform()  # allows body to be moved
            obstacle_body.add_attr(self.obstacle_bodytrans)
            obstacle_body.set_color(.7, .3, .1)  # sets color of obstacle

            b, t, l, r = -obdim, obdim, -obdim, obdim
            obstacle_body1 = rendering.FilledPolygon(
                [(l, b), (l, t), (r, t), (r, b)])  # creates deputy body polygon
            self.obstacle_bodytrans1 = rendering.Transform()  # allows body to be moved
            obstacle_body1.add_attr(self.obstacle_bodytrans1)
            obstacle_body1.set_color(.7, .3, .1)  # sets color of obstacle

            b, t, l, r = -obdim, obdim, -obdim, obdim
            obstacle_body2 = rendering.FilledPolygon(
                [(l, b), (l, t), (r, t), (r, b)])  # creates deputy body polygon
            self.obstacle_bodytrans2 = rendering.Transform()  # allows body to be moved
            obstacle_body2.add_attr(self.obstacle_bodytrans2)
            obstacle_body2.set_color(.7, .3, .1)  # sets color of obstacle

            b, t, l, r = -obdim, obdim, -obdim, obdim
            obstacle_body3 = rendering.FilledPolygon(
                [(l, b), (l, t), (r, t), (r, b)])  # creates deputy body polygon
            self.obstacle_bodytrans3 = rendering.Transform()  # allows body to be moved
            obstacle_body3.add_attr(self.obstacle_bodytrans3)
            obstacle_body3.set_color(.7, .3, .1)  # sets color of obstacle

            b, t, l, r = -obdim, obdim, -obdim, obdim
            obstacle_body4 = rendering.FilledPolygon(
                [(l, b), (l, t), (r, t), (r, b)])  # creates deputy body polygon
            self.obstacle_bodytrans4 = rendering.Transform()  # allows body to be moved
            obstacle_body4.add_attr(self.obstacle_bodytrans4)
            obstacle_body4.set_color(.7, .3, .1)  # sets color of obstacle

            b, t, l, r = -obdim, obdim, -obdim, obdim
            obstacle_body5 = rendering.FilledPolygon(
                [(l, b), (l, t), (r, t), (r, b)])  # creates deputy body polygon
            self.obstacle_bodytrans5 = rendering.Transform()  # allows body to be moved
            obstacle_body5.add_attr(self.obstacle_bodytrans5)
            obstacle_body5.set_color(.7, .3, .1)  # sets color of obstacle

            b, t, l, r = -obdim, obdim, -obdim, obdim
            obstacle_body6 = rendering.FilledPolygon(
                [(l, b), (l, t), (r, t), (r, b)])  # creates deputy body polygon
            self.obstacle_bodytrans6 = rendering.Transform()  # allows body to be moved
            obstacle_body6.add_attr(self.obstacle_bodytrans6)
            obstacle_body6.set_color(.7, .3, .1)  # sets color of obstacle

            b, t, l, r = -obdim, obdim, -obdim, obdim
            obstacle_body7 = rendering.FilledPolygon(
                [(l, b), (l, t), (r, t), (r, b)])  # creates deputy body polygon
            self.obstacle_bodytrans7 = rendering.Transform()  # allows body to be moved
            obstacle_body7.add_attr(self.obstacle_bodytrans7)
            obstacle_body7.set_color(.7, .3, .1)  # sets color of obstacle

            b, t, l, r = -obdim, obdim, -obdim, obdim
            obstacle_body8 = rendering.FilledPolygon(
                [(l, b), (l, t), (r, t), (r, b)])  # creates deputy body polygon
            self.obstacle_bodytrans8 = rendering.Transform()  # allows body to be moved
            obstacle_body8.add_attr(self.obstacle_bodytrans8)
            obstacle_body8.set_color(.7, .3, .1)  # sets color of obstacle

            b, t, l, r = -obdim, obdim, -obdim, obdim
            obstacle_body9 = rendering.FilledPolygon(
                [(l, b), (l, t), (r, t), (r, b)])  # creates deputy body polygon
            self.obstacle_bodytrans9 = rendering.Transform()  # allows body to be moved
            obstacle_body9.add_attr(self.obstacle_bodytrans9)
            obstacle_body9.set_color(.7, .3, .1)  # sets color of obstacle

            #CHIEF BODY#
            b, t, l, r = -bodydim, bodydim, -bodydim, bodydim
            chief_body = rendering.FilledPolygon(
                [(l, b), (l, t), (r, t), (r, b)])  # creates body polygon
            self.chief_bodytrans = rendering.Transform()  # allows body to be moved
            chief_body.add_attr(self.chief_bodytrans)
            chief_body.set_color(.5, .5, .5)  # sets color of body

            #CHIEF SOLAR PANEL#
            b, t, l, r = -panelhei, panelhei, -panelwid * 2, panelwid * 2
            chief_panel = rendering.FilledPolygon(
                [(l, b), (l, t), (r, t), (r, b)])  # creates solar panel polygon
            self.chief_panel_trans = rendering.Transform()  # allows panel to be moved
            chief_panel.add_attr(self.chief_panel_trans)
            # sets panel as part of chief object
            chief_panel.add_attr(self.chief_bodytrans)
            chief_panel.set_color(.2, .2, .5)  # sets color of panel

            #VELOCITY ARROW#
            if self.velocityArrow:
                velocityArrow = rendering.Line((0, 0), (panelwid * 2, 0))
                self.velocityArrowTrans = rendering.Transform()
                velocityArrow.add_attr(self.velocityArrowTrans)
                velocityArrow.add_attr(self.deputy_bodytrans)
                velocityArrow.set_color(.8, .1, .1)

            #FORCE ARROW#
            if self.forceArrow:
                forceArrow = rendering.Line((0, 0), (panelwid * 2, 0))
                self.forceArrowTrans = rendering.Transform()
                forceArrow.add_attr(self.forceArrowTrans)
                forceArrow.add_attr(self.deputy_bodytrans)
                forceArrow.set_color(.1, .8, .1)

            #THRUST BLOCKS#
            if self.thrustVis == 'Block':
                b, t, l, r = -bodydim / 2, bodydim / 2, - \
                    panelwid, panelwid  # half the panel dimensions
                L_thrust = rendering.FilledPolygon(
                    [(l, b), (l, t), (r, t), (r, b)])  # creates thrust polygon
                self.L_thrust_trans = rendering.Transform()  # allows thrust to be moved
                L_thrust.add_attr(self.L_thrust_trans)
                L_thrust.add_attr(self.deputy_bodytrans)
                L_thrust.set_color(.7, .3, .1)  # sets color of thrust
                R_thrust = rendering.FilledPolygon(
                    [(l, b), (l, t), (r, t), (r, b)])  # creates thrust polygon
                self.R_thrust_trans = rendering.Transform()  # allows thrust to be moved
                R_thrust.add_attr(self.R_thrust_trans)
                R_thrust.add_attr(self.deputy_bodytrans)
                R_thrust.set_color(.7, .3, .1)  # sets color of thrust

                b, t, l, r = -bodydim / 2, bodydim / 2, -bodydim / 2, bodydim / 2
                T_thrust = rendering.FilledPolygon(
                    [(l, b), (l, t), (r, t), (r, b)])  # creates thrust polygon
                self.T_thrust_trans = rendering.Transform()  # allows thrust to be moved
                T_thrust.add_attr(self.T_thrust_trans)
                T_thrust.add_attr(self.deputy_bodytrans)
                T_thrust.set_color(.7, .3, .1)  # sets color of thrust
                B_thrust = rendering.FilledPolygon(
                    [(l, b), (l, t), (r, t), (r, b)])  # creates thrust polygon
                self.B_thrust_trans = rendering.Transform()  # allows thrust to be moved
                B_thrust.add_attr(self.B_thrust_trans)
                B_thrust.add_attr(self.deputy_bodytrans)
                B_thrust.set_color(.7, .3, .1)  # sets color of thrust

            #STARS#
            if self.stars > 0:
                for i in range(self.stars):
                    x, y = random.random() * (x_thresh * 2), random.random() * (y_thresh * 2)
                    dim = bodydim / 10
                    if dim <= 0:
                        dim = 1
                    star = rendering.make_circle(dim)  # creates trace dot
                    self.startrans = rendering.Transform()  # allows trace to be moved
                    star.add_attr(self.startrans)
                    star.set_color(.9, .9, .9)  # sets color of trace
                    self.viewer.add_geom(star)  # adds trace into render
                    self.startrans.set_translation(x, y)

            #ELLIPSES#
            if self.ellipse_quality > 0:
                thetaList = []
                i = 0
                while i <= math.pi * 2:
                    thetaList.append(i)
                    i += (1 / 100) * math.pi
                dotsize = int(self.scale_factor) + 1
                if dotsize < 0:
                    dotsize += 1

                for i in range(0, len(thetaList)):  # ellipse 1
                    x, y = self.ellipse_b1 * \
                        math.cos(thetaList[i]), self.ellipse_a1 * \
                        math.sin(thetaList[i])
                    x = (x * self.scale_factor) + x_thresh
                    y = (y * self.scale_factor) + y_thresh
                    dot1 = rendering.make_circle(dotsize)  # creates dot
                    self.dot1trans = rendering.Transform()  # allows dot to be moved
                    dot1.add_attr(self.dot1trans)
                    dot1.set_color(.1, .9, .1)  # sets color of dot
                    self.dot1trans.set_translation(x, y)
                    self.viewer.add_geom(dot1)  # adds dot into render

                for i in range(0, len(thetaList)):  # ellipse 2
                    x, y = self.ellipse_b2 * \
                        math.cos(thetaList[i]), self.ellipse_a2 * \
                        math.sin(thetaList[i])
                    x = (x * self.scale_factor) + x_thresh
                    y = (y * self.scale_factor) + y_thresh
                    dot2 = rendering.make_circle(dotsize)  # creates dot
                    self.dot2trans = rendering.Transform()  # allows dot to be moved
                    dot2.add_attr(self.dot2trans)
                    dot2.set_color(.8, .9, .1)  # sets color of dot
                    self.dot2trans.set_translation(x, y)
                    self.viewer.add_geom(dot2)  # adds dot into render

            self.viewer.add_geom(chief_panel)  # adds solar panel to viewer
            self.viewer.add_geom(chief_body)  # adds satellites to viewer

            if self.thrustVis == 'Block':
                self.viewer.add_geom(L_thrust)  # adds solar panel to viewer
                self.viewer.add_geom(R_thrust)  # adds solar panel to viewer
                self.viewer.add_geom(T_thrust)  # adds solar panel to viewer
                self.viewer.add_geom(B_thrust)  # adds thrust into viewer

            self.viewer.add_geom(deputy_panel)  # adds solar panel to viewer

            if self.velocityArrow:
                # adds velocityArrow to viewer
                self.viewer.add_geom(velocityArrow)

            if self.forceArrow:
                self.viewer.add_geom(forceArrow)  # adds forceArrow to viewer

            self.viewer.add_geom(deputy_body)  # adds body to viewer
            self.viewer.add_geom(obstacle_body)
            self.viewer.add_geom(obstacle_body1)
            self.viewer.add_geom(obstacle_body2)
            self.viewer.add_geom(obstacle_body3)
            self.viewer.add_geom(obstacle_body4)
            self.viewer.add_geom(obstacle_body5)
            self.viewer.add_geom(obstacle_body6)
            self.viewer.add_geom(obstacle_body7)
            self.viewer.add_geom(obstacle_body8)
            self.viewer.add_geom(obstacle_body9)
        # if there is no state (either the simulation has not begun or it has ended), end
        if self.state is None:
            print('No state')
            return None

        x = self.state
        tx, ty = (x[0] + self.x_threshold) * self.scale_factor, (x[1] + self.y_threshold) * \
            self.scale_factor  # pulls the state of the x and y coordinates
        self.deputy_bodytrans.set_translation(tx, ty)  # translate deputy
        self.chief_bodytrans.set_translation(
            self.x_chief + x_thresh, self.y_chief + y_thresh)  # translate chief

        # ! OBSTACLES
        p = self.obstacle

        xx, yy = (p[2] + self.x_threshold) * \
            self.scale_factor, (p[3] + self.y_threshold) * self.scale_factor
        self.obstacle_bodytrans.set_translation(xx, yy)

        xx, yy = (p[4] + self.x_threshold) * \
            self.scale_factor, (p[5] + self.y_threshold) * self.scale_factor
        self.obstacle_bodytrans1.set_translation(xx, yy)

        xx, yy = (p[6] + self.x_threshold) * \
            self.scale_factor, (p[7] + self.y_threshold) * self.scale_factor
        self.obstacle_bodytrans2.set_translation(xx, yy)

        xx, yy = (p[8] + self.x_threshold) * \
            self.scale_factor, (p[9] + self.y_threshold) * self.scale_factor
        self.obstacle_bodytrans3.set_translation(xx, yy)

        xx, yy = (p[10] + self.x_threshold) * \
            self.scale_factor, (p[11] + self.y_threshold) * self.scale_factor
        self.obstacle_bodytrans4.set_translation(xx, yy)

        xx, yy = (p[12] + self.x_threshold) * \
            self.scale_factor, (p[13] + self.y_threshold) * self.scale_factor
        self.obstacle_bodytrans5.set_translation(xx, yy)

        xx, yy = (p[14] + self.x_threshold) * \
            self.scale_factor, (p[15] + self.y_threshold) * self.scale_factor
        self.obstacle_bodytrans6.set_translation(xx, yy)

        xx, yy = (p[16] + self.x_threshold) * \
            self.scale_factor, (p[17] + self.y_threshold) * self.scale_factor
        self.obstacle_bodytrans7.set_translation(xx, yy)

        xx, yy = (p[18] + self.x_threshold) * \
            self.scale_factor, (p[19] + self.y_threshold) * self.scale_factor
        self.obstacle_bodytrans8.set_translation(xx, yy)

        xx, yy = (p[20] + self.x_threshold) * \
            self.scale_factor, (p[21] + self.y_threshold) * self.scale_factor
        self.obstacle_bodytrans9.set_translation(xx, yy)

        #PARTICLE DYNAMICS#
        if self.thrustVis == 'Particle':
            lx, ly = (x[0]) * self.scale_factor, (x[1]) * self.scale_factor
            v = random.randint(-self.p_var, self.p_var)
            if self.x_force > 0:
                DockingRender.create_particle(
                    self, self.p_velocity, 180 + v, lx, ly, self.p_ttl)
            elif self.x_force < 0:
                DockingRender.create_particle(
                    self, self.p_velocity, 0 + v, lx, ly, self.p_ttl)
            if self.y_force > 0:
                DockingRender.create_particle(
                    self, self.p_velocity, 270 + v, lx, ly, self.p_ttl)
            elif self.y_force < 0:
                DockingRender.create_particle(
                    self, self.p_velocity, 90 + v, lx, ly, self.p_ttl)

            for i in range(0, len(self.particles)):
                #velocity, theta, x, y, ttl
                self.particles[i][4] -= 1  # decrement the ttl
                r = (self.particles[i][1] * math.pi) / 180
                self.particles[i][2] += (self.particles[i][0] * math.cos(r))
                self.particles[i][3] += (self.particles[i][0] * math.sin(r))

            DockingRender.clean_particles(self, False)

            # translate & rotate all particles
            for i in range(0, len(self.p_obj)):
                self.trans[i].set_translation(
                    x_thresh + self.particles[i][2], y_thresh + self.particles[i][3])  # translate particle
                self.trans[i].set_rotation(self.particles[i][1])

        #TRACE DOTS#
        if self.trace != 0:  # if trace enabled, draw trace
            if self.tracectr == self.trace:  # if time to draw a trace, draw, else increment counter
                if self.traceMin:
                    tracewidth = 1
                else:
                    tracewidth = int(bodydim / 8) + 1

                trace = rendering.make_circle(tracewidth)  # creates trace dot
                self.tracetrans = rendering.Transform()  # allows trace to be moved
                trace.add_attr(self.tracetrans)
                trace.set_color(.9, .1, .9)  # sets color of trace
                self.viewer.add_geom(trace)  # adds trace into render
                self.tracectr = 0
            else:
                self.tracectr += 1

        self.tracetrans.set_translation(tx, ty)  # translate trace

        #BLOCK THRUSTERS#
        if self.thrustVis == 'Block':
            inc_l, inc_r, inc_b, inc_t = -25, 25, -5, 5  # create block dimensions
            # calculate block translations
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

            # translate blocks
            self.L_thrust_trans.set_translation(inc_l, 0)
            self.R_thrust_trans.set_translation(inc_r, 0)
            self.T_thrust_trans.set_translation(0, inc_t)
            self.B_thrust_trans.set_translation(0, inc_b)

        #VELOCITY ARROW#
        if self.velocityArrow:
            tv = math.atan(x[3] / x[2])  # angle of velocity
            if x[2] < 0:  # arctan adjustment
                tv += math.pi
            self.velocityArrowTrans.set_rotation(tv)

        #FORCE ARROW#
        if self.forceArrow:
            if self.x_force == 0:
                tf = math.atan(0)  # angle of velocity
            else:
                # angle of velocity
                tf = math.atan(self.y_force / self.x_force)
            if self.x_force < 0:  # arctan adjustment
                tf += math.pi
            self.forceArrowTrans.set_rotation(tf)

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def create_particle(self, velocity, theta, x, y, ttl):
        p = [velocity, theta, x, y, ttl]
        obj_len = len(self.p_obj)  # position of particle in list
        p_len = len(self.particles)  # position of particle in list
        trans_len = len(self.trans)  # position of particle in list

        self.particles.append(p)
        self.p_obj.append(self.particles[p_len])
        self.p_obj[obj_len] = rendering.make_circle(1)  # creates particle dot
        self.trans.append(rendering.Transform())  # allows particle to be moved
        self.p_obj[obj_len].add_attr(self.trans[trans_len])
        self.p_obj[obj_len].set_color(.9, .9, .6)  # sets color of particle

        self.trans[trans_len].set_translation(
            self.particles[p_len][2], self.particles[p_len][3])  # translate particle
        self.trans[trans_len].set_rotation(self.particles[p_len][1])
        self.viewer.add_geom(self.p_obj[obj_len])  # adds particle into render

        DockingRender.clean_particles(self, False)
        return p

    def clean_particles(self, all):
        # if all or if the first particle has reached its ttl
        while self.particles and (all or self.particles[0][4] < 0):
            # sets color of particle
            self.p_obj[0].set_color(
                self.bg_color[0], self.bg_color[1], self.bg_color[2])
            self.particles.pop(0)  # delete particle at beginning of list
            self.p_obj.pop(0)  # position of particle in list
            self.trans.pop(0)  # position of particle in list

    def close(self):  # if a viewer exists, close and kill it
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
