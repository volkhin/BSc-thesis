#!/usr/bin/env python
#-*- coding: utf-8 -*-

# Copyright (C) 2013 Artem Volkhin <artem@volkhin.com>

from __future__ import division
import copy
import os
import math
import numbers
import sys
import matplotlib.pyplot as pyplot
import time
import random
import pygame


DF = 1.8
DIMENSIONS = 3

class Config(object):
    MAX_TICKS = 600
    NUMBER_OF_PARTICLES = 400
    INIT_PARTICLE_RADIUS = 0.02
    INIT_PARTICLE_RADIUS_VISUAL = INIT_PARTICLE_RADIUS
    BOARD_SIZE = 6.
    IMMOBILE_THRESHOLD = 0.1
    IMMOBILE_THRESHOLD_MT = IMMOBILE_THRESHOLD
    DIFFUSION_COEFFICIENT = 1. * 10 ** -4
    DIFFUSION_COEFFICIENT_MT = DIFFUSION_COEFFICIENT
    MICROTUBULES_NUMBER = 2
    MICROTUBULE_RADIUS = 0.025
    MICROTUBULE_LENGTH = BOARD_SIZE * 0.5
    FUSE_PARTICLES = True

    def __init__(self, **kws):
        for k, v in kws.items():
            if hasattr(self, k):
                setattr(self, k, v)

    def __repr__(self):
        return '\n'.join("{0}: {1}".format(k, getattr(self, k))
                         for k in dir(self) if not k.startswith('__'))


class ResultWriter(object):
    SEPARATOR = '\t'
    OUTPUT_DIRECTORY = '../results'

    def __init__(self, filename):
        filepath = os.path.join(self.OUTPUT_DIRECTORY, filename + '.txt')
        self.writer = open(filepath, 'w')

    def write(self, *args):
        values = ['{:.08f}'.format(x) if isinstance(x, numbers.Number)
                else str(x) for x in args]
        output = self.SEPARATOR.join(map(str, values))
        print output
        print >>self.writer, output
        self.writer.flush()

    def __del__(self):
        self.writer.close()


def track_time(func):
    def f(*args, **kws):
        t = time.time()
        result = func(*args, **kws)
        print 'Run {}, elapsed time: {}'.format(func.func_name, time.time() - t)
        return result
    return f


class EventTracker(object):
    def __init__(self):
        self.events = []
        self.total_speed = 0
        self.total_speed_num = 0
        self.tracked_time = 0

    def track(self, tick, event, *args):
        # print "Event", tick, event, args
        self.events.append((tick, event, args))

    def track_speed(self, tick, step):
        global total_speed, total_speed_num
        speed = math.sqrt(step[0]**2 + step[1]**2 + step[2]**2)
        self.total_speed += speed
        self.total_speed_num += 1

    def get_last_event_time(self):
        return self.events[-1][0]

    def get_average_speed(self):
        return self.total_speed / self.total_speed_num

    def get_average_event_time(self, event):
        try:
            last_event = [x for x in self.events if x[1] == event][-1][0]
            number_of_events = sum(1 for x in self.events if x[1] == event) 
            return last_event / number_of_events
        except:
            return None

    def get_average_p2p_collision_time(self):
        return self.get_average_event_time('p2p_collistion')

    def get_average_p2m_collision_time(self):
        return self.get_average_event_time('p2m_collistion')

global_event_tracker = EventTracker()


class Point(object):
    def __init__(self, (x, y, z)):
        self.x, self.y, self.z = x, y, z

    def coords(self):
        return [self.x, self.y, self.z]

    def __add__(self, o):
        return Point((self.x + o.x, self.y + o.y, self.z + o.z))

    def __sub__(self, o):
        return Point((self.x - o.x, self.y - o.y, self.z - o.z))

    def __mul__(self, k):
        return Point((self.x * k, self.y * k, self.z * k))

    def __truediv__(self, k):
        return Point((self.x / k, self.y / k, self.z / k))

    def __repr__(self):
        return "(" + ','.join('{0:.2}'.format(x) for x in self.coords()) + ')'


class Particle(object):
    def __init__(self, center, r):
        self.center = center
        self.r = r
        self.mt = None

    def __repr__(self):
        return "Particle({0}, {1})".format(self.center, self.r)

    def collide(self, o):
        dx = abs(self.center.x - o.center.x)
        dy = abs(self.center.y - o.center.y)
        dz = abs(self.center.z - o.center.z)
        dist = (dx ** DF + dy ** DF + dz ** DF) ** (1. / DF)
        return dist < self.r + o.r

    def get_mass(self):
        return self.r

    def get_visual_radii(self, config):
        enlargment = (self.r / config.INIT_PARTICLE_RADIUS) ** DF
        return enlargment * config.INIT_PARTICLE_RADIUS_VISUAL


class Microtubule(object):
    def __init__(self, x, z, min_y, max_y, r):
        self.x = x
        self.z = z
        self.min_y = min_y
        self.max_y = max_y
        self.r = r

    def __repr__(self):
        return "Microtubule({0}, {1}, {2} - {3}, {4})".format(
                self.x, self.z, self.min_y, self.max_y, self.r)


class GUI(object):
    def __init__(self, config):
        self.config = config
        self.is_running = True
        self.screen = pygame.display.set_mode((600, 600))
        self.clock = pygame.time.Clock()

    def process_events(self):
        for event in pygame.event.get():
            if (event.type == pygame.QUIT or
                event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                self.quit()

    def plot_pygame(self, env):
        COLOR_BACKGROUND = 255, 255, 255
        COLOR_PARTICLE = 16, 46, 55
        COLOR_PARTICLE_ON_MT = 243, 40, 40
        COLOR_MICROTUBULE = 16, 46, 55

        if not self.is_running:
            return
        BOARD_SIZE = self.config.BOARD_SIZE
        fps = self.clock.get_fps()
        pygame.display.set_caption(
                'SG simulation, {:.0f} fps, {}/{} tick'.format(
                    fps, env.tick, self.config.MAX_TICKS))
        self.screen.fill(COLOR_BACKGROUND)
        width, height = self.screen.get_width(), self.screen.get_height()
        for particle in env.particles:
            x = int(particle.center.x * width / BOARD_SIZE)
            y = height - int(particle.center.y * height / BOARD_SIZE)
            r = int(particle.get_visual_radii(self.config) * width / BOARD_SIZE)
            color = COLOR_PARTICLE_ON_MT if particle.mt else COLOR_PARTICLE
            pygame.draw.circle(self.screen, color, (x, y), r, 1)
        for mt in env.vertical_mt:
            x1 = mt.x * width / BOARD_SIZE
            y1 = height - mt.min_y * height / BOARD_SIZE
            x2 = mt.x * width / BOARD_SIZE
            y2 = height - mt.max_y * height / BOARD_SIZE
            pygame.draw.line(self.screen, COLOR_MICROTUBULE, (x1, y1), (x2, y2))
        pygame.display.flip()
        self.clock.tick(100)

    def quit(self):
        self.is_running = False
        pygame.quit()


class Environment(object):
    def __init__(self, config, stop_condition=None):
        self.event_tracker = EventTracker()
        self.stop_condition = stop_condition
        config = config or Config()
        self.config = config
        self.vertical_mt = []
        microtubules_number = self.config.MICROTUBULES_NUMBER
        x_pos = range(microtubules_number)
        y_pos = range(microtubules_number)
        random.shuffle(y_pos)
        for x, y in zip(x_pos, y_pos):
            x = self.config.BOARD_SIZE * ((x + 1) / (microtubules_number + 1))
            # y = self.config.BOARD_SIZE * ((y + 1) / (microtubules_number + 1))
            y = self.config.BOARD_SIZE * 0.5
            z = 0
            length = self.config.MICROTUBULE_LENGTH
            mt = Microtubule(x, y, z, length, self.config.MICROTUBULE_RADIUS)
            self.vertical_mt.append(mt)
        self.particles = []
        for _ in range(self.config.NUMBER_OF_PARTICLES):
            center = Point([random.uniform(0, self.config.BOARD_SIZE)
                            for _ in range(DIMENSIONS)])
            particle = Particle(center, self.config.INIT_PARTICLE_RADIUS)
            self.particles.append(particle)
        self.is_running = False

    def move_particles(self):
        for particle in self.particles:
            if particle.mt and particle.r < self.config.IMMOBILE_THRESHOLD_MT:
                # sigma = get_sigma(self.config.DIFFUSION_COEFFICIENT_MT)
                d = self.config.DIFFUSION_COEFFICIENT_MT / particle.r
                sigma = math.sqrt(2 * d)
                y_step = random.gauss(0, sigma)
                step = [0, y_step, 0]
                particle.center += Point(step)
            elif not particle.mt and particle.r < self.config.IMMOBILE_THRESHOLD:
                d = self.config.DIFFUSION_COEFFICIENT / particle.r
                sigma = math.sqrt(2 * d)
                step = [random.gauss(0, sigma) for _ in range(DIMENSIONS)]
                self.event_tracker.track_speed(self.tick, step)
                particle.center += Point(step)
            self.bound_particle(particle)
            self.find_nearest_mt_for_particle(particle)

    def bound_particle(self, particle):
        if particle.mt:
            bounds = [(particle.mt.x, particle.mt.x),
                      (particle.mt.min_y, particle.mt.max_y),
                      (particle.mt.z, particle.mt.z)]
        else:
            bounds = [(0, self.config.BOARD_SIZE) for _ in range(DIMENSIONS)]
        coords = particle.center.coords()
        for i, limits in enumerate(bounds):
            len = limits[1] - limits[0]
            if coords[i] > limits[1]:
                coords[i] = limits[1] - min(coords[i] - limits[1], len)
            if coords[i] < limits[0]:
                coords[i] = limits[0] + min(limits[0] - coords[i], len)
        particle.center.x, particle.center.y, particle.center.z = coords

    def find_nearest_mt_for_particle(self, particle):
        if not particle.mt:
            for mt in self.vertical_mt:
                dx = abs(particle.center.x - mt.x)
                if dx > particle.r + mt.r:
                    continue
                dz = abs(particle.center.z - mt.z)
                if dz > particle.r + mt.r:
                    continue
                dy = max(0, particle.center.y - mt.max_y,
                        mt.min_y - particle.center.y)
                dist = (dx ** DF + dy ** DF + dz ** DF) ** (1. / DF)
                if dist < particle.r + mt.r:
                    particle.mt = mt
                    self.event_tracker.track(self.tick, "p2m_collistion")
                    break

    def fuse_particles(self):
        def weighted_mean(a, wa, b, wb):
            return (a * wa + b * wb) / (wa + wb)
        self.particles.sort(key=lambda particle: particle.center.x)
        for i, particle in enumerate(self.particles):
            if hasattr(particle, 'dead'):
                continue
            x, r = particle.center.x, particle.r
            j = i
            while j > 0 and self.particles[j].center.x > x - 2 * r:
                j -= 1
            while j < len(self.particles) and self.particles[j].center.x < x + 2 * r:
                particle2 = self.particles[j]
                if (i != j and particle.collide(particle2) and
                        not hasattr(particle2, 'dead') and
                        not (particle.mt and particle2.mt and particle.mt != particle2.mt)):
                    self.event_tracker.track(self.tick, "p2p_collistion")
                    particle.center = weighted_mean(particle.center,
                                                    particle.get_mass(),
                                                    particle2.center,
                                                    particle2.get_mass())
                    particle.r = (particle.r ** DF + particle2.r ** DF) ** (1. / DF)
                    particle2.r = 0
                    self.particles[j].dead = True
                j += 1
        self.particles[:] = [particle for particle in self.particles
                             if not hasattr(particle, 'dead')]

    def run_simulation(self, gui_enabled=True, verbose=True):
        t = time.time()
        if gui_enabled:
            gui = GUI(self.config)
        for self.tick in range(self.config.MAX_TICKS):
            if gui_enabled and not gui.is_running:
                break
            if self.stop_condition and self.stop_condition(self):
                break
            self.event_tracker.track(self.tick, "tick")
            self.move_particles()
            if self.config.FUSE_PARTICLES:
                self.fuse_particles()
            if verbose and self.tick % 10 == 0:
                print self.tick, len(self.particles)
            if gui_enabled:
                gui.process_events()
                gui.plot_pygame(self)
        if verbose:
            print 'top 5 radiuses:', sorted(["{:.4f}".format(particle.r)
                for particle in self.particles], reverse=True)[:5]
            print 'particles left:', len(self.particles)
            print 'total time:', time.time() - t
            print 'average speed:', self.event_tracker.get_average_speed()
            print 'average particle/particle collision time:',\
                    self.event_tracker.get_average_event_time('p2p_collistion')
            print 'average particle/microtubule collision time:',\
                    self.event_tracker.get_average_event_time('p2m_collistion')

    def plot(self):
        BOARD_SIZE = self.config.BOARD_SIZE
        pyplot.clf()
        for particle in self.particles:
            circle = pyplot.Circle(particle.center.coords(),
                    particle.get_visual_radii(self.config), fill=False)
            pyplot.gcf().gca().add_artist(circle)
        for mt in self.vertical_mt:
            pyplot.plot([mt.x, mt.x], [mt.min_y, mt.max_y], 'b')
        pyplot.axis([0, BOARD_SIZE, 0, BOARD_SIZE])
        pyplot.axes().set_aspect('equal')
        # pyplot.ion()
        # pyplot.show()
        timestamp = int(time.time())
        image_filename = "../output/{0}.eps".format(timestamp)
        pyplot.savefig(image_filename)
        description_filename = "../output/{0}.txt".format(timestamp)
        with open(description_filename, 'w') as writer:
            print >>writer, self.config

@track_time
def run_visualizer():
    env = Environment(Config(MAX_TICKS=600))
    env.run_simulation()
    env.plot()

def get_average(data):
    size = len(data[0])
    result = []
    for i in range(size):
        values = sorted(x[i] for x in data if x[i] is not None)
        sz = max(3, len(values) / 2)
        while len(values) >= sz + 2:
            del values[0]
            del values[-1]
        try:
            value = sum(values) / len(values)
        except:
            value = None
        result.append(value)
    return result


class Experiment(object):
    arguments = []
    runs = 1
    writer = None

    def __init__(self):
        self.setup()
        self.run()

    def setup(self):
        pass

    def get_name(self):
        return self.__class__.__name__

    @track_time 
    def run(self):
        self.arguments = sorted(self.arguments)
        print '{} experiment'.format(self.get_name())
        print 'arguments:', self.arguments
        for arg in self.arguments:
            data = []
            for _ in range(self.runs):
                result = self.calculate(arg)
                data.append(result)
            avg_result = get_average(data)
            self.writer.write(*avg_result)

    def calculate(self, arg):
        pass


class P2pCollisionTimeOnDiffusionExperiment(Experiment):
    runs = 30
    arguments = {0.001 * x for x in range(1, 101, 5)}

    def setup(self):
        self.config = Config(MAX_TICKS=300, MICROTUBULES_NUMBER=0)
        self.writer = ResultWriter(self.get_name())
        self.writer.write('diffusion', 'speed', 'collision_time')

    def calculate(self, arg):
        config = copy.copy(self.config)
        config.DIFFUSION_COEFFICIENT *= arg
        config.DIFFUSION_COEFFICIENT_MT *= arg
        env = Environment(config)
        env.run_simulation(False, False)
        result = (config.DIFFUSION_COEFFICIENT,
                  env.event_tracker.get_average_speed(),
                  env.event_tracker.get_average_event_time('p2p_collistion'))
        return result


class P2pCollisionTimeOnConcetrationExperiment(Experiment):
    runs = 9
    arguments = {x for x in range(200, 601, 10)}

    def setup(self):
        self.config = Config(MAX_TICKS=300, MICROTUBULES_NUMBER=0)
        self.writer = ResultWriter(self.get_name())
        self.writer.write('number_of_particles', 'speed', 'collision_time')

    def calculate(self, arg):
        config = copy.copy(self.config)
        config.NUMBER_OF_PARTICLES = arg
        env = Environment(config)
        env.run_simulation(False, False)
        result = (config.NUMBER_OF_PARTICLES,
                  env.event_tracker.get_average_speed(),
                  env.event_tracker.get_average_event_time('p2p_collistion'))
        return result


class P2mCollisionTimeOnLengthExperiment(Experiment):
    runs = 5
    arguments = {0.1 * x for x in range(1, 21)} | {0.3 * x for x in range(1, 21)}

    def setup(self):
        self.config = Config(MAX_TICKS=300, FUSE_PARTICLES=False)
        self.writer = ResultWriter(self.get_name())
        self.writer.write('microtubule_length', 'collision_time')

    def calculate(self, arg):
        config = copy.copy(self.config)
        config.MICROTUBULE_LENGTH = arg
        env = Environment(config)
        env.run_simulation(False, False)
        result = (config.MICROTUBULE_LENGTH,
                  env.event_tracker.get_average_event_time('p2m_collistion'))
        return result

class P2mCollisionTimeOnRadiusExperiment(Experiment):
    runs = 50
    arguments = {0.01 * x for x in range(1, 21)} |\
            {0.002 * x for x in range(2, 11)}

    def setup(self):
        self.config = Config(
                MAX_TICKS=300,
                NUMBER_OF_PARTICLES=100,
                FUSE_PARTICLES=False,
                IMMOBILE_THRESHOLD=1,
                IMMOBILE_THRESHOLD_MT=1)
        self.writer = ResultWriter(self.get_name())
        self.writer.write('radius', 'collision_time')

    def calculate(self, arg):
        config = copy.copy(self.config)
        config.INIT_PARTICLE_RADIUS = arg
        config.MICROTUBULE_RADIUS = arg
        env = Environment(config)
        env.run_simulation(False, False)
        result = (config.INIT_PARTICLE_RADIUS,
                  env.event_tracker.get_average_event_time('p2m_collistion'))
        return result

class CreatingLargeGranulesExperiment(Experiment):
    runs = 10
    arguments = {0.03 + 0.001 * x for x in range(31)}

    def setup(self):
        self.config = Config(
                MAX_TICKS=1000,
                IMMOBILE_THRESHOLD=1,
                IMMOBILE_THRESHOLD_MT=1)
        self.writer = ResultWriter(self.get_name())
        self.writer.write('desired_radius', 'time')

    def calculate(self, arg):
        def stop_condition(env):
            R = max(particle.r for particle in env.particles)
            return R > arg
        config = copy.copy(self.config)
        env = Environment(config, stop_condition=stop_condition)
        env.run_simulation(False, False)
        result = (arg,
                  env.event_tracker.get_last_event_time())
        return result

class P2pCollisionTimeOnVolumeExperiment(Experiment):
    runs = 10
    arguments = {0.1 * x for x in range(5, 21)}

    def setup(self):
        self.config = Config(MAX_TICKS=300, MICROTUBULES_NUMBER=0)
        self.writer = ResultWriter(self.get_name())
        self.writer.write('board_size', 'number_of_particles', 'collision_time')

    def calculate(self, arg):
        config = copy.copy(self.config)
        config.NUMBER_OF_PARTICLES = int(config.NUMBER_OF_PARTICLES * arg ** 3)
        config.BOARD_SIZE *= arg
        env = Environment(config)
        env.run_simulation(False, False)
        result = (config.BOARD_SIZE,
                  config.NUMBER_OF_PARTICLES,
                  env.event_tracker.get_average_event_time('p2p_collistion'))
        return result


class P2pCollisionTimeOnImmobilityExperiment(Experiment):
    runs = 3
    arguments = {0.02 * x for x in range(5, 21)}

    def setup(self):
        self.config = Config(MAX_TICKS=300, MICROTUBULES_NUMBER=0)
        self.writer = ResultWriter(self.get_name())
        self.writer.write('immobile_threshold', 'collision_time')

    def calculate(self, arg):
        config = copy.copy(self.config)
        config.IMMOBILE_THRESHOLD = arg
        env = Environment(config)
        env.run_simulation(False, False)
        result = (config.IMMOBILE_THRESHOLD,
                  env.event_tracker.get_average_event_time('p2p_collistion'))
        return result


if __name__ == "__main__":
    random.seed(1)
    run_visualizer()
    # P2pCollisionTimeOnDiffusionExperiment()
    # P2pCollisionTimeOnConcetrationExperiment()
    # P2mCollisionTimeOnLengthExperiment()
    # P2mCollisionTimeOnRadiusExperiment()
    # CreatingLargeGranulesExperiment()
    # P2pCollisionTimeOnVolumeExperiment()
    # P2pCollisionTimeOnImmobilityExperiment()
