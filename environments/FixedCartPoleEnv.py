import math
import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np


class FixedCartPoleEnv(gym.Env):
    """
    Description:
        A modified Cart pole example, where the cart does not move and only the pole can be pushed.

    Source:
        This environment corresponds to the version of the cart-pole problem
        described by Barto, Sutton, and Anderson

    Observation:
        Type: Box(4)
        Num     Observation               Min                     Max
        0       Pole Angle                bounded                 bounded
        1       Pole Angular Velocity     -Inf                    Inf

    Actions:
        Type: Discrete(2)
        Num   Action
        0     Push pole to the left
        1     Push pole to the right

        Note: The amount the velocity that is reduced or increased is not
        fixed; it depends on the angle the pole is pointing. This is because
        the center of gravity of the pole increases the amount of energy needed
        to move the cart underneath it

    Reward:
        Reward is 1 for every step taken, including the termination step

    Starting State:
        All observations are assigned a uniform random value in [-0.05..0.05]

    Note:
        The size of theta greater than 0.201 makes the pole impossible to come back
        to the upright position (because the force cannot overcome the rotation caused
        by the gravity)

    Episode Termination:
        Pole Angle is more than 20 degrees.
        Solved Requirements:
        Considered solved when the average return is greater than or equal to
        195.0 over 100 consecutive trials.
    """

    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 50}

    def __init__(self, seed = None):
        self.gravity = 9.8
        self.masspole = 2
        self.length = 0.5  # actually half the pole's length
        self.force_mag = 2.0
        self.tau = 0.1  # seconds between state updates
        self.kinematics_integrator = "euler"

        # Angle at which to fail the episode
        self.theta_threshold_radians = 20 * 2 * math.pi / 360 # in degree 20, in rad 0.349
        self.step_threshold = 200

        # Angle limit set to 2 * theta_threshold_radians so failing observation
        # is still within bounds.
        high = np.array([self.theta_threshold_radians * 2,np.finfo(np.float32).max,], dtype=np.float32)
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        self.seed(seed)
        self.viewer = None
        self.state = None

        self.action_names = ['left', 'right']

        self.steps_beyond_done = None
        self.step_count = 0
        self.normalizer=[2.5, 0.5]
        self.denormalizer=[0.40, 2.0]

    def test_agent_accuracy(self, agent):
        error = 0

        return (1.0 - error/10) * 100

    def normalize(self, state):
        return (state[0] * self.normalizer[0], state[1] * self.normalizer[1])

    def denormalize(self, state):
        return (state[0] * self.denormalizer[0], state[1] * self.denormalizer[1])

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        err_msg = "%r (%s) invalid" % (action, type(action))
        assert self.action_space.contains(action), err_msg

        theta, theta_dot = self.state

        force = self.force_mag if action == 1 else -self.force_mag
        costheta = math.cos(theta)
        sintheta = math.sin(theta)

        gravity_torque = sintheta * (self.gravity * self.masspole) * self.length
        force_torque = costheta * force * self.length * 2
        net_torque = gravity_torque + force_torque

        inertia = (1 / 3) * self.masspole * (2 * self.length) ** 2
        thetaacc = net_torque / inertia
        #print("{0}, {1}, {2}, {3}".format(action, gravity_torque, force_torque, thetaacc))

        if self.kinematics_integrator == "euler":
            theta = theta + self.tau * theta_dot
            theta_dot = theta_dot + self.tau * thetaacc
        else:  # semi-implicit euler
            theta_dot = theta_dot + self.tau * thetaacc
            theta = theta + self.tau * theta_dot

        self.state = (theta, theta_dot)
        self.step_count += 1

        done = bool(
            theta < -self.theta_threshold_radians
            or theta > self.theta_threshold_radians
            or self.step_count >= self.step_threshold
        )

        if not done:
            reward = 1.0
        elif self.steps_beyond_done is None:
            # Pole just fell!
            self.steps_beyond_done = 0
            reward = 1.0
        else:
            if self.steps_beyond_done == 0:
                logger.warn(
                    "You are calling 'step()' even though this "
                    "environment has already returned done = True. You "
                    "should always call 'reset()' once you receive 'done = "
                    "True' -- any further steps are undefined behavior."
                )
            self.steps_beyond_done += 1
            reward = 0.0

        return np.array(self.normalize(self.state), dtype=np.float32), reward, done, {}

    def reset(self):
        self.state = self.np_random.uniform(low=-0.15, high=0.15, size=(2,))
        #self.state = [-0.20, -0.05]
        self.steps_beyond_done = None
        self.step_count = 0
        return np.array(self.normalize(self.state), dtype=np.float32)

    def render(self, mode="human"):
        screen_width = 600
        screen_height = 500

        world_width = 1 * 2
        scale = screen_width / world_width
        carty = 100  # TOP OF CART
        polewidth = 10.0
        polelen = scale * (2 * self.length)
        cartwidth = 50.0
        cartheight = 30.0

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)
            l, r, t, b = -cartwidth / 2, cartwidth / 2, cartheight / 2, -cartheight / 2
            axleoffset = cartheight / 4.0
            cart = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            self.carttrans = rendering.Transform()
            cart.add_attr(self.carttrans)
            self.viewer.add_geom(cart)
            l, r, t, b = (
                -polewidth / 2,
                polewidth / 2,
                polelen - polewidth / 2,
                -polewidth / 2,
            )
            pole = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            pole.set_color(0.8, 0.6, 0.4)
            self.poletrans = rendering.Transform(translation=(0, axleoffset))
            pole.add_attr(self.poletrans)
            pole.add_attr(self.carttrans)
            self.viewer.add_geom(pole)
            self.axle = rendering.make_circle(polewidth / 2)
            self.axle.add_attr(self.poletrans)
            self.axle.add_attr(self.carttrans)
            self.axle.set_color(0.5, 0.5, 0.8)
            self.viewer.add_geom(self.axle)
            self.track = rendering.Line((0, carty), (screen_width, carty))
            self.track.set_color(0, 0, 0)
            self.viewer.add_geom(self.track)

            self._pole_geom = pole

        if self.state is None:
            return None

        # Edit the pole polygon vertex
        pole = self._pole_geom
        l, r, t, b = (
            -polewidth / 2,
            polewidth / 2,
            polelen - polewidth / 2,
            -polewidth / 2,
        )
        pole.v = [(l, b), (l, t), (r, t), (r, b)]

        x = self.state
        cartx = 0 * scale + screen_width / 2.0  # MIDDLE OF CART
        self.carttrans.set_translation(cartx, carty)
        self.poletrans.set_rotation(-x[0])

        return self.viewer.render(return_rgb_array=mode == "rgb_array")

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None