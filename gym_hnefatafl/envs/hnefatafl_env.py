import numpy as np
import gym
from gym import error, spaces, utils
from gym.utils import seeding
from gym_hnefatafl.envs.render_utils import Render_utils
from gym_hnefatafl.envs.board import HnefataflBoard
import gym_hnefatafl.envs.render_utils

from gym_hnefatafl.envs.board import HnefataflBoard, TileBattleState
from gym_hnefatafl.envs.board import Player


class HnefataflEnv(gym.Env):
    """The main OpenAI Gym class. It encapsulates an environment with
    arbitrary behind-the-scenes dynamics. An environment can be
    partially or fully observed.
    The main API methods that users of this class need to know are:
        step
        reset
        render
        close
        seed
    And set the following attributes:
        action_space: The Space object corresponding to valid actions
        observation_space: The Space object corresponding to valid observations
        reward_range: A tuple corresponding to the min and max possible rewards
    Note: a default reward range set to [-inf,+inf] already exists. Set it if you want a narrower range.
    The methods are accessed publicly as "step", "reset", etc.. The
    non-underscored versions are wrapper methods to which we may add
    functionality over time.
    """

    # Set this in SOME subclasses
    metadata = {'render.modes': []}
    reward_range = (-float('inf'), float('inf'))
    spec = None

    # Set these in ALL subclasses
    observation_space = None

    # recalculates the action space for the agent whose turn it is next
    def recalculate_action_space(self):
        def turn_player_board():
            return self._hnefatafl.black_board if self._blackTurn else self._hnefatafl.white_board

        self.action_space = []
        for (x, y), tileBattleState in np.ndenumerate(turn_player_board()):
            if tileBattleState == TileBattleState.allied:
                # first direction
                for x_other in reversed(range(1, x)):
                    if self._hnefatafl.move_board[x_other, y] == 0:
                        self.action_space.append(((x, y), (x_other, y)))
                    else:
                        break
                # second direction
                for x_other in range(x + 1, 13):
                    if self._hnefatafl.move_board[x_other, y] == 0:
                        self.action_space.append(((x, y), (x_other, y)))
                    else:
                        break
                # third direction
                for y_other in reversed(range(1, y)):
                    if self._hnefatafl.move_board[x, y_other] == 0:
                        self.action_space.append(((x, y), (x, y_other)))
                    else:
                        break
                # forth direction
                for y_other in range(y + 1, 13):
                    if self._hnefatafl.move_board[x, y_other] == 0:
                        self.action_space.append(((x, y), (x, y_other)))
                    else:
                        break

    def __init__(self):
        self.viewer = None
        self._hnefatafl = HnefataflBoard()
        self._blackTurn = True
        self.recalculate_action_space()

    def step(self, action):
        """Run one timestep of the environment's dynamics. When end of
        episode is reached, you are responsible for calling `reset()`
        to reset this environment's state.
        Accepts an action and returns a tuple (observation, reward, done, info).
        Args:
            action (object): an action provided by the environment
        Returns:
            observation (object): agent's observation of the current environment
            reward (float) : amount of reward returned after previous action
            done (boolean): whether the episode has ended, in which case further step() calls will return undefined results
            info (dict): contains auxiliary diagnostic information (helpful for debugging, and sometimes learning)
        """

        self._hnefatafl.do_action(action, Player.black if self._blackTurn else Player.white)
        self._blackTurn = not self._blackTurn
        self.recalculate_action_space()
        return self, 0, False, None

    def reset(self):
        """Resets the state of the environment and returns an initial observation.
        Returns: observation (object): the initial observation of the
            space.
        """
        self._hnefatafl = HnefataflBoard()

        raise NotImplementedError

    def render(self, mode='human'):
        #assert mode in RENDERING_MODES

        img = self.get_image(mode)

        if 'rgb_array' in mode:
            return img

        elif 'human' in mode:
            from gym.envs.classic_control import rendering
            print(img)
            if self.viewer is None:
                self.viewer = rendering.SimpleImageViewer()
            self.viewer.imshow(img)
            return self.viewer.isopen

        else:
            super(HnefataflEnv, self).render(mode=mode)  # just raise an exception

    def get_image(self, mode):
        print(self._hnefatafl.board)
        img = Render_utils.room_to_rgb(self._hnefatafl.board)
        #if mode.startswith('tiny_'):
           # img = Render_utils.room_to_tiny_world_rgb(self.room_state, self.room_fixed, scale=4)

        return img
        """Renders the environment.
        The set of supported modes varies per environment. (And some
        environments do not support rendering at all.) By convention,
        if mode is:
        - human: render to the current display or terminal and
          return nothing. Usually for human consumption.
        - rgb_array: Return an numpy.ndarray with shape (x, y, 3),
          representing RGB values for an x-by-y pixel image, suitable
          for turning into a video.
        - ansi: Return a string (str) or StringIO.StringIO containing a
          terminal-style text representation. The text can include newlines
          and ANSI escape sequences (e.g. for colors).
        Note:
            Make sure that your class's metadata 'render.modes' key includes
              the list of supported modes. It's recommended to call super()
              in implementations to use the functionality of this method.
        Args:
            mode (str): the mode to render with
            close (bool): close all open renderings
        Example:
        class MyEnv(Env):
            metadata = {'render.modes': ['human', 'rgb_array']}
            def render(self, mode='human'):
                if mode == 'rgb_array':
                    return np.array(...) # return RGB frame suitable for video
                elif mode is 'human':
                    ... # pop up a window and render
                else:
                    super(MyEnv, self).render(mode=mode) # just raise an exception
        """
        raise NotImplementedError

    def close(self):
        """Override _close in your subclass to perform any necessary cleanup.
        Environments will automatically close() themselves when
        garbage collected or when the program exits.
        """
        return

    # prints the tile state board
    def __str__(self):
        return str(self._hnefatafl.board)
