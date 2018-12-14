
import pkg_resources
from gym_hnefatafl.envs import HnefataflEnv


class ReplayAgent(object):
    def __init__(self, player):
        self.player = player
    def read_file(self):
        url = __file__[:-37]
        game_filename = url + '/' + 'game.txt'
        file = open(game_filename)
        print(file.readline())
        file.close()
    def make_move(self, env: HnefataflEnv) -> ((int, int), (int, int)):
        print(__file__)
        url = __file__[:-37]
        print(url)
        game_filename = url+ '/'+'game.txt'
        file = open(game_filename)
        print(file.readline())
        file.close()
