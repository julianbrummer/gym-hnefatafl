
import pkg_resources
from gym_hnefatafl.envs import HnefataflEnv


class ReplayAgent(object):
    def __init__(self, player):
        self.player = player
        if self.player == player.white:
            self.globalcounter=1
        else:
            self.globalcounter=0
        self.actions=[]
        url = __file__[:-37]
        game_filename = url + '/' + 'game.txt'
        file = open(game_filename)
        for action in file:
            self.actions.append(action)
            print(action)
        file.close()
    def make_move(self, env: HnefataflEnv) -> ((int, int), (int, int)):
            if self.actions.__sizeof__()>=self.globalcounter:
                action=self.actions[self.globalcounter]
                action = action.replace('(', '').replace(')','')
                actionlist=action.split(',')
                intacts=[int(x) for x in actionlist]
                fromX=intacts[0]
                fromY=intacts[1]
                toX=intacts[2]
                toY=intacts[3]
                self.globalcounter = self.globalcounter + 2
                return ((fromX,fromY), (toX,toY))
            else:
                return NotImplementedError
    def give_reward(self,reward):
        pass
