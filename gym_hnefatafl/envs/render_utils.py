import numpy as np
import pkg_resources
from scipy import misc
from gym_hnefatafl.envs.board import TileState


class Render_utils:
    def room_to_rgb(board):
        """
        Creates an RGB image of the room.
        :param room:
        :param room_structure:
        :return:
        """
        resource_package = __name__

        # room = np.array(room)
        # if not room_structure is None:
        # Change the ID of a player on a target
        #   room[(room == 5) & (room_structure == 2)] = 6

        # Load images, representing the corresponding situation
        white_filename = pkg_resources.resource_filename(resource_package, '/'.join(('surface', 'whiteplayer.png')))
        white = misc.imread(white_filename)

        black_filename = pkg_resources.resource_filename(resource_package, '/'.join(('surface', 'blackplayer.png')))
        black = misc.imread(black_filename)

        king_filename = pkg_resources.resource_filename(resource_package, '/'.join(('surface', 'king.png')))
        king = misc.imread(king_filename)

        corner_filename = pkg_resources.resource_filename(resource_package, '/'.join(('surface', 'corner.png')))
        corner = misc.imread(corner_filename)

        border_filename = pkg_resources.resource_filename(resource_package, '/'.join(('surface', 'border.png')))
        border = misc.imread(border_filename)

        empty_filename = pkg_resources.resource_filename(resource_package,
                                                         '/'.join(('surface', 'emptyfield.png')))
        empty = misc.imread(empty_filename)

        throne_filename = pkg_resources.resource_filename(resource_package, '/'.join(('surface', 'throne.png')))
        throne = misc.imread(throne_filename)

        surfaces = [empty, white, black, king, throne, corner, border]
        # Assemble the new rgb_room, with all loaded images
        room_rgb = np.zeros(shape=(13 * 32, 13 * 32, 3), dtype=np.uint8)
        for i in range(13):
            x_i = i * 32
            for j in range(13):
                y_j = j * 32
                surfaces_id = int(board[i, j])
                # print(type(board[i, j]))
                # print(surfaces_id)
                room_rgb[x_i:(x_i + 32), y_j:(y_j + 32), :] = surfaces[surfaces_id]

        return room_rgb
