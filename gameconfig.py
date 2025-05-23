import numpy as np

board_size = 9
p1_start = (2, 2)
p2_start = (board_size - 3, board_size - 3)

all_pieces = {}
pieces = {}
all_pieces["monomino"] = tuple([0])
pieces["monomino"] = ["monomino"]

all_pieces["domino1"] = (0, 1)
all_pieces["domino2"] = (0, 14)
pieces["domino"] = ["domino1", "domino2"]

all_pieces["tromino_I1"] = (0, 1, 2)
all_pieces["tromino_I2"] = (0, 14, 28)
pieces["tromino_I"] = ["tromino_I1", "tromino_I2"]

all_pieces["tromino_L1"] = (0, 1, 14)
all_pieces["tromino_L2"] = (0, 1, 15)
all_pieces["tromino_L3"] = (0, 14, 15)
all_pieces["tromino_L4"] = (1, 14, 15)
pieces["tromino_L"] = ["tromino_L1", "tromino_L2", "tromino_L3", "tromino_L4"]

all_pieces["tetromino_I1"] = (0, 1, 2, 3)
all_pieces["tetromino_I2"] = (0, 14, 28, 42)
pieces["tetromino_I"] = ["tetromino_I1", "tetromino_I2"]

# all_pieces["tetromino_L1"] = (0, 1, 2, 14)
# all_pieces["tetromino_L2"] = (0, 1, 15, 29)
# all_pieces["tetromino_L3"] = (2, 14, 15, 16)
# all_pieces["tetromino_L4"] = (0, 14, 28, 29)
# all_pieces["tetromino_L5"] = (0, 1, 2, 16)
# all_pieces["tetromino_L6"] = (1, 15, 28, 29)
# all_pieces["tetromino_L7"] = (0, 14, 15, 16)
# all_pieces["tetromino_L8"] = (0, 1, 14, 28)
# pieces["tetromino_L"] = [
#     "tetromino_L1",
#     "tetromino_L2",
#     "tetromino_L3",
#     "tetromino_L4",
#     "tetromino_L5",
#     "tetromino_L6",
#     "tetromino_L7",
#     "tetromino_L8",
# ]

all_pieces["tetromino_T1"] = (0, 1, 2, 15)
all_pieces["tetromino_T2"] = (1, 14, 15, 29)
all_pieces["tetromino_T3"] = (1, 14, 15, 16)
all_pieces["tetromino_T4"] = (0, 14, 15, 28)
pieces["tetromino_T"] = ["tetromino_T1", "tetromino_T2", "tetromino_T3", "tetromino_T4"]

all_pieces["tetromino_S1"] = (1, 2, 14, 15)
all_pieces["tetromino_S2"] = (0, 14, 15, 29)
all_pieces["tetromino_S3"] = (1, 14, 15, 28)
all_pieces["tetromino_S4"] = (0, 1, 15, 16)
pieces["tetromino_S"] = ["tetromino_S1", "tetromino_S2", "tetromino_S3", "tetromino_S4"]

all_pieces["tetromino_square"] = (0, 1, 14, 15)
pieces["tetromino_square"] = ["tetromino_square"]

# all_pieces["pentomino_F1"] = (0, 14, 15, 16, 29)
# all_pieces["pentomino_F2"] = (1, 15, 16, 28, 29)
# all_pieces["pentomino_F3"] = (1, 14, 15, 16, 30)
# all_pieces["pentomino_F4"] = (1, 2, 14, 15, 29)
# all_pieces["pentomino_F5"] = (2, 14, 15, 16, 29)
# all_pieces["pentomino_F6"] = (1, 14, 15, 29, 30)
# all_pieces["pentomino_F7"] = (1, 14, 15, 16, 28)
# all_pieces["pentomino_F8"] = (0, 1, 15, 16, 29)
# pieces["pentomino_F"] = [
#     "pentomino_F1",
#     "pentomino_F2",
#     "pentomino_F3",
#     "pentomino_F4",
#     "pentomino_F5",
#     "pentomino_F6",
#     "pentomino_F7",
#     "pentomino_F8",
# ]

# all_pieces["pentomino_I1"] = (0, 1, 2, 3, 4)
# all_pieces["pentomino_I2"] = (0, 14, 28, 42, 56)
# pieces["pentomino_I"] = ["pentomino_I1", "pentomino_I2"]

# all_pieces["pentomino_L1"] = (0, 1, 2, 3, 14)
# all_pieces["pentomino_L2"] = (0, 1, 15, 29, 43)
# all_pieces["pentomino_L3"] = (3, 14, 15, 16, 17)
# all_pieces["pentomino_L4"] = (0, 14, 28, 42, 43)
# all_pieces["pentomino_L5"] = (0, 1, 2, 3, 17)
# all_pieces["pentomino_L6"] = (1, 15, 29, 42, 43)
# all_pieces["pentomino_L7"] = (0, 14, 15, 16, 17)
# all_pieces["pentomino_L8"] = (0, 1, 14, 28, 42)
# pieces["pentomino_L"] = [
#     "pentomino_L1",
#     "pentomino_L2",
#     "pentomino_L3",
#     "pentomino_L4",
#     "pentomino_L5",
#     "pentomino_L6",
#     "pentomino_L7",
#     "pentomino_L8",
# ]

##### ADD THESE BACK
all_pieces["pentomino_N1"] = (0, 14, 15, 16, 30)
all_pieces["pentomino_N2"] = (2, 14, 15, 16, 28)
all_pieces["pentomino_N3"] = (1, 2, 15, 28, 29)
all_pieces["pentomino_N4"] = (0, 1, 15, 29, 30)
pieces["pentomino_N"] = ["pentomino_N1", "pentomino_N2", "pentomino_N3", "pentomino_N4"]

# all_pieces["pentomino_P1"] = (0, 1, 14, 15, 29)
# all_pieces["pentomino_P2"] = (0, 1, 14, 15, 28)
# all_pieces["pentomino_P3"] = (0, 14, 15, 28, 29)
# all_pieces["pentomino_P4"] = (1, 14, 15, 28, 29)
# all_pieces["pentomino_P5"] = (0, 1, 2, 15, 16)
# all_pieces["pentomino_P6"] = (1, 2, 14, 15, 16)
# all_pieces["pentomino_P7"] = (0, 1, 2, 14, 15)
# all_pieces["pentomino_P8"] = (0, 1, 14, 15, 16)
# pieces["pentomino_P"] = [
#     "pentomino_P1",
#     "pentomino_P2",
#     "pentomino_P3",
#     "pentomino_P4",
#     "pentomino_P5",
#     "pentomino_P6",
#     "pentomino_P7",
#     "pentomino_P8",
# ]

# all_pieces["pentomino_T1"] = (0, 1, 2, 15, 29)
# all_pieces["pentomino_T2"] = (0, 14, 15, 16, 28)
# all_pieces["pentomino_T3"] = (1, 15, 28, 29, 30)
# all_pieces["pentomino_T4"] = (2, 14, 15, 16, 30)
# pieces["pentomino_T"] = ["pentomino_T1", "pentomino_T2", "pentomino_T3", "pentomino_T4"]

all_pieces["pentomino_U1"] = (0, 1, 14, 28, 29)
all_pieces["pentomino_U2"] = (0, 2, 14, 15, 16)
all_pieces["pentomino_U3"] = (0, 1, 15, 28, 29)
all_pieces["pentomino_U4"] = (0, 1, 2, 14, 16)
pieces["pentomino_U"] = ["pentomino_U1", "pentomino_U2", "pentomino_U3", "pentomino_U4"]

# all_pieces["pentomino_V1"] = (0, 14, 28, 29, 30)
# all_pieces["pentomino_V2"] = (0, 1, 2, 14, 28)
# all_pieces["pentomino_V3"] = (0, 1, 2, 16, 30)
# all_pieces["pentomino_V4"] = (2, 16, 28, 29, 30)
# pieces["pentomino_V"] = ["pentomino_V1", "pentomino_V2", "pentomino_V3", "pentomino_V4"]

all_pieces["pentomino_W1"] = (0, 1, 15, 16, 30)
all_pieces["pentomino_W2"] = (2, 15, 16, 28, 29)
all_pieces["pentomino_W3"] = (0, 14, 15, 29, 30)
all_pieces["pentomino_W4"] = (1, 2, 14, 15, 28)
pieces["pentomino_W"] = ["pentomino_W1", "pentomino_W2", "pentomino_W3", "pentomino_W4"]

all_pieces["pentomino_X1"] = (1, 14, 15, 16, 29)
pieces["pentomino_X"] = ["pentomino_X1"]

# all_pieces["pentomino_Y1"] = (0, 1, 2, 3, 15)
# all_pieces["pentomino_Y2"] = (1, 14, 15, 16, 17)
# all_pieces["pentomino_Y3"] = (0, 14, 15, 28, 42)
# all_pieces["pentomino_Y4"] = (1, 14, 15, 29, 43)
# all_pieces["pentomino_Y5"] = (0, 1, 2, 3, 16)
# all_pieces["pentomino_Y6"] = (2, 14, 15, 16, 17)
# all_pieces["pentomino_Y7"] = (0, 14, 28, 29, 42)
# all_pieces["pentomino_Y8"] = (1, 15, 28, 29, 43)
# pieces["pentomino_Y"] = [
#     "pentomino_Y1",
#     "pentomino_Y2",
#     "pentomino_Y3",
#     "pentomino_Y4",
#     "pentomino_Y5",
#     "pentomino_Y6",
#     "pentomino_Y7",
#     "pentomino_Y8",
# ]

# all_pieces["pentomino_Z1"] = (0, 1, 15, 16, 17)
# all_pieces["pentomino_Z2"] = (2, 3, 14, 15, 16)
# all_pieces["pentomino_Z3"] = (0, 14, 15, 29, 43)
# all_pieces["pentomino_Z4"] = (1, 14, 15, 28, 42)
# all_pieces["pentomino_Z5"] = (1, 2, 3, 14, 15)
# all_pieces["pentomino_Z6"] = (0, 1, 2, 16, 17)
# all_pieces["pentomino_Z7"] = (1, 15, 28, 29, 42)
# all_pieces["pentomino_Z8"] = (0, 14, 28, 29, 43)
# pieces["pentomino_Z"] = [
#     "pentomino_Z1",
#     "pentomino_Z2",
#     "pentomino_Z3",
#     "pentomino_Z4",
#     "pentomino_Z5",
#     "pentomino_Z6",
#     "pentomino_Z7",
#     "pentomino_Z8",
# ]

# This should remain 14 to get coordinates, then ok to restrict to 8x8
for piece in all_pieces:
    all_pieces[piece] = np.array(all_pieces[piece])
    rows = all_pieces[piece] // 14
    cols = all_pieces[piece] % 14
    all_pieces[piece] = np.stack((rows, cols), axis=-1)


def remove_duplicate_pos(arr1, arr2):
    """
    Removes any rows in arr1 that are also present in arr2.

    Parameters:
    - arr1: (n, 2) numpy array of coordinates
    - arr2: (m, 2) numpy array of coordinates

    Returns:
    - A filtered numpy array containing only the rows from arr1 that are not in arr2.
    """
    # Convert to set of tuples for fast lookup
    arr2_set = set(map(tuple, arr2))

    # Filter arr1 to only include rows not in arr2
    filtered_arr = np.array([row for row in arr1 if tuple(row) not in arr2_set])

    return filtered_arr


boundaries = {}
corners = {}
for piece in all_pieces:
    # take each set of coords, add +1, -1, +14, -14 then concat results make a set of it, creating boundary for piece
    t = np.unique(
        np.concatenate(
            (
                all_pieces[piece] + (1, 0),
                all_pieces[piece] - (1, 0),
                all_pieces[piece] + (0, 1),
                all_pieces[piece] - (0, 1),
            )
        ),
        axis=0,
    )
    boundaries[piece] = remove_duplicate_pos(t, all_pieces[piece])
    c = np.unique(
        np.concatenate(
            (
                all_pieces[piece] + (1, 1),
                all_pieces[piece] + (-1, -1),
                all_pieces[piece] + (-1, 1),
                all_pieces[piece] + (1, -1),
            )
        ),
        axis=0,
    )
    corners[piece] = remove_duplicate_pos(c, np.concatenate((boundaries[piece], all_pieces[piece])))

piece_canonical_coord_options = {}
piece_canonical_boundaries = {}
piece_canonical_corners = {}
for piece_name in all_pieces:
    piece_canonical_coords = all_pieces[piece_name]
    piece_shifts = -1 * piece_canonical_coords
    piece_canonical_coord_options[piece_name] = np.array([piece_canonical_coords + shift for shift in piece_shifts])
    piece_canonical_boundaries[piece_name] = np.array([boundaries[piece_name] + shift for shift in piece_shifts])
    piece_canonical_corners[piece_name] = np.array([corners[piece_name] + shift for shift in piece_shifts])

# pcco is a dictionary from piece name to every possible way to place that piece on (0, 0)
# pcb is a dictionary from piece name to every possible boundary of that piece on (0, 0)
# pcc is a dictionary from piece name to every possible corner of that piece on (0, 0)
# To get all possible placements of a piece on a grid, shift the coords by the grid spot value

pcco = {}
pcb = {}
pcc = {}
for piece_name in pieces:
    pcco[piece_name] = np.vstack([piece_canonical_coord_options[piece] for piece in pieces[piece_name]])
    pcb[piece_name] = np.vstack([piece_canonical_boundaries[piece] for piece in pieces[piece_name]])
    pcc[piece_name] = np.vstack([piece_canonical_corners[piece] for piece in pieces[piece_name]])

piece_bdd_corner_pos_dict = {}
for r in range(board_size):
    for c in range(board_size):
        position = (r, c)
        piece_bdd_corner_pos_dict[position] = {}
        for piece in pcco.keys():
            piece_bdd_corner_pos_dict[position][piece] = []
            placement_options = pcco[piece] + position
            # Remove any options out of bounds
            mask = np.any((placement_options < 0) | (placement_options >= board_size), axis=(1, 2))
            placement_options = placement_options[~mask]

            boundary_options = pcb[piece] + position
            boundary_options = boundary_options[~mask]
            # corner_options = pcc[piece] + position
            # corner_options = corner_options[~mask]

            for option, boundary in zip(placement_options, boundary_options):
                b_mask = np.any((boundary < 0) | (boundary >= board_size), axis=1)
                boundary = tuple(boundary[~b_mask].T)
                option = tuple(option.T)
                # c_mask = np.any((corner < 0) | (corner >= 14), axis=1)
                # corner = tuple(corner[~c_mask].T)
                piece_bdd_corner_pos_dict[position][piece].append((option, boundary))

coords_to_id = {}
id_to_boundary = {}
inverted_coord_str_map = {}
i = 1
for pos in piece_bdd_corner_pos_dict:
    for piece_name in piece_bdd_corner_pos_dict[pos]:
        for option, boundary in piece_bdd_corner_pos_dict[pos][piece_name]:
            if str(option) not in coords_to_id:
                inv_option = tuple(board_size - 1 - x for x in option)
                inverted_coord_str_map[str(inv_option)] = i
                coords_to_id[str(option)] = i
                id_to_boundary[i] = boundary
                i += 1

pos_piece_to_id = {}
id_to_piece_coords = {}
id_to_coords = {}
for pos in piece_bdd_corner_pos_dict:
    pos_piece_to_id[pos] = {}
    for piece_name in piece_bdd_corner_pos_dict[pos]:
        pos_piece_to_id[pos][piece_name] = []
        for option, _ in piece_bdd_corner_pos_dict[pos][piece_name]:
            action_id = coords_to_id[str(option)]
            pos_piece_to_id[pos][piece_name].append(action_id)
            id_to_piece_coords[action_id] = (piece_name, option)
            id_to_coords[action_id] = option

piecenames = list(pcco.keys())
action_size = len(id_to_coords) + 1

reverse_id_perspective = {0: 0}
for a_id in id_to_coords:
    piece_name, coords = id_to_piece_coords[a_id]
    flipped_coords = tuple(board_size - 1 - x for x in id_to_coords[a_id])
    pos = (flipped_coords[0][0], flipped_coords[1][0])
    candidate_ids = pos_piece_to_id[pos][piece_name]
    # Board placing flipped coords, then go through each candidate id, get its coords, then see if np.all true
    test_board = np.zeros((board_size, board_size), dtype=bool)
    test_board[flipped_coords] = True
    for c_id in candidate_ids:
        if np.all(test_board[id_to_coords[c_id]]):
            reverse_id_perspective[a_id] = c_id
            break

num_channels = 4 + 2 * len(piecenames)
