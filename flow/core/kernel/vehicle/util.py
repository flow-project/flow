import math

def observed(position, orientation, target_position, fov=90, looking_distance=20):

    """ Checks if a single vehicle/pedestrian can see another vehicle/pedestrian

    Parameters
    ---------------------
    position : 2D tuple
        (x, y) position of observer
    orientation : double
        angle of observer in degrees (0 deg is West, 90 deg is North)
    target_position: 2D tuple
        (x, y) position of the target
    fov: double
        the field of view of the observer
    looking_distance: double
        how far the observer can see

    Return: boolean
        whether or not the target can be observed
    """

    delta_x = target_position[0] - position[0]
    delta_y = target_position[1] - position[1]

    # edge case where both objects are at the same position
    if delta_x == 0 and delta_y == 0:
        return True
    
    # object is too far
    if euclidian_distance(delta_x, delta_y) > looking_distance:
        return False

    angle = get_angle(delta_x, delta_y)
    right_angle = (orientation - angle) % 360
    left_angle = (angle - orientation) % 360

   # object is not in FOV
    if left_angle > fov/2.0 and right_angle > fov/2.0:
        return False
    
    return True

def euclidian_distance(x, y):
    return math.sqrt(x**2 + y **2)

def get_angle(x, y):
    if x == 0:
        if y > 0:
            return 90
        else:
            return 270
    elif x < 0:
        return math.degrees(math.atan(y / x)) + 180

    return math.degrees(math.atan(y / x))

pos_1 = (2.4, -1.23)
theta_1 = 45
pos_2 = (2.48, -0.3)

print(observed(pos_1, theta_1, pos_2))
print(observed(pos_2, theta_1, pos_1))
