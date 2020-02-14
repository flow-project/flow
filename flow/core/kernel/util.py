import math

def observed(position, orientation, target_position, fov=90, looking_distance=50):

    """ Checks if a single vehicle/pedestrian can see another vehicle/pedestrian

    Parameters
    ---------------------
    position : 2D tuple
        (x, y) position of observer
    orientation : double
        angle of observer in degrees (0 deg is East, 90 deg is North)
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

    # change orientation from SUMO's (clockwise, zeroed to North)
    # to the standard unit circle
    orientation = orientation_unit_circle(orientation)

    angle = get_angle(delta_x, delta_y)
    right_angle = (orientation - angle) % 360
    left_angle = (angle - orientation) % 360

    # object is not in FOV
    if left_angle > fov/2.0 and right_angle > fov/2.0:
        return False
    
    return True

def orientation_unit_circle(angle):
    
    """ Converts SUMO's angle to the standard unit circle
    SUMO defines 0 degrees facing North with the angle increasing clockwise (90 is East,
    180 is South, etc.). This method converts SUMO's defintion to the standard unit circle
    where 0 is East, 90 is North, etc.

    Parameters
    ---------------------
    angle : double
        SUMO's angle value

    Return: double
        the angle represented in the standard unit circle
    """

    return (360 - (angle - 90)) % 360

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

def get_blocked_segments(position, target_position, target_orientation, target_length, target_width):

    target_orientation = orientation_unit_circle(target_orientation)
    corner_angle = math.degrees(math.atan(target_width / target_length))
    corner_dist = euclidian_distance(target_length, target_width)

    corners = []
    angles = []

    t_angle = math.radians(target_orientation + corner_angle)
    corners.append((target_position[0] + math.cos(t_angle) * corner_dist, \
            target_position[1] + math.sin(t_angle) * corner_dist))

    t_angle = math.radians(target_orientation + (180 - corner_angle))
    corners.append((target_position[0] + math.cos(t_angle) * corner_dist, \
            target_position[1] + math.sin(t_angle) * corner_dist))


    t_angle = math.radians(target_orientation + (180 + corner_angle))
    corners.append((target_position[0] + math.cos(t_angle) * corner_dist, \
            target_position[1] + math.sin(t_angle) * corner_dist))


    t_angle = math.radians(target_orientation + (360 - corner_angle))
    corners.append((target_position[0] + math.cos(t_angle) * corner_dist, \
            target_position[1] + math.sin(t_angle) * corner_dist))

    for i, c in enumerate(corners):
        angles.append((i, get_angle(position[0] - c[0], position[1] - c[1])))

    max_angle = corners[max(angles, key=lambda x: x[1])[0]]
    min_angle = corners[min(angles, key=lambda x: x[1])[0]]

    return(max_angle, min_angle)

def check_blocked(position, target_position, blocked):
    for b in blocked:
        line_of_sight = (position, target_position)
        if lines_intersect(line_of_sight, b):
            return True
    return False

def lines_intersect(line1, line2):

    '''
    line intersection algorithm
    https://bryceboe.com/2006/10/23/line-segment-intersection-algorithm/
    '''

    def ccw(A, B, C):
            return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1])*(C[0]-A[0])

    a, b, c, d = line1[0], line1[1], line2[0], line2[1]
    return ccw(a, c, d) != ccw(b, c, d) and ccw(a, b, c) != ccw(a, b, d)
