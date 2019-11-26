import numpy as np

L = 25
dx = 5
positions = np.array([10, 15, 20, 25])
velocities = np.array([11, 5, 4, 5])


def agg_func(car_positions, vel, L, dx,looking_distance=10):
    """ Agrregate microscopic positions to densities and
        microscopic velocities with look ahead and behind distance

        Parameters
        ----------
        car_positions: array_like
            positions of cars on road

        vel: array_like
            velocities of cars on road

        looking_distance: int
            distance to look ahead

        dx : float
            discrete distance steps

        L : int
            length of rode

        Returns
        -------
        tuple: (array_like, array_like)
            relative densities and velocities at every specified point on road

        """

    car_length = 5
    # collect full lengths covered by car
    car_dict = {"position":[]}
    n = 0
    for car in car_positions:
        car_dict["position"].append((car-car_length, car, vel[n]))
        n += 1
    print(car_dict)
    discrete_points = np.arange(0, L+dx, dx)
    rho = []
    final_vel = []

    # car_regions = np.sort(np.append(car_positions, car_length))

    for i in discrete_points:
        # check if region contains car
        # print(str(i) + " <<<<<")
        if i == max(discrete_points):
            # print("end of list")
            break
        if i+looking_distance > L:
            right_bound =(i + looking_distance)-L #region to looking distance
            # print(right_bound)
            # if car_positions in right_bound: #look ahead
            #     """TODO:"""
        else:
            right_bound = i + looking_distance
            # print(right_bound)

        # # left bound
        if i-looking_distance < 0:
            left_bound = L + (i - looking_distance) #region to looking distance
            # print(left_bound)
            # if car_positions in right_bound: #look ahead
            #     """TODO:"""
        else:
            left_bound = i - looking_distance
            # print(left_bound)

        d_r = 0
        d_l = 0
        speed = []

        print(str(i) + "<<<<<<<")
        for points in car_dict["position"]:
            a = left_bound
            b = right_bound
            m = i
            l = points[0]
            # speed = points[2]

            if (b == max(discrete_points)) & (l < 0):
                l = l + max(discrete_points)

            r = points[1]
            right = False

            if i+looking_distance > max(discrete_points):
                b = b + max(discrete_points)
                right = True
            if l > r:
                r = r + max(discrete_points)

            if right == True:
                if (m - l) > 0:
                    l = l + max(discrete_points)
                    r = r + max(discrete_points)

            # right half
            if (m == l) & (b == r):
                d_r += car_length
                speed = np.append(speed, points[2])

            elif l <= m < r <= b:
                d_r += r - m
                speed = np.append(speed, points[2])

            elif m <= l < r <= b:
                d_r += car_length
                speed = np.append(speed, points[2])

            elif m <= l < b < r:
                d_r += b - l
                speed = np.append(speed, points[2])

            elif b <= l:
                d_r += 0

            elif r <= m:
                d_r += 0

            elif l < m < b < r:
                d_r += b - m
                speed = np.append(speed, points[2])

            # reset values
            l = points[0]
            r = points[1]
            b = right_bound

            # left half
            if i-looking_distance < 0:
                m = max(discrete_points) + i

            if l > r:
                r = r + max(discrete_points)

            if l + max(discrete_points) < m:
                l = l + max(discrete_points)
                r = r + max(discrete_points)

            if (m == r) & (b == l):
                d_l += car_length
                speed = np.append(speed, points[2])
                continue

            elif l <= a < r <= m:
                d_l += r - a
                speed = np.append(speed, points[2])
                continue

            elif a <= l < r <= m:
                d_l += car_length
                speed = np.append(speed, points[2])
                continue

            elif a <= l < m < r:
                d_l += m - l
                speed = np.append(speed, points[2])
                continue

            elif m <= l:
                d_l += 0

            elif r <= a:
                d_l += 0

            elif l < a < m < r:
                d_l += m - a
                speed = np.append(speed, points[2])

        # print(d_r + d_l)
        print("speed:  " + str(speed))
        avg_vel = np.mean(speed)
        avg_density = np.mean([d_r / looking_distance, d_l / looking_distance])

        rho = np.append(rho, avg_density)
        final_vel = np.append(final_vel, avg_vel)
    return rho, final_vel

if __name__ == "__main__":
    # run function
    rho, vel = agg_func(positions, velocities, L, dx)
    # velocity = agg_func(positions, 10, L, dx)
