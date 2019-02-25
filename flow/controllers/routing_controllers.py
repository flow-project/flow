import random

"""Contains a list of custom routing controllers."""

from flow.controllers.base_routing_controller import BaseRouter

import numpy as np

np.random.seed(204)


class ContinuousRouter(BaseRouter):
    """A router used to continuously re-route of the vehicle in a closed loop.

    This class is useful if vehicles are expected to continuously follow the
    same route, and repeat said route once it reaches its end.
    """

    def choose_route(self, env):
        """Adopt the current edge's route if about to leave the network."""
        if len(env.vehicles.get_route(self.veh_id)) == 0:
            # this occurs to inflowing vehicles, whose information is not added
            # to the subscriptions in the first step that they departed
            return None
        elif env.vehicles.get_edge(self.veh_id) == \
                env.vehicles.get_route(self.veh_id)[-1]:
            return env.available_routes[env.vehicles.get_edge(self.veh_id)]
        else:
            return None


class MinicityRouter(BaseRouter):
    """A router used to continuously re-route vehicles in minicity scenario.

    This class allows the vehicle to pick a random route at junctions.
    """

    def choose_route(self, env):
        """See parent class."""
        vehicles = env.vehicles
        veh_id = self.veh_id
        veh_edge = vehicles.get_edge(veh_id)
        veh_route = vehicles.get_route(veh_id)
        veh_next_edge = env.scenario.next_edge(veh_edge,
                                               vehicles.get_lane(veh_id))
        not_an_edge = ":"
        no_next = 0

        if len(veh_next_edge) == no_next:
            next_route = None
        elif veh_route[-1] == veh_edge:
            random_route = random.randint(0, len(veh_next_edge) - 1)
            while veh_next_edge[0][0][0] == not_an_edge:
                veh_next_edge = env.scenario.next_edge(
                    veh_next_edge[random_route][0],
                    veh_next_edge[random_route][1])
            next_route = [veh_edge, veh_next_edge[0][0]]
        else:
            next_route = None

        if veh_edge in ['e_37', 'e_51']:
            next_route = [veh_edge, 'e_29_u', 'e_21']

        return next_route

class IntersectionRouter(MinicityRouter):

    def choose_route(self, env):
        type_id = env.vehicles.get_state(self.veh_id, 'type')
        cur_route = env.vehicles.get_route(self.veh_id)
        cur_edge = env.vehicles.get_edge(self.veh_id)
        cur_lane = env.vehicles.get_lane(self.veh_id)
        route_assigned = False

        if len(cur_route) > 1:
            route_assigned = True
        if 'xiao' in type_id and not route_assigned:
            if cur_edge == 'e_7':
                if cur_lane == 0:
                    route = ['e_7', 'e_6']
            elif cur_edge == 'e_3':
                if cur_lane == 0:
                    route = ['e_3', 'e_2']
        elif 'idm' in type_id:
            route = MinicityRouter.choose_route(self, env)
        else:
            route = None

        return route

class IntersectionRandomRouter(MinicityRouter):

    def choose_route(self, env):
        type_id = env.vehicles.get_state(self.veh_id, 'type')
        cur_route = env.vehicles.get_route(self.veh_id)
        cur_edge = env.vehicles.get_edge(self.veh_id)
        cur_lane = env.vehicles.get_lane(self.veh_id)
        straight_routes = [
            ['e_1_inflow', 'e_1_sbc+', 'e_1', 'e_6', 'e_6_sbc-'],
            ['e_3_inflow', 'e_3_sbc+', 'e_3', 'e_8', 'e_8_sbc-'],
            ['e_5_inflow', 'e_5_sbc+', 'e_5', 'e_2', 'e_2_sbc-'],
            ['e_7_inflow', 'e_7_sbc+', 'e_7', 'e_4', 'e_4_sbc-'],
            ['e_1_sbc+', 'e_1', 'e_6', 'e_6_sbc-'],
            ['e_3_sbc+', 'e_3', 'e_8', 'e_8_sbc-'],
            ['e_5_sbc+', 'e_5', 'e_2', 'e_2_sbc-'],
            ['e_7_sbc+', 'e_7', 'e_4', 'e_4_sbc-']]
        right_turn_routes = [
            ['e_1_inflow', 'e_1_sbc+', 'e_1', 'e_8', 'e_8_sbc-'],
            ['e_7_inflow', 'e_7_sbc+', 'e_7', 'e_6', 'e_6_sbc-'],
            ['e_5_inflow', 'e_5_sbc+', 'e_5', 'e_4', 'e_4_sbc-'],
            ['e_3_inflow', 'e_3_sbc+', 'e_3', 'e_2', 'e_2_sbc-'],
            ['e_1_sbc+', 'e_1', 'e_8', 'e_8_sbc-'],
            ['e_7_sbc+', 'e_7', 'e_6', 'e_6_sbc-'],
            ['e_5_sbc+', 'e_5', 'e_4', 'e_4_sbc-'],
            ['e_3_sbc+', 'e_3', 'e_2', 'e_2_sbc-'],
        ]
        left_turn_routes = [
            ['e_1_inflow', 'e_1_sbc+', 'e_1', 'e_4', 'e_4_sbc-'],
            ['e_7_inflow', 'e_7_sbc+', 'e_7', 'e_2', 'e_2_sbc-'],
            ['e_5_inflow', 'e_5_sbc+', 'e_5', 'e_8', 'e_8_sbc-'],
            ['e_3_inflow', 'e_3_sbc+', 'e_3', 'e_6', 'e_6_sbc-'],
            ['e_1_sbc+', 'e_1', 'e_4', 'e_4_sbc-'],
            ['e_7_sbc+', 'e_7', 'e_2', 'e_2_sbc-'],
            ['e_5_sbc+', 'e_5', 'e_8', 'e_8_sbc-'],
            ['e_3_sbc+', 'e_3', 'e_6', 'e_6_sbc-'],
        ]        # request all vehicles start from sbc+ edges, not any edge
        all_routes = 7 * straight_routes + \
                     2 * right_turn_routes + \
                     1 * left_turn_routes;
        all_routes_dict = {}
        for route in all_routes:
            if route[0] in all_routes_dict:
                all_routes_dict[route[0]] += [route]
            else:
                all_routes_dict[route[0]] = [route]

        if len(cur_route) == 1:
            route = random.choice(all_routes_dict[cur_edge])
        else:
            route = None
        return route

class MinicityTrainingRouter_9(MinicityRouter):

    def choose_route(self, env):
        type_id = env.vehicles.get_state(self.veh_id, 'type')
        cur_route = env.vehicles.get_route(self.veh_id)
        cur_edge = env.vehicles.get_edge(self.veh_id)
        cur_lane = env.vehicles.get_lane(self.veh_id)
        route_assigned = False

        if len(cur_route) > 1:
            route_assigned = True
        if 'section1' in type_id and not route_assigned:
            if cur_edge == 'e_3':
                route = ['e_3', 'e_2', 'e_1', 'e_7', 'e_8_b', 'e_8_u', 'e_9',
                        'e_10','e_11']
            elif cur_edge == 'e_2':
                route = ['e_2', 'e_1', 'e_7', 'e_8_b', 'e_8_u', 'e_9', 'e_92']
            elif cur_edge == 'e_7':
                route = ['e_7', 'e_8_b', 'e_8_u', 'e_9', 'e_92', 'e_7']
            elif cur_edge == 'e_29_u':
                route = ['e_29_u', 'e_21', 'e_8_b', 'e_8_u', 'e_9', 'e_10']
        elif 'section2' in type_id and not route_assigned:
            route = ['e_3', 'e_25', 'e_30', 'e_31', 'e_32', 'e_21', 'e_8_u']
        elif 'section3' in type_id and not route_assigned:
            if cur_edge == 'e_41':
                if cur_lane == 0:
                    route = ['e_41', 'e_88']
                else:
                    route = ['e_41', 'e_39', 'e_37']
            elif cur_edge == 'e_25':
                route = ['e_25', 'e_87', 'e_50']
            elif cur_edge == 'e_54':
                route = ['e_54', 'e_88', 'e_26']
            elif cur_edge == 'e_38':
                if cur_lane == 0:
                    route = ['e_38', 'e_50']
                else:
                    route = ['e_38', 'e_40', 'e_42']
        elif 'section4' in type_id and not route_assigned:
            route = ['e_39', 'e_37', 'e_29_u', 'e_21']
        elif 'section6' in type_id and not route_assigned:
            route = ['e_60', 'e_69', 'e_72', 'e_68', 'e_66', 'e_63', 'e_94',
                     'e_52', 'e_38', 'e_50', 'e_60']
        elif 'section5' in type_id and not route_assigned:
            route = ['e_34', 'e_23', 'e_15', 'e_16', 'e_20', 'e_47', 'e_34']
        elif 'section7' in type_id and not route_assigned:
            route = ['e_42', 'e_44', 'e_46', 'e_48', 'e_78', 'e_86', 'e_59']
        elif 'section8' in type_id and not route_assigned:
            route = ['e_73', 'e_75', 'e_77', 'e_81', 'e_84', 'e_85', 'e_90']
        elif 'idm' in type_id:
            route = MinicityRouter.choose_route(self, env)
        else:
            route = None

        return route


class MinicityTrainingRouter_4(MinicityRouter):
    #top right corner routes
    def choose_route(self, env):
        type_id = env.vehicles.get_state(self.veh_id, 'type')
        edge = env.vehicles.get_edge(self.veh_id)
        cur_route = env.vehicles.get_route(self.veh_id)

        routes = {}
        overlap_routes = {}  # assuming we only have
        # upper-right
        some_routes = [
            ['e_80', 'e_83', 'e_82', 'e_79', 'e_47', 'e_49', 'e_55', 'e_56',
             'e_89'],
            ['e_45', 'e_43', 'e_41', 'e_50', 'e_60', 'e_69', 'e_73', 'e_75',
             'e_86', 'e_59'],
            ['e_45', 'e_43', 'e_41', 'e_50', 'e_60', 'e_69', 'e_73', 'e_75',
             'e_77', 'e_79', 'e_47'],
            ['e_48', 'e_81', 'e_84', 'e_85', 'e_90', 'e_62', 'e_57', 'e_59',
             'e_46'],
            ['e_49', 'e_58', 'e_76', 'e_74', 'e_70', 'e_61', 'e_54', 'e_40',
             'e_42', 'e_44'],
            ['e_46','e_48', 'e_78', 'e_76', 'e_74', 'e_70', 'e_61', 'e_54', 
            'e_40', 'e_42', 'e_44']
        ]

        for some_route in some_routes:
            for i in range(len(some_route)):
                # Routes through the top edge going right will continue in the
                # first path, while those in the center top edge will follow
                # the second path. This is to prevent vehicles in these routes
                # from converging onto one path.
                if some_route[-i] in routes:
                    overlap_routes[some_route[-i]] = \
                        some_route[-i:] + some_route[:-i]
                else:
                    routes[some_route[-i]] = some_route[-i:] + some_route[:-i]

        if 'idm' in type_id:
            route = MinicityRouter.choose_route(self, env)
        elif edge == cur_route[-1]:
            if edge in overlap_routes:
                # pick randomly among possible choices given multiple routes
                possible_routes = [overlap_routes[edge], routes[edge]]
                route = random.choice(possible_routes)
            else:
                # choose the only available route
                route = routes[edge]
        else:
            route = None

        return route

class MinicityTrainingRouter_5(MinicityRouter):
    #top left corner routes
    def choose_route(self, env):
        type_id = env.vehicles.get_state(self.veh_id, 'type')
        edge = env.vehicles.get_edge(self.veh_id)
        cur_route = env.vehicles.get_route(self.veh_id)

        routes = {}
        overlap_routes = {}  # assuming we only have
        #top left corner
        some_routes = [
            ['e_12', 'e_18', 'e_19', 'e_24', 'e_33', 'e_45', 'e_43', 'e_41',
             'e_88', 'e_26'],
            ['e_34', 'e_23', 'e_5', 'e_4', 'e_3', 'e_25', 'e_87', 'e_40',
             'e_42', 'e_44'],
            ['e_15', 'e_16', 'e_20', 'e_47', 'e_45', 'e_43', 'e_41', 'e_88',
             'e_26', 'e_12', 'e_13', 'e_14'],
            ['e_46', 'e_35', 'e_27', 'e_6', 'e_22', 'e_33'],
            # ['e_46', 'e_35', 'e_27', 'e_6', 'e_5', 'e_4', 'e_3', 'e_25',
            # 'e_87', 'e_40', 'e_42', 'e_44'],
            ['e_15', 'e_16', 'e_20', 'e_47', 'e_34', 'e_23']
        ]

        for some_route in some_routes:
            for i in range(len(some_route)):
                # Routes through the top edge going right will continue in the
                # first path, while those in the center top edge will follow
                # the second path. This is to prevent vehicles in these routes
                # from converging onto one path.
                if some_route[-i] in routes:
                    overlap_routes[some_route[-i]] = \
                        some_route[-i:] + some_route[:-i]
                else:
                    routes[some_route[-i]] = some_route[-i:] + some_route[:-i]

        if 'idm' in type_id:
            route = MinicityRouter.choose_route(self, env)
        elif edge == cur_route[-1]:
            if edge in overlap_routes:
                # pick randomly among possible choices given multiple routes
                possible_routes = [overlap_routes[edge], routes[edge]]
                route = random.choice(possible_routes)
            else:
                # choose the only available route
                route = routes[edge]
        else:
            route = None

        return route
class MinicityTrainingRouter_6(MinicityRouter):
    #bottom half routes
    def choose_route(self, env):
        type_id = env.vehicles.get_state(self.veh_id, 'type')
        edge = env.vehicles.get_edge(self.veh_id)
        cur_route = env.vehicles.get_route(self.veh_id)

        routes = {}
        overlap_routes = {}  # assuming we only have
        # bottom-left
        some_routes = [
            ['e_25', 'e_30', 'e_31', 'e_32', 'e_21', 'e_8_u', 'e_9', 'e_10',
             'e_11'],
            ['e_87', 'e_39', 'e_37', 'e_29_u', 'e_21', 'e_8_u', 'e_9', 'e_92',
             'e_7', 'e_8_b', 'e_8_u', 'e_9', 'e_10', 'e_11', 'e_25']
        ]
        # bottom right corner
        some_routes += [
            ['e_50', 'e_60', 'e_69', 'e_72', 'e_68', 'e_66', 'e_63', 'e_94',
             'e_52', 'e_38']
            # ['e_50', 'e_60', 'e_69', 'e_72', 'e_68', 'e_66', 'e_91','e_64',
            # 'e_65', 'e_66', 'e_63', 'e_94', 'e_52', 'e_38']
        ]
        # bottom half outer loop
        some_routes += [
            ['e_67', 'e_71', 'e_70', 'e_61', 'e_54', 'e_88', 'e_26', 'e_2',
             'e_1', 'e_7', 'e_17', 'e_28_b', 'e_36', 'e_93', 'e_53', 'e_64']
        ]
        # bottom right inner loop
        some_routes += [
            ['e_50', 'e_60', 'e_69', 'e_72', 'e_68', 'e_66', 'e_63', 'e_94',
             'e_52', 'e_38']
        ]

        for some_route in some_routes:
            for i in range(len(some_route)):
                # Routes through the top edge going right will continue in the
                # first path, while those in the center top edge will follow
                # the second path. This is to prevent vehicles in these routes
                # from converging onto one path.
                if some_route[-i] in routes:
                    overlap_routes[some_route[-i]] = \
                        some_route[-i:] + some_route[:-i]
                else:
                    routes[some_route[-i]] = some_route[-i:] + some_route[:-i]

        if 'idm' in type_id:
            route = MinicityRouter.choose_route(self, env)
        elif edge == cur_route[-1]:
            if edge in overlap_routes:
                # pick randomly among possible choices given multiple routes
                possible_routes = [overlap_routes[edge], routes[edge]]
                route = random.choice(possible_routes)
            else:
                # choose the only available route
                route = routes[edge]
        else:
            route = None

        return route



class LoopyEightRouter(BaseRouter):

    def choose_route(self, env):
        type_id = env.vehicles.get_state(self.veh_id, 'type')
        edge = env.vehicles.get_edge(self.veh_id)
        cur_route = env.vehicles.get_route(self.veh_id)

        routes = {}
        overlap_routes = {}  # assuming we only have

        all_routes = [
                ['e1', 'e2', 'e3', 'e4'],
                ['e2', 'e5', 'e8', 'e9', 'e12', 'e13', 'e14', 'e10', 'e11',
                 'e6', 'e1'],
                ['e8', 'e9', 'e12', 'e13', 'e14', 'e10', 'e11', 'e7']
        ]

        # the other direction
        all_routes_op = [
                ['e4_op', 'e3_op', 'e2_op', 'e1_op'],
                ['e1_op', 'e6_op', 'e11_op', 'e10_op', 'e14_op', 'e13_op',
                 'e12_op', 'e9_op', 'e8_op', 'e5_op', 'e2_op'],
                ['e7_op', 'e11_op', 'e10_op', 'e14_op', 'e13_op', 'e12_op',
                 'e9_op', 'e8_op']
        ]

        all_routes += all_routes_op

        for route in all_routes:
            for i in range(len(route)):
                # Routes through the top edge going right will continue in the
                # first path, while those in the center top edge will follow
                # the second path. This is to prevent vehicles in these routes
                # from converging onto one path.
                if route[-i] in routes:
                    overlap_routes[route[-i]] = route[-i:] + route[:-i]
                else:
                    routes[route[-i]] = route[-i:] + route[:-i]

        if edge == cur_route[-1]:
            if edge in overlap_routes:
                # pick randomly among possible choices given multiple routes
                possible_routes = [overlap_routes[edge], routes[edge]]
                route = random.choice(possible_routes)
            else:
                # choose the only available route
                route = routes[edge]
        else:
            route = None

        return route


class GridRouter(BaseRouter):
    """A router used to re-route a vehicle within a grid environment."""

    def choose_route(self, env):
        if len(env.vehicles.get_route(self.veh_id)) == 0:
            # this occurs to inflowing vehicles, whose information is not added
            # to the subscriptions in the first step that they departed
            return None
        elif env.vehicles.get_edge(self.veh_id) == \
                env.vehicles.get_route(self.veh_id)[-1]:
            return [env.vehicles.get_edge(self.veh_id)]
        else:
            return None


class BayBridgeRouter(ContinuousRouter):
    """Assists in choosing routes in select cases for the Bay Bridge scenario.

    Extension to the Continuous Router.
    """

    def choose_route(self, env):
        """See parent class."""
        edge = env.vehicles.get_edge(self.veh_id)
        lane = env.vehicles.get_lane(self.veh_id)

        if edge == "183343422" and lane in [2] \
                or edge == "124952179" and lane in [1, 2]:
            new_route = env.available_routes[edge + "_1"]
        else:
            new_route = super().choose_route(env)

        return new_route
