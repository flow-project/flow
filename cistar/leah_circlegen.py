from CircleGenerator import CircleGenerator
from SumoExperiment import Generator

net_path = Generator.NET_PATH
cfg_path = Generator.CFG_PATH
data_prefix = Generator.DATA_PREFIX
base = "leah-test"
generator = CircleGenerator(net_path, cfg_path, data_prefix, base)

net_params = {"length": 840, "lanes": 1, "speed_limit":35, "resolution": 4}
cfg_params = {"num_cars":12, "max_speed":35, "start_time": 0, "end_time":3000}

generator.generate_net(net_params)
cfg, outs = generator.generate_cfg(cfg_params)