import unittest
import os
import pandas as pd

from pandas.testing import assert_series_equal

from flow.energy_models.power_demand import PDMCombustionEngine
# from flow.energy_models.power_demand import PDMElectric

os.environ["TEST_FLAG"] = "True"


class TestEnergyModels(unittest.TestCase):
    """
    Tests all energy models used in flow
    """

    def setUp(self):
        """
        Load verification data from csv
        """
        dir_path = os.path.dirname(os.path.realpath(__file__))
        self.data = pd.read_csv(os.path.join(dir_path, "test_files/energy_model_verification_data.csv"))
        print(self.data)

    def test_pdm_combustion_engine_power(self):
        energy_model = PDMCombustionEngine()
        power_value = energy_model.get_instantaneous_power
        power = self.data.apply(lambda x: power_value(x['a(m/s^2)'], x['v(m/s)'], x['rg(%)']), axis=1)
        assert_series_equal(self.data['Tacoma_fit(KW)']*1000, power, check_less_precise=True, check_names=False)

    def test_pdm_combustion_engine_fuel(self):
        energy_model = PDMCombustionEngine()
        fuel_value = energy_model.get_instantaneous_fuel_consumption
        fuel = self.data.apply(lambda x: fuel_value(x['a(m/s^2)'], x['v(m/s)'], x['rg(%)']), axis=1)
        assert_series_equal(self.data['Tacoma_fit(gal/hr)'], fuel, check_less_precise=True, check_names=False)

    """
    # Joy TODO: uncomment when this are fixed.
    def test_pdm_electric_power(self):
        energy_model = PDMElectric()
        power_value = energy_model.get_instantaneous_power
        power = self.data.apply(lambda x: power_value(x['a(m/s^2)'], x['v(m/s)'], x['rg(%)']), axis=1)
        assert_series_equal(self.data['Prius_fit(KW)']*1000, power, check_less_precise=True, check_names=False)
    """


if __name__ == '__main__':
    unittest.main()
