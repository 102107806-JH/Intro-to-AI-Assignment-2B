from jh_ml_models.model_deployment_abstractions.deployment_data_testing.deployment_data_model_tester import DeploymentDataModelTester
from jh_ml_models.model_deployment_abstractions.deployment_data_testing.multi_scat_site_tester import MultiScatSiteDeploymentDataModelTester
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("TkAgg")


# 970, 2000, 2200, 2820, 2825, 2827, 2846, 3001, 3002, 3120, 3122, 3126, 3127, 3180, 3662, 3682, 3685, 3804,
# 3812, 4030, 4032, 4034, 4035, 4040, 4043, 4051, 4057, 4063, 4262, 4263, 4264, 4266, 4270, 4272, 4273, 4321,
# 4324, 4335, 4812, 4821,

if __name__ == "__main__":
    scats_site_list = \
        [970, 2000, 2200, 2820, 2825, 2827, 2846, 3001,
         3002, 3120, 3122, 3126, 3127, 3180, 3662, 3682,
         3685, 3804, 3812, 4030, 4032, 4034, 4035, 4040,
         4043, 4051, 4057, 4063, 4262, 4263, 4264, 4266,
         4270, 4272, 4273, 4321, 4324, 4335, 4812, 4821]

    tester = MultiScatSiteDeploymentDataModelTester(scat_site_list=scats_site_list)
    start_datetime = datetime(year=2025, month=8, day=1, hour=0, minute=0)
    end_datetime = datetime(year=2025, month=8, day=2, hour=0, minute=0)
    results = tester.test_models(
        start_time=start_datetime,
        end_time=end_datetime,
        predicition_depth=1,
        sequence_length=12
    )
    plt.plot(results[970]["Targets"], 'g')
    plt.plot(results[970]["GRU"], 'r')
    plt.plot(results[970]["TCN"], 'b')
    plt.plot(results[970]["LSTM"], 'm')
    plt.show()
    print("End")