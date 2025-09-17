import matplotlib.pyplot as plt
import matplotlib
import torch
matplotlib.use("TkAgg")

if __name__ == "__main__":
    """
    Different prediction depths can be accessed by changing the last number in the path
    to the desired depth.
    """
    results = torch.load("testing/deployment_data_test_results/results_dictionary_depth_3")
    """
    'results' is a nested dictionary the outer dictionary links to specific scats sites
    the inner to the different results for that scat site. The example bellow illustrates this
    by using a pyplot. You can also access the times of all predictions eg:
    results[970]["Prediction_Times"].
    Below is the available inner dictionary keys.
    "Prediction_Time" -> The time for the prediction
    "Targets" -> The target tfv at that prediction time
    "GRU", "TCN", "LSTM" -> The tfv predictions for the prediction time
    You probably wont need the prediction time as all predictions are 15 minutes apart.
    Its just there for your convenience. I have also provided a scat site list below in case you need it
    (its currently unused).
    """
    scats_site_list = \
        [970, 2000, 2200, 2820, 2825, 2827, 2846, 3001,
         3002, 3120, 3122, 3126, 3127, 3180, 3662, 3682,
         3685, 3804, 3812, 4030, 4032, 4034, 4035, 4040,
         4043, 4051, 4057, 4063, 4262, 4263, 4264, 4266,
         4270, 4272, 4273, 4321, 4324, 4335, 4812, 4821]

    plt.plot(results[970]["Targets"], 'g')
    plt.plot(results[970]["GRU"], 'r')
    plt.plot(results[970]["TCN"], 'b')
    plt.plot(results[970]["LSTM"], 'm')
    plt.show()
    print("End")