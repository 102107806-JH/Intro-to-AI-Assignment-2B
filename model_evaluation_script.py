import matplotlib.pyplot as plt
import matplotlib
import torch
import numpy as np
matplotlib.use("TkAgg")

def calculate_metrics(results_dict, remove_sites_list = None):
    for site in remove_sites_list:
        del results_dict[site]

    overall_GRU_MSE = 0
    overall_TCN_MSE = 0
    overall_LSTM_MSE = 0

    overall_GRU_MAE = 0
    overall_TCN_MAE = 0
    overall_LSTM_MAE = 0
    for scats_site in results_dict:
        # Calculate the residuals
        GRU_residuals = np.asarray(results[scats_site]["Targets"]) - np.asarray(results[scats_site]["GRU"])
        TCN_residuals = np.asarray(results[scats_site]["Targets"]) - np.asarray(results[scats_site]["TCN"])
        LSTM_residuals = np.asarray(results[scats_site]["Targets"]) - np.asarray(results[scats_site]["LSTM"])

        # Calculate the MSE
        results_dict[scats_site]["GRU_MSE"] = (np.mean(np.square(GRU_residuals))).tolist()
        results_dict[scats_site]["TCN_MSE"] = (np.mean(np.square(TCN_residuals))).tolist()
        results_dict[scats_site]["LSTM_MSE"] = (np.mean(np.square(LSTM_residuals))).tolist()

        # Calculate the MAE
        results_dict[scats_site]["GRU_MAE"] = (np.mean(np.absolute(GRU_residuals))).tolist()
        results_dict[scats_site]["TCN_MAE"] = (np.mean(np.absolute(TCN_residuals))).tolist()
        results_dict[scats_site]["LSTM_MAE"] = (np.mean(np.absolute(LSTM_residuals))).tolist()

        # Sum the MSE and MAE in there respective varaibles (will later be used to calculate the average)
        overall_GRU_MSE += results_dict[scats_site]["GRU_MSE"]
        overall_TCN_MSE += results_dict[scats_site]["TCN_MSE"]
        overall_LSTM_MSE += results_dict[scats_site]["LSTM_MSE"]

        overall_GRU_MAE += results_dict[scats_site]["GRU_MAE"]
        overall_TCN_MAE += results_dict[scats_site]["TCN_MAE"]
        overall_LSTM_MAE += results_dict[scats_site]["LSTM_MAE"]

    results_dict["Overall_Results"] = {}  # Set up another nested dictionary to store the overall results
    # Calculate the averages of the MSE and MAE for each model which go inside the newly created dictionary stored in the "Overall_Results" key of the result_dict
    results_dict["Overall_Results"]["GRU_MSE"] = overall_GRU_MSE / len(results_dict)
    results_dict["Overall_Results"]["TCN_MSE"] = overall_TCN_MSE / len(results_dict)
    results_dict["Overall_Results"]["LSTM_MSE"] = overall_LSTM_MSE / len(results_dict)
    results_dict["Overall_Results"]["GRU_MAE"] = overall_GRU_MAE / len(results_dict)
    results_dict["Overall_Results"]["TCN_MAE"] = overall_TCN_MAE / len(results_dict)
    results_dict["Overall_Results"]["LSTM_MAE"] = overall_LSTM_MAE / len(results_dict)

    return results_dict

if __name__ == "__main__":
    fig, axs = plt.subplots(10) # To present the plots
    for i in range(10):
        results = torch.load(f"testing/deployment_data_test_results/results_dictionary_depth_{i+1}")

        results = calculate_metrics(results, remove_sites_list=[4035])  # Calculates the MAE and MSE for each of the models and adds them to the dictionary

        print(f"Averages are taken over all the deployment data. Prediction depth: {i+1}")
        print(f"GRU MSE: {results['Overall_Results']['GRU_MSE']}")
        print(f"TCN MSE: {results['Overall_Results']['TCN_MSE']}")
        print(f"LSTM MSE: {results['Overall_Results']['LSTM_MSE']}")

        print(f"GRU MAE: {results['Overall_Results']['GRU_MAE']}")
        print(f"TCN MAE: {results['Overall_Results']['TCN_MAE']}")
        print(f"LSTM MAE: {results['Overall_Results']['LSTM_MAE']}")
        print("-----------------------------------------")

        plotBounds = (0, 96)  # Corresponds to the start (inclusive) and end (exclusive) 15 minute blocks which are to be plotted
        axs[i].plot(results[970]["Targets"][plotBounds[0]:plotBounds[1]], 'g')
        axs[i].plot(results[970]["GRU"][plotBounds[0]:plotBounds[1]], 'r')
        axs[i].plot(results[970]["TCN"][plotBounds[0]:plotBounds[1]], 'b')
        axs[i].plot(results[970]["LSTM"][plotBounds[0]:plotBounds[1]], 'm')
    plt.show()
    print("End")