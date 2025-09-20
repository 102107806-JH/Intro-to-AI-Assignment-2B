import matplotlib.pyplot as plt
import matplotlib
import torch
import numpy as np
from matplotlib.ticker import MaxNLocator
from matplotlib import ticker
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
    n_plot_rows = 2
    n_plot_cols = 5
    fig, axs = plt.subplots(n_plot_rows,n_plot_cols) # To present the plots
    for i in range(10):
        results = torch.load(f"testing/deployment_data_test_results/results_dictionary_depth_{i+1}", weights_only=False)

        results = calculate_metrics(results, remove_sites_list=[4035])  # Calculates the MAE and MSE for each of the models and adds them to the dictionary

        print(f"Averages are taken over all the deployment data. Prediction depth: {i+1}")
        print("MSE:{", end="")
        print(f"GRU: {round(results['Overall_Results']['GRU_MSE'], 2)}", end=" ")
        print(f"TCN: {round(results['Overall_Results']['TCN_MSE'], 2)}", end=" ")
        print(f"LSTM: {round(results['Overall_Results']['LSTM_MSE'], 2)}", end="}")
        print(" MAE:{ ", end="")
        print(f"GRU: {round(results['Overall_Results']['GRU_MAE'], 2)}",  end=" ")
        print(f"TCN: {round(results['Overall_Results']['TCN_MAE'], 2)}",  end=" ")
        print(f"LSTM: {round(results['Overall_Results']['LSTM_MAE'], 2)}", end="}")
        print("\n-------------------------------------------------------------------------------------")

        plotBounds = (0, 96)  # Corresponds to the start (inclusive) and end (exclusive) 15 minute blocks which are to be plotted
        plot_row = i//n_plot_cols
        plot_col = i%n_plot_cols
        plot_scat_site = 970

        axs[plot_row,plot_col].plot(results[plot_scat_site]["Targets"][plotBounds[0]:plotBounds[1]], 'g', label="TARGET")
        axs[plot_row,plot_col].plot(results[plot_scat_site]["GRU"][plotBounds[0]:plotBounds[1]], 'r', label="GRU")
        axs[plot_row,plot_col].plot(results[plot_scat_site]["TCN"][plotBounds[0]:plotBounds[1]], 'b', label="TCN")
        axs[plot_row,plot_col].plot(results[plot_scat_site]["LSTM"][plotBounds[0]:plotBounds[1]], 'm', label="LSTM")
        axs[plot_row, plot_col].set_title(f"Depth {i+1}")

        if plot_row != n_plot_rows -1:
            axs[plot_row,plot_col].set_xticks([0,31,63,95], labels=[])
        else:
            axs[plot_row,plot_col].set_xticks([0,31,63,95], labels=[0,31,63,95])
            axs[plot_row, plot_col].set(xlabel="Time\n(Number of 15 min intervals\nsince 2025/08/01 00:00:00)")



        if plot_col != 0:
            axs[plot_row,plot_col].set_yticks([0,50,100,150,200,250], labels=[])
        else:
            axs[plot_row,plot_col].set_yticks([0,50,100,150,200,250], labels=[0,50,100,150,200,250])
            axs[plot_row, plot_col].set(ylabel="Traffic Flow Volume (TFV)")


    han, lab = plt.gca().get_legend_handles_labels()
    fig.legend(han, lab, loc='upper right')
    fig.suptitle('Model Predictions vs Targets at Different Prediction Depths')
    plt.show()
    print("Program End")