from file_handling.graph_vertex_edge_init import GraphVertexEdgeInit
from textbook_abstractions.problem import Problem
from search_algorithms.dijkstra_search import dijkstras_serach
from ml_models.lstm_model import train_model
from ml_models.lstm_data_handler import preprocess_data
import pandas as pd
import torch

if __name__ == "__main__":
    print("CUDA available:", torch.cuda.is_available())
    print("Device count:", torch.cuda.device_count())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
    print("Loading and preparing data...")

    data_file_path = "data/model_data.xlsx"
    df_scats = pd.read_excel(data_file_path)
    all_scats_numbers = df_scats['SCATS_Number'].unique().tolist()

    #  Hyperparameter search loop used to find the best settings
    """
    param_grid = {
        'epochs': [10, 20, 30, 40, 50, 100],
        'learning_rate': [0.00001, 0.0005, 0.0001, 0.001, 0.01],
        'hidden_size': [50, 100, 200, 300, 400],
        'num_layers': [1, 2, 3, 4],
        'optimizer': ['Adam', 'RMSprop', 'Adagrad', 'Adadelta', 'Adamax', 'NAdam', 'NAG']
    }

    best_models = {}
    print("Starting Hyperparameter Random Search for All SCATS Sites")

    for scats_number in all_scats_numbers:
        print(f"\nTraining models for SCATS site: {scats_number}")
        train_loader, min_val, max_val = preprocess_data(data_file_path, scats_number)
        best_error = float("inf")
        best_params = {}
        samples = sample_hyperparameters(param_grid, n_samples=15)  # 15 random configs

        for hyperparameters in samples:
            _, current_error = train_model(train_loader, min_val, max_val, hyperparameters)
            if current_error < best_error:  # lower error = better
                best_error = current_error
                best_params = hyperparameters

        final_flowrate_model, final_error = train_model(train_loader, min_val, max_val, best_params)
        best_models[scats_number] = {
            'model': final_flowrate_model,
            'error': final_error,
            'hyperparameters': best_params
        }
        print(f"Final training for SCATS {scats_number} complete.")
        print(f"Best sMAPE: {final_error:.2f}%")
        print(f"Best Parameters: {best_params}")
    print("Random Search and Final Model Training Complete for All Sites")

    """

    final_hyperparameters = {
        'epochs': 100,           # with early stopping
        'learning_rate': 0.0005,
        'hidden_size': 150,
        'num_layers': 3,
        'optimizer': 'Adam'
    }

    best_models = {}
    for scats_number in all_scats_numbers:
        print(f"\nTraining model for SCATS site: {scats_number}")
        train_loader, min_val, max_val = preprocess_data(data_file_path, scats_number)
        final_flowrate_model, final_accuracy = train_model(train_loader, min_val, max_val, final_hyperparameters)
        best_models[scats_number] = {
            'model': final_flowrate_model,
            'accuracy': final_accuracy,
            'hyperparameters': final_hyperparameters
        }
    print("Model Training Complete for All Sites")

    # Graph and Pathfinding Integration
    print("\nInitializing the graph structure...")
    testFileExtractor = GraphVertexEdgeInit(r"data/graph_init_data.xlsx")
    graph = testFileExtractor.extract_file_contents()
    chosen_scats_for_path_cost = 4032
    chosen_model = best_models[chosen_scats_for_path_cost]['model']
    print(f"\nCreating problem and running Dijkstra's search using model for SCATS {chosen_scats_for_path_cost}")
    test_problem = Problem(graph, initial_state=3662, goal_state=2000,
                           flowrate_prediction_model=chosen_model, traffic_data=df_scats)
    solution_nodes = dijkstras_serach(test_problem, k_val=100)
    print("\nDijkstra's Search Results:")
    if solution_nodes:
        for node in solution_nodes:
            print(f"Path found to node: {node.state} | Path Cost: {node.path_cost: .2f}")
    else:
        print("No path found.")
    print("\nProgram End")