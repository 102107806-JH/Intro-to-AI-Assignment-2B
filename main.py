from file_handling.graph_vertex_edge_init import GraphVertexEdgeInit
from textbook_abstractions.problem import Problem
from search_algorithms.dijkstra_search import dijkstras_serach
from ml_models.lstm_model import train_model
from ml_models.lstm_data_handler import preprocess_data
import pandas as pd
import itertools

if __name__ == "__main__":
    # Load the dataset
    print("Loading and preparing data...")
    data_file_path = "data/model_data.xlsx"
    df_scats = pd.read_excel(data_file_path)

    # Get a list of all unique SCATS numbers from the dataset
    all_scats_numbers = df_scats['SCATS_Number'].unique().tolist()

    # Define the hyperparameter grid for a coarse search
    param_grid = {
        'epochs': [10, 20, 50, 100],
        'learning_rate': [0.001, 0.0005, 0.0001],
        'hidden_size': [50, 100, 200, 400, 800],
        'num_layers': [1, 2, 3, 4, 8],
        'optimizer': ['Adam', 'SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adamax', 'NAdam', 'NAG']
    }

    best_models = {}  # Dictionary to store the best model and its accuracy for each SCATS site
    keys, values = zip(*param_grid.items())  # Get the keys and values from the hyperparameter grid
    print("Starting Hyperparameter Grid Search for All SCATS Sites")

    # Loop through each SCATS number to train a dedicated model
    for scats_number in all_scats_numbers:
        print(f"\nTraining models for SCATS site: {scats_number}")
        # Preprocess data specifically for the current SCATS number
        train_loader, min_val, max_val = preprocess_data(data_file_path, scats_number)
        best_accuracy = -1
        best_params = {}
        for v in itertools.product(*values):
            hyperparameters = dict(zip(keys, v))
            # Train the model and get the accuracy
            _, current_accuracy = train_model(train_loader, min_val, max_val, hyperparameters)
            if current_accuracy > best_accuracy:   # Check if the current model is the best one so far for this site
                best_accuracy = current_accuracy
                best_params = hyperparameters
        # After searching all combinations, train the final model for this site
        final_flowrate_model, final_accuracy = train_model(train_loader, min_val, max_val, best_params)
        # Store the best model and its accuracy
        best_models[scats_number] = {
            'model': final_flowrate_model,
            'accuracy': final_accuracy,
            'hyperparameters': best_params
        }
        print(f"Final training for SCATS {scats_number} complete.")
        print(f"Best Accuracy: {final_accuracy:.2f}%")
        print(f"Best Parameters: {best_params}")
    print("Grid Search and Final Model Training Complete for All Sites")

    # Initialize the graph
    print("\nInitializing the graph structure...")
    testFileExtractor = GraphVertexEdgeInit(r"data/graph_init_data.xlsx")
    graph = testFileExtractor.extract_file_contents()

    # Create a problem object and run a search with a single trained model
    # To use a different model, simply change the SCATS number here
    chosen_scats_for_path_cost = 4032
    chosen_model = best_models[chosen_scats_for_path_cost]['model']
    print(f"\nCreating problem and running Dijkstra's search using model for SCATS {chosen_scats_for_path_cost}")
    test_problem = Problem(graph, initial_state=3662, goal_state=2000,
                           flowrate_prediction_model=chosen_model, traffic_data=df_scats)

    # Run the Dijkstra's search algorithm
    solution_nodes = dijkstras_serach(test_problem, k_val=100)

    # Print the Dijkstra's search results
    print("\nDijkstra's Search Results:")
    if solution_nodes:
        for node in solution_nodes:
            print(f"Path found to node: {node.state} | Path Cost: {node.path_cost: .2f}")
    else:
        print("No path found.")
    print("\nProgram End")