from file_handling.graph_vertex_edge_init import GraphVertexEdgeInit

if __name__ == "__main__":
    testFileExtractor = GraphVertexEdgeInit(r"data/graph_init_data.xlsx")
    test_graph = testFileExtractor.extract_file_contents()
    print("fin")