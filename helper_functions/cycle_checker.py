def cycle_checker(node):
    cur_node = node.parent  # Set cur node to nodes parent #

    while cur_node is not None:  # Keep going until at the root #
        if cur_node.state == node.state:  # Cycle detected return true #
            return True
        cur_node = cur_node.parent

    return False  # Cycle not detected return false #
