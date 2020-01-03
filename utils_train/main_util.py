

def logger_print(logger, vals_pre, vals_post):
    
    logger.info("yy1_truth_sequence_len: {}".format(vals_pre.pop("yy1_truth_sequence_len")))
    logger.info("f_max_sequence_length_y1: {}".format(vals_pre.pop("f_max_sequence_length_y1")))
    logger.info("s_max_sequence_length_y1: {}".format(vals_post.pop("s_max_sequence_length_y1")))
    logger.info("f_to_y_new_sequence_length_y1: {}".format(vals_pre.pop("f_to_y_new_sequence_length_y1")))
    logger.info("f_to_yy_new_sequence_length_y1: {}".format(vals_pre.pop("f_to_yy_new_sequence_length_y1")))
    logger.info("s_to_y_new_sequence_length_y1: {}".format(vals_post.pop("s_to_y_new_sequence_length_y1")))
    logger.info("s_to_yy_new_sequence_length_y1: {}".format(vals_post.pop("s_to_yy_new_sequence_length_y1")))
    logger.info("f_to_y_node_add_y1: {}".format(vals_pre.pop("f_to_y_node_add_y1")))
    logger.info("f_to_y_node_decrease_y1: {}".format(vals_pre.pop("f_to_y_node_decrease_y1")))
    logger.info("f_to_y_node_change_y1: {}".format(vals_pre.pop("f_to_y_node_change_y1")))
    logger.info("f_to_y_edge_change_y1: {}".format(vals_pre.pop("f_to_y_edge_change_y1")))
    logger.info("f_to_yy_node_add_y1: {}".format(vals_pre.pop("f_to_yy_node_add_y1")))
    logger.info("f_to_yy_node_decrease_y1: {}".format(vals_pre.pop("f_to_yy_node_decrease_y1")))
    logger.info("f_to_yy_node_change_y1: {}".format(vals_pre.pop("f_to_yy_node_change_y1")))
    logger.info("f_to_yy_edge_change_y1: {}".format(vals_pre.pop("f_to_yy_edge_change_y1")))
    logger.info("s_to_y_node_add_y1: {}".format(vals_post.pop("s_to_y_node_add_y1")))
    logger.info("s_to_y_node_decrease_y1: {}".format(vals_post.pop("s_to_y_node_decrease_y1")))
    logger.info("s_to_y_node_change_y1: {}".format(vals_post.pop("s_to_y_node_change_y1")))
    logger.info("s_to_y_edge_change_y1: {}".format(vals_post.pop("s_to_y_edge_change_y1")))
    logger.info("s_to_yy_node_add_y1: {}".format(vals_post.pop("s_to_yy_node_add_y1")))
    logger.info("s_to_yy_node_decrease_y1: {}".format(vals_post.pop("s_to_yy_node_decrease_y1")))
    logger.info("s_to_yy_node_change_y1: {}".format(vals_post.pop("s_to_yy_node_change_y1")))
    logger.info("s_to_yy_edge_change_y1: {}".format(vals_post.pop("s_to_yy_edge_change_y1")))

    logger.info("yy2_truth_sequence_len: {}".format(vals_pre.pop("yy2_truth_sequence_len")))
    logger.info("f_max_sequence_length_y2: {}".format(vals_pre.pop("f_max_sequence_length_y2")))
    logger.info("s_max_sequence_length_y2: {}".format(vals_post.pop("s_max_sequence_length_y2")))
    logger.info("f_to_y_new_sequence_length_y2: {}".format(vals_pre.pop("f_to_y_new_sequence_length_y2")))
    logger.info("f_to_yy_new_sequence_length_y2: {}".format(vals_pre.pop("f_to_yy_new_sequence_length_y2")))
    logger.info("s_to_y_new_sequence_length_y2: {}".format(vals_post.pop("s_to_y_new_sequence_length_y2")))
    logger.info("s_to_yy_new_sequence_length_y2: {}".format(vals_post.pop("s_to_yy_new_sequence_length_y2")))
    logger.info("f_to_y_node_add_y2: {}".format(vals_pre.pop("f_to_y_node_add_y2")))
    logger.info("f_to_y_node_decrease_y2: {}".format(vals_pre.pop("f_to_y_node_decrease_y2")))
    logger.info("f_to_y_node_change_y2: {}".format(vals_pre.pop("f_to_y_node_change_y2")))
    logger.info("f_to_y_edge_change_y2: {}".format(vals_pre.pop("f_to_y_edge_change_y2")))
    logger.info("f_to_yy_node_add_y2: {}".format(vals_pre.pop("f_to_yy_node_add_y2")))
    logger.info("f_to_yy_node_decrease_y2: {}".format(vals_pre.pop("f_to_yy_node_decrease_y2")))
    logger.info("f_to_yy_node_change_y2: {}".format(vals_pre.pop("f_to_yy_node_change_y2")))
    logger.info("f_to_yy_edge_change_y2: {}".format(vals_pre.pop("f_to_yy_edge_change_y2")))
    logger.info("s_to_y_node_add_y2: {}".format(vals_post.pop("s_to_y_node_add_y2")))
    logger.info("s_to_y_node_decrease_y2: {}".format(vals_post.pop("s_to_y_node_decrease_y2")))
    logger.info("s_to_y_node_change_y2: {}".format(vals_post.pop("s_to_y_node_change_y2")))
    logger.info("s_to_y_edge_change_y2: {}".format(vals_post.pop("s_to_y_edge_change_y2")))
    logger.info("s_to_yy_node_add_y2: {}".format(vals_post.pop("s_to_yy_node_add_y2")))
    logger.info("s_to_yy_node_decrease_y2: {}".format(vals_post.pop("s_to_yy_node_decrease_y2")))
    logger.info("s_to_yy_node_change_y2: {}".format(vals_post.pop("s_to_yy_node_change_y2")))
    logger.info("s_to_yy_edge_change_y2: {}".format(vals_post.pop("s_to_yy_edge_change_y2")))

    logger.info("yy3_truth_sequence_len: {}".format(vals_pre.pop("yy3_truth_sequence_len")))
    logger.info("f_max_sequence_length_y3: {}".format(vals_pre.pop("f_max_sequence_length_y3")))
    logger.info("s_max_sequence_length_y3: {}".format(vals_post.pop("s_max_sequence_length_y3")))
    logger.info("f_to_y_new_sequence_length_y3: {}".format(vals_pre.pop("f_to_y_new_sequence_length_y3")))
    logger.info("f_to_yy_new_sequence_length_y3: {}".format(vals_pre.pop("f_to_yy_new_sequence_length_y3")))
    logger.info("s_to_y_new_sequence_length_y3: {}".format(vals_post.pop("s_to_y_new_sequence_length_y3")))
    logger.info("s_to_yy_new_sequence_length_y3: {}".format(vals_post.pop("s_to_yy_new_sequence_length_y3")))
    logger.info("f_to_y_node_add_y3: {}".format(vals_pre.pop("f_to_y_node_add_y3")))
    logger.info("f_to_y_node_decrease_y3: {}".format(vals_pre.pop("f_to_y_node_decrease_y3")))
    logger.info("f_to_y_node_change_y3: {}".format(vals_pre.pop("f_to_y_node_change_y3")))
    logger.info("f_to_y_edge_change_y3: {}".format(vals_pre.pop("f_to_y_edge_change_y3")))
    logger.info("f_to_yy_node_add_y3: {}".format(vals_pre.pop("f_to_yy_node_add_y3")))
    logger.info("f_to_yy_node_decrease_y3: {}".format(vals_pre.pop("f_to_yy_node_decrease_y3")))
    logger.info("f_to_yy_node_change_y3: {}".format(vals_pre.pop("f_to_yy_node_change_y3")))
    logger.info("f_to_yy_edge_change_y3: {}".format(vals_pre.pop("f_to_yy_edge_change_y3")))
    logger.info("s_to_y_node_add_y3: {}".format(vals_post.pop("s_to_y_node_add_y3")))
    logger.info("s_to_y_node_decrease_y3: {}".format(vals_post.pop("s_to_y_node_decrease_y3")))
    logger.info("s_to_y_node_change_y3: {}".format(vals_post.pop("s_to_y_node_change_y3")))
    logger.info("s_to_y_edge_change_y3: {}".format(vals_post.pop("s_to_y_edge_change_y3")))
    logger.info("s_to_yy_node_add_y3: {}".format(vals_post.pop("s_to_yy_node_add_y3")))
    logger.info("s_to_yy_node_decrease_y3: {}".format(vals_post.pop("s_to_yy_node_decrease_y3")))
    logger.info("s_to_yy_node_change_y3: {}".format(vals_post.pop("s_to_yy_node_change_y3")))
    logger.info("s_to_yy_edge_change_y3: {}".format(vals_post.pop("s_to_yy_edge_change_y3")))


    logger.info("f_new_graph_01_y1: {}".format(vals_pre.pop("f_new_graph_01_y1")))
    logger.info("f_new_graph_01_y2: {}".format(vals_pre.pop("f_new_graph_01_y2")))
    logger.info("f_new_graph_01_y3: {}".format(vals_pre.pop("f_new_graph_01_y3")))
    logger.info("f_new_graph_logits_y1: {}".format(vals_pre.pop("f_new_graph_logits_y1")))
    logger.info("f_new_graph_logits_y2: {}".format(vals_pre.pop("f_new_graph_logits_y2")))
    logger.info("f_new_graph_logits_y3: {}".format(vals_pre.pop("f_new_graph_logits_y3")))
    logger.info("f_adjust_graph_y1: {}".format(vals_pre.pop("f_adjust_graph_y1")))
    logger.info("f_adjust_graph_y2: {}".format(vals_pre.pop("f_adjust_graph_y2")))
    logger.info("f_adjust_graph_y3: {}".format(vals_pre.pop("f_adjust_graph_y3")))

    logger.info("s_new_graph_01_y1: {}".format(vals_post.pop("s_new_graph_01_y1")))
    logger.info("s_new_graph_01_y2: {}".format(vals_post.pop("s_new_graph_01_y2")))
    logger.info("s_new_graph_01_y3: {}".format(vals_post.pop("s_new_graph_01_y3")))
    logger.info("s_new_graph_logits_y1: {}".format(vals_post.pop("s_new_graph_logits_y1")))
    logger.info("s_new_graph_logits_y2: {}".format(vals_post.pop("s_new_graph_logits_y2")))
    logger.info("s_new_graph_logits_y3: {}".format(vals_post.pop("s_new_graph_logits_y3")))
    logger.info("s_adjust_graph_y1: {}".format(vals_post.pop("s_adjust_graph_y1")))
    logger.info("s_adjust_graph_y2: {}".format(vals_post.pop("s_adjust_graph_y2")))
    logger.info("s_adjust_graph_y3: {}".format(vals_post.pop("s_adjust_graph_y3")))

    logger.info("adjs_y1_undirt: {}".format(vals_pre.pop("adjs_y1_undirt")))
    logger.info("adjs_y2_undirt: {}".format(vals_pre.pop("adjs_y2_undirt")))
    logger.info("adjs_y3_undirt: {}".format(vals_pre.pop("adjs_y3_undirt")))
    logger.info("adjs_yy1_undirt: {}".format(vals_pre.pop("adjs_yy1_undirt")))
    logger.info("adjs_yy2_undirt: {}".format(vals_pre.pop("adjs_yy2_undirt")))
    logger.info("adjs_yy3_undirt: {}".format(vals_pre.pop("adjs_yy3_undirt")))

    return vals_pre, vals_post

def logger_print2(logger, vals_pre):
    
    logger.info("yy1_truth_sequence_len: {}".format(vals_pre.pop("yy1_truth_sequence_len")))
    logger.info("f_max_sequence_length_y1: {}".format(vals_pre.pop("f_max_sequence_length_y1")))
    logger.info("s_max_sequence_length_y1: {}".format(vals_pre.pop("s_max_sequence_length_y1")))
    logger.info("f_to_y_new_sequence_length_y1: {}".format(vals_pre.pop("f_to_y_new_sequence_length_y1")))
    logger.info("f_to_yy_new_sequence_length_y1: {}".format(vals_pre.pop("f_to_yy_new_sequence_length_y1")))
    logger.info("s_to_y_new_sequence_length_y1: {}".format(vals_pre.pop("s_to_y_new_sequence_length_y1")))
    logger.info("s_to_yy_new_sequence_length_y1: {}".format(vals_pre.pop("s_to_yy_new_sequence_length_y1")))
    logger.info("f_to_y_node_add_y1: {}".format(vals_pre.pop("f_to_y_node_add_y1")))
    logger.info("f_to_y_node_decrease_y1: {}".format(vals_pre.pop("f_to_y_node_decrease_y1")))
    logger.info("f_to_y_node_change_y1: {}".format(vals_pre.pop("f_to_y_node_change_y1")))
    logger.info("f_to_y_edge_change_y1: {}".format(vals_pre.pop("f_to_y_edge_change_y1")))
    logger.info("f_to_yy_node_add_y1: {}".format(vals_pre.pop("f_to_yy_node_add_y1")))
    logger.info("f_to_yy_node_decrease_y1: {}".format(vals_pre.pop("f_to_yy_node_decrease_y1")))
    logger.info("f_to_yy_node_change_y1: {}".format(vals_pre.pop("f_to_yy_node_change_y1")))
    logger.info("f_to_yy_edge_change_y1: {}".format(vals_pre.pop("f_to_yy_edge_change_y1")))
    logger.info("s_to_y_node_add_y1: {}".format(vals_pre.pop("s_to_y_node_add_y1")))
    logger.info("s_to_y_node_decrease_y1: {}".format(vals_pre.pop("s_to_y_node_decrease_y1")))
    logger.info("s_to_y_node_change_y1: {}".format(vals_pre.pop("s_to_y_node_change_y1")))
    logger.info("s_to_y_edge_change_y1: {}".format(vals_pre.pop("s_to_y_edge_change_y1")))
    logger.info("s_to_yy_node_add_y1: {}".format(vals_pre.pop("s_to_yy_node_add_y1")))
    logger.info("s_to_yy_node_decrease_y1: {}".format(vals_pre.pop("s_to_yy_node_decrease_y1")))
    logger.info("s_to_yy_node_change_y1: {}".format(vals_pre.pop("s_to_yy_node_change_y1")))
    logger.info("s_to_yy_edge_change_y1: {}".format(vals_pre.pop("s_to_yy_edge_change_y1")))

    logger.info("yy2_truth_sequence_len: {}".format(vals_pre.pop("yy2_truth_sequence_len")))
    logger.info("f_max_sequence_length_y2: {}".format(vals_pre.pop("f_max_sequence_length_y2")))
    logger.info("s_max_sequence_length_y2: {}".format(vals_pre.pop("s_max_sequence_length_y2")))
    logger.info("f_to_y_new_sequence_length_y2: {}".format(vals_pre.pop("f_to_y_new_sequence_length_y2")))
    logger.info("f_to_yy_new_sequence_length_y2: {}".format(vals_pre.pop("f_to_yy_new_sequence_length_y2")))
    logger.info("s_to_y_new_sequence_length_y2: {}".format(vals_pre.pop("s_to_y_new_sequence_length_y2")))
    logger.info("s_to_yy_new_sequence_length_y2: {}".format(vals_pre.pop("s_to_yy_new_sequence_length_y2")))
    logger.info("f_to_y_node_add_y2: {}".format(vals_pre.pop("f_to_y_node_add_y2")))
    logger.info("f_to_y_node_decrease_y2: {}".format(vals_pre.pop("f_to_y_node_decrease_y2")))
    logger.info("f_to_y_node_change_y2: {}".format(vals_pre.pop("f_to_y_node_change_y2")))
    logger.info("f_to_y_edge_change_y2: {}".format(vals_pre.pop("f_to_y_edge_change_y2")))
    logger.info("f_to_yy_node_add_y2: {}".format(vals_pre.pop("f_to_yy_node_add_y2")))
    logger.info("f_to_yy_node_decrease_y2: {}".format(vals_pre.pop("f_to_yy_node_decrease_y2")))
    logger.info("f_to_yy_node_change_y2: {}".format(vals_pre.pop("f_to_yy_node_change_y2")))
    logger.info("f_to_yy_edge_change_y2: {}".format(vals_pre.pop("f_to_yy_edge_change_y2")))
    logger.info("s_to_y_node_add_y2: {}".format(vals_pre.pop("s_to_y_node_add_y2")))
    logger.info("s_to_y_node_decrease_y2: {}".format(vals_pre.pop("s_to_y_node_decrease_y2")))
    logger.info("s_to_y_node_change_y2: {}".format(vals_pre.pop("s_to_y_node_change_y2")))
    logger.info("s_to_y_edge_change_y2: {}".format(vals_pre.pop("s_to_y_edge_change_y2")))
    logger.info("s_to_yy_node_add_y2: {}".format(vals_pre.pop("s_to_yy_node_add_y2")))
    logger.info("s_to_yy_node_decrease_y2: {}".format(vals_pre.pop("s_to_yy_node_decrease_y2")))
    logger.info("s_to_yy_node_change_y2: {}".format(vals_pre.pop("s_to_yy_node_change_y2")))
    logger.info("s_to_yy_edge_change_y2: {}".format(vals_pre.pop("s_to_yy_edge_change_y2")))

    logger.info("yy3_truth_sequence_len: {}".format(vals_pre.pop("yy3_truth_sequence_len")))
    logger.info("f_max_sequence_length_y3: {}".format(vals_pre.pop("f_max_sequence_length_y3")))
    logger.info("s_max_sequence_length_y3: {}".format(vals_pre.pop("s_max_sequence_length_y3")))
    logger.info("f_to_y_new_sequence_length_y3: {}".format(vals_pre.pop("f_to_y_new_sequence_length_y3")))
    logger.info("f_to_yy_new_sequence_length_y3: {}".format(vals_pre.pop("f_to_yy_new_sequence_length_y3")))
    logger.info("s_to_y_new_sequence_length_y3: {}".format(vals_pre.pop("s_to_y_new_sequence_length_y3")))
    logger.info("s_to_yy_new_sequence_length_y3: {}".format(vals_pre.pop("s_to_yy_new_sequence_length_y3")))
    logger.info("f_to_y_node_add_y3: {}".format(vals_pre.pop("f_to_y_node_add_y3")))
    logger.info("f_to_y_node_decrease_y3: {}".format(vals_pre.pop("f_to_y_node_decrease_y3")))
    logger.info("f_to_y_node_change_y3: {}".format(vals_pre.pop("f_to_y_node_change_y3")))
    logger.info("f_to_y_edge_change_y3: {}".format(vals_pre.pop("f_to_y_edge_change_y3")))
    logger.info("f_to_yy_node_add_y3: {}".format(vals_pre.pop("f_to_yy_node_add_y3")))
    logger.info("f_to_yy_node_decrease_y3: {}".format(vals_pre.pop("f_to_yy_node_decrease_y3")))
    logger.info("f_to_yy_node_change_y3: {}".format(vals_pre.pop("f_to_yy_node_change_y3")))
    logger.info("f_to_yy_edge_change_y3: {}".format(vals_pre.pop("f_to_yy_edge_change_y3")))
    logger.info("s_to_y_node_add_y3: {}".format(vals_pre.pop("s_to_y_node_add_y3")))
    logger.info("s_to_y_node_decrease_y3: {}".format(vals_pre.pop("s_to_y_node_decrease_y3")))
    logger.info("s_to_y_node_change_y3: {}".format(vals_pre.pop("s_to_y_node_change_y3")))
    logger.info("s_to_y_edge_change_y3: {}".format(vals_pre.pop("s_to_y_edge_change_y3")))
    logger.info("s_to_yy_node_add_y3: {}".format(vals_pre.pop("s_to_yy_node_add_y3")))
    logger.info("s_to_yy_node_decrease_y3: {}".format(vals_pre.pop("s_to_yy_node_decrease_y3")))
    logger.info("s_to_yy_node_change_y3: {}".format(vals_pre.pop("s_to_yy_node_change_y3")))
    logger.info("s_to_yy_edge_change_y3: {}".format(vals_pre.pop("s_to_yy_edge_change_y3")))


    logger.info("f_new_graph_01_y1: {}".format(vals_pre.pop("f_new_graph_01_y1")))
    logger.info("f_new_graph_01_y2: {}".format(vals_pre.pop("f_new_graph_01_y2")))
    logger.info("f_new_graph_01_y3: {}".format(vals_pre.pop("f_new_graph_01_y3")))
    logger.info("f_new_graph_logits_y1: {}".format(vals_pre.pop("f_new_graph_logits_y1")))
    logger.info("f_new_graph_logits_y2: {}".format(vals_pre.pop("f_new_graph_logits_y2")))
    logger.info("f_new_graph_logits_y3: {}".format(vals_pre.pop("f_new_graph_logits_y3")))
    logger.info("f_adjust_graph_y1: {}".format(vals_pre.pop("f_adjust_graph_y1")))
    logger.info("f_adjust_graph_y2: {}".format(vals_pre.pop("f_adjust_graph_y2")))
    logger.info("f_adjust_graph_y3: {}".format(vals_pre.pop("f_adjust_graph_y3")))

    logger.info("s_new_graph_01_y1: {}".format(vals_pre.pop("s_new_graph_01_y1")))
    logger.info("s_new_graph_01_y2: {}".format(vals_pre.pop("s_new_graph_01_y2")))
    logger.info("s_new_graph_01_y3: {}".format(vals_pre.pop("s_new_graph_01_y3")))
    logger.info("s_new_graph_logits_y1: {}".format(vals_pre.pop("s_new_graph_logits_y1")))
    logger.info("s_new_graph_logits_y2: {}".format(vals_pre.pop("s_new_graph_logits_y2")))
    logger.info("s_new_graph_logits_y3: {}".format(vals_pre.pop("s_new_graph_logits_y3")))
    logger.info("s_adjust_graph_y1: {}".format(vals_pre.pop("s_adjust_graph_y1")))
    logger.info("s_adjust_graph_y2: {}".format(vals_pre.pop("s_adjust_graph_y2")))
    logger.info("s_adjust_graph_y3: {}".format(vals_pre.pop("s_adjust_graph_y3")))

    logger.info("adjs_y1_undirt: {}".format(vals_pre.pop("adjs_y1_undirt")))
    logger.info("adjs_y2_undirt: {}".format(vals_pre.pop("adjs_y2_undirt")))
    logger.info("adjs_y3_undirt: {}".format(vals_pre.pop("adjs_y3_undirt")))
    logger.info("adjs_yy1_undirt: {}".format(vals_pre.pop("adjs_yy1_undirt")))
    logger.info("adjs_yy2_undirt: {}".format(vals_pre.pop("adjs_yy2_undirt")))
    logger.info("adjs_yy3_undirt: {}".format(vals_pre.pop("adjs_yy3_undirt")))

    return vals_pre


def logger_print3(logger, vals_pre):
    
    logger.info("yy1_truth_sequence_len: {}".format(vals_pre.pop("yy1_truth_sequence_len")))
    logger.info("f_max_sequence_length_y1: {}".format(vals_pre.pop("f_max_sequence_length_y1")))
    logger.info("f_to_y_new_sequence_length_y1: {}".format(vals_pre.pop("f_to_y_new_sequence_length_y1")))
    logger.info("f_to_yy_new_sequence_length_y1: {}".format(vals_pre.pop("f_to_yy_new_sequence_length_y1")))
    logger.info("f_to_y_node_add_y1: {}".format(vals_pre.pop("f_to_y_node_add_y1")))
    logger.info("f_to_y_node_decrease_y1: {}".format(vals_pre.pop("f_to_y_node_decrease_y1")))
    logger.info("f_to_y_node_change_y1: {}".format(vals_pre.pop("f_to_y_node_change_y1")))
    logger.info("f_to_y_edge_change_y1: {}".format(vals_pre.pop("f_to_y_edge_change_y1")))
    logger.info("f_to_yy_node_add_y1: {}".format(vals_pre.pop("f_to_yy_node_add_y1")))
    logger.info("f_to_yy_node_decrease_y1: {}".format(vals_pre.pop("f_to_yy_node_decrease_y1")))
    logger.info("f_to_yy_node_change_y1: {}".format(vals_pre.pop("f_to_yy_node_change_y1")))
    logger.info("f_to_yy_edge_change_y1: {}".format(vals_pre.pop("f_to_yy_edge_change_y1")))
    

    logger.info("yy2_truth_sequence_len: {}".format(vals_pre.pop("yy2_truth_sequence_len")))
    logger.info("f_max_sequence_length_y2: {}".format(vals_pre.pop("f_max_sequence_length_y2")))
    logger.info("f_to_y_new_sequence_length_y2: {}".format(vals_pre.pop("f_to_y_new_sequence_length_y2")))
    logger.info("f_to_yy_new_sequence_length_y2: {}".format(vals_pre.pop("f_to_yy_new_sequence_length_y2")))
    logger.info("f_to_y_node_add_y2: {}".format(vals_pre.pop("f_to_y_node_add_y2")))
    logger.info("f_to_y_node_decrease_y2: {}".format(vals_pre.pop("f_to_y_node_decrease_y2")))
    logger.info("f_to_y_node_change_y2: {}".format(vals_pre.pop("f_to_y_node_change_y2")))
    logger.info("f_to_y_edge_change_y2: {}".format(vals_pre.pop("f_to_y_edge_change_y2")))
    logger.info("f_to_yy_node_add_y2: {}".format(vals_pre.pop("f_to_yy_node_add_y2")))
    logger.info("f_to_yy_node_decrease_y2: {}".format(vals_pre.pop("f_to_yy_node_decrease_y2")))
    logger.info("f_to_yy_node_change_y2: {}".format(vals_pre.pop("f_to_yy_node_change_y2")))
    logger.info("f_to_yy_edge_change_y2: {}".format(vals_pre.pop("f_to_yy_edge_change_y2")))
    

    logger.info("yy3_truth_sequence_len: {}".format(vals_pre.pop("yy3_truth_sequence_len")))
    logger.info("f_max_sequence_length_y3: {}".format(vals_pre.pop("f_max_sequence_length_y3")))
    logger.info("f_to_y_new_sequence_length_y3: {}".format(vals_pre.pop("f_to_y_new_sequence_length_y3")))
    logger.info("f_to_yy_new_sequence_length_y3: {}".format(vals_pre.pop("f_to_yy_new_sequence_length_y3")))
    logger.info("f_to_y_node_add_y3: {}".format(vals_pre.pop("f_to_y_node_add_y3")))
    logger.info("f_to_y_node_decrease_y3: {}".format(vals_pre.pop("f_to_y_node_decrease_y3")))
    logger.info("f_to_y_node_change_y3: {}".format(vals_pre.pop("f_to_y_node_change_y3")))
    logger.info("f_to_y_edge_change_y3: {}".format(vals_pre.pop("f_to_y_edge_change_y3")))
    logger.info("f_to_yy_node_add_y3: {}".format(vals_pre.pop("f_to_yy_node_add_y3")))
    logger.info("f_to_yy_node_decrease_y3: {}".format(vals_pre.pop("f_to_yy_node_decrease_y3")))
    logger.info("f_to_yy_node_change_y3: {}".format(vals_pre.pop("f_to_yy_node_change_y3")))
    logger.info("f_to_yy_edge_change_y3: {}".format(vals_pre.pop("f_to_yy_edge_change_y3")))
    

    logger.info("f_new_graph_01_y1: {}".format(vals_pre.pop("f_new_graph_01_y1")))
    logger.info("f_new_graph_01_y2: {}".format(vals_pre.pop("f_new_graph_01_y2")))
    logger.info("f_new_graph_01_y3: {}".format(vals_pre.pop("f_new_graph_01_y3")))
    logger.info("f_new_graph_logits_y1: {}".format(vals_pre.pop("f_new_graph_logits_y1")))
    logger.info("f_new_graph_logits_y2: {}".format(vals_pre.pop("f_new_graph_logits_y2")))
    logger.info("f_new_graph_logits_y3: {}".format(vals_pre.pop("f_new_graph_logits_y3")))
    logger.info("f_adjust_graph_y1: {}".format(vals_pre.pop("f_adjust_graph_y1")))
    logger.info("f_adjust_graph_y2: {}".format(vals_pre.pop("f_adjust_graph_y2")))
    logger.info("f_adjust_graph_y3: {}".format(vals_pre.pop("f_adjust_graph_y3")))

    
    logger.info("adjs_y1_undirt: {}".format(vals_pre.pop("adjs_y1_undirt")))
    logger.info("adjs_y2_undirt: {}".format(vals_pre.pop("adjs_y2_undirt")))
    logger.info("adjs_y3_undirt: {}".format(vals_pre.pop("adjs_y3_undirt")))
    logger.info("adjs_yy1_undirt: {}".format(vals_pre.pop("adjs_yy1_undirt")))
    logger.info("adjs_yy2_undirt: {}".format(vals_pre.pop("adjs_yy2_undirt")))
    logger.info("adjs_yy3_undirt: {}".format(vals_pre.pop("adjs_yy3_undirt")))

    return vals_pre

def logger_print4(logger, vals_pre):
    
    logger.info("yy1_truth_sequence_len: {}".format(vals_pre.pop("yy1_truth_sequence_len")))
    logger.info("f_max_sequence_length_y1: {}".format(vals_pre.pop("f_max_sequence_length_y1")))
    logger.info("f_to_y_new_sequence_length_y1: {}".format(vals_pre.pop("f_to_y_new_sequence_length_y1")))
    logger.info("f_to_yy_new_sequence_length_y1: {}".format(vals_pre.pop("f_to_yy_new_sequence_length_y1")))
    logger.info("f_to_y_node_add_y1: {}".format(vals_pre.pop("f_to_y_node_add_y1")))
    logger.info("f_to_y_node_decrease_y1: {}".format(vals_pre.pop("f_to_y_node_decrease_y1")))
    logger.info("f_to_y_node_change_y1: {}".format(vals_pre.pop("f_to_y_node_change_y1")))
    logger.info("f_to_y_edge_change_y1: {}".format(vals_pre.pop("f_to_y_edge_change_y1")))
    logger.info("f_to_yy_node_add_y1: {}".format(vals_pre.pop("f_to_yy_node_add_y1")))
    logger.info("f_to_yy_node_decrease_y1: {}".format(vals_pre.pop("f_to_yy_node_decrease_y1")))
    logger.info("f_to_yy_node_change_y1: {}".format(vals_pre.pop("f_to_yy_node_change_y1")))
    logger.info("f_to_yy_edge_change_y1: {}".format(vals_pre.pop("f_to_yy_edge_change_y1")))
    

    logger.info("yy2_truth_sequence_len: {}".format(vals_pre.pop("yy2_truth_sequence_len")))
    logger.info("f_max_sequence_length_y2: {}".format(vals_pre.pop("f_max_sequence_length_y2")))
    logger.info("f_to_y_new_sequence_length_y2: {}".format(vals_pre.pop("f_to_y_new_sequence_length_y2")))
    logger.info("f_to_yy_new_sequence_length_y2: {}".format(vals_pre.pop("f_to_yy_new_sequence_length_y2")))
    logger.info("f_to_y_node_add_y2: {}".format(vals_pre.pop("f_to_y_node_add_y2")))
    logger.info("f_to_y_node_decrease_y2: {}".format(vals_pre.pop("f_to_y_node_decrease_y2")))
    logger.info("f_to_y_node_change_y2: {}".format(vals_pre.pop("f_to_y_node_change_y2")))
    logger.info("f_to_y_edge_change_y2: {}".format(vals_pre.pop("f_to_y_edge_change_y2")))
    logger.info("f_to_yy_node_add_y2: {}".format(vals_pre.pop("f_to_yy_node_add_y2")))
    logger.info("f_to_yy_node_decrease_y2: {}".format(vals_pre.pop("f_to_yy_node_decrease_y2")))
    logger.info("f_to_yy_node_change_y2: {}".format(vals_pre.pop("f_to_yy_node_change_y2")))
    logger.info("f_to_yy_edge_change_y2: {}".format(vals_pre.pop("f_to_yy_edge_change_y2")))
    

    logger.info("yy3_truth_sequence_len: {}".format(vals_pre.pop("yy3_truth_sequence_len")))
    logger.info("f_max_sequence_length_y3: {}".format(vals_pre.pop("f_max_sequence_length_y3")))
    logger.info("f_to_y_new_sequence_length_y3: {}".format(vals_pre.pop("f_to_y_new_sequence_length_y3")))
    logger.info("f_to_yy_new_sequence_length_y3: {}".format(vals_pre.pop("f_to_yy_new_sequence_length_y3")))
    logger.info("f_to_y_node_add_y3: {}".format(vals_pre.pop("f_to_y_node_add_y3")))
    logger.info("f_to_y_node_decrease_y3: {}".format(vals_pre.pop("f_to_y_node_decrease_y3")))
    logger.info("f_to_y_node_change_y3: {}".format(vals_pre.pop("f_to_y_node_change_y3")))
    logger.info("f_to_y_edge_change_y3: {}".format(vals_pre.pop("f_to_y_edge_change_y3")))
    logger.info("f_to_yy_node_add_y3: {}".format(vals_pre.pop("f_to_yy_node_add_y3")))
    logger.info("f_to_yy_node_decrease_y3: {}".format(vals_pre.pop("f_to_yy_node_decrease_y3")))
    logger.info("f_to_yy_node_change_y3: {}".format(vals_pre.pop("f_to_yy_node_change_y3")))
    logger.info("f_to_yy_edge_change_y3: {}".format(vals_pre.pop("f_to_yy_edge_change_y3")))
    

    logger.info("f_new_graph_01_y1: {}".format(vals_pre.pop("f_new_graph_01_y1")))
    logger.info("self.priority_score_y1: {}".format(vals_pre.pop("self.priority_score_y1")))
    logger.info("f_new_graph_01_y2: {}".format(vals_pre.pop("f_new_graph_01_y2")))
    logger.info("self.priority_score_y2: {}".format(vals_pre.pop("self.priority_score_y2")))
    logger.info("f_new_graph_01_y3: {}".format(vals_pre.pop("f_new_graph_01_y3")))
    logger.info("self.priority_score_y3: {}".format(vals_pre.pop("self.priority_score_y3")))
    logger.info("f_new_graph_logits_y1: {}".format(vals_pre.pop("f_new_graph_logits_y1")))
    logger.info("f_new_graph_logits_y2: {}".format(vals_pre.pop("f_new_graph_logits_y2")))
    logger.info("f_new_graph_logits_y3: {}".format(vals_pre.pop("f_new_graph_logits_y3")))
    logger.info("f_adjust_graph_y1: {}".format(vals_pre.pop("f_adjust_graph_y1")))
    logger.info("f_adjust_graph_y2: {}".format(vals_pre.pop("f_adjust_graph_y2")))
    logger.info("f_adjust_graph_y3: {}".format(vals_pre.pop("f_adjust_graph_y3")))

    
    logger.info("adjs_y1_undirt: {}".format(vals_pre.pop("adjs_y1_undirt")))
    logger.info("adjs_y2_undirt: {}".format(vals_pre.pop("adjs_y2_undirt")))
    logger.info("adjs_y3_undirt: {}".format(vals_pre.pop("adjs_y3_undirt")))
    logger.info("adjs_yy1_undirt: {}".format(vals_pre.pop("adjs_yy1_undirt")))
    logger.info("adjs_yy2_undirt: {}".format(vals_pre.pop("adjs_yy2_undirt")))
    logger.info("adjs_yy3_undirt: {}".format(vals_pre.pop("adjs_yy3_undirt")))

    return vals_pre


def logger_print5(logger, vals_pre):

    logger.info("====================== Start To Report node and edge change ======================")
    logger.info("====================== Start To Report Y1 ======================")
    logger.info("yy1_truth_sequence_len: {}".format(vals_pre.pop("yy1_truth_sequence_len")))
    # first state
    logger.info("f_max_sequence_length_y1_ori: {}".format(vals_pre.pop("f_max_sequence_length_y1_ori")))
    logger.info("f_to_y_node_add_y1_ori: {}".format(vals_pre.pop("f_to_y_node_add_y1_ori")))
    logger.info("f_to_y_node_decrease_y1_ori: {}".format(vals_pre.pop("f_to_y_node_decrease_y1_ori")))
    logger.info("f_to_y_node_change_y1_ori: {}".format(vals_pre.pop("f_to_y_node_change_y1_ori")))
    logger.info("f_to_y_edge_change_y1_ori: {}".format(vals_pre.pop("f_to_y_edge_change_y1_ori")))
    
    logger.info("f_max_sequence_length_y1_cft: {}".format(vals_pre.pop("f_max_sequence_length_y1_cft")))
    logger.info("f_to_yy_node_add_y1_cft: {}".format(vals_pre.pop("f_to_yy_node_add_y1_cft")))
    logger.info("f_to_yy_node_decrease_y1_cft: {}".format(vals_pre.pop("f_to_yy_node_decrease_y1_cft")))
    logger.info("f_to_yy_node_change_y1_cft: {}".format(vals_pre.pop("f_to_yy_node_change_y1_cft")))
    logger.info("f_to_yy_edge_change_y1_cft: {}".format(vals_pre.pop("f_to_yy_edge_change_y1_cft")))
    # sec state
    logger.info("s_max_sequence_length_y1: {}".format(vals_pre.pop("s_max_sequence_length_y1")))
    logger.info("s_to_y_node_add_y1: {}".format(vals_pre.pop("s_to_y_node_add_y1")))
    logger.info("s_to_y_node_decrease_y1: {}".format(vals_pre.pop("s_to_y_node_decrease_y1")))
    logger.info("s_to_y_node_change_y1: {}".format(vals_pre.pop("s_to_y_node_change_y1")))
    logger.info("s_to_y_edge_change_y1: {}".format(vals_pre.pop("s_to_y_edge_change_y1")))
    logger.info("s_to_yy_node_add_y1: {}".format(vals_pre.pop("s_to_yy_node_add_y1")))
    logger.info("s_to_yy_node_decrease_y1: {}".format(vals_pre.pop("s_to_yy_node_decrease_y1")))
    logger.info("s_to_yy_node_change_y1: {}".format(vals_pre.pop("s_to_yy_node_change_y1")))
    logger.info("s_to_yy_edge_change_y1: {}".format(vals_pre.pop("s_to_yy_edge_change_y1")))
    # unusable state
    logger.info("f_to_y_new_sequence_length_y1_ori: {}".format(vals_pre.pop("f_to_y_new_sequence_length_y1_ori")))
    logger.info("f_to_yy_new_sequence_length_y1_cft: {}".format(vals_pre.pop("f_to_yy_new_sequence_length_y1_cft")))
    logger.info("s_to_y_new_sequence_length_y1: {}".format(vals_pre.pop("s_to_y_new_sequence_length_y1")))
    logger.info("s_to_yy_new_sequence_length_y1: {}".format(vals_pre.pop("s_to_yy_new_sequence_length_y1")))




    logger.info("====================== Start To Report Y2 ======================")
    logger.info("yy2_truth_sequence_len: {}".format(vals_pre.pop("yy2_truth_sequence_len")))
    # first state
    logger.info("f_max_sequence_length_y2_ori: {}".format(vals_pre.pop("f_max_sequence_length_y2_ori")))
    logger.info("f_to_y_node_add_y2_ori: {}".format(vals_pre.pop("f_to_y_node_add_y2_ori")))
    logger.info("f_to_y_node_decrease_y2_ori: {}".format(vals_pre.pop("f_to_y_node_decrease_y2_ori")))
    logger.info("f_to_y_node_change_y2_ori: {}".format(vals_pre.pop("f_to_y_node_change_y2_ori")))
    logger.info("f_to_y_edge_change_y2_ori: {}".format(vals_pre.pop("f_to_y_edge_change_y2_ori")))
    
    logger.info("f_max_sequence_length_y2_cft: {}".format(vals_pre.pop("f_max_sequence_length_y2_cft")))
    logger.info("f_to_yy_node_add_y2_cft: {}".format(vals_pre.pop("f_to_yy_node_add_y2_cft")))
    logger.info("f_to_yy_node_decrease_y2_cft: {}".format(vals_pre.pop("f_to_yy_node_decrease_y2_cft")))
    logger.info("f_to_yy_node_change_y2_cft: {}".format(vals_pre.pop("f_to_yy_node_change_y2_cft")))
    logger.info("f_to_yy_edge_change_y2_cft: {}".format(vals_pre.pop("f_to_yy_edge_change_y2_cft")))
    # sec state
    logger.info("s_max_sequence_length_y2: {}".format(vals_pre.pop("s_max_sequence_length_y2")))
    logger.info("s_to_y_node_add_y2: {}".format(vals_pre.pop("s_to_y_node_add_y2")))
    logger.info("s_to_y_node_decrease_y2: {}".format(vals_pre.pop("s_to_y_node_decrease_y2")))
    logger.info("s_to_y_node_change_y2: {}".format(vals_pre.pop("s_to_y_node_change_y2")))
    logger.info("s_to_y_edge_change_y2: {}".format(vals_pre.pop("s_to_y_edge_change_y2")))
    logger.info("s_to_yy_node_add_y2: {}".format(vals_pre.pop("s_to_yy_node_add_y2")))
    logger.info("s_to_yy_node_decrease_y2: {}".format(vals_pre.pop("s_to_yy_node_decrease_y2")))
    logger.info("s_to_yy_node_change_y2: {}".format(vals_pre.pop("s_to_yy_node_change_y2")))
    logger.info("s_to_yy_edge_change_y2: {}".format(vals_pre.pop("s_to_yy_edge_change_y2")))
    # unusable state
    logger.info("f_to_y_new_sequence_length_y2_ori: {}".format(vals_pre.pop("f_to_y_new_sequence_length_y2_ori")))
    logger.info("f_to_yy_new_sequence_length_y2_cft: {}".format(vals_pre.pop("f_to_yy_new_sequence_length_y2_cft")))
    logger.info("s_to_y_new_sequence_length_y2: {}".format(vals_pre.pop("s_to_y_new_sequence_length_y2")))
    logger.info("s_to_yy_new_sequence_length_y2: {}".format(vals_pre.pop("s_to_yy_new_sequence_length_y2")))





    logger.info("====================== Start To Report Y3 ======================")
    logger.info("yy3_truth_sequence_len: {}".format(vals_pre.pop("yy3_truth_sequence_len")))
    # first state
    logger.info("f_max_sequence_length_y3_ori: {}".format(vals_pre.pop("f_max_sequence_length_y3_ori")))
    logger.info("f_to_y_node_add_y3_ori: {}".format(vals_pre.pop("f_to_y_node_add_y3_ori")))
    logger.info("f_to_y_node_decrease_y3_ori: {}".format(vals_pre.pop("f_to_y_node_decrease_y3_ori")))
    logger.info("f_to_y_node_change_y3_ori: {}".format(vals_pre.pop("f_to_y_node_change_y3_ori")))
    logger.info("f_to_y_edge_change_y3_ori: {}".format(vals_pre.pop("f_to_y_edge_change_y3_ori")))
    
    logger.info("f_max_sequence_length_y3_cft: {}".format(vals_pre.pop("f_max_sequence_length_y3_cft")))
    logger.info("f_to_yy_node_add_y3_cft: {}".format(vals_pre.pop("f_to_yy_node_add_y3_cft")))
    logger.info("f_to_yy_node_decrease_y3_cft: {}".format(vals_pre.pop("f_to_yy_node_decrease_y3_cft")))
    logger.info("f_to_yy_node_change_y3_cft: {}".format(vals_pre.pop("f_to_yy_node_change_y3_cft")))
    logger.info("f_to_yy_edge_change_y3_cft: {}".format(vals_pre.pop("f_to_yy_edge_change_y3_cft")))
    # sec state
    logger.info("s_max_sequence_length_y3: {}".format(vals_pre.pop("s_max_sequence_length_y3")))
    logger.info("s_to_y_node_add_y3: {}".format(vals_pre.pop("s_to_y_node_add_y3")))
    logger.info("s_to_y_node_decrease_y3: {}".format(vals_pre.pop("s_to_y_node_decrease_y3")))
    logger.info("s_to_y_node_change_y3: {}".format(vals_pre.pop("s_to_y_node_change_y3")))
    logger.info("s_to_y_edge_change_y3: {}".format(vals_pre.pop("s_to_y_edge_change_y3")))
    logger.info("s_to_yy_node_add_y3: {}".format(vals_pre.pop("s_to_yy_node_add_y3")))
    logger.info("s_to_yy_node_decrease_y3: {}".format(vals_pre.pop("s_to_yy_node_decrease_y3")))
    logger.info("s_to_yy_node_change_y3: {}".format(vals_pre.pop("s_to_yy_node_change_y3")))
    logger.info("s_to_yy_edge_change_y3: {}".format(vals_pre.pop("s_to_yy_edge_change_y3")))
    # unusable state
    logger.info("f_to_y_new_sequence_length_y3_ori: {}".format(vals_pre.pop("f_to_y_new_sequence_length_y3_ori")))
    logger.info("f_to_yy_new_sequence_length_y3_cft: {}".format(vals_pre.pop("f_to_yy_new_sequence_length_y3_cft")))
    logger.info("s_to_y_new_sequence_length_y3: {}".format(vals_pre.pop("s_to_y_new_sequence_length_y3")))
    logger.info("s_to_yy_new_sequence_length_y3: {}".format(vals_pre.pop("s_to_yy_new_sequence_length_y3")))



    logger.info("====================== Start To Report First adj change ======================")
    # y1
    logger.info("f_new_graph_01_y1_ori: {}".format(vals_pre.pop("f_new_graph_01_y1_ori")))
    logger.info("f_new_graph_logits_y1_ori: {}".format(vals_pre.pop("f_new_graph_logits_y1_ori")))
    logger.info("f_adjust_graph_y1_ori: {}".format(vals_pre.pop("f_adjust_graph_y1_ori")))
    logger.info("f_new_graph_01_y1_cft: {}".format(vals_pre.pop("f_new_graph_01_y1_cft")))
    logger.info("f_new_graph_logits_y1_cft: {}".format(vals_pre.pop("f_new_graph_logits_y1_cft")))
    logger.info("f_adjust_graph_y1_cft: {}".format(vals_pre.pop("f_adjust_graph_y1_cft")))
    # y2
    logger.info("f_new_graph_01_y2_ori: {}".format(vals_pre.pop("f_new_graph_01_y2_ori")))
    logger.info("f_new_graph_logits_y2_ori: {}".format(vals_pre.pop("f_new_graph_logits_y2_ori")))
    logger.info("f_adjust_graph_y2_ori: {}".format(vals_pre.pop("f_adjust_graph_y2_ori")))
    logger.info("f_new_graph_01_y2_cft: {}".format(vals_pre.pop("f_new_graph_01_y2_cft")))
    logger.info("f_new_graph_logits_y2_cft: {}".format(vals_pre.pop("f_new_graph_logits_y2_cft")))
    logger.info("f_adjust_graph_y2_cft: {}".format(vals_pre.pop("f_adjust_graph_y2_cft")))
    # y3
    logger.info("f_new_graph_01_y3_ori: {}".format(vals_pre.pop("f_new_graph_01_y3_ori")))
    logger.info("f_new_graph_logits_y3_ori: {}".format(vals_pre.pop("f_new_graph_logits_y3_ori")))
    logger.info("f_adjust_graph_y3_ori: {}".format(vals_pre.pop("f_adjust_graph_y3_ori")))
    logger.info("f_new_graph_01_y3_cft: {}".format(vals_pre.pop("f_new_graph_01_y3_cft")))
    logger.info("f_new_graph_logits_y3_cft: {}".format(vals_pre.pop("f_new_graph_logits_y3_cft")))
    logger.info("f_adjust_graph_y3_cft: {}".format(vals_pre.pop("f_adjust_graph_y3_cft")))




    logger.info("====================== Start To Report Second adj change ======================")
    # y1
    logger.info("s_new_graph_01_y1: {}".format(vals_pre.pop("s_new_graph_01_y1")))
    logger.info("s_new_graph_logits_y1: {}".format(vals_pre.pop("s_new_graph_logits_y1")))
    logger.info("s_adjust_graph_y1: {}".format(vals_pre.pop("s_adjust_graph_y1")))
    # y2
    logger.info("s_new_graph_01_y2: {}".format(vals_pre.pop("s_new_graph_01_y2")))
    logger.info("s_new_graph_logits_y2: {}".format(vals_pre.pop("s_new_graph_logits_y2")))
    logger.info("s_adjust_graph_y2: {}".format(vals_pre.pop("s_adjust_graph_y2")))
    # y3
    logger.info("s_new_graph_01_y3: {}".format(vals_pre.pop("s_new_graph_01_y3")))    
    logger.info("s_new_graph_logits_y3: {}".format(vals_pre.pop("s_new_graph_logits_y3")))
    logger.info("s_adjust_graph_y3: {}".format(vals_pre.pop("s_adjust_graph_y3")))



    logger.info("====================== Start To Report adj gt for y and yy ======================")
    logger.info("adjs_y1_undirt: {}".format(vals_pre.pop("adjs_y1_undirt")))
    logger.info("adjs_y2_undirt: {}".format(vals_pre.pop("adjs_y2_undirt")))
    logger.info("adjs_y3_undirt: {}".format(vals_pre.pop("adjs_y3_undirt")))
    logger.info("adjs_yy1_undirt: {}".format(vals_pre.pop("adjs_yy1_undirt")))
    logger.info("adjs_yy2_undirt: {}".format(vals_pre.pop("adjs_yy2_undirt")))
    logger.info("adjs_yy3_undirt: {}".format(vals_pre.pop("adjs_yy3_undirt")))

    return vals_pre


def logger_print_noF(logger, vals_pre):

    logger.info("====================== Start To Report node and edge change ======================")
    logger.info("====================== Start To Report Y1 ======================")
    logger.info("yy1_truth_sequence_len: {}".format(vals_pre.pop("yy1_truth_sequence_len")))

    # sec state
    logger.info("s_max_sequence_length_y1: {}".format(vals_pre.pop("s_max_sequence_length_y1")))
    logger.info("s_to_y_node_add_y1: {}".format(vals_pre.pop("s_to_y_node_add_y1")))
    logger.info("s_to_y_node_decrease_y1: {}".format(vals_pre.pop("s_to_y_node_decrease_y1")))
    logger.info("s_to_y_node_change_y1: {}".format(vals_pre.pop("s_to_y_node_change_y1")))
    logger.info("s_to_y_edge_change_y1: {}".format(vals_pre.pop("s_to_y_edge_change_y1")))
    logger.info("s_to_yy_node_add_y1: {}".format(vals_pre.pop("s_to_yy_node_add_y1")))
    logger.info("s_to_yy_node_decrease_y1: {}".format(vals_pre.pop("s_to_yy_node_decrease_y1")))
    logger.info("s_to_yy_node_change_y1: {}".format(vals_pre.pop("s_to_yy_node_change_y1")))
    logger.info("s_to_yy_edge_change_y1: {}".format(vals_pre.pop("s_to_yy_edge_change_y1")))
    # unusable state
    logger.info("s_to_y_new_sequence_length_y1: {}".format(vals_pre.pop("s_to_y_new_sequence_length_y1")))
    logger.info("s_to_yy_new_sequence_length_y1: {}".format(vals_pre.pop("s_to_yy_new_sequence_length_y1")))




    logger.info("====================== Start To Report Y2 ======================")
    logger.info("yy2_truth_sequence_len: {}".format(vals_pre.pop("yy2_truth_sequence_len")))

    # sec state
    logger.info("s_max_sequence_length_y2: {}".format(vals_pre.pop("s_max_sequence_length_y2")))
    logger.info("s_to_y_node_add_y2: {}".format(vals_pre.pop("s_to_y_node_add_y2")))
    logger.info("s_to_y_node_decrease_y2: {}".format(vals_pre.pop("s_to_y_node_decrease_y2")))
    logger.info("s_to_y_node_change_y2: {}".format(vals_pre.pop("s_to_y_node_change_y2")))
    logger.info("s_to_y_edge_change_y2: {}".format(vals_pre.pop("s_to_y_edge_change_y2")))
    logger.info("s_to_yy_node_add_y2: {}".format(vals_pre.pop("s_to_yy_node_add_y2")))
    logger.info("s_to_yy_node_decrease_y2: {}".format(vals_pre.pop("s_to_yy_node_decrease_y2")))
    logger.info("s_to_yy_node_change_y2: {}".format(vals_pre.pop("s_to_yy_node_change_y2")))
    logger.info("s_to_yy_edge_change_y2: {}".format(vals_pre.pop("s_to_yy_edge_change_y2")))
    # unusable state
    logger.info("s_to_y_new_sequence_length_y2: {}".format(vals_pre.pop("s_to_y_new_sequence_length_y2")))
    logger.info("s_to_yy_new_sequence_length_y2: {}".format(vals_pre.pop("s_to_yy_new_sequence_length_y2")))





    logger.info("====================== Start To Report Y3 ======================")
    logger.info("yy3_truth_sequence_len: {}".format(vals_pre.pop("yy3_truth_sequence_len")))

    # sec state
    logger.info("s_max_sequence_length_y3: {}".format(vals_pre.pop("s_max_sequence_length_y3")))
    logger.info("s_to_y_node_add_y3: {}".format(vals_pre.pop("s_to_y_node_add_y3")))
    logger.info("s_to_y_node_decrease_y3: {}".format(vals_pre.pop("s_to_y_node_decrease_y3")))
    logger.info("s_to_y_node_change_y3: {}".format(vals_pre.pop("s_to_y_node_change_y3")))
    logger.info("s_to_y_edge_change_y3: {}".format(vals_pre.pop("s_to_y_edge_change_y3")))
    logger.info("s_to_yy_node_add_y3: {}".format(vals_pre.pop("s_to_yy_node_add_y3")))
    logger.info("s_to_yy_node_decrease_y3: {}".format(vals_pre.pop("s_to_yy_node_decrease_y3")))
    logger.info("s_to_yy_node_change_y3: {}".format(vals_pre.pop("s_to_yy_node_change_y3")))
    logger.info("s_to_yy_edge_change_y3: {}".format(vals_pre.pop("s_to_yy_edge_change_y3")))
    # unusable state
    
    logger.info("s_to_y_new_sequence_length_y3: {}".format(vals_pre.pop("s_to_y_new_sequence_length_y3")))
    logger.info("s_to_yy_new_sequence_length_y3: {}".format(vals_pre.pop("s_to_yy_new_sequence_length_y3")))



    logger.info("====================== Start To Report Second adj change ======================")
    # y1
    logger.info("s_new_graph_01_y1: {}".format(vals_pre.pop("s_new_graph_01_y1")))
    logger.info("s_new_graph_logits_y1: {}".format(vals_pre.pop("s_new_graph_logits_y1")))
    logger.info("s_adjust_graph_y1: {}".format(vals_pre.pop("s_adjust_graph_y1")))
    # y2
    logger.info("s_new_graph_01_y2: {}".format(vals_pre.pop("s_new_graph_01_y2")))
    logger.info("s_new_graph_logits_y2: {}".format(vals_pre.pop("s_new_graph_logits_y2")))
    logger.info("s_adjust_graph_y2: {}".format(vals_pre.pop("s_adjust_graph_y2")))
    # y3
    logger.info("s_new_graph_01_y3: {}".format(vals_pre.pop("s_new_graph_01_y3")))    
    logger.info("s_new_graph_logits_y3: {}".format(vals_pre.pop("s_new_graph_logits_y3")))
    logger.info("s_adjust_graph_y3: {}".format(vals_pre.pop("s_adjust_graph_y3")))



    logger.info("====================== Start To Report adj gt for y and yy ======================")
    logger.info("adjs_y1_undirt: {}".format(vals_pre.pop("adjs_y1_undirt")))
    logger.info("adjs_y2_undirt: {}".format(vals_pre.pop("adjs_y2_undirt")))
    logger.info("adjs_y3_undirt: {}".format(vals_pre.pop("adjs_y3_undirt")))
    logger.info("adjs_yy1_undirt: {}".format(vals_pre.pop("adjs_yy1_undirt")))
    logger.info("adjs_yy2_undirt: {}".format(vals_pre.pop("adjs_yy2_undirt")))
    logger.info("adjs_yy3_undirt: {}".format(vals_pre.pop("adjs_yy3_undirt")))

    return vals_pre




def pop(vals_pre):
    
    vals_pre.pop("yy1_truth_sequence_len")
    vals_pre.pop("f_max_sequence_length_y1")
    vals_pre.pop("s_max_sequence_length_y1")
    vals_pre.pop("f_to_y_new_sequence_length_y1")
    vals_pre.pop("f_to_yy_new_sequence_length_y1")
    vals_pre.pop("s_to_y_new_sequence_length_y1")
    vals_pre.pop("s_to_yy_new_sequence_length_y1")
    vals_pre.pop("f_to_y_node_add_y1")
    vals_pre.pop("f_to_y_node_decrease_y1")
    vals_pre.pop("f_to_y_node_change_y1")
    vals_pre.pop("f_to_y_edge_change_y1")
    vals_pre.pop("f_to_yy_node_add_y1")
    vals_pre.pop("f_to_yy_node_decrease_y1")
    vals_pre.pop("f_to_yy_node_change_y1")
    vals_pre.pop("f_to_yy_edge_change_y1")
    vals_pre.pop("s_to_y_node_add_y1")
    vals_pre.pop("s_to_y_node_decrease_y1")
    vals_pre.pop("s_to_y_node_change_y1")
    vals_pre.pop("s_to_y_edge_change_y1")
    vals_pre.pop("s_to_yy_node_add_y1")
    vals_pre.pop("s_to_yy_node_decrease_y1")
    vals_pre.pop("s_to_yy_node_change_y1")
    vals_pre.pop("s_to_yy_edge_change_y1")

    vals_pre.pop("yy2_truth_sequence_len")
    vals_pre.pop("f_max_sequence_length_y2")
    vals_pre.pop("s_max_sequence_length_y2")
    vals_pre.pop("f_to_y_new_sequence_length_y2")
    vals_pre.pop("f_to_yy_new_sequence_length_y2")
    vals_pre.pop("s_to_y_new_sequence_length_y2")
    vals_pre.pop("s_to_yy_new_sequence_length_y2")
    vals_pre.pop("f_to_y_node_add_y2")
    vals_pre.pop("f_to_y_node_decrease_y2")
    vals_pre.pop("f_to_y_node_change_y2")
    vals_pre.pop("f_to_y_edge_change_y2")
    vals_pre.pop("f_to_yy_node_add_y2")
    vals_pre.pop("f_to_yy_node_decrease_y2")
    vals_pre.pop("f_to_yy_node_change_y2")
    vals_pre.pop("f_to_yy_edge_change_y2")
    vals_pre.pop("s_to_y_node_add_y2")
    vals_pre.pop("s_to_y_node_decrease_y2")
    vals_pre.pop("s_to_y_node_change_y2")
    vals_pre.pop("s_to_y_edge_change_y2")
    vals_pre.pop("s_to_yy_node_add_y2")
    vals_pre.pop("s_to_yy_node_decrease_y2")
    vals_pre.pop("s_to_yy_node_change_y2")
    vals_pre.pop("s_to_yy_edge_change_y2")

    vals_pre.pop("yy3_truth_sequence_len")
    vals_pre.pop("f_max_sequence_length_y3")
    vals_pre.pop("s_max_sequence_length_y3")
    vals_pre.pop("f_to_y_new_sequence_length_y3")
    vals_pre.pop("f_to_yy_new_sequence_length_y3")
    vals_pre.pop("s_to_y_new_sequence_length_y3")
    vals_pre.pop("s_to_yy_new_sequence_length_y3")
    vals_pre.pop("f_to_y_node_add_y3")
    vals_pre.pop("f_to_y_node_decrease_y3")
    vals_pre.pop("f_to_y_node_change_y3")
    vals_pre.pop("f_to_y_edge_change_y3")
    vals_pre.pop("f_to_yy_node_add_y3")
    vals_pre.pop("f_to_yy_node_decrease_y3")
    vals_pre.pop("f_to_yy_node_change_y3")
    vals_pre.pop("f_to_yy_edge_change_y3")
    vals_pre.pop("s_to_y_node_add_y3")
    vals_pre.pop("s_to_y_node_decrease_y3")
    vals_pre.pop("s_to_y_node_change_y3")
    vals_pre.pop("s_to_y_edge_change_y3")
    vals_pre.pop("s_to_yy_node_add_y3")
    vals_pre.pop("s_to_yy_node_decrease_y3")
    vals_pre.pop("s_to_yy_node_change_y3")
    vals_pre.pop("s_to_yy_edge_change_y3")


    vals_pre.pop("f_new_graph_01_y1")
    vals_pre.pop("f_new_graph_01_y2")
    vals_pre.pop("f_new_graph_01_y3")
    vals_pre.pop("f_new_graph_logits_y1")
    vals_pre.pop("f_new_graph_logits_y2")
    vals_pre.pop("f_new_graph_logits_y3")
    vals_pre.pop("f_adjust_graph_y1")
    vals_pre.pop("f_adjust_graph_y2")
    vals_pre.pop("f_adjust_graph_y3")

    vals_pre.pop("s_new_graph_01_y1")
    vals_pre.pop("s_new_graph_01_y2")
    vals_pre.pop("s_new_graph_01_y3")
    vals_pre.pop("s_new_graph_logits_y1")
    vals_pre.pop("s_new_graph_logits_y2")
    vals_pre.pop("s_new_graph_logits_y3")
    vals_pre.pop("s_adjust_graph_y1")
    vals_pre.pop("s_adjust_graph_y2")
    vals_pre.pop("s_adjust_graph_y3")

    vals_pre.pop("adjs_y1_undirt")
    vals_pre.pop("adjs_y2_undirt")
    vals_pre.pop("adjs_y3_undirt")
    vals_pre.pop("adjs_yy1_undirt")
    vals_pre.pop("adjs_yy2_undirt")
    vals_pre.pop("adjs_yy3_undirt")

    return vals_pre

def pop2(vals_pre, vals_post):

    vals_pre.pop("yy1_truth_sequence_len")
    vals_pre.pop("f_max_sequence_length_y1")
    vals_post.pop("s_max_sequence_length_y1")
    vals_pre.pop("f_to_y_new_sequence_length_y1")
    vals_pre.pop("f_to_yy_new_sequence_length_y1")
    vals_post.pop("s_to_y_new_sequence_length_y1")
    vals_post.pop("s_to_yy_new_sequence_length_y1")
    vals_pre.pop("f_to_y_node_add_y1")
    vals_pre.pop("f_to_y_node_decrease_y1")
    vals_pre.pop("f_to_y_node_change_y1")
    vals_pre.pop("f_to_y_edge_change_y1")
    vals_pre.pop("f_to_yy_node_add_y1")
    vals_pre.pop("f_to_yy_node_decrease_y1")
    vals_pre.pop("f_to_yy_node_change_y1")
    vals_pre.pop("f_to_yy_edge_change_y1")
    vals_post.pop("s_to_y_node_add_y1")
    vals_post.pop("s_to_y_node_decrease_y1")
    vals_post.pop("s_to_y_node_change_y1")
    vals_post.pop("s_to_y_edge_change_y1")
    vals_post.pop("s_to_yy_node_add_y1")
    vals_post.pop("s_to_yy_node_decrease_y1")
    vals_post.pop("s_to_yy_node_change_y1")
    vals_post.pop("s_to_yy_edge_change_y1")

    vals_pre.pop("yy2_truth_sequence_len")
    vals_pre.pop("f_max_sequence_length_y2")
    vals_post.pop("s_max_sequence_length_y2")
    vals_pre.pop("f_to_y_new_sequence_length_y2")
    vals_pre.pop("f_to_yy_new_sequence_length_y2")
    vals_post.pop("s_to_y_new_sequence_length_y2")
    vals_post.pop("s_to_yy_new_sequence_length_y2")
    vals_pre.pop("f_to_y_node_add_y2")
    vals_pre.pop("f_to_y_node_decrease_y2")
    vals_pre.pop("f_to_y_node_change_y2")
    vals_pre.pop("f_to_y_edge_change_y2")
    vals_pre.pop("f_to_yy_node_add_y2")
    vals_pre.pop("f_to_yy_node_decrease_y2")
    vals_pre.pop("f_to_yy_node_change_y2")
    vals_pre.pop("f_to_yy_edge_change_y2")
    vals_post.pop("s_to_y_node_add_y2")
    vals_post.pop("s_to_y_node_decrease_y2")
    vals_post.pop("s_to_y_node_change_y2")
    vals_post.pop("s_to_y_edge_change_y2")
    vals_post.pop("s_to_yy_node_add_y2")
    vals_post.pop("s_to_yy_node_decrease_y2")
    vals_post.pop("s_to_yy_node_change_y2")
    vals_post.pop("s_to_yy_edge_change_y2")

    vals_pre.pop("yy3_truth_sequence_len")
    vals_pre.pop("f_max_sequence_length_y3")
    vals_post.pop("s_max_sequence_length_y3")
    vals_pre.pop("f_to_y_new_sequence_length_y3")
    vals_pre.pop("f_to_yy_new_sequence_length_y3")
    vals_post.pop("s_to_y_new_sequence_length_y3")
    vals_post.pop("s_to_yy_new_sequence_length_y3")
    vals_pre.pop("f_to_y_node_add_y3")
    vals_pre.pop("f_to_y_node_decrease_y3")
    vals_pre.pop("f_to_y_node_change_y3")
    vals_pre.pop("f_to_y_edge_change_y3")
    vals_pre.pop("f_to_yy_node_add_y3")
    vals_pre.pop("f_to_yy_node_decrease_y3")
    vals_pre.pop("f_to_yy_node_change_y3")
    vals_pre.pop("f_to_yy_edge_change_y3")
    vals_post.pop("s_to_y_node_add_y3")
    vals_post.pop("s_to_y_node_decrease_y3")
    vals_post.pop("s_to_y_node_change_y3")
    vals_post.pop("s_to_y_edge_change_y3")
    vals_post.pop("s_to_yy_node_add_y3")
    vals_post.pop("s_to_yy_node_decrease_y3")
    vals_post.pop("s_to_yy_node_change_y3")
    vals_post.pop("s_to_yy_edge_change_y3")


    vals_pre.pop("f_new_graph_01_y1")
    vals_pre.pop("f_new_graph_01_y2")
    vals_pre.pop("f_new_graph_01_y3")
    vals_pre.pop("f_new_graph_logits_y1")
    vals_pre.pop("f_new_graph_logits_y2")
    vals_pre.pop("f_new_graph_logits_y3")
    vals_pre.pop("f_adjust_graph_y1")
    vals_pre.pop("f_adjust_graph_y2")
    vals_pre.pop("f_adjust_graph_y3")

    vals_post.pop("s_new_graph_01_y1")
    vals_post.pop("s_new_graph_01_y2")
    vals_post.pop("s_new_graph_01_y3")
    vals_post.pop("s_new_graph_logits_y1")
    vals_post.pop("s_new_graph_logits_y2")
    vals_post.pop("s_new_graph_logits_y3")
    vals_post.pop("s_adjust_graph_y1")
    vals_post.pop("s_adjust_graph_y2")
    vals_post.pop("s_adjust_graph_y3")

    vals_pre.pop("adjs_y1_undirt")
    vals_pre.pop("adjs_y2_undirt")
    vals_pre.pop("adjs_y3_undirt")
    vals_pre.pop("adjs_yy1_undirt")
    vals_pre.pop("adjs_yy2_undirt")
    vals_pre.pop("adjs_yy3_undirt")

    return vals_pre, vals_post


import numpy as np
def decode(proc, ids):
    """
    ids: batch,maxlen
    filter BOS and EOS and PAD, and join the char together
    """
    all_text = []
    batch_size = np.shape(ids)[0]
    for i in range(batch_size):
        filter_ids = [id_ for id_ in ids[i] if id_ != 50256]
        texts = proc.decode(filter_ids)
        all_text.append(texts)
    return np.array(all_text)


def pop3(vals_pre):
    
    vals_pre.pop("yy1_truth_sequence_len")
    vals_pre.pop("f_max_sequence_length_y1")
    vals_pre.pop("f_to_y_new_sequence_length_y1")
    vals_pre.pop("f_to_yy_new_sequence_length_y1")
    vals_pre.pop("f_to_y_node_add_y1")
    vals_pre.pop("f_to_y_node_decrease_y1")
    vals_pre.pop("f_to_y_node_change_y1")
    vals_pre.pop("f_to_y_edge_change_y1")
    vals_pre.pop("f_to_yy_node_add_y1")
    vals_pre.pop("f_to_yy_node_decrease_y1")
    vals_pre.pop("f_to_yy_node_change_y1")
    vals_pre.pop("f_to_yy_edge_change_y1")

    vals_pre.pop("yy2_truth_sequence_len")
    vals_pre.pop("f_max_sequence_length_y2")
    vals_pre.pop("f_to_y_new_sequence_length_y2")
    vals_pre.pop("f_to_yy_new_sequence_length_y2")
  
    vals_pre.pop("f_to_y_node_add_y2")
    vals_pre.pop("f_to_y_node_decrease_y2")
    vals_pre.pop("f_to_y_node_change_y2")
    vals_pre.pop("f_to_y_edge_change_y2")
    vals_pre.pop("f_to_yy_node_add_y2")
    vals_pre.pop("f_to_yy_node_decrease_y2")
    vals_pre.pop("f_to_yy_node_change_y2")
    vals_pre.pop("f_to_yy_edge_change_y2")

    vals_pre.pop("yy3_truth_sequence_len")
    vals_pre.pop("f_max_sequence_length_y3")
    vals_pre.pop("f_to_y_new_sequence_length_y3")
    vals_pre.pop("f_to_yy_new_sequence_length_y3")

    vals_pre.pop("f_to_y_node_add_y3")
    vals_pre.pop("f_to_y_node_decrease_y3")
    vals_pre.pop("f_to_y_node_change_y3")
    vals_pre.pop("f_to_y_edge_change_y3")
    vals_pre.pop("f_to_yy_node_add_y3")
    vals_pre.pop("f_to_yy_node_decrease_y3")
    vals_pre.pop("f_to_yy_node_change_y3")
    vals_pre.pop("f_to_yy_edge_change_y3")

    vals_pre.pop("f_new_graph_01_y1")
    vals_pre.pop("f_new_graph_01_y2")
    vals_pre.pop("f_new_graph_01_y3")
    vals_pre.pop("f_new_graph_logits_y1")
    vals_pre.pop("f_new_graph_logits_y2")
    vals_pre.pop("f_new_graph_logits_y3")
    vals_pre.pop("f_adjust_graph_y1")
    vals_pre.pop("f_adjust_graph_y2")
    vals_pre.pop("f_adjust_graph_y3")


    vals_pre.pop("adjs_y1_undirt")
    vals_pre.pop("adjs_y2_undirt")
    vals_pre.pop("adjs_y3_undirt")
    vals_pre.pop("adjs_yy1_undirt")
    vals_pre.pop("adjs_yy2_undirt")
    vals_pre.pop("adjs_yy3_undirt")

    return vals_pre


def pop4(vals_pre):
    
    vals_pre.pop("yy1_truth_sequence_len")
    vals_pre.pop("f_max_sequence_length_y1_ori")
    vals_pre.pop("f_max_sequence_length_y1_cft")
    vals_pre.pop("s_max_sequence_length_y1")
    vals_pre.pop("f_to_y_new_sequence_length_y1_ori")
    vals_pre.pop("f_to_yy_new_sequence_length_y1_cft")
    vals_pre.pop("s_to_y_new_sequence_length_y1")
    vals_pre.pop("s_to_yy_new_sequence_length_y1")
    vals_pre.pop("f_to_y_node_add_y1_ori")
    vals_pre.pop("f_to_y_node_decrease_y1_ori")
    vals_pre.pop("f_to_y_node_change_y1_ori")
    vals_pre.pop("f_to_y_edge_change_y1_ori")
    vals_pre.pop("f_to_yy_node_add_y1_cft")
    vals_pre.pop("f_to_yy_node_decrease_y1_cft")
    vals_pre.pop("f_to_yy_node_change_y1_cft")
    vals_pre.pop("f_to_yy_edge_change_y1_cft")
    vals_pre.pop("s_to_y_node_add_y1")
    vals_pre.pop("s_to_y_node_decrease_y1")
    vals_pre.pop("s_to_y_node_change_y1")
    vals_pre.pop("s_to_y_edge_change_y1")
    vals_pre.pop("s_to_yy_node_add_y1")
    vals_pre.pop("s_to_yy_node_decrease_y1")
    vals_pre.pop("s_to_yy_node_change_y1")
    vals_pre.pop("s_to_yy_edge_change_y1")

    vals_pre.pop("yy2_truth_sequence_len")
    vals_pre.pop("f_max_sequence_length_y2_ori")
    vals_pre.pop("f_max_sequence_length_y2_cft")
    vals_pre.pop("s_max_sequence_length_y2")
    vals_pre.pop("f_to_y_new_sequence_length_y2_ori")
    vals_pre.pop("f_to_yy_new_sequence_length_y2_cft")
    vals_pre.pop("s_to_y_new_sequence_length_y2")
    vals_pre.pop("s_to_yy_new_sequence_length_y2")
    vals_pre.pop("f_to_y_node_add_y2_ori")
    vals_pre.pop("f_to_y_node_decrease_y2_ori")
    vals_pre.pop("f_to_y_node_change_y2_ori")
    vals_pre.pop("f_to_y_edge_change_y2_ori")
    vals_pre.pop("f_to_yy_node_add_y2_cft")
    vals_pre.pop("f_to_yy_node_decrease_y2_cft")
    vals_pre.pop("f_to_yy_node_change_y2_cft")
    vals_pre.pop("f_to_yy_edge_change_y2_cft")
    vals_pre.pop("s_to_y_node_add_y2")
    vals_pre.pop("s_to_y_node_decrease_y2")
    vals_pre.pop("s_to_y_node_change_y2")
    vals_pre.pop("s_to_y_edge_change_y2")
    vals_pre.pop("s_to_yy_node_add_y2")
    vals_pre.pop("s_to_yy_node_decrease_y2")
    vals_pre.pop("s_to_yy_node_change_y2")
    vals_pre.pop("s_to_yy_edge_change_y2")

    vals_pre.pop("yy3_truth_sequence_len")
    vals_pre.pop("f_max_sequence_length_y3_ori")
    vals_pre.pop("f_max_sequence_length_y3_cft")
    vals_pre.pop("s_max_sequence_length_y3")
    vals_pre.pop("f_to_y_new_sequence_length_y3_ori")
    vals_pre.pop("f_to_yy_new_sequence_length_y3_cft")
    vals_pre.pop("s_to_y_new_sequence_length_y3")
    vals_pre.pop("s_to_yy_new_sequence_length_y3")
    vals_pre.pop("f_to_y_node_add_y3_ori")
    vals_pre.pop("f_to_y_node_decrease_y3_ori")
    vals_pre.pop("f_to_y_node_change_y3_ori")
    vals_pre.pop("f_to_y_edge_change_y3_ori")
    vals_pre.pop("f_to_yy_node_add_y3_cft")
    vals_pre.pop("f_to_yy_node_decrease_y3_cft")
    vals_pre.pop("f_to_yy_node_change_y3_cft")
    vals_pre.pop("f_to_yy_edge_change_y3_cft")
    vals_pre.pop("s_to_y_node_add_y3")
    vals_pre.pop("s_to_y_node_decrease_y3")
    vals_pre.pop("s_to_y_node_change_y3")
    vals_pre.pop("s_to_y_edge_change_y3")
    vals_pre.pop("s_to_yy_node_add_y3")
    vals_pre.pop("s_to_yy_node_decrease_y3")
    vals_pre.pop("s_to_yy_node_change_y3")
    vals_pre.pop("s_to_yy_edge_change_y3")


    vals_pre.pop("f_new_graph_01_y1_ori")
    vals_pre.pop("f_new_graph_01_y2_ori")
    vals_pre.pop("f_new_graph_01_y3_ori")
    vals_pre.pop("f_new_graph_01_y1_cft")
    vals_pre.pop("f_new_graph_01_y2_cft")
    vals_pre.pop("f_new_graph_01_y3_cft")

    vals_pre.pop("f_new_graph_logits_y1_ori")
    vals_pre.pop("f_new_graph_logits_y2_ori")
    vals_pre.pop("f_new_graph_logits_y3_ori")
    vals_pre.pop("f_new_graph_logits_y1_cft")
    vals_pre.pop("f_new_graph_logits_y2_cft")
    vals_pre.pop("f_new_graph_logits_y3_cft")
    vals_pre.pop("f_adjust_graph_y1_ori")
    vals_pre.pop("f_adjust_graph_y2_ori")
    vals_pre.pop("f_adjust_graph_y3_ori")
    vals_pre.pop("f_adjust_graph_y1_cft")
    vals_pre.pop("f_adjust_graph_y2_cft")
    vals_pre.pop("f_adjust_graph_y3_cft")

    vals_pre.pop("s_new_graph_01_y1")
    vals_pre.pop("s_new_graph_01_y2")
    vals_pre.pop("s_new_graph_01_y3")
    vals_pre.pop("s_new_graph_logits_y1")
    vals_pre.pop("s_new_graph_logits_y2")
    vals_pre.pop("s_new_graph_logits_y3")
    vals_pre.pop("s_adjust_graph_y1")
    vals_pre.pop("s_adjust_graph_y2")
    vals_pre.pop("s_adjust_graph_y3")

    vals_pre.pop("adjs_y1_undirt")
    vals_pre.pop("adjs_y2_undirt")
    vals_pre.pop("adjs_y3_undirt")
    vals_pre.pop("adjs_yy1_undirt")
    vals_pre.pop("adjs_yy2_undirt")
    vals_pre.pop("adjs_yy3_undirt")

    return vals_pre

def pop_noF(vals_pre):
    
    vals_pre.pop("yy1_truth_sequence_len")
    vals_pre.pop("s_max_sequence_length_y1")
    vals_pre.pop("s_to_y_new_sequence_length_y1")
    vals_pre.pop("s_to_yy_new_sequence_length_y1")
    vals_pre.pop("s_to_y_node_add_y1")
    vals_pre.pop("s_to_y_node_decrease_y1")
    vals_pre.pop("s_to_y_node_change_y1")
    vals_pre.pop("s_to_y_edge_change_y1")
    vals_pre.pop("s_to_yy_node_add_y1")
    vals_pre.pop("s_to_yy_node_decrease_y1")
    vals_pre.pop("s_to_yy_node_change_y1")
    vals_pre.pop("s_to_yy_edge_change_y1")

    vals_pre.pop("yy2_truth_sequence_len")
    vals_pre.pop("s_max_sequence_length_y2")
    vals_pre.pop("s_to_y_new_sequence_length_y2")
    vals_pre.pop("s_to_yy_new_sequence_length_y2")
    vals_pre.pop("s_to_y_node_add_y2")
    vals_pre.pop("s_to_y_node_decrease_y2")
    vals_pre.pop("s_to_y_node_change_y2")
    vals_pre.pop("s_to_y_edge_change_y2")
    vals_pre.pop("s_to_yy_node_add_y2")
    vals_pre.pop("s_to_yy_node_decrease_y2")
    vals_pre.pop("s_to_yy_node_change_y2")
    vals_pre.pop("s_to_yy_edge_change_y2")

    vals_pre.pop("yy3_truth_sequence_len")
    vals_pre.pop("s_max_sequence_length_y3")
    vals_pre.pop("s_to_y_new_sequence_length_y3")
    vals_pre.pop("s_to_yy_new_sequence_length_y3")
    vals_pre.pop("s_to_y_node_add_y3")
    vals_pre.pop("s_to_y_node_decrease_y3")
    vals_pre.pop("s_to_y_node_change_y3")
    vals_pre.pop("s_to_y_edge_change_y3")
    vals_pre.pop("s_to_yy_node_add_y3")
    vals_pre.pop("s_to_yy_node_decrease_y3")
    vals_pre.pop("s_to_yy_node_change_y3")
    vals_pre.pop("s_to_yy_edge_change_y3")

   

    vals_pre.pop("s_new_graph_01_y1")
    vals_pre.pop("s_new_graph_01_y2")
    vals_pre.pop("s_new_graph_01_y3")
    vals_pre.pop("s_new_graph_logits_y1")
    vals_pre.pop("s_new_graph_logits_y2")
    vals_pre.pop("s_new_graph_logits_y3")
    vals_pre.pop("s_adjust_graph_y1")
    vals_pre.pop("s_adjust_graph_y2")
    vals_pre.pop("s_adjust_graph_y3")

    vals_pre.pop("adjs_y1_undirt")
    vals_pre.pop("adjs_y2_undirt")
    vals_pre.pop("adjs_y3_undirt")
    vals_pre.pop("adjs_yy1_undirt")
    vals_pre.pop("adjs_yy2_undirt")
    vals_pre.pop("adjs_yy3_undirt")

    return vals_pre



def map_ids_to_strs(samples_ids, tokenizer, config):
    sentence_ids = []

    for sample_id_sen in samples_ids:
        tokens_ids = []
        for word_id in sample_id_sen:
            if word_id == config.bos_token_id or word_id == config.eos_token_id or word_id == config.pad_id:
                continue    
            tokens_ids.append(word_id)
        sentence_ids.append(tokens_ids)
    
    input_texts = []
    for sentence in sentence_ids:#[]
        input_text = tokenizer.convert_ids_to_tokens(sentence)
        input_texts.append(' '.join(input_text))
    #assert np.shape(input_texts)[0] == np.array([config.batch_size])
    return input_texts



def map_ids_to_strs_xlnet(samples_ids, sp, config):
    sentence_ids = []

    for sample_id_sen in samples_ids:
        tokens_ids = []
        for word_id in sample_id_sen:
            if word_id == config.bos_token_id or word_id == config.eos_token_id or word_id == config.pad_id:
                continue    
            tokens_ids.append(int(word_id))
        sentence_ids.append(tokens_ids)
    
    input_texts = []
    for sentence in sentence_ids:#[]
        input_text = sp.DecodeIds(sentence)
        input_texts.append(input_text)
    return input_texts
