"""Text style transfer Under Linguistic Constraints
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# pylint: disable=invalid-name, too-many-locals

import tensorflow as tf

def compute_countFact_priority_scoreV2(priority_score, new_graph_01, max_sequence_length):
    """Use for node priority score computation by graph_y_ctx_embd
    Use:
        priority_score: [batch, maxlen-1] max_sequence_length: [batch] 
        new_graph_01: [batch, maxlen-1, maxlen-1]
    Create:
        sorted_sequence [batch, maxlen-1]
    """
    sorted_sequence = tf.map_fn( 
            fn=lambda inp: BFS(inp[0], inp[1], inp[2]),
            elems=(priority_score, new_graph_01, max_sequence_length-1),                                       # [batch, maxlen-1] [batch, maxlen-1, maxlen-1] [batch] 
            dtype=tf.int32
    )
    return sorted_sequence                                                                                     # [batch, maxlen-1]

def compute_countFact_priority_score(new_graph_01, max_sequence_length, graph_y_ctx_embd, hidden_state):
    """Use for node priority score computation by graph_y_ctx_embd
    Use:
        graph_y_ctx_embd: [batch, maxlen-1, dim] max_sequence_length: [batch]
        hidden_state [batch, 1, dim] new_graph_01: [batch, maxlen-1, maxlen-1]
    Create:
        sorted_sequence [batch, maxlen-1]
    """
    priority_score = tf.sigmoid(tf.squeeze(tf.matmul(graph_y_ctx_embd, tf.transpose(hidden_state,[0,2,1])), 2)) # [batch, maxlen-1] ###check 可能有更好的算atten score的方法 mlp？

    sorted_sequence = tf.map_fn( 
            fn=lambda inp: BFS(inp[0], inp[1], inp[2]),
            elems=(priority_score, new_graph_01, max_sequence_length-1),                                       # [batch, maxlen-1] [batch, maxlen-1, maxlen-1] [batch] 
            dtype=tf.int32
    )
    return sorted_sequence                                                                                     # [batch, maxlen-1]

def BFS(scores, adj, max_seqlen):
    """Use for attain sorted_sequence for node in a sentence
    [maxlen-1] [maxlen-1, maxlen-1] int
    """
    Q = tf.cast(tf.constant([]), tf.int32)                                                                  # scores本身是不能动的，不能增加或者删除，不然下一个取出来的node_id就不对了
    sorted_sequence = tf.cast(tf.constant([]), tf.int32)                                                    # empty list 用于存储id 排序是为了聚合，所以pad部分没必要放在这里面，没人和他聚合 
    padding_num = tf.shape(scores)[0] - max_seqlen + 1                                                      # 求出pading部分总和 +1代表EOS得部分也要当作pad，因为他没有领边啊
    unsorted_id_num = tf.shape(scores)[0] - padding_num                                                     # 求出未排序的总数
    top_k = tf.cast(tf.constant(0), tf.int32)
    node_id = tf.cast(tf.constant(-1), tf.int32)
    _, _, _, _, sorted_sequence, _, _, _ = tf.while_loop(mainCond, mainBody, 
                                                        loop_vars = [scores, max_seqlen, unsorted_id_num, top_k, sorted_sequence, Q, node_id, adj],
                                                        shape_invariants=[scores.get_shape(), max_seqlen.get_shape(),unsorted_id_num.get_shape(),
                                                                        top_k.get_shape(),tf.TensorShape([None]),tf.TensorShape([None]),
                                                                        node_id.get_shape(),adj.get_shape()]
                                                        )
    sorted_sequence = tf.pad(sorted_sequence+1, [[0,padding_num]])                                          # 把sorted_sequence pad成一样的maxlen
    return sorted_sequence                                                                                  # 返回排序好的sequence [2,1,16.....12,0,0,0] index+1

def mainCond(scores, max_seqlen, unsorted_id_num, top_k, sorted_sequence, Q, node_id, adj):
    return tf.greater(unsorted_id_num, 0)
def mainBody(scores, max_seqlen, unsorted_id_num, top_k, sorted_sequence, Q, node_id, adj):
    top_k, sorted_sequence, node_id, _, _ = tf.while_loop(
                                                    topk_cond, topk_body, 
                                                    [top_k, sorted_sequence, node_id, max_seqlen, scores])  # 在这里取出top后，先判断是否在pad里面，在pad里面的我们不要，已经在sorted_sequence的我们也不要
    Q = tf.cond(tf.greater_equal(node_id, max_seqlen-1), lambda:Q, lambda:tf.concat([Q, [node_id]],0))
    sorted_sequence, adj, Q, unsorted_id_num, node_id, max_seqlen = tf.while_loop(sortCond, sortBody, 
                                                            loop_vars = [sorted_sequence, adj, Q, unsorted_id_num, node_id, max_seqlen],
                                                            shape_invariants=[tf.TensorShape([None]),adj.get_shape(),tf.TensorShape([None]),
                                                            unsorted_id_num.get_shape(),node_id.get_shape(),max_seqlen.get_shape()])
    return scores, max_seqlen, unsorted_id_num, top_k, sorted_sequence, Q, node_id, adj

def topk_cond(top_k, sorted_sequence, node_id, max_seqlen, scores):
    return tf.logical_or(
                tf.cond(tf.equal(top_k,0), lambda: True, lambda: tf.greater(tf.reduce_sum(tf.cast(tf.equal(sorted_sequence, node_id), dtype=tf.int32)),0)),
                tf.greater_equal(node_id, max_seqlen-1))                                                    # 已经在sorted_squence | 不小心取到了pad的部分的值
def topk_body(top_k, sorted_sequence, node_id, max_seqlen, scores):
    top_k+=1                                                                                                # 取出来的node_id还不能在padding里面
    node_id = tf.nn.top_k(scores, k=top_k, sorted=True)[1][-1]                                              # value,index
    return top_k, sorted_sequence, node_id, max_seqlen, scores

def sortCond(sorted_sequence, adj, Q, unsorted_id_num, node_id, max_seqlen):
    return tf.greater(tf.shape(Q)[0],0)
def sortBody(sorted_sequence, adj, Q, unsorted_id_num, node_id, max_seqlen):
    node_id = Q[0]                                                                                          # dequeue operator # 把第一个id pop出来
    Q = Q[1:]

    sorted_sequence = tf.concat([sorted_sequence, [node_id]], 0)                                            # Q dequeue的时候, 再放入sorted_sequence
    unsorted_id_num -= 1

    node_child = tf.squeeze(tf.where(tf.equal(adj[node_id,:],1)), 1)                                        # 把当前点的所有子节点找出来 #出度 [None] ###check?

    Q, _, _ = tf.while_loop(pushChild_cond,
                            pushChild_body,
                            loop_vars = [Q, node_child, sorted_sequence],
                            shape_invariants=[tf.TensorShape([None]), node_child.get_shape(), tf.TensorShape([None])])
    return sorted_sequence, adj, Q, unsorted_id_num, node_id, max_seqlen

def pushChild_cond(Q, node_child, sorted_sequence):
    return tf.greater(tf.shape(node_child)[0], 0)                                                           # node_child中还有子节点 #这里没必要对子节点进行排序
def pushChild_body(Q, node_child, sorted_sequence):
    node_child_id = node_child[0]                                                                           # dequeue操作
    node_child = node_child[1:]                                                                             

    Q = tf.cond(tf.logical_or(tf.greater(tf.reduce_sum(tf.cast(tf.equal(sorted_sequence, tf.cast(node_child_id, tf.int32)), tf.int32)),0),
            tf.greater(tf.reduce_sum(tf.cast(tf.equal(Q, tf.cast(node_child_id, tf.int32)), tf.int32)),0)), lambda:Q, lambda:tf.concat([Q, [node_child_id]],0))
                                                                                                            # 把child依次存入Q队列中 ###check 
                                                                                                            # Q子元素是否也按照分数排序？待,这里要把已经在sorted_sequence的去掉
    return Q, node_child, sorted_sequence


def calculate_node_add_decrease(refer_truth_len, refer_adjs, new_graph_01):
    """calculate seqence_len change for visualization
    Use: refer_adjs: [batch, maxlen-1, maxlen-1]
         refer_truth_len: [batch], new_graph_01:[batch, maxlen-1, maxlen-1]
    Create: node_add_sum, node_decrease_sum, node_change, new_sequence_length, edge_change: int
    """
    with tf.name_scope('calculate_node_add_decrease') as scope:
        # for story Rewrite
        # for new_graph_01 and self.y_truth_len
        ### mask diag
        mask_diag = tf.tile(tf.expand_dims(1 - tf.cast(tf.eye(tf.shape(refer_adjs)[1]-1), dtype=tf.int32), 0), [tf.shape(refer_adjs)[0], 1, 1]) #[b,maxlen-1,maxlen-1]
        new_graph_01_nomask = new_graph_01
        new_graph_01 = tf.multiply(new_graph_01_nomask, mask_diag) #[b,maxlen-1,maxlen-1]

        ## for node add
        ### sequence_len mask
        mask_seqlen = tf.sequence_mask(
            refer_truth_len-1, tf.shape(refer_adjs)[1], dtype=tf.int32) #[batch, seq_len]
        mask_seqlen_2d = 1-mask_seqlen
        mask_seqlen_expand_1 = tf.expand_dims(mask_seqlen, 1) #[batch, 1, seq_len]
        mask_seqlen_expand_2 = tf.transpose(mask_seqlen_expand_1, [0,2,1]) #[batch, seq_len, 1]
        mask_seqlen = 1 - tf.multiply(mask_seqlen_expand_2, mask_seqlen_expand_1) #[batch, seq_len, seq_len]
        adj_for_add = tf.multiply(new_graph_01, mask_seqlen[:,1:,1:])

        #self.mask_seqlen_2d = mask_seqlen_2d
        row_sum = tf.multiply(tf.minimum(tf.reduce_sum(adj_for_add, 2), 1), mask_seqlen_2d[:,1:]) #[batch, maxlen-1]
        col_sum = tf.multiply(tf.minimum(tf.reduce_sum(adj_for_add, 1), 1), mask_seqlen_2d[:,1:]) #[batch, maxlen-1]
        node_add_sum = tf.reduce_sum(tf.minimum((row_sum + col_sum),1),1) #[batch]
        new_sequence_length = refer_truth_len + node_add_sum # add should be compute in padding part, decrease should be compute in sequence part
        
        ## for node decrease
        ### padding mask
        mask_padding = tf.sequence_mask(
            refer_truth_len-1, tf.shape(refer_adjs)[1], dtype=tf.int32) # let EOS is also 0
        mask_padding_expand_1 = tf.expand_dims(mask_padding, 1) #[batch, 1, seq_len]
        mask_padding_expand_2 = tf.transpose(mask_padding_expand_1, [0,2,1]) #[batch, seq_len, 1]
        mask_padding = tf.multiply(mask_padding_expand_2, mask_padding_expand_1) #[batch, seq_len, seq_len]
        adj_for_decrease = tf.multiply(new_graph_01, mask_padding[:,1:,1:])
        row_decrease = tf.minimum(tf.reduce_sum(adj_for_decrease, 2), 1) #[batch, maxlen-1]
        col_decrease = tf.minimum(tf.reduce_sum(adj_for_decrease, 1), 1) #[batch, maxlen-1]
        node_decrease_sum =  refer_truth_len - (tf.reduce_sum((tf.minimum((row_decrease + col_decrease),1)),1) +1 +1) #[batch]
        
        none_adj_mask = 1 - tf.cast(tf.equal(tf.reduce_sum(tf.reduce_sum(adj_for_decrease,2),1), 0), tf.int32) # is none,then 0
        node_decrease_sum = tf.multiply(none_adj_mask, node_decrease_sum)
        
        new_sequence_length = new_sequence_length - node_decrease_sum # contain BOS
        node_change = node_add_sum - node_decrease_sum

        # statistic edge change
        edge_change_mat = (refer_adjs[:,1:,1:])*(-1) + new_graph_01_nomask #[batch, maxlen-1, maxlen-1]
        edge_change = tf.reduce_sum(tf.reduce_sum(edge_change_mat,2),1)

        return node_add_sum, node_decrease_sum, node_change, new_sequence_length, edge_change

def cal_max_idx(inp):
    """calculate seqence_len change for rephraser and adj update
    inp: [maxlen,maxlen] int
    """
    inp = tf.not_equal(inp, 0)
    idxs = tf.where(inp)                                          # [None, 2]
    row_max = tf.reduce_max(idxs[:,0])
    col_max = tf.reduce_max(idxs[:,1])
    max_truth_len = tf.to_int32(tf.maximum(row_max, col_max)) + 1 # index+1
    return max_truth_len

def collect_visualize_var(new_graph_01, max_sequence_length, new_graph_logits, adjust_graph, y_truth_len, adjs_y_undirt, yy_truth_len, adjs_yy_undirt):

    tf.add_to_collection('max_sequence_length_sec', max_sequence_length)

    tf.add_to_collection('new_graph_01_sec', new_graph_01)
    tf.add_to_collection('new_graph_logits_sec', new_graph_logits)
    tf.add_to_collection('adjust_graph_sec', adjust_graph)

    to_y_node_add, to_y_node_decrease, to_y_node_change, to_y_new_sequence_length, to_y_edge_change = calculate_node_add_decrease(y_truth_len, adjs_y_undirt, new_graph_01)
    to_yy_node_add, to_yy_node_decrease, to_yy_node_change, to_yy_new_sequence_length, to_yy_edge_change = calculate_node_add_decrease(yy_truth_len, adjs_yy_undirt, new_graph_01)
    
    tf.add_to_collection('to_y_node_add_sec', to_y_node_add)
    tf.add_to_collection('to_y_node_decrease_sec', to_y_node_decrease)
    tf.add_to_collection('to_y_node_change_sec', to_y_node_change)
    tf.add_to_collection('to_y_new_sequence_length_sec', to_y_new_sequence_length)
    tf.add_to_collection('to_y_edge_change_sec', to_y_edge_change)
    tf.add_to_collection('to_yy_node_add_sec', to_yy_node_add)
    tf.add_to_collection('to_yy_node_decrease_sec', to_yy_node_decrease)
    tf.add_to_collection('to_yy_node_change_sec', to_yy_node_change)
    tf.add_to_collection('to_yy_new_sequence_length_sec', to_yy_new_sequence_length)
    tf.add_to_collection('to_yy_edge_change_sec', to_yy_edge_change)