from policy_gradient import learn_reinforce, policy
from q_learning import learn_q_table, greedy_policy
from visualize import visualize


# visualize(learn_reinforce, policy)
visualize(learn_q_table, greedy_policy)