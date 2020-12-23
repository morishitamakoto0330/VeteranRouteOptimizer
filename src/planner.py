from environment import State

class Planner():

    def __init__(self, env):
        self.env = env
        self.log = []

    def initialize(self):
        self.log = []

    def plan(self, gamma=0.9, threshold=0.0001):
        raise Exception('Planner have to implements plan method.')

    def transitions_at(self, state, action):
        transition_probs = self.env.transit_func(state, action)
        for next_state in transition_probs:
            prob = transition_probs[next_state]
            reward = self.env.reward_func(state, next_state)
            yield prob, next_state, reward

    def dict_to_points(self, state_reward_dict):
        state = None
        # search max Value state
        for s in state_reward_dict:
            if (len(self.env.points) - 1) == len(s.visited_points):
                if state is None:
                    state = s
                else:
                    if state_reward_dict[state] < state_reward_dict[s]:
                        state = s

        if state is None:
            return []
        else:
            last_point = list(set(self.env.points) - set(state.visited_points))
            return state.visited_points + last_point



class ValueIterationPlanner(Planner):
    
    def __init__(self, env):
        super().__init__(env)

    def enum_state(self, env):
        V = {}
        states = []

        # initail state
        state = env.agent_state
        V[state] = 0
        states.append(state)

        while states:
            state = states.pop()
            for s in env.states(state):
                V[s] = 0
                states.append(s)

        return V


    def plan(self, gamma=0.9, threshold=0.0001):
        print('gamma={0}, threshold={1}'.format(gamma, threshold))
        self.initialize()

        V = self.enum_state(self.env)
        print('number of all states={0}'.format(len(V)))

        while True:
            delta = 0
            #self.log.append(self.dict_to_points(V))
            for s in V:
                if not self.env.can_action_at(s):
                    continue
                expected_rewards = []

                for a in self.env.actions(s):
                    r = 0
                    for prob, next_state, reward in self.transitions_at(s, a):
                        r += prob * (reward + gamma * V[next_state])
                    expected_rewards.append(r)

                max_reward = max(expected_rewards)
                delta = max(delta, abs(max_reward - V[s]))
                V[s] = max_reward

            if delta < threshold:
                break

        V_points = self.dict_to_points(V)
        return V_points






