class Planner():

    def __init__(self, env):
        self.env = env
        self.log = []

    def initialize(self):
        self.log = []

    #def plan(self, gamma=0.9, threshold=0.0001):
    def plan(self, gamma=1.0, threshold=0.0001):
        raise Exception('Planner have to implements plan method.')

    def transitions_at(self, state, action):
        transition_probs = self.env.transit_func(state, action)
        for next_state in transition_probs:
            prob = transition_probs[next_state]
            reward = self.env.reward_func(state, next_state)
            yield prob, next_state, reward

    def dict_to_points(self, state_reward_dict):
        points = []
        for i in range(len(self.env.points)):
            p = 0
            points.append(p)

        for s in state_reward_dict:
            for index, p in enumerate(self.env.points):
                if p[0] == s.lon and p[1] == s.lat:
                    points[index] = state_reward_dict[s]

        return points

class ValueIterationPlanner(Planner):
    
    def __init__(self, env):
        super().__init__(env)

    def plan(self, gamma=0.9, threshold=0.0001):
        self.initialize()
        actions = self.env.actions
        V = {}
        for s in self.env.states:
            V[s] = 0

        while True:
            delta = 0
            #self.log.append(self.dict_to_points(V))
            for s in V:
                if not self.env.can_action_at(s):
                    continue
                expected_rewards = []
                for a in actions:
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






