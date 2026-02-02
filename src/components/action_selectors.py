import torch as th
from torch.distributions import Categorical
from .epsilon_schedules import DecayThenFlatSchedule
import torch.nn.functional as F

REGISTRY = {}


class MultinomialActionSelector:

    def __init__(self, args):
        self.args = args

        self.schedule = DecayThenFlatSchedule(args.EPSILON_START, args.EPSILON_FINISH, args.EPSILON_ANNEAL_TIME,
                                              decay="linear")
        self.epsilon = self.schedule.eval(0)
        self.test_greedy = getattr(args, "test_greedy", True)

        self.device = th.device(self.args.DEVICE)

    def update_epsilon(self, steps=0, train_steps=0):
        standard = 0
        if self.args.EPSILON_UPDATE_STANDARD == "steps":
            standard = steps
        elif self.args.EPSILON_UPDATE_STANDARD == "train_steps":
            standard = train_steps
        else:
            Exception("EPSILON_UPDATE_STANDARD is defined wrong")
        self.epsilon = self.schedule.eval(standard)

    def select_action(self, agent_inputs, avail_actions, t_env, test_mode=False):
        masked_policies = agent_inputs.clone()
        masked_policies[avail_actions == 0.0] = 0.0
        if self.args.EPSILON_UPDATE_STANDARD == "steps":
            self.update_epsilon(steps=t_env)

        if test_mode:
            # Greedy action selection only
            self.epsilon = 0.0

        if self.args.ALGORITHM in ['coma', 'liir']:
            if test_mode and self.test_greedy:
                picked_actions = masked_policies.max(dim=2)[1]
            else:
                picked_actions = Categorical(masked_policies).sample().long()
        else:
            if t_env < self.args.START_TRAINING_EPISODE:
                random_actions = th.rand_like(masked_policies)
                picked_actions = random_actions.clamp(-self.args.MAX_ACTION, self.args.MAX_ACTION)
            else:
                if not test_mode:
                    masked_policies += th.normal(
                        0, self.args.MAX_ACTION * self.args.EXPL_NOISE,
                        size=masked_policies.shape
                    ).to(device=self.device)
                picked_actions = masked_policies.clamp(-self.args.MAX_ACTION, self.args.MAX_ACTION)
        return picked_actions


REGISTRY["multinomial"] = MultinomialActionSelector


class EpsilonGreedyActionSelector:

    def __init__(self, args):
        self.args = args

        self.schedule = DecayThenFlatSchedule(args.EPSILON_START, args.EPSILON_FINISH, args.EPSILON_ANNEAL_TIME,
                                              decay="linear")
        self.epsilon = self.schedule.eval(0)

    def update_epsilon(self, steps=0, train_steps=0):
        standard = 0
        if self.args.EPSILON_UPDATE_STANDARD == "steps":
            standard = steps
        elif self.args.EPSILON_UPDATE_STANDARD == "train_steps":
            standard = train_steps
        else:
            Exception("EPSILON_UPDATE_STANDARD is defined wrong")
        self.epsilon = self.schedule.eval(standard)

    def select_action(self, agent_inputs, avail_actions, t_env, test_mode=False):

        # Assuming agent_inputs is a batch of Q-Values for each agent bav
        if self.args.EPSILON_UPDATE_STANDARD == "steps":
            self.update_epsilon(steps=t_env)

        if test_mode:
            # Greedy action selection only
            eps = 0.0
        else:
            eps = self.epsilon

        # mask actions that are excluded from selection
        masked_q_values = agent_inputs.clone()
        masked_q_values[avail_actions == 0.0] = -float("inf")  # should never be selected!

        random_numbers = th.rand_like(agent_inputs[:, :, 0])
        pick_random = (random_numbers < eps).long()
        random_actions = Categorical(avail_actions.float()).sample().long()

        picked_actions = pick_random * random_actions + (1 - pick_random) * masked_q_values.max(dim=2)[1]
        return picked_actions


REGISTRY["epsilon_greedy"] = EpsilonGreedyActionSelector


class UtilEpsilonGreedyActionSelector:

    def __init__(self, args):
        self.args = args

        self.schedule = DecayThenFlatSchedule(args.EPSILON_START, args.EPSILON_FINISH, args.EPSILON_ANNEAL_TIME,
                                              decay="linear")
        self.epsilon = self.schedule.eval(0)

    def update_epsilon(self, steps=0, train_steps=0):
        standard = 0
        if self.args.EPSILON_UPDATE_STANDARD == "steps":
            standard = steps
        elif self.args.EPSILON_UPDATE_STANDARD == "train_steps":
            standard = train_steps
        else:
            Exception("EPSILON_UPDATE_STANDARD is defined wrong")
        self.epsilon = self.schedule.eval(standard)

    def select_action(self, util_inputs, agent_inputs, avail_actions, t_env, test_mode=False):

        # Assuming agent_inputs is a batch of Q-Values for each agent bav
        if self.args.EPSILON_UPDATE_STANDARD == "steps":
            self.update_epsilon(steps=t_env)

        if test_mode:
            # Greedy action selection only
            eps = 0.0
        else:
            eps = self.epsilon

        # mask actions that are excluded from selection
        masked_q_values = agent_inputs.clone()
        masked_q_values[avail_actions == 0.0] = -float("inf")  # should never be selected!

        masked_util_values = util_inputs.clone()
        masked_util_values[avail_actions == 0.0] = -float("inf")  # should never be selected!

        # masked_q_values = th.softmax(masked_q_values, dim=-1)
        # masked_util_values = th.softmax(masked_util_values, dim=-1)
        # mixed_masked_q_values = masked_q_values + masked_util_values
        #
        # random_numbers = th.rand_like(agent_inputs[:, :, 0])
        # pick_random = (random_numbers < eps).long()
        # random_actions = Categorical(avail_actions.float()).sample().long()
        #
        # picked_actions = pick_random * random_actions + (1 - pick_random) * mixed_masked_q_values.max(dim=2)[1]

        masked_q_values = th.softmax(masked_q_values, dim=-1)
        masked_util_values = th.softmax(masked_util_values, dim=-1)

        random_numbers = th.rand_like(agent_inputs[:, :, 0])
        pick_random = (random_numbers < eps).long()

        # if not test_mode:
        #     masked_q_values += masked_util_values
        # masked_util_values = Categorical(masked_util_values).sample().long()

        # random_actions = Categorical(avail_actions.float()).sample().long()
        # picked_actions = pick_random * random_actions + (1 - pick_random) * masked_q_values.max(dim=2)[1]
        picked_actions = pick_random * masked_util_values.max(dim=2)[1] + (1 - pick_random) * masked_q_values.max(dim=2)[1]
        # picked_actions = pick_random * masked_util_values + (1 - pick_random) * masked_q_values.max(dim=2)[1]
        return picked_actions


REGISTRY["util_epsilon_greedy"] = UtilEpsilonGreedyActionSelector


class EpsilonExplActionSelector:
    def __init__(self, args):
        self.args = args

        self.schedule = DecayThenFlatSchedule(
            args.EPSILON_START,
            args.EPSILON_FINISH,
            args.EPSILON_ANNEAL_TIME,
            decay="linear",
        )
        self.epsilon = self.schedule.eval(0)

    def select_action(
        self,
        agent_inputs,
        int_agent_inputs,
        avail_actions,
        t_env,
        int_ratio,
        test_mode=False,
    ):
        # Assuming agent_inputs is a batch of Q-Values for each agent bav
        self.epsilon = self.schedule.eval(t_env)

        if test_mode:
            # Greedy action selection only
            self.epsilon = getattr(self.args, "test_noise", 0.0)
            int_ratio = 0.0

        # mask actions that are excluded from selection
        masked_q_values = agent_inputs.clone()
        masked_q_values[avail_actions == 0] = -float("inf")  # should never be selected!
        masked_int_q_values = int_agent_inputs.clone()
        masked_int_q_values[avail_actions == 0.0] = -float(
            "inf"
        )  # should never be selected!
        masked_int_q_values = F.softmax(masked_int_q_values, dim=-1)

        m = Categorical(masked_int_q_values)
        int_actions = m.sample().long()

        random_numbers = th.rand_like(agent_inputs[:, :, 0])
        pick_random = (random_numbers < self.epsilon).long()
        random_actions = Categorical(avail_actions.float()).sample().long()

        # behavior_actions
        random_numbers = th.rand_like(agent_inputs[:, :, 0])
        pick_int = (random_numbers < int_ratio).long()
        behavior_actions = (
            pick_int * int_actions + (1 - pick_int) * masked_q_values.max(dim=2)[1]
        )
        picked_actions = (
            pick_random * random_actions + (1 - pick_random) * behavior_actions
        )

        return picked_actions, m.entropy()


REGISTRY["epsilon_expl"] = EpsilonExplActionSelector
