""" Launch RL/IL training and evaluation. """

from robot_learning.main import Run

from policy_sequencing_trainer import PolicySequencingTrainer


class SkillChainingRun(Run):
    def __init__(self, parser):
        super().__init__(parser)

    def _set_run_name(self):
        """ Sets run name. """
        config = self._config
        env = config.env
        if hasattr(config, "furniture_name"):
            env = config.furniture_name
        config.run_name = "{}.{}.{}.{}".format(
            env, config.algo, config.run_prefix, config.seed
        )

    def _get_trainer(self):
        if self._config.algo == "ps":
            return PolicySequencingTrainer(self._config)
        else:
            return super()._get_trainer()


if __name__ == "__main__":
    # register furniture environment to Gym
    import furniture

    # default arguments
    from policy_sequencing_config import create_skill_chaining_parser

    parser = create_skill_chaining_parser()

    # change default values
    parser.set_defaults(wandb_entity=None)
    parser.set_defaults(wandb_project=None)

    # execute training code
    SkillChainingRun(parser).run()
