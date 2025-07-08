from aide import Experiment
from aide.utils.config import _load_cfg, prep_cfg, load_task_desc, prep_agent_workspace
from aide.journal import Journal
from aide.agent import Agent
from aide.interpreter import Interpreter
from omegaconf import OmegaConf
from rich.status import Status
from omegaconf import DictConfig


class AideWrapper(Experiment):
    def __init__(self, data_dir: str, goal: str, eval: str | None = None, llm_model:str = "llama-3.3-70b-instruct"):
        """Initialize a new experiment run.

        Args:
            data_dir (str): Path to the directory containing the data files.
            goal (str): Description of the goal of the task.
            eval (str | None, optional): Optional description of the preferred way for the agent to evaluate its solutions.
        """

        _cfg = _load_cfg(use_cli_args=False)
        _cfg = AideWrapper.overwrite_llm_model(llm_model, _cfg)
        _cfg.data_dir = data_dir
        _cfg.goal = goal
        _cfg.eval = eval
        self.cfg = prep_cfg(_cfg)

        self.task_desc = load_task_desc(self.cfg)

        with Status("Preparing agent workspace (copying and extracting files) ..."):
            prep_agent_workspace(self.cfg)

        self.journal = Journal()
        self.agent = Agent(
            task_desc=self.task_desc,
            cfg=self.cfg,
            journal=self.journal,
        )
        self.interpreter = Interpreter(
            self.cfg.workspace_dir, **OmegaConf.to_container(self.cfg.exec)  # type: ignore
        )

    @staticmethod
    def overwrite_llm_model(llm_model:str, _cfg:DictConfig):
        old_config = dict(_cfg)
        for key, value in old_config.items():
            if isinstance(value, DictConfig):
                old_config[key] = AideWrapper.overwrite_llm_model(llm_model, value)
            elif key == "model":
                old_config[key] = llm_model
        return DictConfig(old_config)
            
