from nav.config_models import EnvConfig
from nav.environment import Environment
import yaml

def main():
    config_path = "configs/marl_shared_goal.yaml"
    with open(config_path, "r") as f:
        raw = yaml.safe_load(f)
    config = EnvConfig(**raw)
    
    print(f"Config loaded.")
    print(f"num_agents_per_group: {config.num_agents_per_group}")
    print(f"Number of groups (agents list len): {len(config.agents)}")
    
    env = Environment(config=config)
    print(f"Environment initialized.")
    print(f"Total agents in env: {len(env.agents)}")
    print(f"Agent IDs: {env.agents}")

if __name__ == "__main__":
    main()
