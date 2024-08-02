import os


def get_home_path() -> str | None:
    # load home path
    home_path = os.getenv("HOME")
    if home_path:
        print("home path: ", home_path)
        return home_path
    else:
        raise FileNotFoundError("HOME environment variable is not set.")


def get_isaaclab_path() -> str | None:
    # load home path
    home_path = get_home_path()
    if home_path:
        isaaclab_path = os.path.join(home_path, "Ominverse_RL_platform", "IsaacLab")
        if os.path.isdir(isaaclab_path):
            return isaaclab_path
        else:
            raise FileNotFoundError(f"Dir:{isaaclab_path} is not exists.")
    else:
        print("HOME environment variable is not set.")


def get_policy_dir_path(policy_dir_name: str) -> str | None:
    # load home path
    isaaclab_path = get_isaaclab_path()
    if isaaclab_path:
        policy_dir_path = os.path.join(
            isaaclab_path,
            "logs",
            "rl_games",
            "prtpr",
            policy_dir_name,
        )
        if os.path.isdir(policy_dir_path):
            return policy_dir_path
        else:
            raise FileNotFoundError(f"Policy file:{policy_dir_path} is not exists.")
    else:
        print("HOME environment variable is not set.")
