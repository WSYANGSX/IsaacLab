import os


def get_home_path() -> str | None:
    # load home path
    home_path = os.getenv("HOME")
    if home_path:
        print("home path: ", home_path)
        return home_path
    else:
        print("HOME environment variable is not set.")


def get_isaaclab_path() -> str | None:
    # load home path
    home_path = os.getenv("HOME")
    if home_path:
        isaaclab_path = os.path.join(home_path, "Ominverse_RL_platform", "IsaacLab")
        return isaaclab_path
    else:
        print("HOME environment variable is not set.")


def get_policy_path() -> str | None:
    # load home path
    home_path = os.getenv("HOME")
    if home_path:
        isaaclab_path = os.path.join(
            home_path,
            "Ominverse_RL_platform",
            "IsaacLab",
            "logs",
            "rl_games",
            "prtpr",
            "3",
        )
        return isaaclab_path
    else:
        print("HOME environment variable is not set.")
