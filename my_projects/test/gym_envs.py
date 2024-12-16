from gym import envs

env_specs = envs.registry.all()
env_ids = [env_spec.id for env_spec in env_specs]
print(env_ids)