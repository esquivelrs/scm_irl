# -*- coding: utf-8 -*-

"""Top-level package for scm_irl."""

__author__ = """Rolando Esquivel Sancho"""
__email__ = 'rolando.esq@gmail.com'
__version__ = '0.1.0'


from gymnasium.envs.registration import register

register(
     id="scmirl-v0",
     entry_point="scm_irl/env:ScmIrlEnv",
     max_episode_steps=1000,
)