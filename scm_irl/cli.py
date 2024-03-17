# -*- coding: utf-8 -*-

"""Console script for scm_irl."""
import sys
import click
from sllib.visualisations import seamanship_score


@click.command()
@click.argument('scenario_path', type=click.Path(exists=True))
def main(scenario_path):
    """Console script for scm_irl."""
    seamanship_score.plot_scenario(scenario_path)

    return 0




if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover
