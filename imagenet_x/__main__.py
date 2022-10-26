"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

from imagenet_x.check_prototypes import check_prototypes, add_check_prototypes_args
from imagenet_x.plots import generate_all_plots, add_plot_args
from imagenet_x.aggregate import generate_all_tables, add_aggregate_args

import argparse


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()

    # Create the subparser for the "plot" command.
    plot_parser = subparsers.add_parser("plots")
    add_plot_args(plot_parser)
    plot_parser.set_defaults(func=generate_all_plots)

    # Create the subparser for the "check_prototypes" command.
    check_prototypes_parser = subparsers.add_parser("check_prototypes")
    add_check_prototypes_args(check_prototypes_parser)
    check_prototypes_parser.set_defaults(func=check_prototypes)

    # Create the subparser for the "aggregate" command.
    aggregate_parser = subparsers.add_parser("aggregate")
    add_aggregate_args(aggregate_parser)
    aggregate_parser.set_defaults(func=generate_all_tables)

    # Parse the arguments and run the appropriate function.
    args = parser.parse_args()
    args.func(args)

main()
