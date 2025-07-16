import yaml
import pathlib
import argparse
import random
import sys
import string
import shutil
from datetime import datetime

from scipy import stats
import numpy as np
from hydra import initialize, compose

try:
    import docker
except ImportError:
    pass

"""
Last updated 2024-07-11.
TODO: I am realizing this is super suboptimal. There are two sources of
unnecessary overhead: 

1. starting the image
Possibly ameliorated by addn of something like below with substitution of 
docker exec for docker run:

import docker

# Create a Docker client
client = docker.from_env()

# Run a container in detached mode
container = client.containers.run('my_docker_image', detach=True, name='my_container')

# Execute a command in the running container
exit_code, output = container.exec_run('command_to_run')

# Print the output
print(output.decode())

# Stop and remove the container
container.stop()
container.remove()

2. Loading weights and initializing the actual model itself:
Solved probably by editing the script they are using.

"""


def get_timestamp_now():
    now = datetime.datetime.now()
    year = now.year
    month = str(now.month).zfill(2)
    day = now.day
    hour = now.hour
    second = now.second

    return f"{year}{month}{day}{hour}{second}"


def nb_from_file(file: str):
    """
    File to read params from a negative binomial parameter file in YAML format. If you want to bypass this,
    the script is looking for `n`, `p`, and `loc`.
    """

    fpathlib = pathlib.Path(file)
    if not fpathlib.exists():
        raise FileNotFoundError(f"File {file} does not exist.")

    with open(file, "r") as f:
        config = yaml.safe_load(f)

    return stats.nbinom(
        config.get("n", None), config.get("p", None), config.get("loc", 0)
    )


def nb_from_params(n: int, p: float, loc: int):
    return stats.nbinom(n, p, loc)
  
def generate_random_string(length=6):
    characters = string.ascii_letters + string.digits
    return "".join(random.choice(characters) for _ in range(length))


def setup_docker(gpu: int):
    client = docker.from_env()
    device = [docker.types.DeviceRequest(device_ids=[str(gpu)], capabilities=[["gpu"]])]
    ulimits = [  # docker complains if these are not added + GPU is desired
        docker.types.Ulimit(name="memlock", hard=-1, soft=-1),
        docker.types.Ulimit(name="stack", hard=67108864, soft=67108864),
    ]
    volume = {
        "/home": {"bind": "/home", "mode": "rw"},
        "/data": {"bind": "/data", "mode": "rw"},
    }

    return client, device, ulimits, volume


def parse_args():
    args = argparse.ArgumentParser()
    sampler = args.add_mutually_exclusive_group(required=True)

    sampler.add_argument(
        "--nb_config",
        type=str,
        help="Path to the negative binomial config file",
    )
    sampler.add_argument(
        "--lengths",
        type=str,
        help="Path to file of lengths in `.npy` file format.",
    )

    args.add_argument(
        "--sequential",
        action="store_true",
        help="Run the inference sequentially, \
                      as opposed to selecting n proteins of differing length from the length distribution and\
                       generating those at the same time, meaning all proteins of length l are done in one docker command.",
        required=False,
    )
    args.add_argument(
        "--num",
        type=int,
        default=5000,
        help="Number of proteins to generate.",
        required=True,
    )
    args.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="GPU to use. Will be provides a a `docker.types.DeviceRequest` with given ID.",
        required=True,
    )
    args.add_argument(
        "--output_directory",
        type=str,
        help="Output prefix for the generated proteins. Backbones will then be generated at path: {output_directory}/{DATETIMENOW}{RFDIFFUSION_NORMAL_OUTPUT}\
                    where RFDIFFUSION_NORMAL_OUTPUT is the normal string name; we do this in order to prevent collisions between simultaneous runs of this script.",
        required=True,
    )
    args.add_argument(
        "--container",
        type=str,
        default="rfdiff",
        help="Docker container to use.",
        required=False,
    )
    args.add_argument(
        "--checkpoint",
        type=str,
        help="Path to the checkpoint file to use.",
        required=True,
    )
    args.add_argument(
        "--num_threads",
        type=str,
        help="Number of threads to force one instance to use (CPU).",
        required=False,
        default=8,
    )
    args.add_argument(
        "--cleanup",
        action="store_true",
        required=False,
        help="Whether to delete `traj` files after running. ",
    )
    args.add_argument('--maxlen', type=int, default=1024, help='Maximum length of protein to generate.')
    args.add_argument(
        "--placeholder",
        type=str,
        help="Doesn't do anything. Just a placeholder to allow for multiple simultaneous amulet runs.",
    )

    args = parse_args()
    return args


def main():
    args = parse_args()

    if pathlib.Path(args.checkpoint).exists() is False:
        raise FileNotFoundError(f"Checkpoint file {args.checkpoint} does not exist.")

    if pathlib.Path(args.output_prefix.parent).exists() is False:
        raise ValueError(f"The output prefix desired's path directories do not exist. Please fix; parent is: {args.output_prefix.parent}.\n\
                         Fix by running `mkdir -p {args.output_prefix.parent}`")

    datetime_stamp = get_timestamp_now()

    container = args.container
    if not ((container is None) or (container == "none") or (container == "")):
        try:
            import docker

            client, device, ulimits, volume = setup_docker(args.gpu)
        except ImportError:
            raise ImportError(
                'You passed a container name but do not have the docker library installed. \
                Please install the docker library or pass "" and then run this script using amulet.'
            )
    else:
        client = None
        try:
            # assume we are on amulet and an appropriate container has been uploaded
            # in Sergey's RFDiffusion repo (which we are forking for this image+script)
            # the requisite scripting files are moved to /root
            # the file structure is roughly something like:
            # /root
            # ...py files
            # /root/config
            # /root/config/inference/
            # /root/config/inference/base.yaml
            # /root/config/inference/symmetry.yaml

            sys.path.append("/root/")
            import run_inference  # type: ignore
            # this is inside the container

        except ImportError:
            raise ImportError(
                'You passed "" for the container meaning you wanted to run on amulet: for this \
                we need to be able to find the `run_inference.py` script in /root on the \
                container filesystem. \
                Please specify a container or install the script.'
            )

    print(f"Client is: {client}")

    rand_str = generate_random_string(length=4)
    prefix = pathlib.Path(args.output_directory) / f"{datetime_stamp}_{rand_str}"
    prefix.mkdir(parents=True, exist_ok=True)
    print("Completed making parent dir:", prefix)

    prefix = prefix.as_posix()  # convert to string

    overrides = [  # baseline hydra opts
        f"inference.ckpt_override_path={args.checkpoint}",
        f"++num_threads={args.num_threads}",
    
    ]

    if args.nb_config:
        sampler = nb_from_file(
            args.nb_config
        )  # will be stats.nbinom object; see documentation for scipy
        random_lengths = sampler.rvs(size=args.num)
    else:
        random_lengths = np.load(args.lengths)
        random_lengths = np.random.choice(random_lengths, size=args.num, replace=True)

    random_lengths = np.minimum(random_lengths, args.maxlen)
    random_lengths = np.maximum(random_lengths, 40)

    unique, counts = np.unique(random_lengths, return_counts=True)

    stacked = np.vstack((unique, counts)).T  # will be 2 x TOT_NUM

    if random.random() <= 1:  # flip half the time
        stacked = stacked[::-1]

    for idx, (rand_length, count) in enumerate(stacked):
        tnow = datetime.now()
        rand_length_zfill = str(int(rand_length)).zfill(5)

        print(f"{tnow} || Generating: {count} {rand_length}-long backbone(s).")
        random_tag = generate_random_string(length=6)

        if client is not None:
            #             cmd = f"python /root/run_inference.py 'contigmap.contigs=[{rand_length}-{rand_length}]' \
            # inference.output_prefix={prefix}/{datetime_stamp}_{rand_length_zfill}AA \
            # inference.num_designs={count} inference.ckpt_override_path={args.checkpoint} ++num_threads={args.num_threads}"
            cmd = "python /root/run_inference.py"
            extra_cfgs = [
                f"inference.output_prefix={prefix}/{datetime_stamp}_{random_tag}_{rand_length_zfill}AA",
                f"contigmap.contigs=[{rand_length}-{rand_length}]",
                f"inference.num_designs={count}",
            ]
            for cfg in extra_cfgs + overrides:
                cmd += f" {cfg}"
            print(f"Running cmd: {cmd}")

            client.containers.run(
                container,
                cmd,
                device_requests=device,
                volumes=volume,
                ulimits=ulimits,
            )

        else:
            from os import chdir

            chdir(
                "/root/"
            )  # base IPD & SOvchinnikov code gets copied to /root for some reason
            with initialize(
                version_base=None, config_path="../../root/config/inference"
            ):  # file struct is config/inference/{base.yaml, symmetry.yaml}
                # see explanation in comment at start of this file to see more details about this
                # this is equivalent roughly to:
                # python /root/run_inference.py ...[settings]... (example inference.num_designs=3)
                cfg = compose(
                    "base",
                    overrides=overrides
                    + [
                        f"inference.output_prefix={prefix}/{datetime_stamp}_{random_tag}_{rand_length_zfill}AA",
                        f"contigmap.contigs=[{rand_length}-{rand_length}]",
                        f"inference.num_designs={count}",
                    ],
                )

            run_inference.main(cfg)
    if args.cleanup is True:
        print("args.cleanup is True: deleting `traj` files.")
        pth_to_delete = pathlib.Path(prefix)/'traj'
        if pth_to_delete.exists() is False:
            print(f"Path to delete {pth_to_delete} does not exist. Continuing.")
        else:
            shutil.rmtree(pth_to_delete)
        print("Finished deleting `traj` files.")

    return 0


if __name__ == "__main__":
    main()
