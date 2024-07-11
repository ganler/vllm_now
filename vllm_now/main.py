# Generate docker-compose.yml and nginx.conf files to run vLLM servers with load balancing.

import os
import re
import sys
from typing import List

import GPUtil
from fire import Fire


def _save_file(file_name, content):
    with open(file_name, "w") as f:
        f.write(content)


def make_nginx_conf(worker_connections=4096, n_servers=8):
    return r"""events {
    worker_connections [WORKER_CONNECTIONS];
}
http {
    upstream vllm_servers {
        [SERVER_LIST]
    }
    server {
        gzip on;
        keepalive_timeout 600s;
        proxy_read_timeout 600s;
        proxy_connect_timeout 600s;
        proxy_send_timeout 600s;
        fastcgi_read_timeout 600s;
        listen 80;
        location / {
            proxy_pass http://vllm_servers;
        }
    }
}
""".replace(
        "[WORKER_CONNECTIONS]", str(worker_connections)
    ).replace(
        "[SERVER_LIST]",
        "\n        ".join([f"server vllm-server-{i}:8000;" for i in range(n_servers)]),
    )


def get_available_physical_gpus() -> List[int]:
    if os.getenv("CUDA_VISIBLE_DEVICES", None):
        return list(map(int, os.getenv("CUDA_VISIBLE_DEVICES").split(",")))

    return [gpu.id for gpu in GPUtil.getGPUs()]


def make_docker_compose_yml(gpu_groups, vllm_version, fire_kwargs):
    _kwarg_split = "\n    - "
    flag_kwargs_field = _kwarg_split.join(
        [
            f"--{k.replace('_', '-')}"
            for k, v in fire_kwargs.items()
            if isinstance(v, bool)
        ]
    )
    if flag_kwargs_field:
        flag_kwargs_field = _kwarg_split + flag_kwargs_field

    per_instance_template = (
        lambda i, gpus: f"""\
  vllm-server-{i}:
    <<: *vllm-server-base
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              capabilities: [gpu]
              device_ids: {gpus}
"""
    )

    hf_home = os.getenv("HF_HOME", "~/.cache/huggingface/")

    _double_newline = "\n\n"
    _depends_on_split = "\n      "

    return f"""\
version: "3"
x-vllm-server-base: &vllm-server-base
  image: vllm/vllm-openai:{vllm_version}
  command:
    - --disable-log-requests   # To save your eyes from the logs
    - {_kwarg_split.join([f"--{k.replace('_', '-')}={v}" for k, v in fire_kwargs.items() if not isinstance(v, bool)])}{flag_kwargs_field}
  volumes:
    - {hf_home}:/root/.cache/huggingface:rw
services:
{_double_newline.join([per_instance_template(i, list(map(str, gpus))) for i, gpus in enumerate(gpu_groups)])}
  load-balancer:
    image: nginx:latest
    ports:
      - "80:80"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf:ro
    depends_on:
      {_depends_on_split.join([f"- vllm-server-{i}" for i in range(len(gpu_groups))])}
"""


def _clean_argv(fire_kwargs):
    # Processing vLLM arguments
    vllm_args = sys.argv[1:]
    # Pop the `vllm_version` argument
    regexpr = re.compile(r"^--vllm[_-]version($|=.*$)")
    if "vllm_version" in fire_kwargs:
        assert isinstance(
            fire_kwargs["vllm_version"], str
        ), "vllm_version must be a string"
        # the index of the `vllm_version` argument
        idx = next(i for i, arg in enumerate(vllm_args) if regexpr.match(arg))
        val = vllm_args.pop(idx)
        if not re.compile(r"^--vllm[_-]version=.*$").match(val):
            vllm_args.pop(idx)


def main(**fire_kwargs):
    n_gpus_per_group = fire_kwargs.get("tensor_parallel_size", 1)
    assert n_gpus_per_group >= 1
    assert "model" in fire_kwargs, "model argument is required"

    gpus_idx = get_available_physical_gpus()
    # split them into groups by `n_gpus_per_group`
    gpu_groups = [
        gpus_idx[i : i + n_gpus_per_group]
        for i in range(0, len(gpus_idx), n_gpus_per_group)
        if len(gpus_idx[i : i + n_gpus_per_group]) == n_gpus_per_group
    ]

    vllm_version = fire_kwargs.get("vllm_version", "latest")
    fire_kwargs.pop("vllm_version", None)

    # assert kwargs to only include numerical values or strings
    for k, v in fire_kwargs.items():
        assert isinstance(v, (int, float, str)), f"Invalid type for {k}: {type(v)}"
    _save_file(
        "docker-compose.yml",
        make_docker_compose_yml(gpu_groups, vllm_version, fire_kwargs),
    )
    _save_file("nginx.conf", make_nginx_conf(n_servers=len(gpu_groups)))
    print(
        """\
# [Launch] vllm servers for your favorite models
docker-compose -d up
# [Check] the status
docker-compose logs -t -f
# [Stop] the servers
docker-compose stop
"""
    )


def run():
    Fire(main)
