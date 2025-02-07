```bash
pip install "vllm_now @ git+https://github.com/ganler/vllm_now@main" --upgrade
vllm_now --model "ise-uiuc/Magicoder-S-DS-6.7B"

# [Launch] vllm servers for your favorite models
docker compose up -d
# [Check] the status
docker compose logs -t -f
# [Stop] the servers
docker compose stop
```
The model will be served via `http://localhost:${VLLM_NOW_PORT}` where if `VLLM_NOW_PORT` is not set, it defaults to `80`.

> [!Note]
>
> To install `docker compose`: https://docs.docker.com/compose/install/linux/#install-using-the-repository
>
> Quick command for Ubuntu: `sudo apt-get install docker-compose-plugin`
>
> Note sometimes your `docker compose` can be `docker-compose` ([Why?](https://stackoverflow.com/questions/66514436/difference-between-docker-compose-and-docker-compose)).

> [!Tip]
>
> `vllm_now` arguments are synchronized with `vllm.entrypoints.openai.api_server` [arguments](https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html).
>
> In addition, you can set `--vllm-version` to specify the version of `vllm` to install.
>
> Supported and compatible environment variables:
>
> - `HF_HOME`
> - `CUDA_VISIBLE_DEVICES`
