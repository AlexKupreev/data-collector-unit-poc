# Data Collector Unit - PoC

-----

A python app that handles both data collection and usage scenarios in a single container.


## Table of Contents

- [Installation](#installation)
- [License](#license)

## Architecture

- dagster for data pipeline
- sqLite as a data storage/data warehouse

## Installation



## Development

### Basic use cases

- see all environments:

    ```console
    hatch env show
    ```

- spawn a shell within an dev environment

    ```console
    hatch shell --name dev
    ```

- run dagster on Linux: from the project folder - enter the shell and 

    ```cli
    export DAGSTER_HOME=`pwd`/src/data_collector_unit_poc/dagster/service
    dagster dev -h 0.0.0.0 -p 3000 -w ./src/data_collector_unit_poc/dagster/workspace.yaml

    ```

- create new migration:

  ```cli
  alembic revision --autogenerate -m "<new migration name>"
  ```

- apply the last migration

  ```cli
  cd src/data_collector_unit_poc/migrations && alembic upgrade head
  ```

- run fastapi server in dev mode:

  ```commandline
  fastapi dev src/data_collector_unit_poc/web/main.py
  ```

- create app in fly.io (needed for deployment to fly.io):

  ```commandline
  flyctl apps create data-collector-unit-poc
  ```

- set environment variables:

  ```commandline
  fly secrets set APP_SECRET=...
  fly secrets set ADMIN_USERNAME=...
  fly secrets set ADMIN_PASSWORD=...
  optionally:
  fly secrets set AWS_ACCESS_KEY_ID=...
  fly secrets set AWS_SECRET_ACCESS_KEY=...
  fly secrets set BUCKET_NAME=...
  ```

- deploy to fly.io:

  ```commandline
  fly deploy -a <app name>
  ```

  To deploy on a single machine use flag `--ha=false`.

- scale to 1 machine after deploy:

  ```commandline
  flyctl scale count 1
  ```

- create 1 persistent volume (1 GB by default)
  
  ```commandline
  fly volumes create data_collector_unit_poc_volume -n 1 -r <region code>
  ```

## License

`data-collector-unit-poc` is distributed under the terms of the [MIT](https://spdx.org/licenses/MIT.html) license.
