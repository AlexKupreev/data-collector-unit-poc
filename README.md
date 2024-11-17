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


## License

`data-collector-unit-poc` is distributed under the terms of the [MIT](https://spdx.org/licenses/MIT.html) license.
