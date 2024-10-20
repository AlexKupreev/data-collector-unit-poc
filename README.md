# Dagster DataBox

-----

A small python app that handles both data collection and usage scenarios in a single container.

## Table of Contents

- [Installation](#installation)
- [License](#license)

## Installation

```console
pip install dagster-databox
```

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


## License

`data-collector-unit-poc` is distributed under the terms of the [MIT](https://spdx.org/licenses/MIT.html) license.
