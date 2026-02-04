# model_eval

<p align="center">
    <em></em>
</p>

[![build](https://github.com/mullenkamp/model_eval/workflows/Build/badge.svg)](https://github.com/mullenkamp/model_eval/actions)
[![codecov](https://codecov.io/gh/mullenkamp/model_eval/branch/master/graph/badge.svg)](https://codecov.io/gh/mullenkamp/model_eval)
[![PyPI version](https://badge.fury.io/py/model_eval.svg)](https://badge.fury.io/py/model_eval)

---

**Documentation**: <a href="https://mullenkamp.github.io/model_eval/" target="_blank">https://mullenkamp.github.io/model_eval/</a>

**Source Code**: <a href="https://github.com/mullenkamp/model_eval" target="_blank">https://github.com/mullenkamp/model_eval</a>

---

## Development

### Setup environment

We use [Hatch](https://hatch.pypa.io/latest/install/) to manage the development environment and production build. Ensure it's installed on your system.

### Run unit tests

You can run all the tests with:

```bash
hatch run test
```

### Format the code

Execute the following command to apply linting and check typing:

```bash
hatch run lint
```

### Publish a new version

You can bump the version, create a commit and associated tag with one command:

```bash
hatch version patch
```

```bash
hatch version minor
```

```bash
hatch version major
```

Your default Git text editor will open so you can add information about the release.

When you push the tag on GitHub, the workflow will automatically publish it on PyPi and a GitHub release will be created as draft.

## Serve the documentation

You can serve the Mkdocs documentation with:

```bash
hatch run docs-serve
```

It'll automatically watch for changes in your code.

## License

This project is licensed under the terms of the Apache Software License 2.0.
