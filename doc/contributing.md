# Contributing

This documentation is automatically built and deployed on every push to a tag or on pushes to the `sphinx` branch.

## Generating documentation locally

To generate the documentation locally, run the following commands from the root of the repository:

- `python doc/generate_rst.py`: generates the RST files for the API reference by introspecting the codebase
- `sphinx-apidoc -o doc/api/references mfai --force --templatedir doc/_templates/apidoc`: generates the full package reference using `sphinx-apidoc`
- `mv doc/api/references/modules.rst doc/api/references.rst`: moves the generated index to the expected location
- `make -C doc html`: builds the HTML documentation into `doc/_build/html/`

You can then open the documentation in your browser by opening `doc/_build/html/index.html` directly.

## Live preview

To get a live preview that automatically refreshes on file changes, you can use the [Live Preview](https://marketplace.visualstudio.com/items?itemName=ms-vscode.live-server) extension for VSCode.

Once installed, open `doc/_build/html/index.html` in VSCode and click **Show Preview** — the browser will automatically refresh every time you rebuild the documentation with `make -C doc html`.