# Script to build and publish mfai using runai (MF internal docker wrapper)
runai exec python -m build
# Use first arg if provided, otherwise use testpypi
runai exec python -m twine upload --verbose --repository ${1:-"testpypi"} dist/*
