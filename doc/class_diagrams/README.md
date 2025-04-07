# Class diagrams generation

You can generate mfai's class diagrams by running the command:
```sh
docker build . -f Dockerfile -t mfai
docker run -it --rm mfai python3 doc/class_diagrams/make_diagrams.py
```

The diagrams are generated in PlantUML format by the librairy [py2puml](https://pypi.org/project/py2puml/), and exported to png by [plantuml](https://pypi.org/project/plantuml/).


## Limitations
- py2puml does not support type hinting `Literal[]`
- exporting to svg do not work with the python lib plantuml, png are used instead