all:
	sphinx-build docs docs/_build/_html
docs-init:
	sphinx-apidoc -F -A "Teodor Mihai Moldovan" -H "dpcluster" -V 0.1 -f -o docs dpcluster
develop:
	python setup.py develop --user
