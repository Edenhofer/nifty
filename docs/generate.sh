rm -rf docs/build docs/source/mod
sphinx-apidoc -e -o docs/source/mod nifty7
sphinx-build -b html docs/source/ docs/build/
