
#!/bin/bash

if [[ ! -f "$(pwd)/setup.bash" ]]
then
  echo "please launch from the proxfly folder!"
  exit
fi

PROJECT_PATH=$(pwd)
echo "project path: $PROJECT_PATH"

echo "Setting up the project..."
git submodule update --init --recursive

cd $PROJECT_PATH
conda env create -f environment.yaml
conda init