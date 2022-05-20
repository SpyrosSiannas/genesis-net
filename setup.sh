#!/bin/bash
COLOR_CYAN="\\033[0;36m"
COLOR_NORMAL="\\033[0m"
COLOR_GREEN='\033[1;32m'

function log_status () {
    echo "${COLOR_CYAN}= = = = $1 = = = =${COLOR_NORMAL}"
}

function log_success () {
    echo "${COLOR_GREEN}= = = = $1 = = = =${COLOR_NORMAL}"
}

if [[ $OSTYPE == "win32" ]]; then
    python_ext=python
    pip_ext=pip
else
    python_ext=python3
    pip_ext=pip3
fi

if [ ! -d env ]; then
    log_status "Generating Virtual Environment"
    $python_ext -m venv env
    source ./activate.bash
    log_status "Upgrading pip"
    $pip_ext install --upgrade pip
    log_status "Installing Dependencies"
    $pip_ext install -r requirements.txt
fi

log_success "Setup Complete"
