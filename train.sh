# Training script
# (still experimental)
COLOR_CYAN="\\033[0;36m"

if [[ $OSTYPE == "win32" ]]; then
    python_ext=python
else
    python_ext=python3
fi

while [[ $# -gt 0 ]]; do
    case $1 in
        --gan)
            MODEL_PATH=GAN/train.py
            echo "${COLOR_CYAN}= = = = TRAINING GAN = = = =${COLOR_NORMAL}"
            TENSORBOARD_ARGS="--logdir_spec=Real:logs/real,Fake:logs/fake"
            shift
            ;;
        # add feature extraction model
        *)
          shift
          ;;

    esac
done

echo "${COLOR_CYAN}Running tensorboard at localhost:6006${COLOR_NORMAL}"
tensorboard $TENSORBOARD_ARGS
$python_ext $MODEL_PATH
