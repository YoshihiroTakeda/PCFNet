# PCFNet
This repository is official implementation of the paper "Mining for Protoclusters at $z\sim4$ from photometric datasets with Deep Learning" ([Takeda et al. 2024](https://doi.org/10.3847/1538-4357/ad8a67)).


## Usage
Clone this repository and follow the instructions below.
```bash
cd your_workdir
git clone git@github.com:YoshihiroTakeda/PCFNet
```

### Setup
We recommend using a virtual environment to run the code.

#### VSCode & Docker
The easiest way to setup the environment is to use VSCode with Docker.
1. Install [VSCode](https://code.visualstudio.com/) & [Docker](https://www.docker.com/)
2. Install [Remote - Containers](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers) extension
3. Open the project folder (`your_workdir/pcfnet`) in VSCode
4. Set up for the docker environment
    - To make the user id and group id consistent between the host and the container, excute the `make_env4docker.sh` script in your terminal. Then, `.env` file will be created.
    ```bash
    bash make_env4docker.sh
    ```
5. Click on the button in the bottom left corner of VScode window and select "Reopen in Container"
    - Automatically builds the Docker image and starts the container. This may take a while the first time.

#### Pip
You can also set up PCFNet as library in your environment with `pip`. Required packages are automatically installed.
```bash
python -m pip install -e .[all]
```

### Data Preparation
1. Prepare the data for training and prediction
    - Simulation data (PCcone): Please contact the authors of Araya-Araya et al. (2021) for the data.
    - Observation data: Please obtain from HSC-SSP data release.
    - Original data: Please prepare the data in the same format as the simulation data and observation data and write codes like `preprocess.py` to convert the format for PCFNet. **You have to carefully check the consistency between the simulation and observation.**
2. Place the data in the `data` directory. For exmaple, 
    - `data/sim/` for simulation data
    - `data/obs/` for observation data
3. Modify the configuration file `config/PCFNet_configure.yaml` to match the data paths. You may also need additional modifications for your computational environment (batchsize, iteration times, etc.)

### Data Initialization & Train & Predict
You make sure that the cofiguration file `config/PCFNet.yaml` is correctly set up.
Then, run the following commands in the terminal.

```bash
python src/preprocess_obs.py --config config/PCFNet_configure.yaml
python src/preprocess.py --config config/PCFNet_configure.yaml
python src/train.py --config config/PCFNet_configure.yaml
python src/predict_obs.py --config config/PCFNet_configure.yaml

### or...
## modified src/run_all.sh and run the following command
bash src/run_all.sh
```

#### [Optional] WandB integration
You can use [WandB](https://wandb.ai/site) to monitor the training process. Turn on the `wandb` flag in the configuration file and login to QanB.

```bash
wandb login
```
To illustrate the PR curve per epoch, you can use the custom chart, `wandb_custom/pr_curve.json`. Select custom chart in the WandB app and register the JSON file as a new Vega-Lite JSON. The table in query should be set to HistoryTable, and `Other Setting` -> `Show step selector` should be turned on. Then, you can easily see the PR curve per epoch by using the slider shown after clicking the gear button.


#### [Optional] Tensorboard
You can also use Tensorboard to monitor the training process. 



### Evaluation
You can quickly view the results in the jupyter notebook.



# References
- Araya-Araya et al. 2021, [MNRAS](https://doi.org/10.1093/mnras/stab1133), 504, 5054. 


# Citation
If you use this code for your research, please cite our paper.

```bibtex
@article{Takeda2024,
  title={Mining for Protoclusters at $z\sim4$ from photometric datasets with Deep Learning},
  author={Takeda, Yoshihiro and Kashikawa, Nobunari and Ito, Kei and Toshikawa, Jun and Momose, Rieko and Fujiwara, Kent and Liang, Yongming and Ishimoto, Rikako and Yoshioka, Takehiro and Arita, Junya and Kubo, Mariko and Uchiyama, Hisakazu},
  year={2024},
  journal={ApJ},
  url={https://doi.org/10.3847/1538-4357/ad8a67}
}
```
