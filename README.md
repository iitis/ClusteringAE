## DESCRIPTION:
Autoencoders pretraining using clustering.

v.1.0

Related to the work:
> Improving Autoencoders Performance for Hyperspectral Unmixing using Clustering

> Source code for the review process of the 14th Asian Conference on Intelligent Information and Database Systems (ACIIDS 2022).

## LICENSE:
Copyright 2021 Institute of Theoretical and Applied Informatics,
Polish Academy of Sciences (ITAI PAS) <https://www.iitis.pl>
Authors:
- Bartosz Grabowski (ITAI PAS, ORCID ID: [0000−0002−2364−6547](https://orcid.org/0000-0002-2364-6547))
- Przemysław Głomb (ITAI PAS, ORCID ID: [0000−0002−0215−4674](https://orcid.org/0000-0002-0215-4674)),
- Kamil Książek (ITAI PAS, ORCID ID: [0000−0002−0201−6220](https://orcid.org/0000-0002-0201-6220)),
- Krisztián Buza (Sapientia Hungarian University of Transylvania, ORCID ID: [0000-0002-7111-6452](https://orcid.org/0000-0002-7111-6452))

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program. If not, see <https://www.gnu.org/licenses/>.

## FUNCTIONALITY:
- Autoencoder training and evaluation for spectral unmixing task
- Autoencoder pretraining using clustering algorithm

## FILES:
- `ATE/*`: [Autoencoder Testing Environment](https://github.com/iitis/AutoencoderTestingEnvironment) files.
- `cfg/*`: Config files.
- `grids/*.py`: Files required to run comparison between baseline and pretraining-based autoencoder training.
- `grids/run_exp.sh`: File to run comparison between baseline and pretraining-based autoencoder training.
- `grids/scripts/*`: Simple scripts used for various purposes.
- `grids/tests/*`: Unit tests.

## DATASETS:
All datasets have to be inserted into <em>./ATE/data/</em> folder.

## USAGE:
Run the script using `./grids/run_exp.sh` file.
The script requires Samson and Jasper datasets in the <em>./ATE/data/</em> folder as well as saved models' weights in <em>mpath</em> (set by default to <em>./models</em>).

To run demo version of the experiment, run `./grids/run_exp_demo.sh` file. The results of the experiment will be generated in the `./results` directory. Please note that this version of the script uses Custom dataset which is composed of random numbers, so the results too are going to be random.

## DEPENDENCIES
The scripts are dependent on [Autoencoder Testing Environment](https://github.com/iitis/AutoencoderTestingEnvironment). Used datasets, as well as loaded models' weights follow the same structure.
