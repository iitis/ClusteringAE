: '
Copyright 2021 Institute of Theoretical and Applied Informatics,
Polish Academy of Sciences (ITAI PAS) https://www.iitis.pl
Authors:
- Bartosz Grabowski (ITAI PAS, ORCID ID: 0000−0002−2364−6547)
- Przemysław Głomb (ITAI PAS, ORCID ID: 0000−0002−0215−4674),
- Kamil Książek (ITAI PAS, ORCID ID: 0000−0002−0201−6220),
- Krisztián Buza (Sapientia Hungarian University of Transylvania, ORCID ID: 0000-0002-7111-6452)
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
---
Autoencoders pretraining using clustering v.1.0
Related to the work:
Improving Autoencoders Performance for Hyperspectral Unmixing using Clustering
Source code for the review process of the 14th Asian Conference on Intelligent
Information and Database Systems (ACIIDS 2022)
'

for dname in Samson Jasper
do
    for grid_shape in '(3,3)' '(5,5)' '(7,7)'
    do
        i=0
        for weights_init in Kaiming_He_normal Kaiming_He_uniform Xavier_Glorot_normal Xavier_Glorot_uniform
        do
            for model_no in {0..49}
            do
                python3 -m grids.exp --exp-name $dname'_'$grid_shape --run-name $i --dname $dname --model-no $model_no --grid-shape $grid_shape --weights-init $weights_init --seed $i --use-mlflow
                i=$((i+1))
            done
        done
    done
done
