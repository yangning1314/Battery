# PINN4SOH


# 1. System requirements
python version: 3.8

|    Package     | Version  |
|:--------------:|:--------:|
|     torch      |  1.10.1   |
|    sklearn     |  1.3.2  |
|     numpy      |  1.24.3  |
|     pandas     |  2.0.3   |
|   matplotlib   |  3.7.5  |



# 2. Installation guide
If you are not familiar with Python and Pytorch framework, 
you can install Anaconda first and use Anaconda to quickly configure the environment.
## 2.1 Create environment
```angular2html
conda create -n new_environment python=3.8
```



## 2.2 Activate environment
```angular2html
conda activate new_environment
```

## 2.3 Install dependencies
```angular2html
conda install pytorch=1.10.1
conda install scikit-learn=1.3.2 numpy=1.24.3 pandas=2.0.3 matplotlib=3.7.5
```

# 3. Demo
We provide a detailed demo of our code running on the XJTU dataset.
Run the `Compare.py` file. The program will generate a folder named `myplot.png` and save the results in it.You can see the comparison results between MLP and PINN.


**Note: As we all know, the training process of neural network models is random, 
and the volatility of regression models is often greater than that of classification models. 
Therefore, the results obtained from the above process are not expected to be exactly identical to those mentioned in our manuscript. 
However, it is evident that the results obtained from our method are superior to those of MLP and CNN.**



# 4.  Additional information
The data in the `data` folder is preprocessed data.
Raw data can be obtained from the following links:
1. XJTU dataset: [link](https://wang-fujin.github.io/)
2. TJU dataset: [link](https://zenodo.org/record/6405084)
3. HUST dataset: [link](https://data.mendeley.com/datasets/nsc7hnsg4s/2)
4. MIT dataset: [link](https://data.matr.io/1/projects/5c48dd2bc625d700019f3204)

The code for **reading and preprocessing** the dataset is publicly available at [https://github.com/wang-fujin/Battery-dataset-preprocessing-code-library](https://github.com/wang-fujin/Battery-dataset-preprocessing-code-library)

---

We generated a comprehensive dataset consisting of 55 lithium-nickel-cobalt-manganese-oxide (NCM) batteries. 

It is available at: [Link](https://wang-fujin.github.io/)

Zenodo link: [https://zenodo.org/records/10963339](https://zenodo.org/records/10963339).

![https://github.com/wang-fujin/PINN4SOH/blob/main/xjtu%20battery%20dataset.png](https://github.com/wang-fujin/PINN4SOH/blob/main/xjtu%20battery%20dataset.png)

![https://github.com/wang-fujin/PINN4SOH/blob/main/6%20batches.png](https://github.com/wang-fujin/PINN4SOH/blob/main/6%20batches.png)
