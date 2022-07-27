#  Covid_19_Prediction_using_LSTM
 <3
 :wave:
 :raised_hand_with_fingers_splayed:
 [![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/)
![TensorFlow](https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)

### The Project's Aim 
To create a deep learning model using LSTM neural network to predict new cases (cases_new) in Malaysia using the past 30 days of number of cases.

## Data Source:
https://github.com/MoH-Malaysia/covid19-public

## Tensorboard Plot

![TensorBoard](https://user-images.githubusercontent.com/109563861/181297356-84e7ca11-375f-40e4-b2d9-7320072cfaf9.PNG)

From the Tensorboard graph:
>>    * From the graph it can be seen that underfit occure 
>>    *  the training loss is lower than the validation loss, and the validation loss has a trend that suggests further improvements are possible.
>>    * performance may be improved by increasing the number of training epochs OR  performance may be improved by increasing the capacity of the model, such as the number of memory cells in a hidden layer or number of hidden layers.

## Performance of the model Graph

![Figure 2022-07-27 235449](https://user-images.githubusercontent.com/109563861/181297486-c314d134-5f53-433c-b28d-8b082d588912.png)
![Figure 2022-07-27 235433](https://user-images.githubusercontent.com/109563861/181297549-baceb7f6-d403-462e-b892-4499f8b9e921.png)
![Figure 2022-07-27 235444](https://user-images.githubusercontent.com/109563861/181297576-34b0416e-26bc-44c1-9751-ef37400f0ec9.png)

## Performance of the model MAPE (𝑀𝑒𝑎𝑛 𝐴𝑏𝑠𝑜𝑙𝑢𝑡𝑒 𝑃𝑒𝑟𝑐𝑒𝑛𝑡𝑎𝑔𝑒 𝐸𝑟𝑟𝑜r)
![MAPE_error](https://user-images.githubusercontent.com/109563861/181297650-e898c6eb-eb57-4f81-93ac-4add788b355d.PNG)

From the model MAPE:
>>    * MAPE = 9 %

## Architecture of the Model
![model](https://user-images.githubusercontent.com/109563861/181297790-282e1165-f08c-4bb0-9937-a507838b83d3.png)
