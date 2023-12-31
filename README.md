# Counterfactual-Integrated_Gradients
## A simple implementation of the counterfactual integrated gradients method: Work In Progress (WIP)

This implementation provides a method to find the nearest counterfactual neighbour.
Then generate explanations using Integrated Gradients [1] between an input reference and counterfactual reference. 

![CF-IG_Example (1)](https://github.com/jamie-duell/Counterfactual-Integrated_Gradients/assets/22540396/ad2071ed-74d2-483b-9c3f-a553ffc23c6a)

This can be extrapolated upon by using other counterfactual generative methods e.g. DiCE [2] or Wachter et. al. [3]. 
Once can then produce interactive visual explanations: 

![CF_Lymph](https://github.com/jamie-duell/Counterfactual-Integrated_Gradients/assets/22540396/84edc940-068e-434e-8400-d6285cac9b07)

## Installation

### TO BE COMPLETED

## Dependencies

### TO BE COMPLETED

## Quick Guide (WIP) 

Instantiate an instance of the CounterfactualIG class to access the required methods with: 

`CF_IG = CounterfactualIG(pytorch_trained_model, data)` 

following this, one can currently utilise the following functions: <br />
<br />
`CF_IG.generate_nearest_CF_neighbour`;
<br />
`CF_IG.generate_counterfactuals`;
<br />
`CF_IG.plot_explanation`. 
<br /><br />
If you do not have access to a counterfactual instance from a generative method (e.g. DiCE), we recommend simply using the nearest counterfactual neighbour using `CF_IG.generate_nearest_CF_neighbour`. <br /> <br /> 
**note: we believe DiCE (from our knowledge) produces best results when counterfactual instances are used in concatenation with our `CF_IG.generate_counterfactuals` method.**

Currently the method can be executed with five parameters `CF_IG.generate_nearest_CF_neighbour(tensor_instance[input_index:input_index+1], input_index, df_columns_to_drop, categorical_features, ohe)`<br /><br />
Here: 
<br />
`tensor_instance` is the tensor instance that we wish to explain. <br /> `input_index` is the same as that used in the tensor_instance.<br /> `df_columns_to_drop` will drop any columns necessary (empty list otherwise).<br /> `categorical_features` for one hot encoding if `ohe=False` (aka. you have not yet done one hot encoding on your dataset, this will be carried out with a simple label encoder).
<br /><br /><br />
Using a conunterfactual reference point (possibly from the generate nearest CF neighbour method), one can obtain feature-attribution values using: `CF_IG.generate_counterfactuals`.
<br />
The method can be executed with 5 parameters: `CF_IG.generate_counterfactuals(tensor_instance[input_index:input_index+1], counterfactual_instance, _K, decision_boundary_proba)`,
<br /><br />Here:
<br />`_K = value` (`Default = 500`) is the number of steps in the Riemann approximation of the line integral. 
<br />`decision_boundary_proba` (`Default = 0.5`) is the probability at which the Riemann integral will terminate on a desired probability towards a class.

<br /><br />
Finally, the `CF_IG.plot_explanation` function generates an interactive feature attribution plot as shown in the images, where one can left click the bars in the bar plot and obtain the original and counterfactual reference feature values.

The method currently takes 3 parameters `CF_IG.plot_explanation(explanaation_array, tensor_instance, counterfactual_instance)`.<br /><br />Here:<br />`explanation array` is generated with the `CF_IG.generate_counterfactuals` function.




<br /><br /><br />

REFERENCES: 
<br />
[1] M. Sundararajan, A. Taly, and Q. Yan, “Axiomatic attribution for deep networks,” ICML’17, pp. 3319–3328, 2017.
<br />
[2] R. K. Mothilal, A. Sharma, and C. Tan, “Explaining machine learning classifiers through diverse counterfactual explanations,” in Proceedings of the 2020 Conference on Fairness, Accountability, and Transparency, ser. FAT* ’20, Barcelona, Spain: Association for Computing Machinery, 2020, pp. 607–617, ISBN: 9781450369367. DOI: 10.1145/3351095.3372850. [Online]. Available: https://doi.org/10.1145/3351095.3372850.
<br />
[3] S. Wachter, B. Mittelstadt, and C. Russell, “Counterfactual explanations without opening the black box: Automated decisions and the gdpr,” 2017. DOI: 10.48550/ARXIV.1711.00399. [Online]. Available: https://arxiv.org/abs/1711.00399.
