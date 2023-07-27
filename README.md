# Counterfactual-Integrated_Gradients
A simple implementation of the counterfactual integrated gradients class and associated methods: WIP

This implementation provides a method to find the nearest counterfactual neighbour.
Then generate explanations using Integrated Gradients [1] between an input reference and counterfactual reference. 

![Example of Counterfactual-Integrated Gradients](https://github.com/jamie-duell/Counterfactual-Integrated_Gradients/edit/main/imgs/example_img.png?raw=true)

This can be extrapolated upon by using other counterfactual generative methods e.g. DiCE [2] or Wachter et. al. [3]. 
Once can then produce interactive visual explanations: 

![Example of Counterfactual-Integrated Gradients](https://github.com/jamie-duell/Counterfactual-Integrated_Gradients/edit/main/imgs/example_exp.png?raw=true)

REFERENCES: 

[1] M. Sundararajan, A. Taly, and Q. Yan, “Axiomatic attribution for deep networks,” ICML’17, pp. 3319–3328, 2017.

[2] R. K. Mothilal, A. Sharma, and C. Tan, “Explaining machine learning classifiers through diverse counterfactual explanations,” in Proceedings of the 2020 Conference on Fairness, Accountability, and Transparency, ser. FAT* ’20, Barcelona, Spain: Association for Computing Machinery, 2020, pp. 607–617, ISBN: 9781450369367. DOI: 10.1145/3351095.3372850. [Online]. Available: https://doi.org/10.1145/3351095.3372850.

[3] S. Wachter, B. Mittelstadt, and C. Russell, “Counterfactual explanations without opening the black box: Automated decisions and the gdpr,” 2017. DOI: 10.48550/ARXIV.1711.00399. [Online]. Available: https://arxiv.org/abs/1711.00399.
